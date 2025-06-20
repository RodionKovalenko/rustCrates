use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_types::neural_network_generic::OperationMode,
    utils::activation::{softmax_complex_padding_real, softmax_last_row},
};

use super::gradient_struct::Gradient;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxLayer {
    learning_rate: f64,
    pub operation_mode: OperationMode,

    #[serde(skip)]
    softmax_output_batch: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    gradient: Option<Gradient>,
    #[serde(skip)]
    padding_mask_batch: Option<Vec<Vec<u32>>>,
}

impl SoftmaxLayer {
    pub fn new(learning_rate: f64, operation_mode: OperationMode) -> Self {
        Self {
            learning_rate,
            operation_mode,
            softmax_output_batch: None,
            input_batch: None,
            gradient: None,
            padding_mask_batch: None,
        }
    }
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, padding_mask_option: Option<Vec<Vec<u32>>>) -> Vec<Vec<Vec<f64>>> {
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        let padding_mask_batch = padding_mask_option.unwrap_or_else(|| vec![vec![1; seq_len]; batch_size]);
        self.padding_mask_batch = Some(padding_mask_batch);

        let layer_output: Vec<Vec<Vec<f64>>> = match self.operation_mode {
            OperationMode::PRODUCTION => {
                input_batch
                    .par_iter()
                    .map(|input| softmax_last_row(input)) // Apply `softmax_last_row` to each input
                    .collect() // Collect results into a Vec
            }
            OperationMode::TRAINING => input_batch
                .par_iter()
                .zip(self.padding_mask_batch.as_ref().unwrap().par_iter())
                .map(|(input, padding_mask_seq)| softmax_complex_padding_real(input, padding_mask_seq))
                .collect::<Vec<_>>(),
        };

        self.softmax_output_batch = Some(layer_output.clone());
        self.input_batch = Some(input_batch.clone());

        layer_output
    }

    pub fn backward(&mut self, target_token_ids: &Vec<Vec<u32>>) -> Gradient {
        let softmax_output_batch: &Vec<Vec<Vec<f64>>> = self.softmax_output_batch.as_ref().expect("Softmax output batch is missing in softmax layer");
        let _input_batch: &Vec<Vec<Vec<Complex<f64>>>> = self.input_batch.as_ref().expect("Input batch is missing in softmax layer");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Input batch is missing in softmax layer");

        let batch_size = softmax_output_batch.len();
        let seq_len = softmax_output_batch[0].len();
        let vocab_dim = softmax_output_batch[0][0].len();
        let mut softmax_gradient;

        let mut gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); vocab_dim]; seq_len]; batch_size];

        for (batch_index, (softmax_output, target_tokens)) in softmax_output_batch.iter().zip(target_token_ids.iter()).enumerate() {
            let padding_mask = &padding_mask_batch[batch_index];

            let mut _sequence_len_unpadded: usize = 0;
            for padding in padding_mask.iter() {
                if *padding != 0 {
                    _sequence_len_unpadded += 1;
                }
            }

            let target_len = target_tokens.len();
            // let seq_ind_start = _sequence_len_unpadded - target_len - 1;
            // let seq_end = _sequence_len_unpadded - 1;

            let seq_ind_start = softmax_output.len() - target_len - 1;
            let seq_end =  softmax_output.len() - 1;

            let mut target_len_unpadded = 0.0;
            for (_t, &target_class) in target_tokens.iter().enumerate() {
                if target_class != 1 {
                    target_len_unpadded += 1.0;
                }
            }

            let normalizer: f64 = batch_size as f64 * target_len_unpadded;

            for (t, &target_class) in target_tokens.iter().enumerate() {
                if target_class as usize >= vocab_dim {
                    panic!("Target token ID {} exceeds vocabulary dimension {}", target_class, vocab_dim);
                }

                if target_class == 1 {
                    continue; // Skip padding token
                }

                let seq_ind = seq_ind_start + t;

                if seq_ind >= seq_end {
                    break;
                }

                for (c, softmax_prob) in softmax_output[seq_ind].iter().enumerate() {
                    // for log softmax
                    // let prob = softmax_prob.exp();
                    // for softmax
                    //let prob = softmax_prob;

                    // let real_part_gradient = input_batch[batch_index][seq_ind][c].re / input_batch[batch_index][seq_ind][c].norm();
                    // let im_part_gradient = input_batch[batch_index][seq_ind][c].im / input_batch[batch_index][seq_ind][c].norm();

                    if target_class == c as u32 {
                        softmax_gradient = (softmax_prob - 1.0) / normalizer;
                    } else {
                        softmax_gradient = softmax_prob / normalizer;
                    };

                    gradient_batch[batch_index][seq_ind][c] += Complex::new(softmax_gradient, 0.0);
                }
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(gradient_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }
}
