use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_types::neural_network_generic::OperationMode,
    utils::{
        activation::{softmax_complex_padding, softmax_last_row},
        matrix::{check_nan_or_inf_3d, is_nan_or_inf},
    },
};

use super::gradient_struct::Gradient;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxLayer {
    learning_rate: f64,
    operation_mode: OperationMode,
    softmax_output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient: Option<Gradient>,
    padding_mask_batch: Option<Vec<Vec<u32>>>,
}

impl SoftmaxLayer {
    pub fn new(learning_rate: f64, operation_mode: OperationMode) -> Self {
        Self {
            learning_rate,
            operation_mode,
            softmax_output_batch: None,
            gradient: None,
            padding_mask_batch: None,
        }
    }
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, padding_mask_option: Option<Vec<Vec<u32>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        let padding_mask_batch = padding_mask_option.unwrap_or_else(|| vec![vec![1; seq_len]; batch_size]);
        self.padding_mask_batch = Some(padding_mask_batch);

        let layer_output: Vec<Vec<Vec<Complex<f64>>>> = match self.operation_mode {
            OperationMode::PRODUCTION => {
                input_batch
                    .par_iter()
                    .map(|input| softmax_last_row(input)) // Apply `softmax_last_row` to each input
                    .collect() // Collect results into a Vec
            }
            OperationMode::TRAINING => input_batch
                .par_iter()
                .zip(self.padding_mask_batch.as_ref().unwrap().par_iter())
                .map(|(input, padding_mask_seq)| softmax_complex_padding(input, padding_mask_seq))
                .collect::<Vec<_>>(),
        };

        self.softmax_output_batch = Some(layer_output.clone());

        layer_output
    }

    pub fn backward(&mut self, target_token_ids: &Vec<Vec<u32>>) -> Gradient {
        let softmax_output_batch: &Vec<Vec<Vec<Complex<f64>>>> = self.softmax_output_batch.as_ref().expect("Input batch is missing in softmax layer");

        let batch_size = softmax_output_batch.len();
        let seq_len = softmax_output_batch[0].len();
        let vocab_dim = softmax_output_batch[0][0].len();

        let mut gradient_batch = vec![vec![vec![Complex::new(0.0, 0.0); vocab_dim]; seq_len]; batch_size];
        let normalizer = (seq_len * batch_size) as f64;

        for (batch_index, (softmax_output, target_tokens)) in softmax_output_batch.iter().zip(target_token_ids.iter()).enumerate() {
            let target_len = target_tokens.len();
            let seq_ind_start = seq_len - target_len;

            for (target_ind, &token_id) in target_tokens.iter().enumerate() {
                if token_id as usize >= vocab_dim {
                    panic!("Target token ID {} exceeds vocabulary dimension {}", token_id, vocab_dim);
                }

                if token_id == 1 {
                    continue; // Skip padding token
                }

                let seq_ind = seq_ind_start + target_ind;

                for (column_index, softmax_prob) in softmax_output[seq_ind].iter().enumerate() {
                    let target = if token_id == column_index as u32 { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 0.0) };
                    let gradient = softmax_prob - target; // ✅ Compute gradient consistently

                    if is_nan_or_inf(&gradient) {
                        panic!("Gradient is not valid: {:?}", &gradient);
                    }

                    gradient_batch[batch_index][seq_ind][column_index] += gradient / normalizer;
                }
            }
        }

        check_nan_or_inf_3d(&mut gradient_batch, "gradient input batch in softmax output layer has None Values");
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(gradient_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }
}
