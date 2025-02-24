use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_types::neural_network_generic::OperationMode,
    utils::activation::{softmax_complex_padding, softmax_last_row},
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

        // println!("------------------------------------------------------------------------------");
        // println!("input batch in softmax forward: {} {} {}", &input_batch.len(), &input_batch[0].len(), input_batch[0][0].len());
        // println!("input batch 0: {:?}", &input_batch[0][0][0..100]);
        // println!("------------------------------------------------------------------------------");

        let layer_output: Vec<Vec<Vec<Complex<f64>>>> = match self.operation_mode {
            OperationMode::PRODUCTION => {
                input_batch
                    .par_iter()
                    .map(|input| softmax_last_row(input)) // Apply `softmax_last_row` to each input
                    .collect() // Collect results into a Vec
            }
            OperationMode::TRAINING => { input_batch
                .par_iter()
                .zip(self.padding_mask_batch.as_ref().unwrap().par_iter())
                .map(|(input, padding_mask_seq)| softmax_complex_padding(input, padding_mask_seq))
                .collect::<Vec<_>>()
            }
        };

        self.softmax_output_batch = Some(layer_output.clone());

        // println!("------------------------------------------------------------------------------");
        // println!("output batch in softmax forward: {} {} {}", &layer_output.len(), &layer_output[0].len(), layer_output[0][0].len());
        // println!("ouptput batch 0: {:?}", &layer_output[0][0][0..100]);
        // println!("------------------------------------------------------------------------------");

        layer_output
    }

    pub fn backward(&mut self, target_token_ids: &Vec<Vec<u32>>) -> Gradient {
        // Ensure softmax_output_batch exists (precomputed during the forward pass)
        let softmax_output_batch: &Vec<Vec<Vec<Complex<f64>>>> = self.softmax_output_batch.as_ref().expect("Input batch is missing in softmax layer");
        let batch_size = softmax_output_batch.len();
        let seq_len = softmax_output_batch[0].len();
        let vocab_dim = softmax_output_batch[0][0].len();

        // Initialize gradient_batch with zeros
        let mut gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); vocab_dim]; seq_len]; batch_size];
        let normalizer = softmax_output_batch.len() * target_token_ids[0].len();

        // Iterate over the batch of softmax outputs and target token IDs
        for (batch_index, (softmax_output, target_tokens)) in softmax_output_batch.iter().zip(target_token_ids.iter()).enumerate() {
            let seq_len = softmax_output.len();
            let target_len = target_tokens.len();

            if target_len > seq_len {
                panic!("Target length exceeds sequence length!");
            }

            let seq_ind_start = seq_len - target_len;

            for (sample_index, softmax_sample) in softmax_output[seq_ind_start..seq_len].iter().enumerate() {
                for (column_index, softmax_prob) in softmax_sample.iter().enumerate() {
                    let target = if target_tokens[sample_index] == column_index as u32 { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 0.0) };

                    // Compute gradient
                    let gradient = softmax_prob - target;

                    // Store in batch-indexed gradient storage
                    gradient_batch[batch_index][sample_index][column_index] = gradient / (normalizer as f64);
                }
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(gradient_batch);
        self.gradient = Some(gradient.clone());

        // Return the final gradient_batch
        gradient
    }
}
