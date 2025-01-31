use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_types::neural_network_generic::OperationMode,
    utils::activation::{softmax_complex, softmax_last_row},
};

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxLayer {
    learning_rate: f64,
    operation_mode: OperationMode,
    softmax_output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
}

impl SoftmaxLayer {
    pub fn new(learning_rate: f64, operation_mode: OperationMode) -> Self {
        Self {
            learning_rate,
            operation_mode,
            softmax_output_batch: None,
        }
    }
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let layer_output: Vec<Vec<Vec<Complex<f64>>>> = match self.operation_mode {
            OperationMode::PRODUCTION => {
                input_batch
                    .par_iter() // Parallel iterator for the input batch
                    .map(|input| softmax_last_row(input)) // Apply `softmax_last_row` to each input
                    .collect() // Collect results into a Vec
            }
            OperationMode::TRAINING => {
                input_batch
                    .par_iter() // Parallel iterator for the input batch
                    .map(|input| softmax_complex(input)) // Apply `softmax_complex` to each input
                    .collect() // Collect results into a Vec
            }
        };

        self.softmax_output_batch = Some(layer_output.clone());

        layer_output
    }

    pub fn backward_batch(&self, target_token_ids: &Vec<Vec<u32>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        // Ensure softmax_output_batch exists (precomputed during the forward pass)
        let softmax_output_batch: &Vec<Vec<Vec<Complex<f64>>>> = self.softmax_output_batch.as_ref().expect("Input batch is missing in softmax layer");
        let batch_size = softmax_output_batch.len();
        let seq_len = softmax_output_batch[0].len();
        let vocab_dim = softmax_output_batch[0][0].len();

        // Initialize gradient_batch with zeros
        let mut gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); vocab_dim]; seq_len]; batch_size];

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
                    let target = if target_tokens[sample_index] == column_index as u32 {
                        Complex::new(1.0, 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };

                    if target.norm() == 1.0 {
                        println!("target is 1: {}", column_index);
                    }

                    // Compute gradient
                    let gradient = softmax_prob - target;

                    // Store in batch-indexed gradient storage
                    gradient_batch[batch_index][sample_index][column_index] = gradient;
                }
            }
        }

        // Return the final gradient_batch
        gradient_batch
    }

    pub fn backward(&self, target_token_ids: &Vec<Vec<u32>>) -> Vec<Vec<Complex<f64>>> {
        // Ensure softmax_output_batch exists (precomputed during the forward pass)
        let softmax_output_batch: &Vec<Vec<Vec<Complex<f64>>>> = self.softmax_output_batch.as_ref().expect("Input batch is missing in softmax layer");
        let seq_len = softmax_output_batch[0].len();
        let vocab_dim = softmax_output_batch[0][0].len();

        // Initialize gradient_batch with zeros
        let mut gradient_batch: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); vocab_dim]; seq_len];

        // Iterate over the batch of softmax outputs and target token IDs in serial
        for (softmax_output, target_tokens) in softmax_output_batch.iter().zip(target_token_ids.iter()) {
            let seq_len = softmax_output.len();
            let target_len: usize = target_tokens.len();
            // println!("softmax output dim: {}, {}", softmax_output.len(), softmax_output[0].len());
            // println!("target_tokens dim: {}", target_tokens.len());

            let seq_ind_start = seq_len - target_len;

            for (sample_index, softmax_sample) in softmax_output[seq_ind_start..seq_len].iter().enumerate() {
                for (column_index, softmax_prob) in softmax_sample.iter().enumerate() {
                    let target = if target_tokens[sample_index] == column_index as u32 {
                        Complex::new(1.0, 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };

                    // Compute the gradient for this position
                    let gradient = softmax_prob - target;

                    if target.norm() == 1.0 {
                        println!("target is 1: {}", sample_index);
                    }

                    // Accumulate the gradients directly into the gradient_batch
                    gradient_batch[sample_index][column_index] += gradient;
                }
            }
        }

        // Return the final gradient_batch
        gradient_batch
    }
}
