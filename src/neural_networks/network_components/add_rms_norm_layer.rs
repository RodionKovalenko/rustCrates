use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::matrix::add_matrix;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer {
    gamma: Vec<f64>,    // Learnable scaling parameter (for each feature)
    epsilon: f64,       // Small constant for numerical stability
    learning_rate: f64, // Learning rate for gamma updates
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
}

impl RMSNormLayer {
    // Initialize the RMSNorm layer with a given feature dimension (e.g., 16 for each token embedding)
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![1.0; feature_dim], // Initialize gamma to 1.0 for all features
            epsilon,
            learning_rate,
            input_batch: None,
        }
    }

    // RMSNorm function that works on a single token embedding (vector of Complex<f64>)
    pub fn rms_norm(&self, input: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        if input.is_empty() {
            panic!("Input to RMSNorm cannot be empty");
        }

        let rms = self.rms(input);

        // Normalize the input and apply the learned gamma scaling
        input.iter().zip(self.gamma.iter()).map(|(x, &g)| (*x / rms * g)).collect()
    }

    pub fn rms(&self, input: &Vec<Complex<f64>>) -> Complex<f64> {
        let mean_square = input.iter().map(|x| x * x).sum::<Complex<f64>>() / input.len() as f64;
        let rms = (mean_square + self.epsilon).sqrt();

        rms
    }

    // Forward pass for a batch of token embeddings (2D input)
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, input_before_transform_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch = Some(input_batch.clone());
        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

        for (batch_ind, input) in input_batch.iter().enumerate() {
            let output = add_matrix(input, &input_before_transform_batch[batch_ind]);
            // let output = input.clone();

            output_batch.push(
                output
                    .iter()
                    .map(|vec| self.rms_norm(vec)) // Normalize each token embedding
                    .collect(),
            );
        }

        output_batch
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found in RMSNorm layer");

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let dim_len = input_batch[0][0].len();

        let mut input_batch_gradients = vec![vec![vec![Complex::new(0.0, 0.0); dim_len]; seq_len]; batch_size];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let rms =  self.rms(&input_batch[b][s]);
                // Compute gradients
                for d_i in 0..dim_len {
                    // Applying RMSNorm backward formula with gamma scaling
                    for d_j in 0..dim_len {
                        if d_i == d_j {
                            let grad = (1.0 / rms) - ((input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (dim_len as f64 * rms.powf(3.0)));
                            input_batch_gradients[b][s][d_j] += grad;
                        } else {
                            let grad = Complex::new(0.0, 0.0) - ((input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (dim_len as f64 * rms.powf(3.0)));
                            input_batch_gradients[b][s][d_j] += grad;
                        }
                    }

                    input_batch_gradients[b][s][d_i] *= previous_gradient_batch[b][s][d_i];
                }
            }
        }

        input_batch_gradients
    }
}
