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

        // Compute the RMS of the input vector
        // norm_sqr() => re^2 + im^2
        let mean_square = input.iter().map(|x| x.norm_sqr()).sum::<f64>() / input.len() as f64;
        let rms = (mean_square + self.epsilon).sqrt();

        // Normalize the input and apply the learned gamma scaling
        input
            .iter()
            .zip(self.gamma.iter())
            .map(|(x, &g)| (*x / Complex::new(rms, 0.0)) * Complex::new(g, 0.0))
            .collect()
    }

    // Forward pass for a batch of token embeddings (2D input)
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, input_before_transform_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch = Some(input_batch.clone());
        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

        for (batch_ind, input) in input_batch.iter().enumerate() {
            let output = add_matrix(input, &input_before_transform_batch[batch_ind]);

            output_batch.push(
                output
                    .iter()
                    .map(|vec| self.rms_norm(vec)) // Normalize each token embedding
                    .collect(),
            );
        }

        output_batch
    }

    pub fn backward(&mut self, gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        // Retrieve the input batch from the forward pass
        let input_batch = self.input_batch.as_ref().expect("Input batch not found in RMSNorm layer");

        println!("RMS Norm backward: input batch shape: {}, {}, {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());
        println!("RMS Norm backward: gradient shape: {}, {}", gradient.len(), gradient[0].len());

        // Initialize a container for the input gradients with the same shape as the input batch
        let mut input_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()];

        // Iterate through the input batch (3D: [batch_size, sequence_length, feature_dim])
        for (seq_ind, input_sequence) in input_batch.iter().enumerate() {
            // Iterate over each token embedding (2D: [sequence_length, feature_dim])
            for (input_vec, grad_vec) in input_sequence.iter().zip(gradient.iter()) {
                let input_len = input_vec.len() as f64;

                // Compute mean square and RMS for the current token embedding
                let mean_square = input_vec.iter().map(|x| x.norm_sqr()).sum::<f64>() / input_len;
                let rms = (mean_square + self.epsilon).sqrt();

                // Compute gradient for gamma (averaged across batch and sequence dimensions)
                let gamma_grad: Vec<f64> = input_vec
                    .iter()
                    .zip(grad_vec.iter())
                    .map(|(input, grad)| (grad * input.conj()).re / rms)
                    .map(|g| g / input_batch.len() as f64) // Normalize by batch size
                    .collect();

                // Update gamma using gradient descent
                for (g, g_grad) in self.gamma.iter_mut().zip(gamma_grad.iter()) {
                    *g -= self.learning_rate * g_grad;
                }

                // Compute gradient for the input vector (and apply normalization)
                for (i, x) in input_vec.iter().enumerate() {
                    let norm_factor = *x / Complex::new(rms, 0.0);
                    let sum_grad = grad_vec.iter().zip(input_vec.iter()).map(|(grad, x_j)| grad * x_j.conj()).sum::<Complex<f64>>().re / (input_len * rms);

                    // Adjust the gradient with normalization
                    let grad_adjustment = grad_vec[i] * norm_factor - Complex::new(sum_grad, 0.0) * x / Complex::new(rms, 0.0);

                    // Update the input gradients for this sequence and dimension
                    input_gradients[seq_ind][i] += grad_adjustment;
                }
            }
        }

        input_gradients
    }
}
