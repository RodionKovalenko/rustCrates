use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer {
    gamma: Vec<f64>,      // Learnable scaling parameter (for each feature)
    epsilon: f64,         // Small constant for numerical stability
    learning_rate: f64,   // Learning rate for gamma updates
}

impl RMSNormLayer {
    // Initialize the RMSNorm layer with a given feature dimension (e.g., 16 for each token embedding)
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![1.0; feature_dim], // Initialize gamma to 1.0 for all features
            epsilon,
            learning_rate,
        }
    }

    // RMSNorm function that works on a single token embedding (vector of Complex<f64>)
    pub fn rms_norm(&self, input: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        if input.is_empty() {
            panic!("Input to RMSNorm cannot be empty");
        }

        // Compute the RMS of the input vector
        let mean_square = input.iter().map(|x| x.norm_sqr()).sum::<f64>() / input.len() as f64;
        let rms = (mean_square + self.epsilon).sqrt();

        // Normalize the input and apply the learned gamma scaling
        input.iter()
            .zip(self.gamma.iter())
            .map(|(x, &g)| (*x / Complex::new(rms, 0.0)) * Complex::new(g, 0.0))
            .collect()
    }
}

impl RMSNormLayer {
    // Forward pass for a batch of token embeddings (2D input)
    pub fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        // Apply RMSNorm for each token (each row in the input)
        input.iter()
            .map(|vec| self.rms_norm(vec)) // Normalize each token embedding
            .collect()
    }

    // Backward pass to compute gradients for input and gamma
    pub fn backward(&mut self, gradient: &Vec<Vec<Complex<f64>>>, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let mut input_gradients = Vec::new(); // Store gradients for each token embedding

        for (grad_vec, input_vec) in gradient.iter().zip(input.iter()) {
            let input_len = input_vec.len() as f64;

            // Compute mean square and RMS for the input token
            let mean_square = input_vec.iter().map(|x| x.norm_sqr()).sum::<f64>() / input_len;
            let rms = (mean_square + self.epsilon).sqrt();

            // Compute gradient for gamma (scaled by norm squared)
            let gamma_grad: Vec<f64> = grad_vec.iter()
                .zip(input_vec.iter())
                .map(|(grad, input)| (grad * input.conj()).re / rms)
                .collect();

            // Update gamma parameters using gradient descent
            for (g, g_grad) in self.gamma.iter_mut().zip(gamma_grad.iter()) {
                *g -= self.learning_rate * g_grad; // Update gamma with gradient
            }

            // Compute gradient for the input token (gradient of the normalized values)
            let input_grad: Vec<Complex<f64>> = input_vec.iter()
                .enumerate()
                .map(|(i, x)| {
                    // Compute the partial derivatives of the RMS normalization
                    let norm_factor = *x / Complex::new(rms, 0.0);
                    let sum_grad = grad_vec.iter()
                        .zip(input_vec.iter())
                        .map(|(grad, x_j)| grad * x_j.conj())
                        .sum::<Complex<f64>>()
                        .re / (input_len * rms);

                    grad_vec[i] * norm_factor - Complex::new(sum_grad, 0.0) * x / Complex::new(rms, 0.0)
                })
                .collect();

            // Add the token's gradients to the batch of gradients
            input_gradients.push(input_grad);
        }

        input_gradients
    }
}
