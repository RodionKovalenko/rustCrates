use super::gradient_struct::Gradient;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::E;

// Use smaller base instead of the original 10000
pub static INITIAL_BASE: f64 = 10000.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionalEncodingLayer {
    pub embedding_dim: usize, // Store the embedding dimension
    pub base: f64,
    pub gradient: Option<Gradient>,
    pub log_scaling_factor: f64,
}

impl PositionalEncodingLayer {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            base: INITIAL_BASE,
            gradient: None,
            log_scaling_factor: -4.5,
        }
    }

    /// Apply positional encoding to a batch of embeddings
    pub fn forward(&self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let scaling_factor = self.get_scaling_factor();

        input_batch
            .par_iter() // Parallel iterator for the outer loop
            .map(|input| {
                let mut output = Vec::with_capacity(input.len());

                for (position, token_embeddings) in input.iter().enumerate() {
                    // Ensure all embeddings have the correct dimension
                    assert_eq!(token_embeddings.len(), self.embedding_dim, "All token embeddings must have the same dimension as specified in the layer.");

                    // Step 1: Add positional encodings to token embeddings
                    let positional_encoding = self.generate_positional_encoding(position, scaling_factor);
                    let token_with_pos_encoding = self.add_positional_encoding(token_embeddings, &positional_encoding);

                    // Step 2: Apply rotary positional encoding
                    let rotated_embeddings = self.apply_rotary_positional_encoding(&token_with_pos_encoding, position, scaling_factor);
                    output.push(rotated_embeddings);
                }

                output
            })
            .collect() // Collect the results from all parallel computations into a single Vec
    }

    pub fn get_scaling_factor(&self) -> f64 {
        let clamped_log_s = self.log_scaling_factor.clamp(-6.9077, -1.609); // Ensures scaling factor is between 10 and 20000
        E.powf(clamped_log_s)
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let mut gradient = Gradient::new_default();

        // Use reference to avoid unnecessary cloning
        gradient.set_gradient_input_batch(previous_gradient_batch.clone());
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self, _learning_rate: f64) {
        // let gradient = self.gradient.as_ref().expect("No gradient found in positional encoding layer");
        // let previous_gradient_batch = gradient.get_gradient_input_batch();
        // let mut total_gradient: f64 = 0.0;
        // let mut count = 0;

        // // Iterate through the batch
        // for (_seq_idx, sequence) in previous_gradient_batch.iter().enumerate() {
        //     for (pos, token_gradients) in sequence.iter().enumerate() {
        //         for (dim_idx, gradient) in token_gradients.iter().enumerate() {
        //             // Compute influence of base on gradients (e.g., positional encoding derivative)
        //             let theta = pos as f64 / self.base.powf(2.0 * (dim_idx as f64 / self.embedding_dim as f64));
        //             let (sin_theta, cos_theta) = theta.sin_cos();

        //             // Compute gradient approximation
        //             let grad_real = gradient.re * cos_theta + gradient.im * sin_theta;
        //             let grad_imag = gradient.im * cos_theta - gradient.re * sin_theta;

        //             // Aggregate the gradient information
        //             total_gradient += grad_real.abs() + grad_imag.abs();
        //             count += 1;
        //         }
        //     }
        // }

        // // Compute mean gradient contribution
        // if count > 0 {
        //     let avg_gradient = total_gradient / count as f64;

        //     // Update base using simple gradient descent
        //     self.log_scaling_factor -= learning_rate * avg_gradient;
        // }
    }

    /// Function to generate a positional encoding (for simplicity, we use a fixed formula here)
    fn generate_positional_encoding(&self, position: usize, scaling_factor: f64) -> Vec<f64> {
        let mut positional_encoding = vec![0.0; self.embedding_dim];

        for i in 0..self.embedding_dim {
            let angle = position as f64 / ((self.base * scaling_factor).powf(2.0 * (i as f64 / 2.0) / self.embedding_dim as f64));

            if i % 2 == 0 {
                positional_encoding[i] = angle.sin();
            } else {
                positional_encoding[i] = angle.cos();
            }
        }

        positional_encoding
    }

    /// Function to add positional encoding to token embeddings
    fn add_positional_encoding(&self, token_embeddings: &Vec<Complex<f64>>, positional_encoding: &Vec<f64>) -> Vec<Complex<f64>> {
        token_embeddings
            .iter()
            .zip(positional_encoding.iter())
            .map(|(embedding, pos_encoding)| {
                // Add positional encoding to the real part of the complex embedding (you can also apply to the imaginary part)
                Complex::new(embedding.re + pos_encoding, embedding.im)
            })
            .collect()
    }

    /// Function to apply rotary positional encoding to a single token's embedding
    fn apply_rotary_positional_encoding(&self, embedding: &[Complex<f64>], position: usize, scaling_factor: f64) -> Vec<Complex<f64>> {
        assert_eq!(embedding.len(), self.embedding_dim);
        assert_eq!(self.embedding_dim % 2, 0);

        let mut rotated_embedding = Vec::with_capacity(self.embedding_dim);
        let half_dim = self.embedding_dim / 2;

        for i in 0..half_dim {
            let theta = position as f64 / ((self.base * scaling_factor).powf(2.0 * i as f64 / self.embedding_dim as f64));
            let (sin_theta, cos_theta) = theta.sin_cos();

            let even_idx = 2 * i;
            let odd_idx = even_idx + 1;
            let even = embedding[even_idx];
            let odd = embedding[odd_idx];

            rotated_embedding.push(Complex::new(even.re * cos_theta - odd.re * sin_theta, even.im * cos_theta - odd.im * sin_theta));
            rotated_embedding.push(Complex::new(even.re * sin_theta + odd.re * cos_theta, even.im * sin_theta + odd.im * cos_theta));
        }

        rotated_embedding
    }
}
