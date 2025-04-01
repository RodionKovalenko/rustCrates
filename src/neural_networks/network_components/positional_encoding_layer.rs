use crate::wavelet_transform::{dwt::transform_1_d, dwt_types::DiscreteWaletetType, modes::WaveletMode};

use super::gradient_struct::Gradient;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// Use smaller base instead of the original 10000
pub static INITIAL_BASE: f64 = 1000.0;

pub static SCALING_FAKTOR: f64 = 1.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionalEncodingLayer {
    pub embedding_dim: usize, // Store the embedding dimension
    pub base: f64,
    #[serde(skip)]
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
        let scaling_factor = SCALING_FAKTOR;

        input_batch
            .par_iter() // Parallel iterator for efficiency
            .map(|input| {
                let mut output = Vec::with_capacity(input.len());

                for (position, token_embeddings) in input.iter().enumerate() {
                    // Ensure correct embedding size
                    assert_eq!(token_embeddings.len(), self.embedding_dim, "All token embeddings must match the specified dimension.");

                    // Step 1: Convert complex embeddings into real & imaginary parts
                    let real_part: Vec<f64> = token_embeddings.iter().map(|c| c.re).collect();
                    let imag_part: Vec<f64> = token_embeddings.iter().map(|c| c.im).collect();

                    // Step 2: Apply wavelet transform separately to real & imaginary parts
                    let transformed_real = transform_1_d(&real_part, &DiscreteWaletetType::DB2, &WaveletMode::SYMMETRIC);
                    let transformed_imag = transform_1_d(&imag_part, &DiscreteWaletetType::DB2, &WaveletMode::SYMMETRIC);

                    // Step 3: Ensure wavelet-transformed outputs have correct dimensions
                    let positional_encoding = self.pad_or_trim_wavelet_output(&transformed_real, &transformed_imag);

                    // Step 4: Add wavelet-based positional encoding
                    let token_with_pos_encoding = self.add_positional_encoding(token_embeddings, &positional_encoding);

                    // Step 5: Apply rotary positional encoding
                    let rotated_embeddings = self.apply_rotary_positional_encoding(&token_with_pos_encoding, position, scaling_factor);
                    output.push(rotated_embeddings);
                }

                // let max = output.iter().flatten().max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));
                // let min = output.iter().flatten().min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));
                // println!("max value positional encoding: {:?}", &max);
                // println!("min value positional encoding: {:?}", &min);

                output
            })
            .collect() // Collect results into a single Vec
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let mut gradient = Gradient::new_default();

        // Use reference to avoid unnecessary cloning
        gradient.set_gradient_input_batch(previous_gradient_batch.clone());
        self.gradient = Some(gradient.clone());

        gradient
    }

    /// Ensures wavelet output matches `embedding_dim`
    fn pad_or_trim_wavelet_output(&self, real: &[f64], imag: &[f64]) -> Vec<Complex<f64>> {
        let mut wavelet_output = Vec::with_capacity(self.embedding_dim);

        for i in 0..self.embedding_dim {
            let real_val = real.get(i).copied().unwrap_or(0.0);
            let imag_val = imag.get(i).copied().unwrap_or(0.0);
            wavelet_output.push(Complex::new(real_val, imag_val));
        }

        wavelet_output
    }

    /// Adds wavelet-based positional encoding to token embeddings
    fn add_positional_encoding(&self, token_embeddings: &Vec<Complex<f64>>, positional_encoding: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        token_embeddings.iter().zip(positional_encoding.iter()).map(|(embedding, pos_encoding)| Complex::new(embedding.re + pos_encoding.re, embedding.im + pos_encoding.im)).collect()
    }

    /// Applies rotary positional encoding to a single token embedding
    fn apply_rotary_positional_encoding(&self, embedding: &[Complex<f64>], position: usize, scaling_factor: f64) -> Vec<Complex<f64>> {
        assert_eq!(embedding.len(), self.embedding_dim);
        assert_eq!(self.embedding_dim % 2, 0, "Embedding dimension must be even for RoPE.");

        let mut rotated_embedding = Vec::with_capacity(self.embedding_dim);
        let half_dim = self.embedding_dim / 2;

        for i in 0..half_dim {
            let mut theta = position as f64 / ((self.base * scaling_factor).powf(2.0 * i as f64 / self.embedding_dim as f64));

            // ✅ Ensure theta is between [-1.0, 1.0]
            theta = theta.clamp(-1.0, 1.0);

            if theta.abs() > 1.0 {
                println!("theta is out of bounds: {}", theta);
            }

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
