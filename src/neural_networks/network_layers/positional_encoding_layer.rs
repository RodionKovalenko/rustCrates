use crate::neural_networks::network_components::gradient_struct::Gradient;
use crate::neural_networks::network_components::layer_input_struct::LayerInput;
use crate::neural_networks::utils::dtype::{r, C, Real, ZERO};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// Use smaller base instead of the original 10000
// maintaining the proportion 10000/512 = 1250/64
pub static INITIAL_BASE: f64 = 1250.0;

pub static SCALING_FAKTOR: f64 = 1.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionalEncodingLayer {
    pub embedding_dim: usize, // Store the embedding dimension
    pub base: f64,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
}

impl PositionalEncodingLayer {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            base: INITIAL_BASE,
            gradient: None,
            input_batch: None,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> Vec<Vec<Vec<C>>> {
        let input_batch = layer_input
            .get_input_batch_ref()
            .expect("PositionalEncodingLayer: input batch missing");

        let scaling_factor = SCALING_FAKTOR;
        let forward_only = layer_input.get_forward_only();

        if layer_input.get_calculate_gradient() {
            self.input_batch = Some(input_batch.to_vec());
        } else {
            self.input_batch = None;
        }

        input_batch
            .par_iter()
            .map(|sequence| {
                sequence
                    .iter()
                    .enumerate()
                    .map(|(position, token_embeddings)| {
                        assert_eq!(token_embeddings.len(), self.embedding_dim, "All token embeddings must match the specified dimension.");
                        let time_step = if forward_only && layer_input.get_time_step() > 0 {
                            layer_input.get_time_step()
                        } else {
                            position
                        };
                        self.apply_rotary_positional_encoding(token_embeddings, time_step, scaling_factor)
                    })
                    .collect::<Vec<Vec<C>>>()
            })
            .collect()
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        if self.input_batch.is_none() {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let input_batch = self
            .input_batch
            .as_ref()
            .expect("Input batch is missing in positional encoding layer");
        assert_eq!(input_batch.len(), previous_gradient_batch.len(), "Batch size mismatch");

        assert_eq!(self.embedding_dim % 2, 0, "Embedding dimension must be even for RoPE.");
        let half_dim = self.embedding_dim / 2;

        let input_gradient_batch: Vec<Vec<Vec<C>>> = input_batch
            .par_iter()
            .zip(previous_gradient_batch.par_iter())
            .map(|(input_sequence, grad_sequence)| {
                assert_eq!(input_sequence.len(), grad_sequence.len(), "Sequence length mismatch");

                grad_sequence
                    .iter()
                    .enumerate()
                    .map(|(position, grad_embedding)| {
                        assert_eq!(grad_embedding.len(), self.embedding_dim);
                        let mut rotated_grad = vec![C::new(ZERO, ZERO); self.embedding_dim];

                        for i in 0..half_dim {
                            let even_idx = 2 * i;
                            let odd_idx = even_idx + 1;

                            let mut theta = position as f64
                                / ((self.base * SCALING_FAKTOR).powf(2.0 * i as f64 / self.embedding_dim as f64));
                            theta = theta.clamp(-1.0, 1.0);
                            let (sin_theta_f64, cos_theta_f64) = theta.sin_cos();
                            let sin_theta: Real = r(sin_theta_f64);
                            let cos_theta: Real = r(cos_theta_f64);

                            let grad_even = grad_embedding[even_idx];
                            let grad_odd = grad_embedding[odd_idx];

                            rotated_grad[even_idx] =
                                C::new(grad_even.re * cos_theta + grad_odd.re * sin_theta, grad_even.im * cos_theta + grad_odd.im * sin_theta);
                            rotated_grad[odd_idx] =
                                C::new(-grad_even.re * sin_theta + grad_odd.re * cos_theta, -grad_even.im * sin_theta + grad_odd.im * cos_theta);
                        }

                        rotated_grad
                    })
                    .collect()
            })
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(input_gradient_batch);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn pad_or_trim_wavelet_output(&self, real: &[f64], imag: &[f64]) -> Vec<C> {
        let mut wavelet_output = Vec::with_capacity(self.embedding_dim);
        for i in 0..self.embedding_dim {
            let real_val = real.get(i).copied().unwrap_or(0.0);
            let imag_val = imag.get(i).copied().unwrap_or(0.0);
            wavelet_output.push(C::new(r(real_val), r(imag_val)));
        }
        wavelet_output
    }

    pub fn add_positional_encoding(&self, token_embeddings: &[C], positional_encoding: &[C]) -> Vec<C> {
        assert_eq!(token_embeddings.len(), positional_encoding.len());
        (0..token_embeddings.len())
            .map(|i| token_embeddings[i] + positional_encoding[i])
            .collect()
    }

    pub fn apply_rotary_positional_encoding(&self, embedding: &[C], position: usize, scaling_factor: f64) -> Vec<C> {
        assert_eq!(embedding.len(), self.embedding_dim);
        assert_eq!(self.embedding_dim % 2, 0, "Embedding dimension must be even for RoPE.");

        let mut rotated_embedding = Vec::with_capacity(self.embedding_dim);
        let half_dim = self.embedding_dim / 2;

        for i in 0..half_dim {
            let mut theta = position as f64
                / ((self.base * scaling_factor).powf(2.0 * i as f64 / self.embedding_dim as f64));
            theta = theta.clamp(-1.0, 1.0);

            let (sin_theta_f64, cos_theta_f64) = theta.sin_cos();
            let sin_theta: Real = r(sin_theta_f64);
            let cos_theta: Real = r(cos_theta_f64);

            let even_idx = 2 * i;
            let odd_idx = even_idx + 1;
            let even = embedding[even_idx];
            let odd = embedding[odd_idx];

            rotated_embedding.push(C::new(even.re * cos_theta - odd.re * sin_theta, even.im * cos_theta - odd.im * sin_theta));
            rotated_embedding.push(C::new(even.re * sin_theta + odd.re * cos_theta, even.im * sin_theta + odd.im * cos_theta));
        }

        rotated_embedding
    }
}
