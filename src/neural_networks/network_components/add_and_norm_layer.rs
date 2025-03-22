use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::matrix::{add_matrix, check_nan_or_inf, check_nan_or_inf_3d, clip_gradients, is_nan_or_inf};

use super::gradient_struct::Gradient;

// LayerNorm (Normal Norm) Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalNormLayer {
    gamma: Vec<Complex<f64>>, // Learnable scaling parameter
    beta: Vec<Complex<f64>>,  // Learnable bias parameter
    epsilon: f64,
    learning_rate: f64,
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient: Option<Gradient>,
}

impl NormalNormLayer {
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![Complex::new(1.0, 0.0); feature_dim],
            beta: vec![Complex::new(0.0, 0.0); feature_dim],
            epsilon,
            learning_rate,
            input_batch: None,
            gradient: None,
        }
    }

    pub fn normalize(&self, input: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        let len = input.len() as f64;

        let mean: Complex<f64> = input.iter().sum::<Complex<f64>>() / len;

        let variance: Complex<f64> = input
            .iter()
            .map(|x| {
                let diff = *x - mean;
                diff * diff
            })
            .sum::<Complex<f64>>()
            / len;

        let stddev = (variance + self.epsilon).sqrt();

        input.iter().enumerate().map(|(i, x)| ((*x - mean) / stddev) * self.gamma[i] + self.beta[i]).collect()
    }

    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, input_before_transform_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let mut output_batch = Vec::new();
        let mut input_batch_added: Vec<Vec<Vec<Complex<f64>>>> = input_batch.clone();

        for (batch_ind, input) in input_batch.iter().enumerate() {
            let output = add_matrix(input, &input_before_transform_batch[batch_ind]);
            input_batch_added[batch_ind] = output.clone();

            output_batch.push(output.iter().map(|vec| self.normalize(vec)).collect());
        }

        self.input_batch = Some(input_batch_added.clone());
        output_batch
    }

    pub fn backward(&mut self, grad_output: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found in NormalNorm layer");

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let feature_dim = input_batch[0][0].len();
        let feature_dim_f64 = feature_dim as f64;
    
        let mut input_gradients = vec![vec![vec![Complex::new(0.0, 0.0); feature_dim]; seq_len]; batch_size];
        let mut gamma_gradients = vec![Complex::new(0.0, 0.0); feature_dim];
        let mut beta_gradients = vec![Complex::new(0.0, 0.0); feature_dim];
    
        for b in 0..batch_size {
            for s in 0..seq_len {
                let x = &input_batch[b][s];
                let dy = &grad_output[b][s];
    
                let mean: Complex<f64> = x.iter().sum::<Complex<f64>>() / feature_dim_f64;
                let variance: Complex<f64> = x.iter().map(|xi| (*xi - mean).powi(2)).sum::<Complex<f64>>() / feature_dim_f64;
                let stddev = (variance + self.epsilon).sqrt();
    
                let normalized: Vec<Complex<f64>> = x.iter().map(|xi| (*xi - mean) / stddev).collect();
    
                // Accumulate gradients w.r.t gamma and beta
                for i in 0..feature_dim {
                    gamma_gradients[i] += dy[i] * normalized[i];
                    beta_gradients[i] += dy[i];
                }
    
                // Intermediate terms
                let dx_hat: Vec<Complex<f64>> = dy.iter().enumerate().map(|(i, &dyi)| dyi * self.gamma[i]).collect();
                let dx_hat_sum: Complex<f64> = dx_hat.iter().sum::<Complex<f64>>();
                let norm_dx_hat_dot: Complex<f64> = normalized.iter().zip(dx_hat.iter()).map(|(normi, dxhati)| *normi * *dxhati).sum();
    
                for i in 0..feature_dim {
                    let term1 = dx_hat[i];
                    let term2 = dx_hat_sum / Complex::new(feature_dim_f64, 0.0);
                    let term3 = normalized[i] * norm_dx_hat_dot / Complex::new(feature_dim_f64, 0.0);
                    let dx = (term1 - term2 - term3) / stddev;
                    input_gradients[b][s][i] = dx;
                }
            }
        }
    
        check_nan_or_inf_3d(&mut input_gradients, "NormalNormLayer backward: NaN/Inf in input gradients");
    
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(input_gradients);
        gradient.set_gradient_gamma_batch(vec![gamma_gradients.clone()]);
        gradient.set_gradient_beta_batch(vec![beta_gradients.clone()]);
    
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &Gradient = self.gradient.as_ref().expect("No gradient found in NormalNormLayer");
        let mut gradient_gamma: Vec<Vec<Complex<f64>>> = gradient.get_gradient_gamma_batch();
        let mut gradient_beta: Vec<Vec<Complex<f64>>> = gradient.get_gradient_beta_batch(); // Get beta gradients

        let threshold = 1.0;
        clip_gradients(&mut gradient_gamma, threshold);
        clip_gradients(&mut gradient_beta, threshold); // Clip beta gradients

        check_nan_or_inf(&mut gradient_gamma, "check weight gradients in linear layer");
        check_nan_or_inf(&mut gradient_beta, "check beta gradients in NormalNormLayer");

        let batch_size = gradient_gamma.len() as f64;

        for batch_ind in 0..gradient_gamma.len() {
            for (i, value) in self.gamma.iter_mut().enumerate() {
                if !is_nan_or_inf(&gradient_gamma[batch_ind][i]) {
                    *value -= self.learning_rate * (gradient_gamma[batch_ind][i] / batch_size);
                }
            }

            for (i, value) in self.beta.iter_mut().enumerate() {
                if !is_nan_or_inf(&gradient_beta[batch_ind][i]) {
                    *value -= self.learning_rate * (gradient_beta[batch_ind][i] / batch_size);
                }
            }
        }

        self.learning_rate *= 0.99;
    }
}
