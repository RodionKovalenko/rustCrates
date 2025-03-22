use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::matrix::{add_matrix, check_nan_or_inf, clip_gradients, is_nan_or_inf};

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
        let input_batch = self.input_batch.as_ref().expect("Input batch not found");
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let feature_dim = input_batch[0][0].len();
        let d = feature_dim as f64;
    
        let mut input_grads = vec![vec![vec![Complex::new(0.0, 0.0); feature_dim]; seq_len]; batch_size];
        let mut gamma_grad = vec![Complex::new(0.0, 0.0); feature_dim];
        let mut beta_grad = vec![Complex::new(0.0, 0.0); feature_dim];
    
        // Iterate over each batch and sequence
        for b in 0..batch_size {
            for s in 0..seq_len {
                let x = &input_batch[b][s];
                let dy = &grad_output[b][s];
    
                // Compute the mean (mean of real and imaginary separately)
                let mean = x.iter().sum::<Complex<f64>>() / d;
                let centered: Vec<_> = x.iter().map(|xi| xi - mean).collect();
    
                // Compute variance and standard deviation
                let var = centered.iter().map(|c| c.norm_sqr()).sum::<f64>() / d;
                let std_dev = (var + self.epsilon).sqrt();
                let inv_std_dev = 1.0 / std_dev;
    
                // Compute normalized values
                let x_hat: Vec<_> = centered.iter().map(|c| c * inv_std_dev).collect();
    
                // Compute gamma and beta gradients
                for i in 0..feature_dim {
                    gamma_grad[i] += dy[i] * x_hat[i]; // Sum of (dy * x_hat) for gamma
                    beta_grad[i] += dy[i]; // Sum of dy for beta
                }
    
                // Compute intermediate values for input gradients
                let dy_gamma: Vec<_> = dy.iter().enumerate().map(|(i, &dyi)| dyi * self.gamma[i]).collect();
                let sum_dy_gamma = dy_gamma.iter().sum::<Complex<f64>>(); // sum(dy * gamma)
                let sum_dy_gamma_xhat: Complex<f64> = dy_gamma.iter().zip(&x_hat).map(|(&dg, &xh)| dg * xh).sum();
    
                // Compute input gradients (dx_hat)
                for i in 0..feature_dim {
                    let dx_hat = dy_gamma[i] - (sum_dy_gamma + x_hat[i] * sum_dy_gamma_xhat) / d;
                    input_grads[b][s][i] = dx_hat * inv_std_dev;
                }
            }
        }
    
        // Create a new Gradient object and set the gradients
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(input_grads);
        gradient.set_gradient_gamma(gamma_grad);
        gradient.set_gradient_beta(beta_grad);
    
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
