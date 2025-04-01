use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::calculate_adam_w_bias,
    matrix::{clip_gradient_1d, is_nan_or_inf},
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

// LayerNorm (Normal Norm) Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalNormLayer {
    gamma: Vec<Complex<f64>>, // Learnable scaling parameter
    beta: Vec<Complex<f64>>,  // Learnable bias parameter
    epsilon: f64,
    pub learning_rate: f64,

    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    gradient: Option<Gradient>,
    #[serde(skip)]
    previous_gradient: Option<Gradient>,
    time_step: usize,
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
            previous_gradient: None,
            time_step: 0,
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

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: &Vec<Vec<Vec<Complex<f64>>>> = &layer_input.get_input_batch();
        let mut output_batch = Vec::new();

        for input in input_batch.iter() {
            output_batch.push(input.iter().map(|vec| self.normalize(vec)).collect());
        }

        self.input_batch = Some(output_batch.clone());
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
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

        for b in 0..batch_size {
            for s in 0..seq_len {
                let input = &input_batch[b][s];
                let grad_out = &grad_output[b][s];

                let mean: Complex<f64> = input.iter().sum::<Complex<f64>>() / d;
                let variance: Complex<f64> = input.iter().map(|x| (*x - mean) * (*x - mean)).sum::<Complex<f64>>() / d;
                let stddev = (variance + self.epsilon).sqrt();

                let norm_input: Vec<Complex<f64>> = input.iter().map(|x| (*x - mean) / stddev).collect();

                for i in 0..feature_dim {
                    beta_grad[i] += grad_out[i];
                    gamma_grad[i] += grad_out[i] * norm_input[i];
                }

                let grad_norm_input: Vec<Complex<f64>> = (0..feature_dim).map(|i| grad_out[i] * self.gamma[i]).collect();

                let sum_grad_norm: Complex<f64> = grad_norm_input.iter().sum();
                let sum_grad_norm_x: Complex<f64> = norm_input.iter().zip(grad_norm_input.iter()).map(|(x, g)| *x * *g).sum();

                for i in 0..feature_dim {
                    input_grads[b][s][i] = (grad_norm_input[i] - sum_grad_norm / d - norm_input[i] * sum_grad_norm_x / d) / stddev;
                }
            }
        }

        // Create a new Gradient object and set the gradients
        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(input_grads);
        gradient.set_gradient_gamma(gamma_grad); // Store gamma gradient as a batch of 1
        gradient.set_gradient_beta(beta_grad); // Store beta gradient as a batch of 1

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No gradient found in NormalNormLayer");
        let mut gradient_gamma: Vec<Complex<f64>> = gradient.get_gradient_gamma();
        let mut gradient_beta: Vec<Complex<f64>> = gradient.get_gradient_beta(); // Get beta gradients

        let threshold = 1.0;
        clip_gradient_1d(&mut gradient_gamma, threshold);
        clip_gradient_1d(&mut gradient_beta, threshold); // Clip beta gradients

        let batch_size = gradient_gamma.len() as f64;
        let learning_rate = self.learning_rate;

        let mut prev_m_gamma: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); gradient_gamma.len()];
        let mut prev_v_gamma: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); gradient_gamma.len()];

        let mut prev_m_beta: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); gradient_gamma.len()];
        let mut prev_v_beta: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); gradient_gamma.len()];

        if let Some(previous_gradient) = &mut self.previous_gradient {
            prev_m_gamma = previous_gradient.get_prev_m_gamma();
            prev_v_gamma = previous_gradient.get_prev_v_gamma();

            prev_m_beta = previous_gradient.get_prev_m_beta();
            prev_v_beta = previous_gradient.get_prev_v_beta();

            self.gamma = calculate_adam_w_bias(&self.gamma, &gradient.get_gradient_gamma(), &mut prev_m_gamma, &mut prev_v_gamma, learning_rate, gradient.get_time_step());
            self.beta = calculate_adam_w_bias(&self.beta, &gradient.get_gradient_beta(), &mut prev_m_beta, &mut prev_v_beta, learning_rate, gradient.get_time_step());
        } else {
                for (i, value) in self.gamma.iter_mut().enumerate() {
                    if !is_nan_or_inf(&gradient_gamma[i]) {
                        *value -= self.learning_rate * (gradient_gamma[i] / batch_size);
                    }

                for (i, value) in self.beta.iter_mut().enumerate() {
                    if !is_nan_or_inf(&gradient_beta[i]) {
                        *value -= self.learning_rate * (gradient_beta[i] / batch_size);
                    }
                }
            }
        }

        gradient.set_prev_m_gamma(prev_m_gamma);
        gradient.set_prev_v_gamma(prev_v_gamma);
        gradient.set_prev_m_beta(prev_m_beta);
        gradient.set_prev_v_beta(prev_v_beta);
        self.previous_gradient = Some(gradient.clone());
    }
}
