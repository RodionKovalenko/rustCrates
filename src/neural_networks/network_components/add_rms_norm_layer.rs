use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{adam_w::calculate_adam_w_bias, matrix::{add_matrix, check_nan_or_inf, check_nan_or_inf_3d, clip_gradients, is_nan_or_inf}};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

pub const EPSILON: f64 = 0.0000000000000000000000001;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer {
    gamma: Vec<Complex<f64>>, // Learnable scaling parameter (for each feature)
    epsilon: f64,             // Small constant for numerical stability
    learning_rate: f64,       // Learning rate for gamma updates
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient: Option<Gradient>,
    previous_gradient: Option<Gradient>,
    time_step: usize,
}

impl RMSNormLayer {
    // Initialize the RMSNorm layer with a given feature dimension (e.g., 16 for each token embedding)
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![Complex::new(1.0, 0.0); feature_dim], // Initialize gamma to 1.0 for all features
            epsilon,
            learning_rate,
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
        }
    }

    // RMSNorm function that works on a single token embedding (vector of Complex<f64>)
    pub fn rms_norm(&self, input: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        if input.is_empty() {
            panic!("Input to RMSNorm cannot be empty");
        }

        let rms = self.rms(input);

        // Normalize the input and apply the learned gamma scaling
        input.iter().zip(self.gamma.iter()).map(|(x, &g)| ((*x / rms) * g)).collect()
    }

    pub fn rms(&self, input: &Vec<Complex<f64>>) -> Complex<f64> {
        let mean_square = input
            .iter()
            .map(|x| {
                // println!("x {:?}, x * x {:?}", x, x * x);
                x * x
            })
            .sum::<Complex<f64>>()
            / input.len() as f64;
        (mean_square + self.epsilon).sqrt()
    }

    // Forward pass for a batch of token embeddings (2D input)
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();
        let input_before_transform_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch_before();

        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
        let mut input_batch_added = input_batch.clone();

        for (batch_ind, input) in input_batch.iter().enumerate() {
            // println!("shape input in rms: {:?}, {:?}", input.len(), input[0].len());
            // println!("shape input before in rms: {:?}, {:?}", input_before_transform_batch[batch_ind].len(), input_before_transform_batch[batch_ind][0].len());
            let output = add_matrix(input, &input_before_transform_batch[batch_ind]);
            input_batch_added[batch_ind] = output.clone();

            output_batch.push(
                output
                    .iter()
                    .map(|vec| self.rms_norm(vec)) // Normalize each token embedding
                    .collect(),
            );
        }

        self.input_batch = Some(input_batch_added.clone());
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found in RMSNorm layer");
        let mut gradient = Gradient::new_default();

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let dim_len = input_batch[0][0].len();

        let mut input_batch_gradients = vec![vec![vec![Complex::new(0.0, 0.0); dim_len]; seq_len]; batch_size];
        let mut gradient_gamma_batch = vec![vec![Complex::new(0.0, 0.0); dim_len]; batch_size];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let rms = self.rms(&input_batch[b][s]);
                let rms_cubed = rms.powf(3.0);
                let dim_f64 = dim_len as f64;
    
                for d_i in 0..dim_len {
                    for d_j in 0..dim_len {
                        let grad = if d_i == d_j {
                            Complex::new(1.0, 0.0) / rms - (input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (dim_f64 * rms_cubed)
                        } else {
                            - (input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (dim_f64 * rms_cubed)
                        };
                        input_batch_gradients[b][s][d_j] += grad * previous_gradient_batch[b][s][d_i];
                    }
    
                    gradient_gamma_batch[b][d_i] += input_batch[b][s][d_i] / rms;
                }
            }
        }

        check_nan_or_inf_3d(&mut input_batch_gradients, "output gradients in rms norm layer has None values");

        gradient.set_gradient_input_batch(input_batch_gradients);
        gradient.set_gradient_gamma_batch(gradient_gamma_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }
    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No gradient found in rms norm layer");
        let mut gradient_gamma: Vec<Vec<Complex<f64>>> = gradient.get_gradient_gamma_batch();

        let threshold = 1.0;
        clip_gradients(&mut gradient_gamma, threshold);
        check_nan_or_inf(&mut gradient_gamma, "check weight gradients in linear layer");

        let batch_size = gradient_gamma.len() as f64;

        let mut prev_m_gamma: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); gradient_gamma[0].len()];
        let mut prev_v_gamma: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); gradient_gamma[0].len()];
        let learning_rate = self.learning_rate;

        if let Some(previous_gradient) = &mut self.previous_gradient {
            prev_m_gamma = previous_gradient.get_prev_v_gamma();
            prev_v_gamma = previous_gradient.get_prev_v_gamma();

            calculate_adam_w_bias(&self.gamma, &gradient.get_gradient_gamma(), &mut prev_m_gamma, &mut prev_v_gamma, learning_rate, gradient.get_time_step());
        } else {
            for batch_ind in 0..gradient_gamma.len() {
                for (i, value) in self.gamma.iter_mut().enumerate() {
                    if !is_nan_or_inf(&gradient_gamma[batch_ind][i]) {
                        *value -= self.learning_rate * (gradient_gamma[batch_ind][i] / batch_size);
                    }
                }
            }
        }
    
        gradient.set_prev_m_gamma(prev_m_gamma);
        gradient.set_prev_v_gamma(prev_v_gamma);
        self.previous_gradient = Some(gradient.clone());
    }
}
