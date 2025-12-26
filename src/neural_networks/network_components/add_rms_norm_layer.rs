use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::calculate_adam_w_bias,
    matrix::{add_matrix, add_matrix_2d_c, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d},
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

pub const EPSILON: f64 = 0.0000000000000000000000001;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer {
    pub gamma: Vec<Complex<f64>>, // Learnable scaling parameter (for each feature)
    pub epsilon: f64,             // Small constant for numerical stability
    pub learning_rate: f64,       // Learning rate for gamma updates
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl RMSNormLayer {
    // Initialize the RMSNorm layer with a given feature dimension (e.g., 16 for each token embedding)
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![Complex::new(1.0, 0.0); feature_dim], // Initialize gamma to 1.0 for all features
            epsilon,
            smoothing: 0.9,
            ema: 0.0,
            learning_rate,
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            global_norm: 0.0,
            max_norm: 0.0,
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
        self.batch_size = layer_input.get_batch_size();

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
                            -(input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (dim_f64 * rms_cubed)
                        };
                        input_batch_gradients[b][s][d_j] += grad.conj() * previous_gradient_batch[b][s][d_i];
                    }

                    gradient_gamma_batch[b][d_i] += input_batch[b][s][d_i] / rms;
                }
            }
        }

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            gradient_gamma_batch = add_matrix_2d_c(&gradient_gamma_batch, &previous_gradient.get_gradient_gamma_batch());
        }

        gradient.set_gradient_input_batch(input_batch_gradients);
        gradient.set_gradient_gamma_batch(gradient_gamma_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No gradient found in rms norm layer");
        let mut gradient_gamma: Vec<Complex<f64>> = gradient.get_gradient_gamma();

        let total_valid_tokens = gradient.get_total_valid_tokens().max(1) as f64;

        gradient_gamma = average_vector_by_scalar(&gradient_gamma, total_valid_tokens);

        clip_all_gradients_by_global_norm_2d(&mut vec![], &mut gradient_gamma, self.global_norm, self.max_norm);

        let learning_rate = self.learning_rate;

        let (mut prev_m_gamma, mut prev_v_gamma, mut prev_v_gamma_hat) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (previous_gradient.get_prev_m_gamma(), previous_gradient.get_prev_v_gamma(), previous_gradient.get_prev_v_gamma_hat())
        } else {
            // Initialize to zeros on first step
            (vec![Complex::new(0.0, 0.0); gradient_gamma.len()], vec![Complex::new(0.0, 0.0); gradient_gamma.len()], vec![Complex::new(0.0, 0.0); gradient_gamma.len()])
        };

        calculate_adam_w_bias(&mut self.gamma, &gradient.get_gradient_gamma(), &mut prev_m_gamma, &mut prev_v_gamma, &mut prev_v_gamma_hat, learning_rate, gradient.get_time_step());

        gradient.set_prev_m_gamma(prev_m_gamma);
        gradient.set_prev_v_gamma(prev_v_gamma);
        gradient.set_prev_v_gamma_hat(prev_v_gamma_hat);
        gradient.set_gradient_gamma(gradient_gamma.clone());
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}
