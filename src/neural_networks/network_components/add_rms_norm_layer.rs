use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::matrix::add_matrix;

use super::gradient_struct::Gradient;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer {
    gamma: Vec<Complex<f64>>, // Learnable scaling parameter (for each feature)
    epsilon: f64,             // Small constant for numerical stability
    learning_rate: f64,       // Learning rate for gamma updates
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient: Option<Gradient>,
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
        }
    }

    // RMSNorm function that works on a single token embedding (vector of Complex<f64>)
    pub fn rms_norm(&self, input: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        if input.is_empty() {
            panic!("Input to RMSNorm cannot be empty");
        }

        let rms = self.rms(input);

        // Normalize the input and apply the learned gamma scaling
        input.iter().zip(self.gamma.iter()).map(|(x, &g)| (*x / rms * g)).collect()
    }

    pub fn rms(&self, input: &Vec<Complex<f64>>) -> Complex<f64> {
        let mean_square = input.iter().map(|x| x * x).sum::<Complex<f64>>() / input.len() as f64;
        let rms = (mean_square + self.epsilon).sqrt();

        rms
    }

    // Forward pass for a batch of token embeddings (2D input)
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, input_before_transform_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
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

        output_batch
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found in RMSNorm layer");
        let mut gradient = Gradient::new_default();

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let dim_len = input_batch[0][0].len();

        let mut input_batch_gradients = vec![vec![vec![Complex::new(0.0, 0.0); dim_len]; seq_len]; batch_size];
        let mut gradient_gamma_batch = vec![vec![Complex::new(0.0, 0.0); dim_len]; batch_size];

        println!("------------------------------------------------------------------------------");
        println!("previous gradient batch  {} {} {}", &previous_gradient_batch.len(), &previous_gradient_batch[0].len(), previous_gradient_batch[0][0].len());
        println!("input batch  {} {} {}", &input_batch.len(), &input_batch[0].len(), input_batch[0][0].len());
        println!("gradient gamma dim: {} {}", &gradient_gamma_batch.len(), &gradient_gamma_batch[0].len());
        println!("gamma dim: {}", &self.gamma.len());
        println!("------------------------------------------------------------------------------");

        for b in 0..batch_size {
            for s in 0..seq_len {
                let rms = self.rms(&input_batch[b][s]);
                // Compute gradients
                for d_i in 0..dim_len {
                    // Applying RMSNorm backward formula with gamma scaling
                    for d_j in 0..dim_len {
                        if d_i == d_j {
                            let grad = (1.0 / rms) - ((input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (dim_len as f64 * rms.powf(3.0)));
                            input_batch_gradients[b][s][d_j] += grad;
                        } else {
                            let grad = Complex::new(0.0, 0.0) - ((input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (dim_len as f64 * rms.powf(3.0)));
                            input_batch_gradients[b][s][d_j] += grad;
                        }
                    }

                    input_batch_gradients[b][s][d_i] *= self.gamma[d_i] * previous_gradient_batch[b][s][d_i];
                    gradient_gamma_batch[b][d_i] = input_batch[b][s][d_i] / rms;
                }
            }
        }

        gradient.set_gradient_input_batch(input_batch_gradients);
        gradient.set_gradient_gamma_batch(gradient_gamma_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }
    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.as_ref().expect("No gradient found in rms norm layer");
        let gradient_gamma = gradient.get_gradient_gamma_batch();

        // println!("------------------------------------------------------------------------------");
        // println!("gradient_gamma batch  {} {}", &gradient_gamma.len(), &gradient_gamma[0].len());
        // println!("gradient gamma dim: {} ", &self.gamma.len());
        // println!("gamma dim: {}", &self.gamma.len());
        // println!("------------------------------------------------------------------------------");

        for batch_ind in 0..gradient_gamma.len() {
            for (i, value) in self.gamma.iter_mut().enumerate() {
                *value -= self.learning_rate * gradient_gamma[batch_ind][i];
            }
        }
    }
}
