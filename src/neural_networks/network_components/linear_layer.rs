use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    matrix::{add_vector, multiply_complex, multiply_scalar_with_matrix},
    weights_initializer::initialize_weights_complex,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub bias: Vec<Complex<f64>>,
    pub gradients: Vec<Vec<Complex<f64>>>,
    pub gradients_bias: Vec<Vec<Complex<f64>>>,
    pub learning_rate: f64,
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
}

impl LinearLayer {
    pub fn new(learning_rate: f64, rows: usize, cols: usize) -> Self {
        let mut weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0); cols];

        initialize_weights_complex(rows, cols, &mut weights);

        Self {
            weights,
            bias,
            learning_rate,
            gradients: vec![],
            gradients_bias: vec![],
            input_batch: None,
        }
    }
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch = Some(input_batch.clone());

        input_batch
            .par_iter() // Use a parallel iterator to process inputs in parallel
            .map(|input| {
                // Perform matrix multiplication
                let mut output = multiply_complex(input, &self.weights);

                // println!("input dim in forward linear layer: {}, {}", input.len(), input[0].len());
                // println!("weights dim in forward linear layer: {}, {}", &self.weights.len(), &self.weights[0].len());
                // println!("output dim in forward linear layer: {}, {}", output.len(), output[0].len());

                // Add the bias vector
                output = add_vector(&output, &self.bias);

                output // Return the processed output for this input
            })
            .collect() // Collect all results into a single Vec
    }

    pub fn backward(&mut self, previous_gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        // Retrieve input batch
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing");

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let dim = input_batch[0][0].len();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()];
        let mut bias_gradients: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.bias.len()];

        // For each input sample in the batch
        for input_sample in input_batch.iter() {
            // Compute the gradient of the loss with respect to the weights and biases for this sample
            let mut sample_input = vec![vec![Complex::new(0.0, 0.0); dim]; seq_len];

            for (j, seq_sample) in input_sample.iter().enumerate() {
                for (k, input_value) in seq_sample.iter().enumerate() {
                    sample_input[j][k] = input_value.clone(); // Use input directly, no need to group
                }
            }

            // Multiply the transposed input sample with previous gradients (for weight gradients)
            let current_sample_weight_gradients = multiply_complex(&sample_input, previous_gradient);

            // Sum gradients for the weights and biases across all samples
            for (i, row) in current_sample_weight_gradients.iter().enumerate() {
                for (j, weight_value) in row.iter().enumerate() {
                    weight_gradients[i][j] += weight_value.clone(); // Accumulate gradients for weights
                }
            }
        }

        // Accumulate gradients for biases
        for grad_value in previous_gradient.iter() {
            for (k, grad_val) in grad_value.iter().enumerate() {
                bias_gradients[k] += grad_val.clone(); // Sum the gradients for biases
            }
        }

        // Normalize the gradients by the batch size (average)
        let batch_scalar = Complex::new(1.0 / batch_size as f64, 0.0);
        weight_gradients = multiply_scalar_with_matrix::<Complex<f64>>(batch_scalar, &weight_gradients);

        for bias in bias_gradients.iter_mut() {
            *bias /= batch_size as f64; // Normalize bias gradients
        }

        // Update weights and biases using gradient descent
        for (i, row) in self.weights.iter_mut().enumerate() {
            for (j, weight_value) in row.iter_mut().enumerate() {
                *weight_value -= self.learning_rate * weight_gradients[i][j];
            }
        }

        for (i, value) in self.bias.iter_mut().enumerate() {
            *value -= self.learning_rate * bias_gradients[i];
        }

        previous_gradient.clone()
    }
}
