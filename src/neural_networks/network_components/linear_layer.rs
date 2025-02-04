use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    matrix::{add_matrix, add_vector, multiply_complex, multiply_scalar_with_matrix, transpose},
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
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

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

    pub fn backward_batch(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<Complex<f64>>) {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing");

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.bias.len()];

        // For each input sample in the batch
        for (batch_ind, (input_sample, gradient_sample)) in input_batch.iter().zip(previous_gradient_batch).enumerate() {
            // Multiply the transposed input sample with previous gradients (for weight gradients)
            weight_gradients[batch_ind] = multiply_complex(input_sample, gradient_sample);

            //Accumulate gradients for biases
            for grad_row in gradient_sample.iter() {
                for (k, grad_val) in grad_row.iter().enumerate() {
                    bias_gradients[k] += grad_val.clone(); // Sum the gradients for biases
                }
            }
        }

        (weight_gradients.clone(), bias_gradients)
    }

    pub fn update_weights(&mut self, weight_gradients: &Vec<Vec<Complex<f64>>>, bias_gradients: &Vec<Complex<f64>>) {
        // Update weights and biases using gradient descent
        for (i, row) in self.weights.iter_mut().enumerate() {
            for (j, weight_value) in row.iter_mut().enumerate() {
                *weight_value -= self.learning_rate * weight_gradients[i][j];
            }
        }

        for (i, value) in self.bias.iter_mut().enumerate() {
            *value -= self.learning_rate * bias_gradients[i];
        }
    }

    pub fn group_gradient_batch(&self, weight_gradients_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Complex<f64>>> {
        let mut weight_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); weight_gradients_batch[0][0].len()]; weight_gradients_batch[0].len()];

        for weight_gradient_batch in weight_gradients_batch {
            for (row, w_gradient) in weight_gradient_batch.iter().enumerate() {
                for (col, gradient_value) in w_gradient.iter().enumerate() {
                    weight_gradients[row][col] += gradient_value;
                }
            }
        }

        weight_gradients
    }
}
