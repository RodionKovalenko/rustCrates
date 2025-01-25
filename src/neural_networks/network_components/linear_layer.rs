use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    matrix::{add_vector, multiply_complex},
    weights_initializer::initialize_weights_complex,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub bias: Vec<Complex<f64>>,
    pub gradients: Vec<Vec<Complex<f64>>>,
    pub learning_rate: f64,
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
        }
    }
    pub fn forward(&self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        input_batch
            .par_iter() // Use a parallel iterator to process inputs in parallel
            .map(|input| {
                // Perform matrix multiplication
                let mut output = multiply_complex(input, &self.weights);

                // Add the bias vector
                output = add_vector(&output, &self.bias);

                output // Return the processed output for this input
            })
            .collect() // Collect all results into a single Vec
    }
}
