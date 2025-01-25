use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_types::neural_network_generic::OperationMode,
    utils::activation::{softmax_complex, softmax_last_row},
};

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxLayer {
    learning_rate: f64,
    operation_mode: OperationMode,
}

impl SoftmaxLayer {
    pub fn new(learning_rate: f64, operation_mode: OperationMode) -> Self {
        Self { learning_rate, operation_mode }
    }
    pub fn forward(&self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let layer_output: Vec<Vec<Vec<Complex<f64>>>> = match self.operation_mode {
            OperationMode::PRODUCTION => {
                input_batch
                    .par_iter() // Parallel iterator for the input batch
                    .map(|input| softmax_last_row(input)) // Apply `softmax_last_row` to each input
                    .collect() // Collect results into a Vec
            }
            OperationMode::TRAINING => {
                input_batch
                    .par_iter() // Parallel iterator for the input batch
                    .map(|input| softmax_complex(input)) // Apply `softmax_complex` to each input
                    .collect() // Collect results into a Vec
            }
        };

        layer_output
    }

    // Backward pass to compute gradients for input and gamma
    pub fn backward(&mut self, gradient: &Vec<Vec<Complex<f64>>>, _input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        gradient.clone()
    }
}
