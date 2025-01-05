use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::activation::softmax_last_row;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxLayer {
    learning_rate: f64
}

impl SoftmaxLayer {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate
        }
    }
    pub fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        softmax_last_row(input)
    }

    // Backward pass to compute gradients for input and gamma
    pub fn backward(&mut self, gradient: &Vec<Vec<Complex<f64>>>, _input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        gradient.clone()
    }
}