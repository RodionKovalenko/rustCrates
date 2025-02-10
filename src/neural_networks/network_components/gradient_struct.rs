use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    gradient_weights_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_bias_batch: Option<Vec<Vec<Complex<f64>>>>,
    gradient_gamma_batch: Option<Vec<Vec<Complex<f64>>>>,

    gradient_weights: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_input: Option<Vec<Vec<Complex<f64>>>>,
    gradient_bias: Option<Vec<Complex<f64>>>,
    gradient_gamma: Option<Vec<Complex<f64>>>,
}

impl Gradient {
    pub fn new_default() -> Self {
        Gradient {
            gradient_weights_batch: None,
            gradient_input_batch: None,
            gradient_bias_batch: None,
            gradient_gamma_batch: None,

            gradient_weights: None,
            gradient_input: None,
            gradient_bias: None,
            gradient_gamma: None,
        }
    }
}