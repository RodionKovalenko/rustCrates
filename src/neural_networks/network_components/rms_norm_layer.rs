use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use super::layer::{BaseLayer, LayerEnum};

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer {
    base_layer: LayerEnum,
}

// Implement BaseLayer for RMSNormLayer
impl BaseLayer for RMSNormLayer {
    fn forward(&mut self, _input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let activated_output: Vec<Vec<Complex<f64>>> = match &mut self.base_layer {
            LayerEnum::Dense(dense) => {
                println!("Dense layer: {:?}", dense);
                dense.activated_output.clone() // Access the field here
            }
            LayerEnum::SelfAttention(attention) => {
                println!("Attention layer: {:?}", attention);
                attention.activated_output.clone() // Access the field here
            }
            _ => {
                println!("Other layer type");
                vec![] // Return empty vector for unsupported types
            }
        };

        activated_output
    }

    fn backward(&mut self, _gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let activated_output: Vec<Vec<Complex<f64>>> = match &mut self.base_layer {
            LayerEnum::Dense(dense) => {
                println!("Dense layer: {:?}", dense);
                dense.activated_output.clone() // Access the field here
            }
            LayerEnum::SelfAttention(attention) => {
                println!("Attention layer: {:?}", attention);
                attention.activated_output.clone() // Access the field here
            }
            _ => {
                println!("Other layer type");
                vec![] // Return empty vector for unsupported types
            }
        };

        activated_output
    }
}
