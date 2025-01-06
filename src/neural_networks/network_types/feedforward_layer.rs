use crate::neural_networks::{
    network_components::layer::{create_default_layer, ActivationType, BaseLayer, Layer, LayerType},
    utils::matrix::add_matrix,
};
use num::Complex;
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardLayer {
    pub layers: Vec<Layer>,
}

impl FeedForwardLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(rows: usize, cols: usize, output_cols: usize) -> Self {
        let mut layers: Vec<Layer> = vec![];
        let dense_layer: Layer = create_default_layer(rows, cols, &ActivationType::GELU, LayerType::DenseLayer);
        let linear_layer: Layer = create_default_layer(cols, output_cols, &ActivationType::LINEAR, LayerType::LinearLayer);

        layers.push(dense_layer);
        layers.push(linear_layer);

        Self { layers }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl BaseLayer for FeedForwardLayer {
    fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let mut output: Vec<Vec<Complex<f64>>> = input.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
            // println!("layer in ffn: {:?}, {}, {}", &layer.layer_type, layer.weights.len(), layer.weights[0].len());
            // println!("output in ffn layer: {:?}, {}, {}", &layer.layer_type, output.len(), output[0].len());
        }
        output = add_matrix(&output, input);

        output
    }

    fn backward(&self, gradients: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        gradients.clone()
    }
}
