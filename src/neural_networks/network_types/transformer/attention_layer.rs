use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::layer::{ActivationType, BaseLayer, LayerType},
    utils::{matrix::multiply_complex, weights_initializer::initialize_weights_complex},
};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionLayer {
    pub weights_q: Vec<Vec<Complex<f64>>>,
    pub weights_k: Vec<Vec<Complex<f64>>>,
    pub weights_v: Vec<Vec<Complex<f64>>>,

    pub bias_q: Vec<Complex<f64>>,
    pub bias_k: Vec<Complex<f64>>,
    pub bias_v: Vec<Complex<f64>>,

    pub activation_type: ActivationType,
    pub layer_type: LayerType,

    pub inactivated_output: Vec<Vec<Complex<f64>>>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub gradient: Vec<Vec<Complex<f64>>>,
    pub gradient_w: Vec<Vec<Complex<f64>>>,
    pub errors: Vec<Vec<Complex<f64>>>,
    pub previous_gradient: Vec<Vec<Complex<f64>>>,

    pub m1: Vec<Vec<Complex<f64>>>,
    pub v1: Vec<Vec<Complex<f64>>>,
}

impl AttentionLayer {
    fn default(rows: usize, cols: usize) -> Self {
        let mut weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);

        let bias_q: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let bias_k: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let bias_v: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

        AttentionLayer {
            weights_q,
            weights_k,
            weights_v,

            bias_q,
            bias_k,
            bias_v,
            activation_type: ActivationType::SIGMOID,
            layer_type: LayerType::InputLayer,

            inactivated_output: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            activated_output: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            gradient: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            gradient_w: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            errors: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            previous_gradient: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
        }
    }

    fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }
    fn set_activation(&mut self, activation: ActivationType) {
        self.activation_type = activation;
    }
    pub fn create_default_attention_layer(
        rows: usize,
        cols: usize,
        activation: ActivationType,
        layer_type: LayerType,
    ) -> AttentionLayer {
        let mut attention_layer: AttentionLayer = AttentionLayer::default(rows, cols);
        attention_layer.set_activation(activation);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl BaseLayer for AttentionLayer {
    fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        println!("output attention weights: {}, {}", &self.weights_k.len(), &self.weights_k[0].len());
        multiply_complex(input, &self.weights_k)
        // Implement forward pass logic for the layer
    }

    fn backward(&self, _gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        // Implement backward pass logic for the layer

        self.gradient.clone()
    }
}
