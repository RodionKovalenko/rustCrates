use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::layer::{ActivationType, BaseLayer, LayerType},
    utils::weights_initializer::initialize_weights_complex,
};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionLayer<const M: usize, const N: usize> {
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

impl<const M: usize, const N: usize> Default for AttentionLayer<M, N> {
    fn default() -> Self {
        let mut weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); M]; N];
        let mut weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); M]; N];
        let mut weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); M]; N];

        initialize_weights_complex::<M, N>(&mut weights_q);
        initialize_weights_complex::<M, N>(&mut weights_k);
        initialize_weights_complex::<M, N>(&mut weights_v);

        let bias_q: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); M];
        let bias_k: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); M];
        let bias_v: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); M];

        AttentionLayer {
            weights_q,
            weights_k,
            weights_v,

            bias_q,
            bias_k,
            bias_v,
            activation_type: ActivationType::SIGMOID,
            layer_type: LayerType::InputLayer,

            inactivated_output: vec![vec![Complex::new(0.0, 0.0); M]; N],
            activated_output: vec![vec![Complex::new(0.0, 0.0); M]; N],
            gradient: vec![vec![Complex::new(0.0, 0.0); M]; N],
            gradient_w: vec![vec![Complex::new(0.0, 0.0); M]; N],
            errors: vec![vec![Complex::new(0.0, 0.0); M]; N],
            previous_gradient: vec![vec![Complex::new(0.0, 0.0); M]; N],
            m1: vec![vec![Complex::new(0.0, 0.0); M]; N],
            v1: vec![vec![Complex::new(0.0, 0.0); M]; N],
        }
    }
}

impl<const M: usize, const N: usize> AttentionLayer<M, N> {
    fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }
    fn set_activation(&mut self, activation: ActivationType) {
        self.activation_type = activation;
    }
}


// Implement BaseLayer for Layer struct
impl<const M: usize, const N: usize> AttentionLayer<M, N> {
    // Layer creation function
    pub fn create_default_attention_layer(
        activation: ActivationType,
        layer_type: LayerType,
    ) -> AttentionLayer<M, N> {
        let mut attention_layer: AttentionLayer<M, N> = AttentionLayer::default();
        attention_layer.set_activation(activation);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}


// Implement BaseLayer for Layer struct
impl<const M: usize, const N: usize> BaseLayer<M, N> for AttentionLayer<M, N> {
    fn forward(&mut self, _input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {

        self.activated_output.clone()
        // Implement forward pass logic for the layer
    }

    fn backward(&mut self, _gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        // Implement backward pass logic for the layer

        self.gradient.clone()
    }
}