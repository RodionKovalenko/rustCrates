use crate::neural_networks::utils::weights_initializer::initialize_weights_complex;
use core::fmt::{Debug};
use num::Complex;
use serde::{Deserialize, Serialize};


// Base Layer trait
pub trait BaseLayer<const M: usize, const N: usize>: Debug {
    fn forward(&mut self, input: &[[Complex<f64>; M]; N]);
    fn backward(&mut self, gradient: &[[Complex<f64>; M]; N]);
}

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::TANH // or any other variant you prefer as default
    }
}

// Activation Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    SIGMOID,
    TANH,
    LINEAR,
    RANDOM,
}

// Layer Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
}

// Implement Default for LayerType
impl Default for LayerType {
    fn default() -> Self {
        LayerType::InputLayer // or another sensible default
    }
}

// Layer Enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerEnum<const M: usize, const N: usize> {
    Dense(Box<Layer<M, N>>),
    RMSNorm(Box<Layer<M, N>>),
    SelfAttention(Box<Layer<M, N>>),
}

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer<const M: usize, const N: usize> {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub layer_bias: Vec<Complex<f64>>,
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

// Implement BaseLayer for Layer struct
impl<const M: usize, const N: usize> BaseLayer<M, N> for Layer<M, N> {
    fn forward(&mut self, input: &[[Complex<f64>; M]; N]) {
        // Implement forward pass logic for the layer
    }

    fn backward(&mut self, gradient: &[[Complex<f64>; M]; N]) {
        // Implement backward pass logic for the layer
    }
}

impl<const M: usize, const N: usize> Default for Layer<M, N> {
    fn default() -> Self {
        Layer {
            weights: vec![vec![Complex::new(0.0, 0.0); M]; N],
            layer_bias: vec![Complex::new(0.0, 0.0); M],
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

// Self-Attention Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer<const M: usize, const N: usize> {
    base_layer: Layer<M, N>,
}

// Implement BaseLayer for SelfAttentionLayer
impl<const M: usize, const N: usize> BaseLayer<M, N> for SelfAttentionLayer<M, N> {
    fn forward(&mut self, input: &[[Complex<f64>; M]; N]) {
        // Implement forward logic for self-attention
    }

    fn backward(&mut self, gradient: &[[Complex<f64>; M]; N]) {
        // Implement backward logic for self-attention
    }
}

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer<const M: usize, const N: usize> {
    base_layer: Layer<M, N>,
}

// Implement BaseLayer for RMSNormLayer
impl<const M: usize, const N: usize> BaseLayer<M, N> for RMSNormLayer<M, N> {
    fn forward(&mut self, input: &[[Complex<f64>; M]; N]) {
        // Implement forward logic for RMSNorm
    }

    fn backward(&mut self, gradient: &[[Complex<f64>; M]; N]) {
        // Implement backward logic for RMSNorm
    }
}

// Layer initialization function with Box<dyn BaseLayer>
pub fn initialize_default_layers<const M: usize, const N: usize>(
    num_outputs: &usize,
    num_h_layers: &usize,
    activation: &ActivationType,
) -> Vec<LayerEnum<M, N>> {
    let mut layers: Vec<LayerEnum<M, N>> = Vec::new();
    let total_layers: usize = *num_h_layers + 2;

    if *num_h_layers == 0 {
        panic!("Number of hidden layers cannot be zero.");
    }

    for l in 0..total_layers {
        let layer_type = get_layer_type(l, total_layers);
        let layer = create_default_layer::<M, N>(activation, layer_type);

        layers.push(LayerEnum::Dense(Box::new(layer)));
    }

    layers
}

// Layer creation function
pub fn create_default_layer<const M: usize, const N: usize>(
    activation: &ActivationType,
    layer_type: LayerType,
) -> Layer<M, N> {
    let mut weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); M]; N];
    let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); M];

    initialize_weights_complex::<M, N>(&mut weights); // 2D matrix

    Layer {
        weights,
        layer_bias: bias, // 1D vector
        activation_type: activation.clone(),
        layer_type,
        ..Default::default() // Fill the rest with default values
    }
}

// Helper function to determine the type of layer
pub fn get_layer_type(layer_idx: usize, total_layers: usize) -> LayerType {
    match layer_idx {
        0 => LayerType::InputLayer,
        x if x == total_layers - 1 => LayerType::OutputLayer,
        _ => LayerType::HiddenLayer,
    }
}
