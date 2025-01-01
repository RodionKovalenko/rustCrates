use crate::neural_networks::{network_types::transformer::attention_layer::AttentionLayer, utils::{
    matrix::multiply_complex, weights_initializer::initialize_weights_complex,
}};
use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use super::{embedding_layer::EmbeddingLayer, rms_norm_layer::RMSNormLayer};

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::TANH // or any other variant you prefer as default
    }
}

// https://encord.com/blog/activation-functions-neural-networks/
// Activation Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    SIGMOID,
    TANH,
    LINEAR,
    SOFTMAX,
    RELU,
    // Leaky Relu
    LEAKYRELU,
    // Exponental Linear Unit Function
    ELU,
    // Scaled Exponental Linear Unit Function
    SELU,
    // Gaussian Error Linear Units used in Chat-GTP-3, Albert und Roberta
    GELU,
    SOFTSIGN,
    SOFTPLUS,
    PROBIT,
    RANDOM,
}

// Base Layer trait
pub trait BaseLayer: Debug + Clone {
    fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>>;
    fn backward(&self, gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>>;
}

// Layer Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
    AttentionLayer,
}

// Implement Default for LayerType
impl Default for LayerType {
    fn default() -> Self {
        LayerType::InputLayer // or another sensible default
    }
}

// Layer Enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerEnum {
    Embedding(Box<EmbeddingLayer>),
    Dense(Box<Layer>),
    RMSNorm(Box<RMSNormLayer>),
    SelfAttention(Box<AttentionLayer>),
}

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub bias: Vec<Complex<f64>>,
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

// Layer initialization function with Box<dyn BaseLayer>
pub fn initialize_default_layers(
    rows: usize,
    cols: usize,
    _num_outputs: &usize,
    num_h_layers: &usize,
    activation: &ActivationType,
) -> Vec<LayerEnum> {
    let mut layers: Vec<LayerEnum> = Vec::new();
    let total_layers: usize = *num_h_layers + 2;

    if *num_h_layers == 0 {
        panic!("Number of hidden layers cannot be zero.");
    }

    for l in 0..total_layers {
        let layer_type = get_layer_type(l, total_layers);
        let layer = create_default_layer(rows, cols, activation, layer_type);

        layers.push(LayerEnum::Dense(Box::new(layer)));
    }

    layers
}

// Layer creation function
pub fn create_default_layer(
    rows: usize,
    cols: usize,
    activation: &ActivationType,
    layer_type: LayerType,
) -> Layer {
    Layer {
        activation_type: activation.clone(),
        layer_type,
        ..Layer::default(rows, cols) // Fill the rest with default values
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

// Implement BaseLayer for Layer struct
impl BaseLayer for Layer {
    fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        multiply_complex(input, &self.weights)
    }

    fn backward(&self, _gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        self.gradient.clone()
    }
}

impl Layer {
    pub fn default( rows: usize, cols: usize) -> Self {
        let mut weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

        initialize_weights_complex(rows, cols,  &mut weights); // 2D matrix

        Layer {
            weights,
            bias,
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
}
