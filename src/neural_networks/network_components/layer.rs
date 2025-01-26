use crate::neural_networks::{
    network_types::{feedforward_layer::FeedForwardLayer, transformer::self_attention_layer::SelfAttentionLayer},
    utils::{
        activation::activate_output_complex,
        matrix::{add_vector, multiply_complex},
        weights_initializer::initialize_weights_complex,
    },
};
use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    add_rms_norm_layer::RMSNormLayer, embedding_layer::EmbeddingLayer, linear_layer::LinearLayer, positional_encoding_layer::PositionalEncodingLayer,
    softmax_output_layer::SoftmaxLayer,
};

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
    fn forward(&self, input: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>>;
    fn backward(&self, gradient: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>>;
}

// Layer Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
    AttentionLayer,
    DenseLayer,
    LinearLayer,
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
    PositionalEncoding(Box<PositionalEncodingLayer>),
    Dense(Box<Layer>),
    FeedForward(Box<FeedForwardLayer>),
    RMSNorm(Box<RMSNormLayer>),
    SelfAttention(Box<SelfAttentionLayer>),
    Linear(Box<LinearLayer>),
    Softmax(Box<SoftmaxLayer>),
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
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
}

// Layer initialization function with Box<dyn BaseLayer>
pub fn initialize_default_layers(rows: usize, cols: usize, _num_outputs: &usize, num_h_layers: &usize, activation: &ActivationType) -> Vec<LayerEnum> {
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
pub fn create_default_layer(rows: usize, cols: usize, activation: &ActivationType, layer_type: LayerType) -> Layer {
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
impl Layer {
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch = Some(input_batch.clone());
    
        // Initialize batch_output to hold the results
        let batch_output: Vec<Vec<Vec<Complex<f64>>>> = input_batch
            .par_iter()  // Process the input batch in parallel
            .map(|input| {
                // Multiply input with weights
                let output: Vec<Vec<Complex<f64>>> = multiply_complex(input, &self.weights);
                
                // Add bias to the result
                let raw_output: Vec<Vec<Complex<f64>>> = add_vector(&output, &self.bias);
    
                // Apply activation if the layer type is DenseLayer
                if &self.layer_type == &LayerType::DenseLayer {
                    let activated_output: Vec<Vec<Complex<f64>>> = activate_output_complex(&raw_output, self.activation_type.clone());
                    activated_output
                } else {
                    raw_output
                }
            })
            .collect();  // Collect the results back into a Vec
    
        batch_output
    }

    pub fn backward(&self, _gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        self.gradient.clone()
    }
}

impl Layer {
    pub fn default(rows: usize, cols: usize) -> Self {
        let mut weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

        initialize_weights_complex(rows, cols, &mut weights); // 2D matrix

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
            input_batch: None,
        }
    }
}
