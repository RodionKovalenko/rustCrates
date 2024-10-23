use crate::neural_networks::utils::weights_initializer::initialize_weights_complex;
use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

// Base Layer trait
pub trait BaseLayer {}

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::TANH // or any other variant you prefer as default
    }
}

// Activation Type Enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    SIGMOID,
    TANH,
    LINEAR,
    RANDOM,
}

// Layer Type Enum
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone)]
pub enum LayerEnum {
    Layer(Layer<32, 32>),
    // Add more types of layers if necessary
}

impl BaseLayer for Layer<32, 32> {}

// Layer struct
#[derive(Debug, Clone)]
pub struct Layer<const N: usize, const M: usize> {
    pub weights: [[Complex<f64>; M]; N],
    pub layer_bias: [Complex<f64>; M],
    pub activation_type: ActivationType,
    pub layer_type: LayerType,
    pub inactivated_output: [[Complex<f64>; M]; N],
    pub activated_output: [[Complex<f64>; M]; N],
    pub gradient: [[Complex<f64>; M]; N],
    pub gradient_w: [[Complex<f64>; M]; N],
    pub errors: [[Complex<f64>; M]; N],
    pub previous_gradient: [[Complex<f64>; M]; N],
    pub m1: [[Complex<f64>; M]; N],
    pub v1: [[Complex<f64>; M]; N],
}

// Implement Default for Layer
impl<const N: usize, const M: usize> Default for Layer<N, M> {
    fn default() -> Self {
        Layer {
            weights: [[Complex::new(0.0, 0.0); M]; N],
            layer_bias: [Complex::new(0.0, 0.0); M],
            activation_type: ActivationType::SIGMOID, // Default activation type
            layer_type: LayerType::InputLayer,         // Default layer type
            inactivated_output: [[Complex::new(0.0, 0.0); M]; N],
            activated_output: [[Complex::new(0.0, 0.0); M]; N],
            gradient: [[Complex::new(0.0, 0.0); M]; N],
            gradient_w: [[Complex::new(0.0, 0.0); M]; N],
            errors: [[Complex::new(0.0, 0.0); M]; N],
            previous_gradient: [[Complex::new(0.0, 0.0); M]; N],
            m1: [[Complex::new(0.0, 0.0); M]; N],
            v1: [[Complex::new(0.0, 0.0); M]; N],
        }
    }
}
impl Layer<32, 32> {
    // Method to set the entire input_weights (2D matrix)
    pub fn set_input_weights(&mut self, new_weights: [[Complex<f64>; 32]; 32]) {
        self.weights = new_weights;
    }

    // Methods for setting outputs and gradients
    pub fn set_inactivated_output(&mut self, new_output: [[Complex<f64>; 32]; 32]) {
        self.inactivated_output = new_output;
    }

    pub fn set_activated_output(&mut self, new_output: [[Complex<f64>; 32]; 32]) {
        self.activated_output = new_output;
    }

    pub fn set_gradient_backward(&mut self, new_gradient: [[Complex<f64>; 32]; 32]) {
        self.gradient = new_gradient;
    }

    pub fn set_gradient_w(&mut self, new_gradient_w: [[Complex<f64>; 32]; 32]) {
        self.gradient_w = new_gradient_w;
    }

    pub fn set_errors(&mut self, new_errors: [[Complex<f64>; 32]; 32]) {
        self.errors = new_errors;
    }

    pub fn set_layer_bias(&mut self, new_bias: [Complex<f64>; 32]) {
        self.layer_bias = new_bias;
    }

    pub fn set_m1(&mut self, new_m1: [[Complex<f64>; 32]; 32]) {
        self.m1 = new_m1;
    }

    pub fn set_v1(&mut self, new_v1: [[Complex<f64>; 32]; 32]) {
        self.v1 = new_v1;
    }
}

// Other structs and methods remain unchanged...
#[derive(Debug, Clone)]
pub struct SelfAttentionLayer {
    base_layer: Layer<32, 32>,
}

#[derive(Debug, Clone)]
pub struct RMSNormLayer {
    base_layer: Layer<32, 32>,
}

// Layer initialization function
pub fn initialize_default_layers(
    num_inputs: &usize,
    num_outputs: &usize,
    num_h_layers: &usize,
    num_h_neurons: &usize,
    activation: &ActivationType,
) -> Vec<Layer<32, 32>> {
    let mut layers: Vec<Layer<32, 32>> = Vec::new();
    let total_layers: usize = *num_h_layers + 2;

    if *num_h_layers == 0 {
        panic!("Number of hidden layers cannot be zero.");
    }

    for l in 0..total_layers {
        let layer_type = get_layer_type(l, total_layers);
        let layer: Layer<32, 32> = create_default_layer(
            match layer_type {
                LayerType::InputLayer => num_inputs,
                LayerType::HiddenLayer => num_h_neurons,
                LayerType::OutputLayer => num_outputs,
            },
            match layer_type {
                LayerType::InputLayer => num_h_neurons,
                LayerType::HiddenLayer => num_h_neurons,
                LayerType::OutputLayer => num_outputs,
            },
            activation,
            layer_type,
        );

        layers.push(layer);
    }

    layers
}

pub fn create_default_layer(
    num_i_neurons: &usize,
    num_o_neurons: &usize,
    activation: &ActivationType,
    layer_type: LayerType,
) -> Layer<32, 32> {
    // Initialize the matrices with the correct dimensions
    let mut weights: [[Complex<f64>; 32]; 32] = [[Complex::new(0.0, 0.0); 32]; 32];

    initialize_weights_complex::<32, 32>(*num_i_neurons, *num_o_neurons, &mut weights); // 2D matrix

    // Create the layer with the initialized matrices
    Layer {
        weights,
        layer_bias: [Complex::new(0.0, 0.0); 32], // 1D vector
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