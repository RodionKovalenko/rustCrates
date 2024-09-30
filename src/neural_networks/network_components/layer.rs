use core::fmt::Debug;
use serde::{Deserialize, Serialize};
use crate::neural_networks::utils::weights_initializer::initialize_weights;

pub trait BaseLayer {}

// Enums for Activation and Layer Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    SIGMOID,
    TANH,
    LINEAR,
    RANDOM,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerEnum {
    Layer(Layer),
    // Add more types of layers if necessary
}

impl BaseLayer for Layer {}

// Layer struct with improved trait bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,       // 2D matrix
    pub layer_bias: Vec<f64>,               // 1D vector
    pub activation_type: ActivationType,    // Activation type
    pub layer_type: LayerType,              // Layer type (Input, Hidden, Output)

    pub inactivated_output: Vec<Vec<f64>>,  // 2D matrix
    pub activated_output: Vec<Vec<f64>>,    // 2D matrix
    pub gradient_backward: Vec<Vec<f64>>,   // 2D matrix
    pub gradient_w: Vec<Vec<f64>>,          // 2D matrix
    pub errors: Vec<Vec<f64>>,              // 2D matrix

    pub previous_layer: Option<Box<LayerEnum>>, // Use LayerEnum instead of dyn BaseLayer
    pub next_layer: Option<Box<LayerEnum>>,     
    pub previous_gradient: Vec<Vec<f64>>,   // 2D matrix for previous gradient
    pub m1: Vec<Vec<f64>>,                  // 2D matrix for momentum
    pub v1: Vec<Vec<f64>>,                  // 2D matrix for velocity
}


impl Layer {
    // Method to set the entire input_weights (2D matrix)
    pub fn set_input_weights(&mut self, new_weights: Vec<Vec<f64>>) {
        self.weights = new_weights;
    }
    // Method to set an individual value in input_weights (2D matrix)
    pub fn set_input_weight(&mut self, i: usize, j: usize, value: f64) {
        self.weights[i][j] = value;
    }

    // Method to set the entire inactivated_output (3D matrix)
    pub fn set_inactivated_output(&mut self, new_output: Vec<Vec<f64>>) {
        self.inactivated_output = new_output;
    }

    // Method to set an individual value in inactivated_output (3D matrix)
    pub fn set_inactivated_output_value(&mut self, i: usize, j: usize, value: f64) {
        self.inactivated_output[i][j] = value;
    }

    // Method to set the entire activated_output (3D matrix)
    pub fn set_activated_output(&mut self, new_output: Vec<Vec<f64>>) {
        self.activated_output = new_output;
    }

    // Method to set an individual value in activated_output (3D matrix)
    pub fn set_activated_output_value(&mut self, i: usize, j: usize, value: f64) {
        self.activated_output[i][j] = value;
    }

    // Method to set the entire gradient_backward (2D matrix)
    pub fn set_gradient_backward(&mut self, new_gradient: Vec<Vec<f64>>) {
        self.gradient_backward = new_gradient;
    }

    // Method to set an individual value in gradient_backward (2D matrix)
    pub fn set_gradient_backward_value(&mut self, i: usize, j: usize, value: f64) {
        self.gradient_backward[i][j] = value;
    }

    // Method to set the entire gradient_w (2D matrix)
    pub fn set_gradient_w(&mut self, new_gradient_w: Vec<Vec<f64>>) {
        self.gradient_w = new_gradient_w;
    }

    // Method to set an individual value in gradient_w (2D matrix)
    pub fn set_gradient_w_value(&mut self, i: usize, j: usize, value: f64) {
        self.gradient_w[i][j] = value;
    }

    // Method to set the entire errors (2D matrix)
    pub fn set_errors(&mut self, new_errors: Vec<Vec<f64>>) {
        self.errors = new_errors;
    }

    // Method to set an individual value in errors (2D matrix)
    pub fn set_error_value(&mut self, i: usize, j: usize, value: f64) {
        self.errors[i][j] = value;
    }

    // Method to set the entire layer_bias (1D vector)
    pub fn set_layer_bias(&mut self, new_bias: Vec<f64>) {
        self.layer_bias = new_bias;
    }

    // Method to set an individual value in layer_bias (1D vector)
    pub fn set_layer_bias_value(&mut self, i: usize, value: f64) {
        self.layer_bias[i] = value;
    }

    // Method to set the entire m1 (2D matrix)
    pub fn set_m1(&mut self, new_m1: Vec<Vec<f64>>) {
        self.m1 = new_m1;
    }

    // Method to set an individual value in m1 (2D matrix)
    pub fn set_m1_value(&mut self, i: usize, j: usize, value: f64) {
        self.m1[i][j] = value;
    }

    // Method to set the entire v1 (2D matrix)
    pub fn set_v1(&mut self, new_v1: Vec<Vec<f64>>) {
        self.v1 = new_v1;
    }

    // Method to set an individual value in v1 (2D matrix)
    pub fn set_v1_value(&mut self, i: usize, j: usize, value: f64) {
        self.v1[i][j] = value;
    }

    // Method to set the entire previous_gradient (2D matrix)
    pub fn set_previous_gradient(&mut self, new_previous_gradient: Vec<Vec<f64>>) {
        self.previous_gradient = new_previous_gradient;
    }

    // Method to set an individual value in previous_gradient (2D matrix)
    pub fn set_previous_gradient_value(&mut self, i: usize, j: usize, value: f64) {
        self.previous_gradient[i][j] = value;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer {
    base_layer: Layer
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer {
    base_layer: Layer
}

// Layer initialization function
pub fn initialize_default_layers(
    num_inputs: &usize,
    num_outputs: &usize,
    num_h_layers: &usize,
    num_h_neurons: &usize,
    activation: &ActivationType,
) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();
    let total_layers: usize = num_h_layers + 2;

    if *num_h_layers == 0 {
        panic!("Number of hidden layers cannot be zero.");
    }

    for l in 0..total_layers {
        let layer_type = get_layer_type(l, total_layers);
       
        let layer;
        match layer_type {
            LayerType::InputLayer => {
                layer = create_default_layer(num_inputs, num_h_neurons, activation, layer_type);
                layers.push(layer);
            }
            LayerType::HiddenLayer => {
                layer = create_default_layer(num_h_neurons, num_h_neurons, activation, layer_type);
                layers.push(layer);
            }
            LayerType::OutputLayer => {
                layer = create_default_layer(num_h_neurons, num_outputs, activation, layer_type);
                layers.push(layer);
            }
        }
    }

    layers
}

pub fn create_default_layer(
    num_i_neurons: &usize,
    num_o_neurons: &usize,
    activation: &ActivationType,
    layer_type: LayerType
) -> Layer {
    // Initialize the matrices with the correct dimensions
    let mut weights: Vec<Vec<f64>> = Vec::new();

    initialize_weights(*num_i_neurons, *num_o_neurons, &mut weights); // 2D matrix

    // Create the layer with the initialized matrices
    let layer = Layer {
        weights,
        inactivated_output: Vec::new(),
        activated_output: Vec::new(),         // 2D matrix
        layer_bias: vec![1.0; *num_o_neurons], // 1D vector
        gradient_backward: Vec::new(),        // 2D matrix
        gradient_w: Vec::new(),               // 2D matrix
        errors: Vec::new(),                   // 2D matrix
        activation_type: activation.clone(),
        layer_type,
        previous_layer: None,
        next_layer: None,
        previous_gradient: Vec::new(), // 2D matrix
        m1: Vec::new(),
        v1: Vec::new(),
    };

    layer
}

// Helper function to determine the type of layer
pub fn get_layer_type(layer_idx: usize, total_layers: usize) -> LayerType {
    match layer_idx {
        0 => LayerType::InputLayer,
        x if x == total_layers - 1 => LayerType::OutputLayer,
        _ => LayerType::HiddenLayer,
    }
}
