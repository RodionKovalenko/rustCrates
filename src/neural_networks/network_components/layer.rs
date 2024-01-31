use core::fmt::Debug;
use num_traits::{Float, FromPrimitive};
use serde::{Serialize, Deserialize};
use crate::neural_networks::network_types::feedforward_network_generic::FeedforwardNetwork;
use crate::neural_networks::utils::matrix::{create_generic, create_generic_3d, create_generic_one_dim};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    SIGMOID,
    TANH,
    LINEAR,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Layer<T> {
    pub input_weights: Vec<Vec<T>>,
    pub inactivated_output: Vec<Vec<Vec<T>>>,
    pub activated_output: Vec<Vec<Vec<T>>>,
    pub gradient: Vec<Vec<T>>,
    pub errors: Vec<Vec<T>>,
    pub layer_bias: Vec<T>,
    pub activation_type: ActivationType,
    pub layer_type: LayerType,
    pub previous_layer: Option<Box<Layer<T>>>,
    pub next_layer: Option<Box<Layer<T>>>,
    pub input_data: Vec<Vec<Vec<T>>>,
    pub previous_gradient: Vec<Vec<T>>,
    pub m1: Vec<Vec<T>>,
    pub v1: Vec<Vec<T>>,
}

impl<T: Debug + Clone> Layer<T> {
    pub fn get_input_weights(&self) -> Vec<Vec<T>> {
        self.input_weights.clone()
    }
    pub fn get_inaktivatede_output(&self) -> Vec<Vec<Vec<T>>> {
        self.inactivated_output.clone()
    }
    pub fn get_aktivated_output(&self) -> Vec<Vec<Vec<T>>> {
        self.activated_output.clone()
    }
    pub fn get_layer_bias(&self) -> Vec<T> {
        self.layer_bias.clone()
    }
    pub fn get_gradient(&self) -> Vec<Vec<T>> {
        self.gradient.clone()
    }
    pub fn get_errors(&self) -> Vec<Vec<T>> {
        self.errors.clone()
    }
    pub fn get_activation_type(&self) -> ActivationType {
        self.activation_type.clone()
    }
    pub fn get_layer_type(&self) -> LayerType {
        self.layer_type.clone()
    }
    pub fn get_previous_layer(&self) -> Option<Box<Layer<T>>> {
        self.previous_layer.clone()
    }
    pub fn get_next_layer(&self) -> Option<Box<Layer<T>>> {
        self.next_layer.clone()
    }
    pub fn get_input_data(&self) -> Vec<Vec<Vec<T>>> {
        self.input_data.clone()
    }
    pub fn get_previous_gradient(&self) -> Vec<Vec<T>> {
        self.previous_gradient.clone()
    }
    pub fn get_m1(&self) -> Vec<Vec<T>> {
        self.m1.clone()
    }
    pub fn get_v1(&self) -> Vec<Vec<T>> {
        self.v1.clone()
    }
}

impl<T: Debug + Clone> Clone for Layer<T> {
    fn clone(&self) -> Self {
        Layer {
            input_weights: self.get_input_weights(),
            inactivated_output: self.get_inaktivatede_output(),
            activated_output: self.get_aktivated_output(),
            layer_bias: self.get_layer_bias(),
            gradient: self.get_gradient(),
            errors: self.get_errors(),
            activation_type: self.get_activation_type(),
            layer_type: self.get_layer_type(),
            previous_layer: self.get_previous_layer(),
            next_layer: self.get_next_layer(),
            input_data: self.get_input_data(),
            previous_gradient: self.get_previous_gradient(),
            m1: self.get_m1(),
            v1: self.get_v1(),
        }
    }
}

pub fn initialize_layer<T: Debug + Clone + Float + FromPrimitive>(feed_net: &mut FeedforwardNetwork<T>) -> &Vec<Layer<T>> {
    let layers = &mut feed_net.layers;
    let total_number_of_layers: i32 = feed_net.number_of_hidden_layers.clone() + 2;
    let num_rows: i32 = feed_net.number_rows_in_data.clone();
    let mut num_columns: i32 = feed_net.number_columns_in_data.clone();
    let mut num_hidden_neurons: i32 = feed_net.number_of_hidden_neurons.clone();
    let mut layer_type;

    if total_number_of_layers == 0 {
        panic!("number of hidden layers cannot be 0");
    }

    for num_layer in 0..total_number_of_layers {
        layer_type = get_layer_type(
            &num_layer,
            &total_number_of_layers,
        );

        if matches!(layer_type, LayerType::HiddenLayer) {
            //num_columns.clone();
        } else if matches!(layer_type, LayerType::OutputLayer) {
            num_columns = num_hidden_neurons.clone();
            num_hidden_neurons = feed_net.number_of_output_neurons.clone();
        }

        let mut weight_matrix: Vec<Vec<T>> = Vec::new();
        let mut weight_matrix_m1: Vec<Vec<T>> = Vec::new();
        let mut weight_matrix_m2: Vec<Vec<T>> = Vec::new();

        let input_layer: Layer<T> = Layer {
            input_weights: weight_matrix,
            inactivated_output: create_generic_3d(num_rows, num_hidden_neurons),
            activated_output: create_generic_3d(num_rows, num_hidden_neurons),
            layer_bias: create_generic_one_dim(),
            gradient: create_generic(num_columns),
            errors: create_generic(feed_net.input_dimensions[2] as i32),
            activation_type: ActivationType::TANH,
            layer_type,
            previous_layer: None,
            next_layer: None,
            input_data: create_generic_3d(num_rows, num_hidden_neurons),
            previous_gradient: create_generic(num_columns),
            m1: weight_matrix_m1,
            v1: weight_matrix_m2,
        };

        layers.push(input_layer);
    }

    layers
}

pub fn get_layer_type(num_layer: &i32, num_hidden_layers: &i32) -> LayerType {
    let layer_type;

    if num_layer == &0 {
        layer_type = LayerType::InputLayer;
    } else if num_layer > &0 && num_layer < &(num_hidden_layers - 1) {
        layer_type = LayerType::HiddenLayer;
    } else {
        layer_type = LayerType::OutputLayer;
    }

    layer_type
}