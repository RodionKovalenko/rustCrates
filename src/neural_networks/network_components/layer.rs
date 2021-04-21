use core::fmt::Debug;
use crate::network_types::feedforward_network::FeedforwardNetwork;
use crate::utils::weights_initializer::initialize_weights;
use num::Zero;
use crate::utils::matrix::{create_generic, create_generic_one_dim, create_generic_3D};

#[derive(Debug, Clone)]
pub enum ActivationType {
    SIGMOID,
    LINEAR,
}

#[derive(Debug, Clone)]
pub enum LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
}

#[derive(Debug)]
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
}

impl<T: Debug + Clone + From<f64>> Layer<T> {
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
}

impl<T: Debug + Clone + From<f64>> Clone for Layer<T> {
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
        }
    }
}

pub fn initialize_layer<T: Debug + Clone + Zero + From<f64>>
(feed_net: &mut FeedforwardNetwork<T>) -> &mut Vec<Layer<T>> {
    let layers = &mut feed_net.layers;
    let total_number_of_layers = feed_net.number_of_hidden_layers + 2;
    let num_layer_inputs_dim1: usize = feed_net.input_dimensions[0];
    let mut num_layer_inputs_dim2: usize = feed_net.input_dimensions[1];
    let mut num_hidden_neurons = feed_net.number_of_hidden_neurons;
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
            num_layer_inputs_dim2 = num_hidden_neurons;
        } else if matches!(layer_type, LayerType::OutputLayer) {
            num_layer_inputs_dim2 = num_hidden_neurons;
            num_hidden_neurons = feed_net.number_of_output_neurons;
        }

        let input_layer: Layer<T> = Layer {
            input_weights: initialize_weights(num_layer_inputs_dim2,
                                              num_hidden_neurons),
            inactivated_output: create_generic_3D(num_layer_inputs_dim1, num_hidden_neurons, feed_net.input_dimensions[2]),
            activated_output: create_generic_3D(num_layer_inputs_dim1, num_hidden_neurons, feed_net.input_dimensions[2]),
            layer_bias: create_generic_one_dim(num_hidden_neurons),
            gradient: create_generic(num_layer_inputs_dim2, num_hidden_neurons),
            errors: create_generic(feed_net.input_dimensions[2], num_hidden_neurons),
            activation_type: ActivationType::SIGMOID,
            layer_type,
            previous_layer: None,
            next_layer: None,
            input_data: create_generic_3D(num_layer_inputs_dim1, num_hidden_neurons, feed_net.input_dimensions[2]),
        };

        layers.push(input_layer);
    }

    // set reference for next and previous layer
    for i in 0..layers.len() {
        if matches!(layers[i].layer_type, LayerType::InputLayer) {
            layers[i].previous_layer = None;
            layers[i].next_layer = Some(Box::<Layer<T>>::new(layers[i + 1].clone()));
        } else if matches!(layers[i].layer_type, LayerType::HiddenLayer) {
            layers[i].previous_layer = Some(Box::<Layer<T>>::new(layers[i - 1].clone()));
            layers[i].next_layer = Some(Box::<Layer<T>>::new(layers[i + 1].clone()));
        } else {
            layers[i].previous_layer = Some(Box::<Layer<T>>::new(layers[i - 1].clone()));
            layers[i].next_layer = None;
        }
    }

    layers
}

fn get_layer_type(num_layer: &i8, num_hidden_layers: &i8) -> LayerType {
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