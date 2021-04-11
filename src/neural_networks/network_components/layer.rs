use core::fmt::Debug;
use std::fmt::{Display, Formatter, Result};
use crate::network_types::feedforward_network::FeedforwardNetwork;
use crate::utils::weights_initializer::initialize_weights;
use std::borrow::{BorrowMut, Borrow};

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
    pub inaktivated_output: Vec<Vec<T>>,
    pub aktivated_output: Vec<Vec<T>>,
    pub layer_bias: Vec<T>,
    pub gradient: Vec<Vec<T>>,
    pub errors: Vec<Vec<T>>,
    pub activation_type: ActivationType,
    pub layer_type: LayerType,
    pub previous_layer: Option<Box<Layer<T>>>,
    pub next_layer: Option<Box<Layer<T>>>,
}

impl Layer<f64> {
    pub fn get_input_weights(&self) -> Vec<Vec<f64>> {
        self.input_weights.clone()
    }
    pub fn get_inaktivatede_output(&self) -> Vec<Vec<f64>> {
        self.inaktivated_output.clone()
    }
    pub fn get_aktivated_output(&self) -> Vec<Vec<f64>> {
        self.aktivated_output.clone()
    }
    pub fn get_layer_bias(&self) -> Vec<f64> {
        self.layer_bias.clone()
    }
    pub fn get_gradient(&self) -> Vec<Vec<f64>> {
        self.gradient.clone()
    }
    pub fn get_errors(&self) -> Vec<Vec<f64>> {
        self.errors.clone()
    }
    pub fn get_activation_type(&self) -> ActivationType {
        self.activation_type.clone()
    }
    pub fn get_layer_type(&self) -> LayerType {
        self.layer_type.clone()
    }
    pub fn get_previous_layer(&self) -> Option<Box<Layer<f64>>> {
        self.previous_layer.clone()
    }
    pub fn get_next_layer(&self) -> Option<Box<Layer<f64>>> {
        self.next_layer.clone()
    }
}

impl Clone for Layer<f64> {
    fn clone(&self) -> Self {
        Layer {
            input_weights: self.get_input_weights(),
            inaktivated_output: self.get_inaktivatede_output(),
            aktivated_output: self.get_aktivated_output(),
            layer_bias: self.get_layer_bias(),
            gradient: self.get_gradient(),
            errors: self.get_errors(),
            activation_type: self.get_activation_type(),
            layer_type: self.get_layer_type(),
            previous_layer: self.get_previous_layer(),
            next_layer: self.get_next_layer(),
        }
    }
}

pub fn initialize_layer(feed_net: &mut FeedforwardNetwork) -> &mut Vec<Layer<f64>> {
    let mut layers = &mut feed_net.layers;
    let total_number_of_layers = feed_net.number_of_hidden_layers + 2;
    let num_layer_inputs_dim1: i32 = feed_net.input_dimensions[0];
    let mut num_layer_inputs_dim2: i32 = feed_net.input_dimensions[1];
    let mut num_hidden_neurons = feed_net.number_of_hidden_neurons;
    let mut input_layer;
    let mut layer_type;

    if total_number_of_layers == 0 {
        panic!("number of hidden layers cannot be 0");
    }

    for numLayer in 0..total_number_of_layers {
        layer_type = get_layer_type(
            &numLayer,
            &total_number_of_layers,
        );

        if matches!(layer_type, LayerType::HiddenLayer) {
            num_layer_inputs_dim2 = num_hidden_neurons;
        } else if matches!(layer_type, LayerType::OutputLayer) {
            num_layer_inputs_dim2 = feed_net.number_of_output_neurons;
        }

        input_layer = Layer {
            input_weights: initialize_weights(num_layer_inputs_dim1,
                                              num_layer_inputs_dim2,
                                              num_hidden_neurons),
            inaktivated_output: vec![vec![]],
            aktivated_output: vec![vec![]],
            layer_bias: vec![],
            gradient: vec![vec![]],
            errors: vec![vec![]],
            activation_type: ActivationType::SIGMOID,
            layer_type,
            previous_layer: None,
            next_layer: None,
        };

        layers.push(input_layer);
    }

    for i in 0..layers.len() {
        layers[i].input_weights = vec![
            vec![0.0f64; 1 as usize];
            1 as usize
        ];
        layers[i].input_weights[0] = vec![123.23];

        if matches!(layers[i].layer_type, LayerType::HiddenLayer) {
            layers[i - 1].input_weights[0] = vec![123.23];
            layers[i - 1].input_weights = vec![
                vec![0.0f64; 1 as usize];
                1 as usize
            ];
            layers[i + 1].input_weights[0] = vec![123.23];
            layers[i + 1].input_weights = vec![
                vec![0.0f64; 1 as usize];
                1 as usize
            ];
        }

        if matches!(layers[i].layer_type, LayerType::InputLayer) {
            layers[i + 1].input_weights[0] = vec![123.23];
            layers[i + 1].input_weights = vec![
                vec![0.0f64; 1 as usize];
                1 as usize
            ];
        }

        if matches!(layers[i].layer_type, LayerType::InputLayer) {
            layers[i].previous_layer = None;
            layers[i].next_layer = Some(Box::<Layer<f64>>::new(layers[i + 1].clone()));
        } else if matches!(layers[i].layer_type, LayerType::HiddenLayer) {
            layers[i].previous_layer = Some(Box::<Layer<f64>>::new(layers[i - 1].clone()));
            layers[i].next_layer = Some(Box::<Layer<f64>>::new(layers[i + 1].clone()));
        } else {
            layers[i].previous_layer = Some(Box::<Layer<f64>>::new(layers[i - 1].clone()));
            layers[i].next_layer = None;
        }
    }

    for i in 0..layers.len() {
        println!("{:?}", layers[i]);

        println!("");
        println!("");
        println!("");
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