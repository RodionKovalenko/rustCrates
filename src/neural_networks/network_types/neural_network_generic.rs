use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::neural_networks::network_components::layer::LayerEnum;

pub const FILE_NAME: &str = "feedforward_network.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<LayerEnum>,
    pub learning_rate: f32,
    pub number_of_input_neurons: usize,
    pub number_of_output_neurons: usize,
    pub number_of_hidden_layers: usize,
    pub number_of_hidden_neurons: usize,
    pub minibatch_size: usize,
}

// Provide more flexible methods for getting properties of the network
impl NeuralNetwork {
    pub fn get_number_of_input_neurons(&self) -> usize {
        self.number_of_input_neurons
    }
    pub fn get_number_of_output_neurons(&self) -> usize {
        self.number_of_output_neurons
    }
    pub fn get_number_of_hidden_neurons(&self) -> usize {
        self.number_of_hidden_neurons
    }
    pub fn get_number_of_hidden_layers(&self) -> usize {
        self.number_of_hidden_layers
    }
    pub fn get_minibatch_size(&self) -> usize {
        self.minibatch_size
    }
}

pub fn create(
    number_inputs: usize,
    number_outputs: usize,
    number_of_hidden_layers: usize,
    number_of_hidden_neurons: usize,
    minibatch_size: usize,
    learning_rate: f32,
) -> NeuralNetwork {
    let feed_net = NeuralNetwork {
        layers: vec![],
        learning_rate,
        number_of_input_neurons: number_inputs,
        number_of_output_neurons: number_outputs,
        number_of_hidden_layers,
        number_of_hidden_neurons,
        minibatch_size,
    };

    feed_net
}
