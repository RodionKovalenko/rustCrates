use crate::neural_networks::network_components::layer::{
    initialize_default_layers, ActivationType, Layer,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::network_trait::Network;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeedforwardNetwork {
    pub layers: Vec<Layer>,
    learning_rate: f32,
    number_of_input_neurons: usize,
    number_of_output_neurons: usize,
    number_of_hidden_layers: usize,
    number_of_hidden_neurons: usize,
    minibatch_size: usize,
}

impl Network for FeedforwardNetwork {
    fn get_layers(&self) -> Vec<Layer> {
        self.layers.clone()
    }
    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

impl FeedforwardNetwork {
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
) -> FeedforwardNetwork {
    let layers = vec![];

    let mut feed_net = FeedforwardNetwork {
        layers,
        learning_rate,
        number_of_input_neurons: number_inputs,
        number_of_output_neurons: number_outputs,
        number_of_hidden_layers,
        number_of_hidden_neurons,
        minibatch_size,
    };

    let layers = initialize_default_layers(
        &number_inputs,
        &number_outputs,
        &number_of_hidden_layers,
        &number_of_hidden_neurons,
        &ActivationType::RANDOM,
    );

    feed_net.layers = layers;
    feed_net
}
