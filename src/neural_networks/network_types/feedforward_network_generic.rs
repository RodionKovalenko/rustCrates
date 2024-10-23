use crate::neural_networks::network_components::layer::{
    initialize_default_layers, ActivationType, Layer, LayerEnum,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::network_trait::Network;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedforwardNetwork<const M: usize, const N: usize> {
    pub layers: Vec<LayerEnum<M, N>>,
    learning_rate: f32,
    number_of_input_neurons: usize,
    number_of_output_neurons: usize,
    number_of_hidden_layers: usize,
    number_of_hidden_neurons: usize,
    minibatch_size: usize,
}

// Implement the Network trait with generics for M and N
impl<const M: usize, const N: usize> Network<M, N> for FeedforwardNetwork<M, N> {
    fn get_layers(&self) -> Vec<Box<Layer<M, N>>> {
        self.layers.iter().filter_map(|layer_enum| {
            match layer_enum {
                LayerEnum::Dense(layer) => Some(layer.clone()), // Ensure layer is cloned properly
                LayerEnum::RMSNorm(layer) => Some(layer.clone()), // Ensure layer is cloned properly
                LayerEnum::SelfAttention(layer) => Some(layer.clone()), // Ensure layer is cloned properly
            }
        }).collect() // Collect into Vec<Box<dyn Layer<M, N>>>
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}
// Provide more flexible methods for getting properties of the network
impl<const M: usize, const N: usize> FeedforwardNetwork<M, N> {
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
    pub fn extract_layers(&self) -> Vec<Layer<M, N>> {
        self.layers.iter().filter_map(|layer_enum| {
            match layer_enum {
                LayerEnum::Dense(layer) => Some(*layer.clone()), // Dereference and clone the inner layer
                LayerEnum::RMSNorm(layer) => Some(*layer.clone()), // Dereference and clone the inner layer
                LayerEnum::SelfAttention(layer) => Some(*layer.clone()), // Dereference and clone the inner layer
            }
        }).collect() // Collect into a Vec<Layer<M, N>>
    }
}

pub fn create<const M: usize, const N: usize>(
    number_inputs: usize,
    number_outputs: usize,
    number_of_hidden_layers: usize,
    number_of_hidden_neurons: usize,
    minibatch_size: usize,
    learning_rate: f32,
    activation: ActivationType, // Allow activation type to be passed as a parameter
) -> FeedforwardNetwork<M, N> {
    let mut feed_net = FeedforwardNetwork::<M, N> {
        layers: vec![],
        learning_rate,
        number_of_input_neurons: number_inputs,
        number_of_output_neurons: number_outputs,
        number_of_hidden_layers,
        number_of_hidden_neurons,
        minibatch_size,
    };

    // Use the passed activation type for layer initialization
    let layers = initialize_default_layers::<M, N>(
        &number_outputs,
        &number_of_hidden_layers,
        &activation, // Use the passed activation
    );

    feed_net.layers = layers;
    feed_net
}
