use std::fmt::Debug;
use num_traits::{Float, FromPrimitive};
#[allow(unused_imports)]
use serde::{Serialize, Deserialize};
use crate::neural_networks::network_components::layer;
use crate::neural_networks::network_components::layer::Layer;

#[derive(Debug, Serialize, Deserialize)]
pub struct FeedforwardNetwork<T> {
    pub layers: Vec<Layer<T>>,
    pub learning_rate: f32,
    pub number_of_hidden_neurons: usize,
    pub input_dimensions: Vec<usize>,
    pub number_of_output_neurons: usize,
    pub number_of_hidden_layers: usize,
    pub number_data_sets: usize,
    pub number_rows_in_data: usize,
    pub number_columns_in_data: usize,
    pub minibatch_size: usize,
    pub data_type_value: T
}

impl<T: Debug + Clone + From<f64>> FeedforwardNetwork<T> {
    pub fn get_layers(&self) -> Vec<Layer<T>> {
        self.layers.clone()
    }
    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate.clone()
    }
    pub fn get_input_dimensions(&self) -> Vec<usize> {
        self.input_dimensions.clone()
    }
    pub fn get_number_of_hidden_neurons(&self) -> usize {
        self.number_of_hidden_neurons.clone()
    }
    pub fn get_number_of_output_neurons(&self) -> usize {
        self.number_of_output_neurons.clone()
    }
    pub fn get_number_of_hidden_layers(&self) -> usize { self.number_of_hidden_layers.clone() }
    pub fn get_number_data_sets(&self) -> usize {
        self.number_data_sets.clone()
    }
    pub fn get_number_rows_in_data(&self) -> usize {
        self.number_rows_in_data.clone()
    }
    pub fn get_number_columns_in_data(&self) -> usize {
        self.number_columns_in_data.clone()
    }
    pub fn get_minibatch_size(&self) -> usize {
        self.minibatch_size.clone()
    }
    pub fn get_data_type_value(&self) -> T { self.data_type_value.clone() }
}

impl<T: Debug + Clone + From<f64>> Clone for FeedforwardNetwork<T> {
    fn clone(&self) -> Self {
        FeedforwardNetwork {
            layers: self.get_layers(),
            learning_rate: self.get_learning_rate(),
            number_of_hidden_neurons: self.get_number_of_hidden_neurons(),
            input_dimensions: self.get_input_dimensions(),
            number_of_output_neurons: self.get_number_of_output_neurons(),
            number_of_hidden_layers: self.get_number_of_hidden_layers(),
            number_data_sets: self.get_number_data_sets(),
            number_rows_in_data: self.get_number_rows_in_data(),
            number_columns_in_data: self.get_number_columns_in_data(),
            minibatch_size: self.get_minibatch_size(),
            data_type_value: self.get_data_type_value(),
        }
    }
}

pub fn create<T: Debug + Clone + Float + FromPrimitive>(
    number_of_hidden_layers: usize,
    number_of_hidden_neurons: usize,
    input_dimensions: Vec<usize>,
    number_of_output_neurons: usize,
    number_data_sets: usize,
    number_rows_in_data: usize,
    number_columns_in_data: usize,
    minibatch_size: usize,
    learning_rate: f32,
    data_type_value: T,
) -> FeedforwardNetwork<T> {
    let layers = vec![];

    let mut feed_net = FeedforwardNetwork {
        layers,
        learning_rate,
        number_of_hidden_neurons,
        input_dimensions,
        number_of_output_neurons,
        number_of_hidden_layers,
        number_data_sets,
        number_rows_in_data,
        number_columns_in_data,
        minibatch_size,
        data_type_value,
    };

    layer::initialize_layer(&mut feed_net);

    feed_net
}