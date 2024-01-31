use std::fmt::Debug;
use num_traits::{Float, FromPrimitive};
#[allow(unused_imports)]
use serde::{Serialize, Deserialize};
use crate::neural_networks::network_components::{layer};
use crate::neural_networks::network_components::layer::{Layer};

#[derive(Debug, Serialize, Deserialize)]
pub struct FeedforwardNetwork<T> {
    pub layers: Vec<Layer<T>>,
    pub learning_rate: f32,
    pub number_of_hidden_neurons: i32,
    // number of input.rs parameters. For example if only 6 inputs, then input.rs dimensions will be
    // [1][6]
    // if there are 25x8 input.rs e.g. then [25][8]
    // or [25][1][3] => means 25 data sets with data input [1][3]
    pub input_dimensions: Vec<usize>,
    pub number_of_output_neurons: i32,
    pub number_of_hidden_layers: i32,
    pub number_data_sets: i32,
    pub number_rows_in_data: i32,
    pub number_columns_in_data: i32,
    pub minibatch_size: i32,
    pub data_type_value: T,
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
    pub fn get_number_of_hidden_neurons(&self) -> i32 {
        self.number_of_hidden_neurons.clone()
    }
    pub fn get_number_of_output_neurons(&self) -> i32 {
        self.number_of_output_neurons.clone()
    }
    pub fn get_number_of_hidden_layers(&self) -> i32 { self.number_of_hidden_layers.clone() }
    pub fn get_number_data_sets(&self) -> i32 {
        self.number_data_sets.clone()
    }
    pub fn get_number_rows_in_data(&self) -> i32 {
        self.number_rows_in_data.clone()
    }
    pub fn get_number_columns_in_data(&self) -> i32 {
        self.number_columns_in_data.clone()
    }
    pub fn get_minibatch_size(&self) -> i32 {
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
    number_of_hidden_layers: i32,
    number_of_hidden_neurons: i32,
    input_dimensions: Vec<usize>,
    number_of_output_neurons: i32,
    number_data_sets: i32,
    number_rows_in_data: i32,
    number_columns_in_data: i32,
    minibatch_size: i32,
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