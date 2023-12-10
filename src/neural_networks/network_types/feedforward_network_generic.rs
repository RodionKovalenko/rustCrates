use std::fmt::Debug;
use std::ops::{Mul, AddAssign, Add, Div, Sub};
use num::{range};
#[allow(unused_imports)]
use serde::{Serialize, Deserialize};
use crate::neural_networks::network_components::input::Data;
use crate::neural_networks::network_components::{input, layer};
use crate::neural_networks::network_components::layer::{Layer, LayerType};
use crate::neural_networks::utils::activation::sigmoid;
use crate::neural_networks::utils::{matrix, train_generic};

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
    pub fn get_number_of_hidden_layers(&self) -> i32 {  self.number_of_hidden_layers.clone()  }
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

pub fn create<T: Clone>(
    number_of_hidden_layers: i32,
    number_of_hidden_neurons: i32,
    input_dimensions: Vec<usize>,
    number_of_output_neurons: i32,
    number_data_sets: i32,
    number_rows_in_data: i32,
    number_columns_in_data: i32,
    minibatch_size: i32,
    learning_rate: f32,
    data_type_value: T
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
        data_type_value
    };

    let data_type_value = feed_net.data_type_value.clone();
    layer::initialize_layer(&mut feed_net, data_type_value);
    feed_net
}

pub fn forward<'a, T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Sub<Output=T> + Add<Output=T> + Div<Output=T>>
(data_structs: &mut Vec<Data<T>>, feed_net: &'a mut FeedforwardNetwork<T>, show_output: bool)
 -> &'a mut FeedforwardNetwork<T> {
    let layers: &mut Vec<Layer<T>> = &mut feed_net.layers;

    for input_index in 0..data_structs.len() {
        for i in 0..layers.len() {
            if matches!(layers[i].layer_type, LayerType::InputLayer) {
                layers[i].input_data[input_index] = data_structs[input_index].get_input();
                layers[i].inactivated_output[input_index] =
                    matrix::multiple_generic_2d(&data_structs[input_index].get_input(),
                                                &layers[i].input_weights.clone());
            } else {
                layers[i].input_data[input_index] = layers[i - 1].activated_output[input_index].clone();
                layers[i].inactivated_output[input_index] =
                    matrix::multiple_generic_2d(&layers[i - 1].activated_output[input_index],
                                                &layers[i].input_weights.clone());
            }

            layers[i].inactivated_output[input_index] = matrix::add(&layers[i].inactivated_output[input_index], &layers[i].layer_bias);
            layers[i].activated_output[input_index] = sigmoid(&layers[i].inactivated_output[input_index]);

            if matches!(layers[i].layer_type, LayerType::OutputLayer) {
                let errors = matrix::get_error(&data_structs[input_index].get_target(),
                                               &layers[i].activated_output[input_index]);

                layers[i].errors[input_index] = errors;

                if input_index == data_structs.len() - 1 {
                    if show_output {
                        if show_output {
                            for ind in 0..data_structs.len() {
                                println!("target: {:?}", &data_structs[ind].get_target());
                                println!("activated output {:?}", layers[i].activated_output[ind]);
                            }
                            println!("");
                        }
                    }
                }
            }
        }
    }

    feed_net
}

pub fn train_generic<'a, T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Sub<Output=T> + Add<Output=T> + Div<Output=T>>
(data_structs: &mut Vec<Data<T>>, feed_net: &'a mut FeedforwardNetwork<T>)
 -> &'a mut FeedforwardNetwork<T> {
    println!("Training beginns");
    for _iter in 0..8000 {
        if _iter % 1000 == 0 {
            forward(data_structs, feed_net, true);
        } else {
            forward(data_structs, feed_net, false);
        }

        for i in range(0, feed_net.layers.len()).rev() {
            train_generic::calculate_gradient(&mut feed_net.layers, i,
                                              data_structs.len(),
                                              feed_net.learning_rate as f64);
        }

        // clear errors and gradients after update
        for i in range(0, feed_net.layers.len()).rev() {
            for k in 0..feed_net.layers[i].errors.len() {
                for j in 0..feed_net.layers[i].errors[0].len() {
                    feed_net.layers[i].errors[k][j] = T::from(0.0);
                }
            }
            for k in 0..feed_net.layers[i].gradient.len() {
                for j in 0..feed_net.layers[i].gradient[0].len() {
                    feed_net.layers[i].gradient[k][j] = T::from(0.0);
                }
            }
        }
    }

    forward(data_structs, feed_net, true);

    feed_net
}

pub fn initialize() {
    let input: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![1.0, 0.0]],
        vec![vec![0.0, 0.0]],
        vec![vec![0.0, 1.0]],
        vec![vec![1.0, 1.0]]
    ];

    let targets = vec![
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
    ];
    let parsed_input = input;
    let mut input_struct;
    let mut data_structs = vec![];

    for i in 0..parsed_input.len() {
        input_struct = input::Data {
            input: parsed_input[i].clone(),
            target: vec![targets[i].clone()],
        };

        data_structs.push(input_struct);
    }

    println!("parsed input size:  {}, {}, {}", parsed_input.len(), parsed_input[0].len(), parsed_input[0][0].len());

    let number_of_hidden_layers = 1;
    let input_dimensions = vec![parsed_input[0].len(), parsed_input[0][0].len(), parsed_input.len()];
    let number_of_output_neurons = targets[0].len() as i32;
    let number_of_hidden_neurons = 30;
    let number_of_data_sets = parsed_input.len() as i32;
    let number_rows_in_set = parsed_input[0].len() as i32;
    let num_columns_in_set = parsed_input[0][0].len() as i32;
    let learning_rate = 0.1;
    let minibatch_size = 50;

    let data_type_value: f64 = 0.0;

    let mut feedforward_network: FeedforwardNetwork<f64> =
        create(
            number_of_hidden_layers,
            number_of_hidden_neurons,
            input_dimensions,
            number_of_output_neurons,
            number_of_data_sets,
            number_rows_in_set,
            num_columns_in_set,
            minibatch_size,
            learning_rate,
            data_type_value
        );

    train_generic(&mut data_structs, &mut feedforward_network);
}