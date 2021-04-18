use crate::network_components::*;
use crate::utils::*;
use crate::network_components::layer::{Layer, LayerType};
use crate::network_components::input::*;
use activation::sigmoid;
use std::fmt::Debug;
use std::ops::{Mul, AddAssign, Add, Div, Sub};
use num::{Zero, range};
#[allow(unused_imports)]
use matrix::parse_2dim_to_float;
use crate::utils::matrix::parse_3_dim_to_float;

pub struct FeedforwardNetwork<T> {
    pub layers: Vec<Layer<T>>,
    pub learning_rate: f32,
    pub number_of_hidden_neurons: usize,
    // number of input.rs parameters. For example if only 6 inputs, then input.rs dimensions will be
    // [1][6]
    // if there are 25x8 input.rs e.g. then [25][8]
    pub input_dimensions: Vec<usize>,
    pub number_of_output_neurons: usize,
    pub number_of_hidden_layers: i8,
}

pub fn create<T: Debug + Clone + Zero + From<f64>>(
    number_of_hidden_layers: i8,
    number_of_hidden_neurons: usize,
    input_dimensions: Vec<usize>,
    number_of_output_neurons: usize,
    learning_rate: f32,
) -> FeedforwardNetwork<T> {
    let layers = vec![];

    let mut feed_net = FeedforwardNetwork {
        layers,
        learning_rate,
        number_of_hidden_neurons,
        input_dimensions,
        number_of_output_neurons,
        number_of_hidden_layers,
    };

    layer::initialize_layer(&mut feed_net);
    feed_net
}

pub fn forward<'a, T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Sub<Output=T> + Add<Output=T> + Div<Output=T>>
(data_structs: &mut Vec<Data<T>>, feed_net: &'a mut FeedforwardNetwork<T>)
 -> &'a mut FeedforwardNetwork<T> {
    let layers: &mut Vec<Layer<T>> = &mut feed_net.layers;

    for input_index in 0..data_structs.len() {
        for i in 0..layers.len() {
            if matches!(layers[i].layer_type, LayerType::InputLayer) {
                layers[i].input_data = data_structs[input_index].get_input();
                layers[i].inactivated_output = matrix::multiple_generic(&data_structs[input_index].get_input(),
                                                                        &layers[i].input_weights.clone());
            } else {
                layers[i].input_data = layers[i - 1].activated_output.clone();
                layers[i].inactivated_output = matrix::multiple_generic(&layers[i - 1].activated_output,
                                                                        &layers[i].input_weights.clone());
            }

            layers[i].activated_output = sigmoid(&layers[i].inactivated_output);


            // println!("actived output size rows: {}, columns: {}", layers[i].aktivated_output.len(),
            //          layers[i].aktivated_output[0].len());
            // println!("inactivated output {:?}",  layers[i].inaktivated_output);
            // println!("");

            // println!("input matrix: rows: {}, columns: {}", layers[i].input_data.len(), layers[i].input_data[0].len());
            // println!("weight matrix: rows: {}, columns: {}", layers[i].input_weights.len(),
            //          layers[i].input_weights[0].len());
            // println!(" --------------\
            // output matrix: rows: {}, columns: {}", layers[i].activated_output.len(), layers[i].activated_output[0].len());

            if matches!(layers[i].layer_type, LayerType::OutputLayer) {
                println!("target: {:?}", &data_structs[input_index].get_target());
                println!("activated output {:?}", layers[i].activated_output);
                println!("errors: {:?}", layers[i].errors);

                let errors = matrix::get_error(&data_structs[input_index].get_target(),
                                              &layers[i].activated_output);

                layers[i].errors = matrix::add(&errors,
                                               &layers[i].errors.clone());

                println!("errors: {:?}", layers[i].errors);
                println!("");
                println!("");
            }
        }
    }

    feed_net
}

pub fn predict() {}

pub fn train<'a, T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Sub<Output=T> + Add<Output=T> + Div<Output=T>>
(data_structs: &mut Vec<Data<T>>, feed_net: &'a mut FeedforwardNetwork<T>)
 -> &'a mut FeedforwardNetwork<T> {


    for iter in 0 .. 10 {
        forward(data_structs, feed_net);

        for i in range(0, feed_net.layers.len()).rev() {
            train::calculate_gradient(&mut feed_net.layers, i);
            //train::update_weights(&mut feed_net.layers, i);

           // println!("updated weights {:?}, {:?}", feed_net.layers[i].layer_type, feed_net.layers[i].input_weights);
        }

        // clear errors and gradients after update
        for i in range(0, feed_net.layers.len()).rev() {
            let num_rows =  feed_net.layers[i].gradient.len();
            let num_columns =  feed_net.layers[i].gradient[0].len();
            feed_net.layers[i].errors = matrix::create_generic(num_rows, num_columns);
            feed_net.layers[i].gradient = matrix::create_generic(num_rows, num_columns);
        }
    }

    feed_net
}

pub fn initialize() {
    let input = vec![
        vec![
            vec![5, 2, 9, 1, 5, 4],
        ],
        vec![
            vec![3, 0, 5, 1, 2, 6],
        ]
    ];

    let targets = vec![
        vec![1.0, 0.3, 1.0, 0.0],
        vec![0.0, 1.0, 0.5, 1.0],
    ];
    let parsed_input = parse_3_dim_to_float(&input);
    let mut input_struct;
    let mut data_structs = vec![];

    for i in 0..parsed_input.len() {
        input_struct = input::Data {
            input: parsed_input[i].clone(),
            target: vec![targets[i].clone()],
        };

        data_structs.push(input_struct);
    }

    let number_of_hidden_layers = 1;
    let input_dimensions = vec![parsed_input[0].len(), parsed_input[0][0].len()];
    let number_of_output_neurons = 4;
    let number_of_hidden_neurons = 5;
    let learning_rate = 0.02;

    let mut feedforward_network: FeedforwardNetwork<f64> =
        create(
            number_of_hidden_layers,
            number_of_hidden_neurons,
            input_dimensions,
            number_of_output_neurons,
            learning_rate,
        );

    train(&mut data_structs, &mut feedforward_network);
}