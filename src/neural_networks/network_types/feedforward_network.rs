use crate::network_components::*;
use crate::utils::*;
use crate::network_components::layer::{Layer, LayerType};
use crate::network_components::input::*;
use activation::sigmoid;
use std::fmt::Debug;
use std::ops::{Mul, AddAssign, Add, Div};
use num::Zero;
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

    println!("number of layers before init {}", feed_net.layers.len());

    layer::initialize_layer(&mut feed_net);

    println!("number of layers after init {}", feed_net.layers.len());

    feed_net
}

pub fn forward<'b, T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Add<Output=T> + Div<Output=T>>
(input_structs: &mut Vec<Data<T>>, feed_net: &'b mut FeedforwardNetwork<T>)
 -> &'b mut FeedforwardNetwork<T> {
    let layers: &mut Vec<Layer<T>> = &mut feed_net.layers;

    for inputIndex in 0..input_structs.len() {
        for i in 0..layers.len() {
            if matches!(layers[i].layer_type, LayerType::InputLayer) {
                layers[i].inaktivated_output = matrix::multiple_generic(&input_structs[inputIndex].get_input(),
                                                                        &layers[i].input_weights.clone());
            } else {
                layers[i].inaktivated_output = matrix::multiple_generic(&layers[i - 1].aktivated_output,
                                                                        &layers[i].input_weights.clone());
            }

            layers[i].aktivated_output = sigmoid(&layers[i].inaktivated_output);

            // println!("weight matrix size rows: {}, columns: {}", layers[i].input_weights.len(),
            //          layers[i].input_weights[0].len());
            // println!("actived output size rows: {}, columns: {}", layers[i].aktivated_output.len(),
            //          layers[i].aktivated_output[0].len());
            // println!("inactivated output {:?}",  layers[i].inaktivated_output);
            // println!("");

            if matches!(layers[i].layer_type, LayerType::OutputLayer) {
                println!("");
                println!("");
                println!("activated output {:?}", layers[i].aktivated_output);
            }
        }
    }

    feed_net
}

pub fn predict() {}

pub fn train() {}

pub fn initialize() {
    let input = vec![
        vec![
            vec![1, 2, 3, 1, 3, 4]
        ],
        vec![
            vec![2, 4, 6, 1, 3, 4]
        ]
    ];

    let targets = vec![
        vec![1.0, 0.0, 0.5, 0.6],
        vec![0.0, 1.0, 0.8, 1.0],
    ];
    let mut parsed_input = parse_3_dim_to_float(&input);
    let mut input_struct;
    let mut input_structs = vec![];

    for i in 0..parsed_input.len() {
        input_struct = input::Data {
            input: parsed_input[i].clone(),
            target: vec![targets[i].clone()],
        };

        input_structs.push(input_struct);
    }

    let number_of_hidden_layers = 5;
    let input_dimensions = vec![parsed_input[0].len(), parsed_input[0][0].len()];
    let number_of_output_neurons = 4;
    let number_of_hidden_neurons = 15;
    let learning_rate = 0.02;

    let mut feedforward_network =
        create(
            number_of_hidden_layers,
            number_of_hidden_neurons,
            input_dimensions,
            number_of_output_neurons,
            learning_rate,
        );

    forward(&mut input_structs, &mut feedforward_network);
}