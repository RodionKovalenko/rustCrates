use crate::network_components::*;
use crate::utils::*;
use crate::network_components::layer::{Layer, LayerType};
use std::fmt::Debug;
use std::ops::{Mul, AddAssign};
use num::Zero;

pub struct FeedforwardNetwork<T> {
    pub layers: Vec<Layer<T>>,
    pub learning_rate: f32,
    pub number_of_hidden_neurons: usize,
    // number of input parameters. For example if only 6 inputs, then input dimensions will be
    // [1][6]
    // if there are 25x8 input e.g. then [25][8]
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

pub fn forward<'a, T: Debug + Clone + Mul<Output=T> + AddAssign + From<f64>>
(input_vec: &mut Vec<Vec<T>>, feed_net: &'a mut FeedforwardNetwork<T>)
 -> &'a mut FeedforwardNetwork<T> {
    let layers: &mut Vec<Layer<T>> = &mut feed_net.layers;
    let mut layer_output: Vec<Vec<T>>;

    for i in 0..layers.len() {
        if matches!(layers[i].layer_type, LayerType::InputLayer) {
            layer_output = matrix::multiple_generic(&input_vec,
                                                    &layers[i].input_weights.clone());
        } else if matches!(layers[i].layer_type, LayerType::HiddenLayer) {
            layer_output = matrix::multiple_generic(&layers[i - 1].inaktivated_output,
                                                    &layers[i].input_weights.clone());
        } else {
            layer_output = matrix::multiple_generic(&layers[i - 1].inaktivated_output,
                                                    &layers[i].input_weights.clone());
        }

        println!("output matrix {:?}", layer_output);
        println!("");

        layers[i].inaktivated_output = layer_output;
    }

    feed_net
}