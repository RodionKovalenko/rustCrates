use crate::network_components::*;
use crate::network_components::layer::{Layer, ActivationType};

pub struct FeedforwardNetwork {
    pub layers: Vec<Layer<f64>>,
    pub learning_rate: f32,
    pub number_of_hidden_neurons: i32,
    // number of input parameters. For example if only 6 inputs, then input dimensions will be
    // [1][6]
    // if there are 25x8 input e.g. then [25][8]
    pub input_dimensions: Vec<i32>,
    pub number_of_output_neurons: i32,
    pub number_of_hidden_layers: i8,
}

pub fn create(
    number_of_hidden_layers: i8,
    number_of_hidden_neurons: i32,
    input_dimensions: Vec<i32>,
    number_of_output_neurons: i32,
    learning_rate: f32,
) -> FeedforwardNetwork {
    let layers = vec![];
    let mut feed_net =  FeedforwardNetwork {
        layers,
        learning_rate,
        number_of_hidden_neurons,
        input_dimensions,
        number_of_output_neurons,
        number_of_hidden_layers,
    };

    println!("number of layers before init {}", feed_net.layers.len());

    let layers = layer::initialize_layer(&mut feed_net);

    println!("number of layers after init {}", feed_net.layers.len());

    feed_net
}