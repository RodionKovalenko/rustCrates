use crate::network_components::*;
use crate::utils::*;
use crate::network_components::layer::{Layer, ActivationType, LayerType};

pub struct FeedforwardNetwork {
    pub layers: Vec<Layer<f64>>,
    pub learning_rate: f32,
    pub number_of_hidden_neurons: usize,
    // number of input parameters. For example if only 6 inputs, then input dimensions will be
    // [1][6]
    // if there are 25x8 input e.g. then [25][8]
    pub input_dimensions: Vec<usize>,
    pub number_of_output_neurons: usize,
    pub number_of_hidden_layers: i8,
}

pub fn create(
    number_of_hidden_layers: i8,
    number_of_hidden_neurons: usize,
    input_dimensions: Vec<usize>,
    number_of_output_neurons: usize,
    learning_rate: f32,
) -> FeedforwardNetwork {
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

    let layers = layer::initialize_layer(&mut feed_net);

    println!("number of layers after init {}", feed_net.layers.len());

    feed_net
}

pub fn forward<'a>(inputVec: &mut Vec<Vec<f64>>, feed_net: &'a mut FeedforwardNetwork) -> &'a mut FeedforwardNetwork {
    let mut layers = &mut feed_net.layers;
    let mut currentLayer: Layer<f64>;
    let mut layerWeights: Vec<Vec<f64>>;
    let mut layerOutput: Vec<Vec<f64>>;
    let mut num_columns: usize;
    let mut num_rows: usize;

    for i in 0..layers.len() {
        if matches!(layers[i].layer_type, LayerType::InputLayer) {
            num_rows = feed_net.input_dimensions[1] as usize;
            num_columns = feed_net.number_of_hidden_neurons as usize;

            layerOutput = matrix::multiple_generic(&inputVec,
                                           &layers[i].input_weights.clone());
        } else if matches!(layers[i].layer_type, LayerType::HiddenLayer) {
            num_rows = feed_net.number_of_hidden_neurons as usize;
            num_columns = feed_net.number_of_hidden_neurons as usize;

            layerOutput = matrix::multiple_generic(&layers[i - 1].inaktivated_output,
                                           &layers[i].input_weights.clone());
        } else {
            num_rows = feed_net.number_of_output_neurons as usize;
            num_columns = feed_net.number_of_output_neurons as usize;
            layerOutput = matrix::multiple_generic(&layers[i - 1].inaktivated_output,
                                           &layers[i].input_weights.clone());
        }

        println!("output matrix {:?}", layerOutput);
        println!("");

        layers[i].inaktivated_output = layerOutput;
    }

    feed_net
}