use num::{range};
use crate::neural_networks::network_components::input::Data;
use crate::neural_networks::network_components::layer;
use crate::neural_networks::network_components::layer::{Layer, LayerType};
use crate::neural_networks::network_types::feedforward_network_generic::{create, FeedforwardNetwork};
use crate::neural_networks::utils::file::{deserialize, serialize};
use crate::neural_networks::utils::matrix::create_generic_3d;

#[allow(unused_imports)]
#[allow(unused_imports)]
pub const FILE_NAME: &str = "feedforward_network.json";

pub fn forward(feed_net: &mut FeedforwardNetwork<f64>,
               _data_input: &Data<f64, f64>) {
    let layers: &mut Vec<Layer<f64>> = &mut feed_net.layers;


    for _i in 0..layers.len() {

    }

    feed_net.layers = layers.clone();
}

pub fn backward(_feed_net: &mut FeedforwardNetwork<f64>,
                _data_input: &Data<f64, f64>) {

    // train::calculate_gradient(&mut feed_net.layers, &i,
    //                           &data_structs.len(),
    //                           &(feed_net.learning_rate as f64),
    //                           &_iter,
    //                           &minibatch_start,
    //                           minibatch_size,
    // );
}

pub fn calculate_total_loss(data_structs: &mut Vec<Data<f64, f64>>,
                            feed_net: &mut FeedforwardNetwork<f64>) -> f64 {
    let mut total_loss = 0.0;
    for ind in 0..data_structs.len() {
        for e in 0..feed_net.layers[feed_net.layers.len() - 1].errors[ind.clone()].len() {
            total_loss += (feed_net.layers[feed_net.layers.len() - 1].errors[ind.clone()][e]).powf(2.0);
        }
    }
    println!("total loss: {}", total_loss);

    total_loss
}

pub fn clear_network_errors(feed_net: &mut FeedforwardNetwork<f64>) {
    // clear errors and gradients after update
    for i in range(0, feed_net.layers.len()).rev() {
        for k in 0..feed_net.layers[i].errors.len() {
            for j in 0..feed_net.layers[i.clone()].errors[0].len() {
                feed_net.layers[i.clone()].errors[k.clone()][j.clone()] = 0.0;
            }
        }
        for k in 0..feed_net.layers[i.clone()].gradient.len() {
            for j in 0..feed_net.layers[i.clone()].gradient[0].len() {
                feed_net.layers[i.clone()].gradient[k.clone()][j] = 0.0;
            }
        }
    }
}

pub fn train(data_structs: &mut Vec<Data<f64, f64>>,
             feed_net: &mut FeedforwardNetwork<f64>,
             num_iteration: i32) {
    println!("Training beginns");

    let  mut minibach_index = 1;
    let mut data_input: &Data<f64, f64>;
    for _iter in 0..num_iteration {
        // loop through all dataset
        for data_index in 0..data_structs.len() {
            data_input = &data_structs[data_index];
            forward(feed_net, data_input);
            backward(feed_net, data_input);
        }

        if &minibach_index  % 25 == 0 {
            let total_loss = calculate_total_loss(data_structs, feed_net);
            println!("propress: {} %", ((&_iter / &num_iteration) as f64) * 100.0);
            if total_loss <= 0.05 {
                serialize(&feed_net);
                break;
            }

            clear_network_errors(feed_net);
            minibach_index = 0;
        }

        minibach_index += 1;
    }

    // serialize network
    serialize(&feed_net);
}

pub fn initialize_network(data_structs: &mut Vec<Data<f64, f64>>,
                          num_hidden_layers: usize,
                          num_hidden_neurons: usize,
                          num_output_neurons: usize,
                          minibatch_size: usize,
                          learning_rate: f32) -> FeedforwardNetwork<f64> {
    let number_of_hidden_layers: usize = num_hidden_layers;
    let number_of_output_neurons = num_output_neurons;
    let mut _number_of_hidden_neurons = num_hidden_neurons;
    let number_of_data_sets = data_structs.len();
    let number_rows_in_set = data_structs[0].input.len();
    let num_columns_in_set = data_structs[0].input[0].len();
    let input_dimensions = vec![number_rows_in_set as usize,
                                num_columns_in_set as usize,
                                number_of_data_sets as usize];

    let data_type_value: f64 = 0.0;

    println!("input dimenstions: {}, {}, {}", input_dimensions[0], input_dimensions[1], input_dimensions[2]);


    let feedforward_network: FeedforwardNetwork<f64> =
        create(
            number_of_hidden_layers,
            _number_of_hidden_neurons,
            input_dimensions,
            number_of_output_neurons,
            number_of_data_sets.clone(),
            number_rows_in_set.clone(),
            num_columns_in_set.clone(),
            minibatch_size,
            learning_rate,
            data_type_value,
        );

    let mut saved_network: FeedforwardNetwork<f64> = deserialize(feedforward_network);

    for i in 0..saved_network.layers.len() {
        let layer_type = layer::get_layer_type(
            &i,
            &number_of_hidden_layers,
        );
        if matches!(layer_type, LayerType::OutputLayer) {
            _number_of_hidden_neurons = number_of_output_neurons.clone();
        }
        saved_network.learning_rate = learning_rate.clone();
        saved_network.layers[i.clone()].input_data = create_generic_3d(number_rows_in_set.clone(),
                                                               number_of_data_sets.clone());
    }

    saved_network
}