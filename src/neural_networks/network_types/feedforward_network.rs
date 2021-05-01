use crate::network_components::*;
use crate::utils::*;
use crate::network_components::layer::{Layer, LayerType};
use crate::network_components::input::*;
use activation::sigmoid;
use num::{range, abs};
#[allow(unused_imports)]
use matrix::parse_2dim_to_float;
use crate::network_types::feedforward_network_generic::FeedforwardNetwork;
use crate::utils::file::{serialize, deserialize};
use crate::utils::matrix::{create_generic_3d, create_generic};
use crate::utils::activation::{tanh_value, tanh};

#[allow(unused_imports)]
pub const FILE_NAME: &str = "feedforward_network.json";

pub fn create(
    number_of_hidden_layers: i8,
    number_of_hidden_neurons: usize,
    input_dimensions: Vec<usize>,
    number_of_output_neurons: usize,
    number_data_sets: i32,
    number_rows_in_data: i32,
    number_columns_in_data: i32,
    minibatch_size: usize,
    learning_rate: f32,
) -> FeedforwardNetwork<f64> {
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
    };

    layer::initialize_layer(&mut feed_net);
    feed_net
}

pub fn forward(data_structs: &mut Vec<Data<f64>>,
               feed_net: &'a mut FeedforwardNetwork<f64>,
               minibatch_start: usize, minibatch_size: usize)
               -> &'a mut FeedforwardNetwork<f64> {
    let mut minibatch_end = minibatch_start + minibatch_size;

    if minibatch_end > data_structs.len() {
        minibatch_end = data_structs.len();
    }

    for input_index in minibatch_start..minibatch_end {
        for i in 0..feed_net.layers.len() {
            if matches!(feed_net.layers[i].layer_type, LayerType::InputLayer) {
                feed_net.layers[i].input_data[input_index] = data_structs[input_index].get_input();
                feed_net.layers[i].inactivated_output[input_index] =
                    matrix::multiple(&data_structs[input_index].get_input(),
                                     &feed_net.layers[i].input_weights);
            } else {
                feed_net.layers[i].input_data[input_index] = feed_net.layers[i - 1].activated_output[input_index].clone();
                feed_net.layers[i].inactivated_output[input_index] =
                    matrix::multiple(&feed_net.layers[i - 1].activated_output[input_index],
                                     &feed_net.layers[i].input_weights);
            }

            // add bias
            feed_net.layers[i].inactivated_output[input_index] = matrix::add(
                &feed_net.layers[i].inactivated_output[input_index],
                &feed_net.layers[i].layer_bias);

            feed_net.layers[i].activated_output[input_index] = tanh(
                &feed_net.layers[i].inactivated_output[input_index]);

            if matches!(feed_net.layers[i].layer_type, LayerType::OutputLayer) {
                let errors = matrix::get_error(&data_structs[input_index].get_target(),
                                               &feed_net.layers[i].activated_output[input_index]);

                feed_net.layers[i].errors[input_index] = errors;
            }
        }
    }

    feed_net
}

pub fn train(data_structs: &mut Vec<Data<f64>>,
             feed_net: &'a mut FeedforwardNetwork<f64>,
             minibatch_size: usize,
             num_iteration: i32) {
    println!("Training beginns");

    let mut minibatch_start: usize = 0;
    for _iter in 0..num_iteration {
        for minibatch_ind in 0..(data_structs.len() / minibatch_size) + 1 {
            minibatch_start = minibatch_ind * minibatch_size;

            forward(data_structs, feed_net, minibatch_start, minibatch_size);

            for i in range(0, feed_net.layers.len()).rev() {
                train::calculate_gradient(&mut feed_net.layers, i,
                                          data_structs.len(),
                                          feed_net.learning_rate as f64,
                                          _iter,
                                          minibatch_start,
                                          minibatch_size,
                );
            }
        }

        if _iter % 100 == 0 {
            let mut total_loss = 0.0;
            for ind in 0..data_structs.len() {
                if _iter % 500 == 0 {
                    println!("target: {:?}", &data_structs[ind].get_target());
                    println!("activated output {:?}",
                             feed_net.layers[feed_net.layers.len() - 1].activated_output[ind]);
                }
                for e in 0..feed_net.layers[feed_net.layers.len() - 1].errors[ind].len() {
                    total_loss += (feed_net.layers[feed_net.layers.len() - 1].errors[ind][e]).powf(2.0);
                }
            }
            println!("total loss: {}", total_loss);
            println!("propress: {} %", (_iter as f64 / num_iteration as f64) * 100.0);
            if total_loss <= 0.05 {
                serialize(&feed_net);
                break;
            }
        }

        // clear errors and gradients after update
        for i in range(0, feed_net.layers.len()).rev() {
            for k in 0..feed_net.layers[i].errors.len() {
                for j in 0..feed_net.layers[i].errors[0].len() {
                    feed_net.layers[i].errors[k][j] = 0.0;
                }
            }
            for k in 0..feed_net.layers[i].gradient.len() {
                for j in 0..feed_net.layers[i].gradient[0].len() {
                    feed_net.layers[i].gradient[k][j] = 0.0;
                }
            }
        }
    }

    // serialize network
    serialize(&feed_net);
}

pub fn initialize_network(data_structs: &mut Vec<Data<f64>>,
                          num_hidden_layers: i8,
                          num_hidden_neurons: usize,
                          num_output_neurons: usize,
                          minibatch_size: usize,
                          learning_rate: f32) -> FeedforwardNetwork<f64> {
    let number_of_hidden_layers = num_hidden_layers;
    let number_of_output_neurons = num_output_neurons;
    let mut number_of_hidden_neurons = num_hidden_neurons;
    let number_of_data_sets = data_structs.len() as i32;
    let number_rows_in_set = data_structs[0].input.len() as i32;
    let num_columns_in_set = data_structs[0].input[0].len() as i32;
    let input_dimensions = vec![number_rows_in_set as usize,
                                num_columns_in_set as usize,
                                number_of_data_sets as usize];

    println!("input dimenstions: {}, {}, {}", input_dimensions[0], input_dimensions[1], input_dimensions[2]);

    let feedforward_network: FeedforwardNetwork<f64> =
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
        );

    let mut saved_network: FeedforwardNetwork<f64> = deserialize(feedforward_network);

    for i in 0..saved_network.layers.len() {
        let layer_type = layer::get_layer_type(
            &(i as i8),
            &number_of_hidden_layers,
        );
        if matches!(layer_type, LayerType::OutputLayer) {
            number_of_hidden_neurons = number_of_output_neurons;
        }
        println!("number of hidden neurons: {}", number_of_hidden_neurons);
        saved_network.learning_rate = learning_rate;
        saved_network.layers[i].input_data = create_generic_3d(number_rows_in_set as usize,
                                                               number_of_hidden_neurons,
                                                               number_of_data_sets as usize);
        saved_network.layers[i].inactivated_output = create_generic_3d(number_rows_in_set as usize, number_of_output_neurons, number_of_data_sets as usize);
        saved_network.layers[i].activated_output = create_generic_3d(number_rows_in_set as usize,
                                                                     number_of_output_neurons,
                                                                     number_of_data_sets as usize);
    }

    saved_network
}