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
use crate::utils::matrix::create_generic_3d;
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
    learning_rate: f32,
    minibatch_size: usize,
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
               mini_bacht_start: usize, minibatch_size: usize)
               -> &'a mut FeedforwardNetwork<f64> {
    let layers: &mut Vec<Layer<f64>> = &mut feed_net.layers;
    let minibatch_end;

    if (mini_bacht_start + minibatch_size) < data_structs.len() {
        minibatch_end = mini_bacht_start + minibatch_size;
    } else {
        minibatch_end = data_structs.len();
    }

    for input_index in mini_bacht_start..minibatch_end {
        for i in 0..layers.len() {
            // println!("layer  {:?}", layers[i].layer_type);
            // println!("input size: {}, {}", layers[i].input_data[input_index].len(), layers[i].input_data[input_index][0].len());
            // println!("weight size: {}, {}", layers[i].input_weights.len(), layers[i].input_weights[0].len());

            if matches!(layers[i].layer_type, LayerType::InputLayer) {
                layers[i].input_data[input_index % minibatch_size] = data_structs[input_index].get_input();
                layers[i].inactivated_output[input_index % minibatch_size] =
                    matrix::multiple_generic_2d(&data_structs[input_index].get_input(),
                                                &layers[i].input_weights.clone());
            } else {
                layers[i].input_data[input_index % minibatch_size] = layers[i - 1].activated_output[input_index % minibatch_size].clone();
                layers[i].inactivated_output[input_index % minibatch_size] =
                    matrix::multiple_generic_2d(&layers[i - 1].activated_output[input_index % minibatch_size].clone(),
                                                &layers[i].input_weights.clone());
            }

            layers[i].inactivated_output[input_index % minibatch_size] = matrix::add(&layers[i].inactivated_output[input_index % minibatch_size], &layers[i].layer_bias);

            //   if !matches!(layers[i].layer_type, LayerType::OutputLayer) {
            layers[i].activated_output[input_index % minibatch_size] = tanh(&layers[i].inactivated_output[input_index % minibatch_size]);
            // } else {
            //     layers[i].activated_output[input_index] = layers[i].inactivated_output[input_index].clone();
            // }

            // println!("output size: {}, {}", layers[i].inactivated_output[input_index].len(),
            //          layers[i].inactivated_output[input_index][0].len());
            //
            // println!("");

            if matches!(layers[i].layer_type, LayerType::OutputLayer) {
                let errors = matrix::get_error(&data_structs[input_index].get_target(),
                                               &layers[i].activated_output[input_index % minibatch_size]);

                layers[i].errors[input_index % minibatch_size] = errors;
            }
        }
    }

    feed_net.layers = layers.clone();
    feed_net
}

pub fn train(data_structs: &mut Vec<Data<f64>>,
             feed_net: &'a mut FeedforwardNetwork<f64>,
             num_iteration: i32) {
    println!("Training beginns");

    let mini_batch_size = 50;
    let mut mini_batch_end;
    for _iter in 0..num_iteration {
        let mut batch_start: usize = 0;

        for minibatch_start in 0..data_structs.len() / mini_batch_size {
            batch_start = minibatch_start * mini_batch_size;
            mini_batch_end = batch_start + mini_batch_size;

            if _iter % 100 == 0 {
                forward(data_structs, feed_net, minibatch_start, mini_batch_size);
                let mut total_loss = 0.0;

                if _iter % 1000 == 0 {
                    for ind in 0..data_structs.len() {
                        if _iter % 5000 == 0 {
                            println!("target: {:?}", &data_structs[ind].get_target());
                            println!("activated output {:?}",
                                     feed_net.layers[feed_net.layers.len() - 1].activated_output[ind % mini_batch_size]);
                        }
                        for e in 0..feed_net.layers[feed_net.layers.len() - 1].errors[ind % mini_batch_size].len() {
                            total_loss += (feed_net.layers[feed_net.layers.len() - 1].errors[ind % mini_batch_size][e]).powf(2.0);
                        }
                    }
                    println!("total loss: {}", total_loss);
                    println!("propress: {} %", (_iter as f64 / num_iteration as f64) * 100.0);
                    if total_loss <= 0.009 {
                        serialize(&feed_net);
                        break;
                    }
                }
            } else {
                forward(data_structs, feed_net, minibatch_start, mini_batch_size);
            }

            for i in range(0, feed_net.layers.len()).rev() {
                train::calculate_gradient(&mut feed_net.layers, i,
                                          mini_batch_size,
                                          feed_net.learning_rate as f64,
                                          _iter,
                );
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
            learning_rate,
            minibatch_size,
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
        saved_network.learning_rate = learning_rate;
        saved_network.layers[i].input_data = create_generic_3d(number_rows_in_set as usize,
                                                               number_of_hidden_neurons,
                                                               number_of_data_sets as usize);
    }

    saved_network
}