use num::{range, abs};
use crate::neural_networks::network_components::input::Data;
use crate::neural_networks::network_components::layer;
use crate::neural_networks::network_components::layer::LayerType;
use crate::neural_networks::network_types::feedforward_network_generic::FeedforwardNetwork;
use crate::neural_networks::utils::file::{deserialize, serialize};
use crate::neural_networks::utils::matrix::create_generic_3d;

#[allow(unused_imports)]

#[allow(unused_imports)]
pub const FILE_NAME: &str = "feedforward_network.json";

pub fn create(
    number_of_hidden_layers: i32,
    number_of_hidden_neurons: i32,
    input_dimensions: Vec<usize>,
    number_of_output_neurons: i32,
    number_data_sets: i32,
    number_rows_in_data: i32,
    number_columns_in_data: i32,
    minibatch_size: i32,
    learning_rate: f32,
    data_type_value: f64
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
        data_type_value
    };

    let data_type_value = feed_net.data_type_value.clone();

    layer::initialize_layer(&mut feed_net, data_type_value);
    feed_net
}

pub fn forward(data_structs: &mut Vec<Data<f64>>,
               feed_net: &mut FeedforwardNetwork<f64>,
               minibatch_start: i32, minibatch_size: i32) {
    // let layers: &mut Vec<Layer<f64>> = &mut feed_net.layers;
    //
    // let mut minibatch_end = minibatch_start + minibatch_size;
    //
    // if minibatch_end > data_structs.len() {
    //     minibatch_end = data_structs.len();
    // }
    //
    // for input_index in minibatch_start..minibatch_end {
    //     for i in 0..layers.len() {
    //         // // println!("layer  {:?}", layers[i].layer_type);
    //         // // println!("input size: {}, {}", layers[i].input_data[input_index].len(), layers[i].input_data[input_index][0].len());
    //         // // println!("weight size: {}, {}", layers[i].input_weights.len(), layers[i].input_weights[0].len());
    //         //
    //         // if matches!(layers[i].layer_type, LayerType::InputLayer) {
    //         //     layers[i].input_data[input_index] = data_structs[input_index].get_input();
    //         //     layers[i].inactivated_output[input_index] =
    //         //         matrix::multiple_generic_2d(&data_structs[input_index].get_input(),
    //         //                                     &layers[i].input_weights.clone());
    //         // } else {
    //         //     layers[i].input_data[input_index] = layers[i - 1].activated_output[input_index].clone();
    //         //     layers[i].inactivated_output[input_index] =
    //         //         matrix::multiple_generic_2d(&layers[i - 1].activated_output[input_index].clone(),
    //         //                                     &layers[i].input_weights.clone());
    //         // }
    //         //
    //         // layers[i].inactivated_output[input_index] = matrix::add(&layers[i].inactivated_output[input_index], &layers[i].layer_bias);
    //         //
    //         // //   if !matches!(layers[i].layer_type, LayerType::OutputLayer) {
    //         // layers[i].activated_output[input_index] = tanh(&layers[i].inactivated_output[input_index]);
    //         // // } else {
    //         // //     layers[i].activated_output[input_index] = layers[i].inactivated_output[input_index].clone();
    //         // // }
    //         //
    //         // // println!("output size: {}, {}", layers[i].inactivated_output[input_index].len(),
    //         // //          layers[i].inactivated_output[input_index][0].len());
    //         // //
    //         // // println!("");
    //         //
    //         // if matches!(layers[i].layer_type, LayerType::OutputLayer) {
    //         //     let errors = matrix::get_error(&data_structs[input_index].get_target(),
    //         //                                    &layers[i].activated_output[input_index]);
    //         //
    //         //     layers[i].errors[input_index] = errors;
    //         // }
    //     }
    // }
    //
    // feed_net.layers = layers.clone();
}

pub fn train(data_structs: &mut Vec<Data<f64>>,
             feed_net: & mut FeedforwardNetwork<f64>,
             minibatch_size: i32,
             num_iteration: i32) {
    println!("Training beginns");

    let mut minibatch_start: usize = 0;
    for _iter in 0..num_iteration {
        for minibatch_ind in 0..((data_structs.len() / (minibatch_size.clone() as usize)) + 1) {
            minibatch_start = (minibatch_ind * (minibatch_size.clone() as usize));
            forward(data_structs, feed_net, (minibatch_start.clone() as i32), minibatch_size.clone());

            for i in range(0, feed_net.layers.len()).rev() {
                // train::calculate_gradient(&mut feed_net.layers, &i,
                //                           &data_structs.len(),
                //                           &(feed_net.learning_rate as f64),
                //                           &_iter,
                //                           &minibatch_start,
                //                           minibatch_size,
                // );
            }
        }

        if _iter % 100 == 0 {
            let mut total_loss = 0.0;
            for ind in 0..data_structs.len() {
                if _iter % 5000 == 0 {
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
                          num_hidden_neurons: i32,
                          num_output_neurons: i32,
                          minibatch_size: i32,
                          learning_rate: f32) -> FeedforwardNetwork<f64> {
    let number_of_hidden_layers:i32 = num_hidden_layers as i32;
    let number_of_output_neurons = num_output_neurons;
    let mut number_of_hidden_neurons = num_hidden_neurons;
    let number_of_data_sets = data_structs.len() as i32;
    let number_rows_in_set = data_structs[0].input.len() as i32;
    let num_columns_in_set = data_structs[0].input[0].len() as i32;
    let input_dimensions = vec![number_rows_in_set as usize,
                                num_columns_in_set as usize,
                                number_of_data_sets as usize];

    let data_type_value: f64 = 0.0;

    println!("input dimenstions: {}, {}, {}", input_dimensions[0], input_dimensions[1], input_dimensions[2]);


    let mut feedforward_network: FeedforwardNetwork<f64> =
        create(
            number_of_hidden_layers,
            number_of_hidden_neurons,
            input_dimensions,
            number_of_output_neurons,
            number_of_data_sets.clone(),
            number_rows_in_set.clone(),
            num_columns_in_set.clone(),
            minibatch_size,
            learning_rate,
            data_type_value
        );

    let mut saved_network: FeedforwardNetwork<f64> = deserialize(feedforward_network);

    for i in 0..saved_network.layers.len() {
        let layer_type = layer::get_layer_type(
            &(i as i32),
            &number_of_hidden_layers,
        );
        if matches!(layer_type, LayerType::OutputLayer) {
            number_of_hidden_neurons = number_of_output_neurons;
        }
        saved_network.learning_rate = learning_rate;
        saved_network.layers[i].input_data = create_generic_3d(number_rows_in_set,
                                                               number_of_hidden_neurons,
                                                               number_of_data_sets);
    }

    saved_network
}