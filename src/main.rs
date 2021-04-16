extern crate serde_derive;
extern crate serde;
extern crate serde_json;

extern crate neural_networks;

#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]
use neural_networks::network_types::*;
use neural_networks::network_components::layer::initialize_layer;

pub mod uphold_api;

use uphold_api::collect_data_task;

/**
* start with cargo run -- --server
* connect client: cargo run -- --client
tcp::test_connection();
 */
fn main() {
    println!("Test beginns");

    //collect_data_task::update_json_data_from_uphold_api();

    let mut input = vec![
        vec![1.0, 2.0, 3.0, 1.0, 3.3, 4.4]
    ];

    let number_of_hidden_layers = 5;
    let input_dimensions = vec![input.len(), input[0].len()];
    let number_of_output_neurons = 25;
    let number_of_hidden_neurons = 15;
    let learning_rate = 0.02;

    let mut feedforward_network =
        feedforward_network::create(
            number_of_hidden_layers,
            number_of_hidden_neurons,
            input_dimensions,
            number_of_output_neurons,
            learning_rate,
        );

    feedforward_network::forward(&mut input, &mut feedforward_network);
}