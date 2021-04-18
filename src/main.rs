extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate neural_networks;

#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]
use neural_networks::network_types::*;

pub mod uphold_api;

#[allow(unused_imports)]
use uphold_api::collect_data_task;
#[allow(unused_imports)]
use neural_networks::utils::matrix::parse_to_float;
#[allow(unused_imports)]
use neural_networks::network_types::feedforward_network::FeedforwardNetwork;

/**
* start with cargo run -- --server
* connect client: cargo run -- --client
tcp::test_connection();
 */
fn main() {
    println!("Test beginns");

    // collect_data_task::update_json_data_from_uphold_api();

    let input = vec![
        vec![1, 2, 3, 1, 3, 4],
    ];
    let mut parsed_input = parse_to_float(&input);

    let number_of_hidden_layers = 5;
    let input_dimensions = vec![parsed_input.len(), parsed_input[0].len()];
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

    feedforward_network::forward(&mut parsed_input, &mut feedforward_network);
}