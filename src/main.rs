extern crate serde_derive;
extern crate serde;
extern crate serde_json;

extern crate neural_networks;
#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]

use neural_networks::*;
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

    // collect_data_task::update_json_data_from_uphold_api();

    let number_of_hidden_layers = 2;
    let input_dimenstions = vec![1, 6];
    let number_of_output_neurons = 6;
    let number_of_hidden_neurons= 100;
    let learning_rate = 0.02;

    let feedforward_network =
        network_types::feedforward_network::create(
            number_of_hidden_layers,
            number_of_hidden_neurons,
            input_dimenstions,
            number_of_output_neurons,
            learning_rate
        );

}