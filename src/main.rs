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
use std::time::Instant;


/**
* start with cargo run -- --server
* connect client: cargo run -- --client
tcp::test_connection();
 */
fn main() {
    println!("Test beginns");

    //collect_data_task::update_json_data_from_uphold_api();
    let now = Instant::now();

    feedforward_network::initialize();
    println!("time elapsed {}", now.elapsed().as_secs());
}