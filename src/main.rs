#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

pub mod uphold_api;
use uphold_api::collect_data_task;

#[allow(unused_variables)]
#[allow(unused_assignments)]
/**
* start with cargo run -- --server
* connect client: cargo run -- --client
tcp::test_connection();
 */
fn main() {

    println!("Test beginns");

    collect_data_task::update_json_data_from_uphold_api();
}