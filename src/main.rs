extern crate serde_derive;
extern crate serde;
extern crate serde_json;

use std::env;

#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]

pub mod uphold_api;

#[allow(unused_imports)]
use uphold_api::*;
#[allow(unused_imports)]
#[allow(unused_imports)]
use std::time::Instant;
use chrono::{Datelike, Timelike, DurationRound};
#[allow(unused_imports)]
#[allow(unused_imports)]
use rand::Rng;
use neural_networks::neural_networks::training::network_train_ffn::train_ffn;

pub enum ARGUMENTS {
    UPHOLD,
    NETWORK,
}

/**
* start with cargo run -- --server
* connect client: cargo run -- --client
tcp::test_connection();
 */
fn main() {
    println!("Test beginns");

    let args: Vec<String> = env::args().collect();
    println!("{:?}", args[1]);

    if args.len() > 1 {
        let arg1: &str = &*args[1].clone();

        match arg1 {
            "uphold" => collect_data_task::update_json_data_from_uphold_api(),
            "network" => train_ffn(),
            _ => println!(" no argument recognized"),
        }
    }
}

