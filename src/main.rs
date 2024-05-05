extern crate serde_derive;
extern crate serde;
extern crate serde_json;

use std::env;

#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]
pub mod uphold_api;

use neural_networks::utils::array::{arange, linspace};
#[allow(unused_imports)]
use uphold_api::*;
#[allow(unused_imports)]
#[allow(unused_imports)]
use std::time::Instant;
#[allow(unused_imports)]
#[allow(unused_imports)]
use rand::Rng;
use neural_networks::neural_networks::network_types::wavelet_network::test_decomposition;
use neural_networks::wavelet_transform::cwt::{cwt_1d};
use neural_networks::wavelet_transform::cwt_types::ContiousWaletetType;

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
    //println!("{:?}", args[1]);

    // if args.len() > 1 {
    //     let arg1: &str = &*args[1].clone();
    //
    //     match arg1 {
    //         "uphold" => collect_data_task::update_json_data_from_uphold_api(),
    //         "network" => train_ffn(),
    //         _ => println!(" no argument recognized"),
    //     }
    // }


    let scales: Vec<f64> = (5..6).map(|x| x as f64).collect();

    // let n: Vec<f64> = (1..2).map(|x| x as f64).collect();
    let n: Vec<f64> = (1..8).map(|x| x as f64).collect();

    println!("scales: {:?}", &scales);
    println!("data: {:?}", &n);

    let transform_cwt = cwt_1d(n, scales, &ContiousWaletetType::MEXH, &1.0);

    println!("transformed: {:?}", &transform_cwt);

    test_decomposition();
}