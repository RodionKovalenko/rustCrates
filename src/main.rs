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
#[allow(unused_imports)]
#[allow(unused_imports)]
use rand::Rng;
use neural_networks::wavelet_transform::cwt::{cwt_2d};
use neural_networks::wavelet_transform::cwt_types::ContinuousWaletetType;

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

    let _args: Vec<String> = env::args().collect();
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


    let scales: Vec<f64> = (1..6).map(|x| x as f64).collect();

    // let n: Vec<f64> = (1..2).map(|x| x as f64).collect();
    let n: Vec<f64> = (1..4).map(|x| x as f64).collect();

    println!("scales: {:?}", &scales);
    println!("data: {:?}", &n);

    let scales: Vec<f64> = (2..5).map(|x| x as f64).collect();
    let n: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    let (transform_cwt, frequencies) = cwt_2d(&n, &scales, &ContinuousWaletetType::MEXH, &1.0);
    println!("transformed: {:?}", &transform_cwt);
    println!("frequencies: {:?}", &frequencies);

}