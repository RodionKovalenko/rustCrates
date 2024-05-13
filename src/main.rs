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
use neural_networks::wavelet_transform::cwt::{cwt_2d, cwt, cwt_3d};
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


    let scales: Vec<f64> = (5..100).map(|x| x as f64).collect();
    println!("scales: {:?}", &scales);

    let n: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let (transform_cwt, _frequencies) = cwt_2d(&n, &scales, &ContinuousWaletetType::GAUS8, &1.0);
    println!("transformed: {:?}", &transform_cwt);

    // let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    // let scales = vec![1.0, 2.0, 3.0];
    // let sampling_period = &1.0;
    //
    // let (wavelets, frequencies) = cwt(&data, &scales, &ContinuousWaletetType::GAUS3, sampling_period);
    //
    // println!("wavelets: {:?}", &wavelets);
    // println!("frequencies: {:?}", &frequencies);
}