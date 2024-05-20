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
use neural_networks::wavelet_transform::dwt::{get_ll_lh_hl_hh, transform_2_d};
use neural_networks::wavelet_transform::dwt_types::DiscreteWaletetType;
use neural_networks::wavelet_transform::modes::WaveletMode;

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

    let data_1ddata_1d = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let data_2d = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

    // let dwt_1d = transform_1_d(&data_1ddata_1d, &DiscreteWaletetType::DB3, &WaveletMode::SYMMETRIC);
    //
    // println!("{:?}", dwt_1d);

    let dwt = transform_2_d(&data_2d, &DiscreteWaletetType::DB5, &WaveletMode::CONSTANT);
    let lllhhlhh = get_ll_lh_hl_hh(&dwt);
    println!("{:?}", lllhhlhh);
}