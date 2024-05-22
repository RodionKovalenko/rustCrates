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
use neural_networks::utils::array::arange;
use neural_networks::utils::data_converter::{convert_to_c_array_f64_2d};
use neural_networks::wavelet_transform::cwt::{cwt};
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


    let scale = arange(&1.0, &3.0, &1.0);
    let data_1d = vec![1, 2, 3];

    let (transformed, frequencies) = cwt(&data_1d, &scale, &ContinuousWaletetType::CGAU5, &1.0).unwrap();
    let result = convert_to_c_array_f64_2d(transformed);
    println!("gauss 5{:?}", result);
    println!("{:?}", frequencies);

    let (transformed, frequencies) = cwt(&data_1d, &scale, &ContinuousWaletetType::CGAU6, &1.0).unwrap();
    let result = convert_to_c_array_f64_2d(transformed);
    println!(" CGA6 {:?}", result);
    println!("{:?}", frequencies);

    let (transformed, frequencies) = cwt(&data_1d, &scale, &ContinuousWaletetType::CGAU7, &1.0).unwrap();
    let result = convert_to_c_array_f64_2d(transformed);
    println!(" CGAU7 {:?}", result);
    println!("{:?}", frequencies);

    let (transformed, frequencies) = cwt(&data_1d, &scale, &ContinuousWaletetType::CGAU8, &1.0).unwrap();
    let result = convert_to_c_array_f64_2d(transformed);
    println!(" CGAU8 {:?}", result);
    println!("{:?}", frequencies);
}