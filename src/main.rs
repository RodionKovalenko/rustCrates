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
use neural_networks::utils::data_converter::extract_array_data;
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


    let scales: Vec<f64> = (1..6).map(|x| x as f64).collect();

    let data_1d = &[1, 2, 3];
    let data_2d: Vec<Vec<i64>> = vec![vec![1, 2], vec![3, 4]];
    let data_3d: Vec<Vec<Vec<i32>>> = vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]];
    let data_4d: Vec<Vec<Vec<Vec<i32>>>> = vec![vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]], vec![vec![vec![9, 10], vec![11, 12]], vec![vec![13, 14], vec![15, 16]]]];
    let data_5d: Vec<Vec<Vec<Vec<Vec<i32>>>>> = vec![vec![vec![vec![vec![1, 2], vec![3, 4]], vec![vec![5, 6], vec![7, 8]]], vec![vec![vec![9, 10], vec![11, 12]], vec![vec![13, 14], vec![15, 16]]]],
                                                     vec![vec![vec![vec![17, 18], vec![19, 20]], vec![vec![21, 22], vec![23, 24]]], vec![vec![vec![25, 26], vec![27, 28]], vec![vec![29, 30], vec![31, 32]]]]];


    let (transform_cwt, _frequencies) = cwt(data_1d, &scales, &ContinuousWaletetType::GAUS1, &1.0).unwrap();
    let transform_cwt: Vec<Vec<f64>> = extract_array_data::<>(&transform_cwt).unwrap();
    println!("\n\ntransformed: {:?}", &transform_cwt);

    let (transform_cwt, _frequencies) = cwt(&data_2d, &scales, &ContinuousWaletetType::GAUS2, &1.0).unwrap();
    println!("\n\ntransformed: {:?}", &transform_cwt);

    let (transform_cwt, _frequencies) = cwt(&data_3d, &scales, &ContinuousWaletetType::MORL, &1.0).unwrap();
    println!("\n\ntransformed: {:?}", &transform_cwt);

    let (transform_cwt, _frequencies) = cwt(&data_4d, &scales, &ContinuousWaletetType::GAUS3, &1.0).unwrap();

    println!("\n\ntransformed: {:?}", &transform_cwt);
    let (transform_cwt, _frequencies) = cwt(&data_5d, &scales, &ContinuousWaletetType::GAUS4, &1.0).unwrap();
    let result = extract_array_data::<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>(&transform_cwt).unwrap();

    println!("result: {:?}", result);
    println!("_frequencies: {:?}", _frequencies);

}