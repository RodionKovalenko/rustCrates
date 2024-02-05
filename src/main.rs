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
use neural_networks::wavelet_transform::dwt::{inverse_transform_2_d, transform_2_d};
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

    let args: Vec<String> = env::args().collect();
    println!("{:?}", args[1]);

    // if args.len() > 1 {
    //     let arg1: &str = &*args[1].clone();
    //
    //     match arg1 {
    //         "uphold" => collect_data_task::update_json_data_from_uphold_api(),
    //         "network" => train_ffn(),
    //         _ => println!(" no argument recognized"),
    //     }
    // }

    //get_pixels_from_images("training_data");

    let n1: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
   // let n1: Vec<i32> = vec![1, 2, 3, 4, 5];
   //  let n2: Vec<i32> = vec![1, 2, 3, 4, 5];
   //  let n3: Vec<i32> = vec![1, 2, 3, 4, 5];
   //  let n4: Vec<i32> = vec![1, 2, 3, 4, 5];
   //  let n5: Vec<i32> = vec![1, 2, 3, 4, 5];

    let mut n: Vec<Vec<i32>> = vec![];
    n.push(n1.clone());
    // n.push(n2.clone());
    // n.push(n3.clone());
    // n.push(n4.clone());
    // n.push(n5.clone());

    println!("n : {:?}", &n);
    println!("DB1 : ==================================================================");
    let dw_transformed = transform_2_d(&n, &DiscreteWaletetType::SYM16, &WaveletMode::ANTISYMMETRIC);
    println!("length: {:?}", dw_transformed[0].len());
    println!("DB1 wavelet transform: {:?}", dw_transformed);
    let inverse_transformed = inverse_transform_2_d(&dw_transformed, &DiscreteWaletetType::SYM16, &WaveletMode::ANTISYMMETRIC);
    println!("length: {:?}", inverse_transformed[0].len());
    println!("DB1 inverse transformed: {:?}", inverse_transformed);
    println!("==================================================================");
}
