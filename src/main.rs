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
use neural_networks::wavelet_transform::dwt::{inverse_transform_1_d, inverse_transform_2_d, transform_1_d, transform_2_d};
use neural_networks::wavelet_transform::dwt_types::DiscreteWaletetType;

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

    let n1: Vec<i32> = vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
    let n2: Vec<i32> = vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
    let mut n: Vec<Vec<i32>> = vec![];
    n.push(n1.clone());
    n.push(n2.clone());

    println!("n : {:?}", &n);

    println!("DB1 : ==================================================================");
    let dw_transformed = transform_2_d(&n, &DiscreteWaletetType::DB_1);
    println!("length: {:?}", dw_transformed.len());
    println!("DB1 wavelet transform: {:?}", dw_transformed);
    let inverse_transformed = inverse_transform_2_d(&dw_transformed, &DiscreteWaletetType::DB_1);
    println!("length: {:?}", inverse_transformed.len());
    println!("DB1 inverse transformed: {:?}", inverse_transformed);
    println!("==================================================================");

    // println!("db 1: ==================================================================");
    // let dw_transformed = transform_1_d(&n, &DiscreteWaletetType::DB_1);
    // println!("length: {:?}", dw_transformed.len());
    // println!("db 1 wavelet transform: {:?}", dw_transformed);
    //
    // let inverse_transformed = inverse_transform_1_d(&dw_transformed, &DiscreteWaletetType::DB_1);
    // println!("length: {:?}", inverse_transformed.len());
    // println!("inverse transformed: {:?}", inverse_transformed);
    // println!("==================================================================");
    //
    // println!("db 2: ==================================================================");
    // let dw_transformed = transform_1_d(&n1, &DiscreteWaletetType::DB_2);
    // println!("length: {:?}", dw_transformed.len());
    // println!("db 2 wavelet transform: {:?}", dw_transformed);
    //
    // let inverse_transformed = inverse_transform_1_d(&dw_transformed, &DiscreteWaletetType::DB_2);
    // println!("length: {:?}", inverse_transformed.len());
    // println!("inverse transformed: {:?}", inverse_transformed);
    // println!("==================================================================");
    //
    // println!("db 4: ==================================================================");
    // let dw_transformed = transform_1_d(&n, &DiscreteWaletetType::DB_4);
    // println!("length: {:?}", dw_transformed.len());
    // println!("{:?}", dw_transformed);
    //
    // let inverse_transformed = inverse_transform_1_d(&dw_transformed, &DiscreteWaletetType::DB_4);
    // println!("length: {:?}", inverse_transformed.len());
    // println!("inverse transformed: {:?}", inverse_transformed);
    // println!("==================================================================");
    //
    // println!("db 8: ==================================================================");
    // let dw_transformed = transform_1_d(&n, &DiscreteWaletetType::DB_8);
    // println!("length: {:?}", dw_transformed.len());
    // println!("{:?}", dw_transformed);
    //
    // let inverse_transformed = inverse_transform_1_d(&dw_transformed, &DiscreteWaletetType::DB_8);
    // println!("length: {:?}", inverse_transformed.len());
    // println!("inverse transformed: {:?}", inverse_transformed);
    // println!("==================================================================");
    //
    // println!("db 16: ==================================================================");
    // let dw_transformed = transform_1_d(&n, &DiscreteWaletetType::DB_16);
    // println!("length: {:?}", dw_transformed.len());
    // println!("{:?}", dw_transformed);
    //
    // let inverse_transformed = inverse_transform_1_d(&dw_transformed, &DiscreteWaletetType::DB_16);
    // println!("length: {:?}", inverse_transformed.len());
    // println!("inverse transformed: {:?}", inverse_transformed);
    // println!("==================================================================");
    //
    // println!("db 25: ==================================================================");
    // let dw_transformed = transform_1_d(&n, &DiscreteWaletetType::DB_25);
    // println!("length: {:?}", dw_transformed.len());
    // println!("{:?}", dw_transformed);
    //
    // let inverse_transformed = inverse_transform_1_d(&dw_transformed, &DiscreteWaletetType::DB_25);
    // println!("length: {:?}", inverse_transformed.len());
    // println!("inverse transformed: {:?}", inverse_transformed);
    // println!("==================================================================");
}
