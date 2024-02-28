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
use neural_networks::neural_networks::network_types::wavelet_network::test_decomposition;
use neural_networks::neural_networks::utils::image::save_image_from_pixels;
use neural_networks::wavelet_transform::dwt::{get_ll_lh_hl_hh, transform_1_d, transform_2_d};
use neural_networks::wavelet_transform::dwt_type_resolver::{get_high_pass_filter, get_low_pass_filter};
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


    let n1: Vec<f64> = vec![9.0, 7.0, 6.0, 2.0];
    let n2: Vec<f64> = vec![5.0, 3.0, 4.0, 4.0];
    let n3: Vec<f64> = vec![8.0, 2.0, 4.0, 0.0];
    let n4: Vec<f64> = vec![6.0, 0.0, 2.0, 2.0];
    let n5: Vec<f64> = vec![3.0, 0.0, 25.0, 3.0];

    let mut n = vec![];
    n.push(n1.clone());
    n.push(n2.clone());
    n.push(n3.clone());
    n.push(n4.clone());
    n.push(n5.clone());

    test_decomposition();

    // // println!("n : {:?}", &n);
    // println!("Transform : ==================================================================");
    // println!("length: height: {}, width: {}", n.len(), n[1].len());
    //
    // let dw_transformed = transform_1_d(&n1, &DiscreteWaletetType::DB1, &WaveletMode::SYMMETRIC);
    // println!("transformed: {:?}\n", &dw_transformed);
    //
    // let mut original_signal: Vec<f64> = Vec::new();
    //
    // let detail_coefficients = &dw_transformed[2..4];
    // println!("detail coefficients: {:?}\n", &detail_coefficients);
    // let high_pass_filter = get_low_pass_filter(&DiscreteWaletetType::DB1);
    //
    // for i in 0..detail_coefficients.len() {
    //     for j in 0..high_pass_filter.len() {
    //         if i.clone() * j.clone() >= original_signal.len() {
    //             original_signal.push(detail_coefficients[i.clone()].clone() * high_pass_filter[j.clone()].clone());
    //         } else {
    //             original_signal[i.clone() * j.clone()] += detail_coefficients[i.clone()].clone() * high_pass_filter[j.clone()].clone();
    //         }
    //     }
    // }
    // println!("origal signal: {:?}", &original_signal);
}
