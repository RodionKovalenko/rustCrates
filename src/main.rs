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
use neural_networks::wavelet_transform::dwt::{get_ll_lh_hl_hh, insert_padding_after, insert_padding_before, inverse_transform_2_d, transform_1_d, transform_2_d};
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


    let _scale = arange(&1.0, &3.0, &1.0);
    let _data_1d = vec![1.0, 0.0, 2.0, 3.0];
    let data_2d = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]];

    // let data_2d = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //                    vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]];

    let dwt_type = DiscreteWaletetType::COIF2;
    let mode = WaveletMode::ANTIREFLECT;
    let transformed = transform_2_d(&data_2d, &dwt_type, &mode);

    println!(" DB2 transformed {:?}", transformed);
    let llhllhhh = get_ll_lh_hl_hh(&transformed);

    println!(" llhllhhh");
    for i in 0..llhllhhh.len() {
        for j in 0..llhllhhh[i].len() {
            println!("\n llhllhhh {} {:?}", i, llhllhhh[i][j]);
        }
    }

    let inversed = inverse_transform_2_d(&transformed, &dwt_type, &mode, 1);
    println!(" DB2 inversed {:?}", inversed);


    let mut array = vec![5.0, 7.0, 2.0];

    insert_padding_before(&mut array, &WaveletMode::ANTIREFLECT, 10);
    insert_padding_after(&mut array, &WaveletMode::ANTIREFLECT, 10, 6);

    println!("array {:?}", array);
}