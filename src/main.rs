use std::env;

#[allow(unused_imports)]
pub mod uphold_api;

#[allow(unused_imports)]
use uphold_api::*;
#[allow(unused_imports)]
use std::time::Instant;
#[allow(unused_imports)]
use rand::Rng;
use neural_networks::neural_networks::optimization::expecation_maximization::exp_max_2d;

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

    //decompose_in_wavelet_2d_default("training_data/1.jpg");

    // let dwt_type = DiscreteWaletetType::DB2;
    // let mode = WaveletMode::SYMMETRIC;
    //
    // let data_2d = vec![vec![1.1515151515151515, 2.2626262626262626, 3.36363636363636363636, 4.454545454545454545, 5.5959595959595959],
    //                    vec![6.616161616161616161, 7.7272727272727272, 8.818181818181818181, 9.9191919191919191, 10.1313131313131313]];
    //
    // let transformed = transform_2_d(&data_2d, &dwt_type, &mode);
    //
    // println!("transformed {:?}", &transformed);
    //
    // let scales: Vec<f64> = (1..6).map(|x| x as f64).collect();
    // let n: Vec<f64> = vec![1.0, 2.0, 3.0];
    //
    // let (transform_cwt, frequencies) = cwt_1d(&n, &scales, &ContinuousWaletetType::MEXH, &1.0);
    //
    // println!("transform_cwt {:?}", &transform_cwt);


    let numbers_2d = vec![vec![1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0], vec![4.0, 5.5, 3.0, 6.5, 1.0, 3.0, 2.5, 5.0, 3.5, 9.0]];

    let (p, m, v) = exp_max_2d(&numbers_2d, &vec![1.0, 4.0, 3.0, 2.0], &vec![1.0, 1.0, 1.0, 1.0], &100);

    println!("p {:?}", p);
    println!("m_k {:?}", m);
    println!("s_k {:?}", v);
}