use std::env;

#[allow(unused_imports)]
pub mod uphold_api;

#[allow(unused_imports)]
use uphold_api::*;
#[allow(unused_imports)]
use std::time::Instant;
#[allow(unused_imports)]
use rand::Rng;
use neural_networks::neural_networks::{network_types::{feedforward_network_generic::{create, FeedforwardNetwork}, network_trait::Network, wavelet_network::decompose_in_wavelet_2d_default}, utils::matrix::{multiple, multiple_complex}};

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

    let number_inputs: usize = 2;
    let number_outputs = 32;
    let number_of_hidden_layers: usize = 1;
    let number_of_hidden_neurons:  usize = 32;
    let minibatch_size: usize = 50;
    let learning_rate: f32 = 0.5;

    let feedforward_network: FeedforwardNetwork = create(
        number_inputs,
        number_outputs,
        number_of_hidden_layers,
        number_of_hidden_neurons,
        minibatch_size,
        learning_rate,
    );

    // println!("{:?}", &feedforward_network);

    println!("learning rate {:?}", &feedforward_network.get_learning_rate());
    println!("layers len {:?}", &feedforward_network.get_layers()[0].weights);
    println!("minibatch size {:?}", &feedforward_network.get_minibatch_size());

    // let m1: Vec<f64> = vec![1.0, 2.0, 3.0];
    // let m2: Vec<Vec<f64>> = vec![vec![5.0, 6.0, 7.0], vec![7.0, 8.0, 9.0]];

    // let result = decompose_in_wavelet_2d_default(&m1);

    // println!("1 d input: {:?}", result);

    // let result = decompose_in_wavelet_2d_default(&m2);

    // println!("2d input {:?}", result);

}