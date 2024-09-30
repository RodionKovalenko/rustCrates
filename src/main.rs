use std::env;

#[allow(unused_imports)]
pub mod uphold_api;

#[allow(unused_imports)]
use uphold_api::*;
#[allow(unused_imports)]
use std::time::Instant;
#[allow(unused_imports)]
use rand::Rng;
use neural_networks::neural_networks::network_types::{feedforward_network_generic::{create, FeedforwardNetwork}, network_trait::Network};

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
    let number_outputs = 10;
    let number_of_hidden_layers: usize = 1;
    let number_of_hidden_neurons:  usize = 10;
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
    println!("layers len {:?}", &feedforward_network.get_layers());
    println!("minibatch size {:?}", &feedforward_network.get_minibatch_size());

}