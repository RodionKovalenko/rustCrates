use std::env;

#[allow(unused_imports)]
pub mod uphold_api;

use neural_networks::{
    database::crud_service::{get_value, insert_token},
    neural_networks::{
        network_components::layer::{create_default_layer, ActivationType, LayerEnum, LayerType},
        network_types::{
            feedforward_network_generic::{create, FeedforwardNetwork},
            network_trait::Network,
            transformer::attention_layer::AttentionLayer,
            wavelet_network::decompose_in_wavelet_2d_default,
        }, utils::tokenizer::{detokenize, tokenize},
    },
};
#[allow(unused_imports)]
use rand::Rng;
#[allow(unused_imports)]
use std::time::Instant;
#[allow(unused_imports)]
use uphold_api::*;

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

    let number_inputs: usize = 32;
    let number_outputs = 32;
    let number_of_hidden_layers: usize = 1;
    let number_of_hidden_neurons: usize = 32;
    let minibatch_size: usize = 50;
    let learning_rate: f32 = 0.5;

    const M: usize = 5;
    const N: usize = 5;
    let feedforward_network: FeedforwardNetwork<M, N> = create::<M, N>(
        number_inputs,
        number_outputs,
        number_of_hidden_layers,
        number_of_hidden_neurons,
        minibatch_size,
        learning_rate,
        ActivationType::TANH,
    );

    let layer = create_default_layer::<4, 4>(&ActivationType::LEAKYRELU, LayerType::HiddenLayer);
    let layer = AttentionLayer::<4, 4>::create_default_attention_layer(
        ActivationType::LEAKYRELU,
        LayerType::HiddenLayer,
    );

    //println!("{:?}", layer);
    let _ = insert_token("A");
    let value = get_value("A").unwrap();
    println!("value is {}", &value);

    let _ = insert_token("B");
    let value = get_value("B").unwrap();
    println!("value is {}", &value);


    let _ = insert_token("C");
    let value = get_value("C").unwrap();
    println!("value is {}", &value);

    let _ = insert_token("D");
    let value = get_value("D").unwrap();
    println!("value is {}", &value);


    let (tokens, ids) = tokenize("Hallo, wie geht es dir? Как твои дела? В мене справи добре").unwrap();

    println!("tokens: {:?}", &tokens.len());
    println!("ids: {:?}", &ids.len());

    let decoded_text = detokenize(&ids).unwrap();
    println!("decoded text: {:?}", decoded_text);

    // let m1: Vec<f64> = vec![10000002222233334343434.0, 20000000000000000000000000.0, 3013333333333333333333333333333333333333333333.0];
    // let m2: Vec<Vec<f64>> = vec![vec![56666666666666666666666666.0, 64444444444444444444444.0, 74463357654666.0], vec![776444545454545454545.0, 8545454545444.0, 954545454545454545.0]];

    // let result = decompose_in_wavelet_2d_default(&m1);

    // println!("1 d input: {:?}", result);

    // let result = decompose_in_wavelet_2d_default(&m2);

    // println!("2d input {:?}", result);
}
