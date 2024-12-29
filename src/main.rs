use std::env;

use neural_networks::neural_networks::{
    network_components::layer::{ActivationType, LayerType},
    network_types::{
        feedforward_network_generic::{create, FeedforwardNetwork},
        transformer::attention_layer::AttentionLayer,
    },
};

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

    let layer = AttentionLayer::<4, 4>::create_default_attention_layer(
        ActivationType::LEAKYRELU,
        LayerType::HiddenLayer,
    );

    println!("ffn: {:?}", feedforward_network.layers.len());
    println!("layer: {:?}", layer.layer_type);
}
