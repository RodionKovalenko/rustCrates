use crate::neural_networks::{
    network_components::layer::{
        create_default_layer, ActivationType, Layer, LayerEnum, LayerType,
    },
    network_types::neural_network_generic::{create, NeuralNetwork},
    utils::tokenizer::tokenize,
};

use super::attention_layer::AttentionLayer;

pub fn create_transformer() {
    let number_inputs: usize = 32;
    let number_outputs = 32;
    let number_of_hidden_layers: usize = 1;
    let number_of_hidden_neurons: usize = 32;
    let minibatch_size: usize = 50;
    let learning_rate: f32 = 0.5;

    let mut transformer_network: NeuralNetwork = create(
        number_inputs,
        number_outputs,
        number_of_hidden_layers,
        number_of_hidden_neurons,
        minibatch_size,
        learning_rate,
    );

    let rows: usize = 3;
    let cols: usize = 4;
    let dense_layer: Layer = create_default_layer(
        rows,
        cols,
        &ActivationType::LEAKYRELU,
        LayerType::AttentionLayer,
    );

    let rows: usize = 5;
    let cols: usize = 2;
    let attention_layer: AttentionLayer = AttentionLayer::create_default_attention_layer(
        rows,
        cols,
        ActivationType::LEAKYRELU,
        LayerType::AttentionLayer,
    );

    // Add layers to the network
    transformer_network
        .layers
        .push(LayerEnum::Dense(Box::new(dense_layer)));
    transformer_network
        .layers
        .push(LayerEnum::SelfAttention(Box::new(attention_layer)));

    let (tokens, ids) = tokenize("Hallo, wie geht es dir? Как твои дела? В мене справи добре").unwrap();
    

    // Iterate through and print layer details
    for layer in transformer_network.layers.iter() {
        match layer {
            LayerEnum::Dense(dense) => {
                println!("Dense layer: {:?}", Some(dense).unwrap().weights);
            }
            LayerEnum::SelfAttention(attention) => {
                println!("Attention layer: {:?}", Some(attention).unwrap().weights_k);
            }
            _ => {
                println!("Other layer type");
            }
        }
    }

    println!("token ids: {:?}", &ids);
}
