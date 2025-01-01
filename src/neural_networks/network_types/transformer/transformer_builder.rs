use crate::{neural_networks::{
    network_components::{
        embedding_layer::EmbeddingLayer,
        layer::{create_default_layer, ActivationType, BaseLayer, Layer, LayerEnum, LayerType},
    },
    network_types::neural_network_generic::{create, NeuralNetwork},
    utils::tokenizer::tokenize,
}, utils::data_converter::convert_to_c_f64_2d};

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

    let embedding_dim: usize = 512;
    let vocab_size: usize = 50254;
    let embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim);

    let rows: usize = 30;
    let cols: usize = 512;
    let dense_layer: Layer = create_default_layer(
        rows,
        cols,
        &ActivationType::LEAKYRELU,
        LayerType::AttentionLayer,
    );

    let rows: usize = 50;
    let cols: usize = 512;
    let attention_layer: AttentionLayer = AttentionLayer::create_default_attention_layer(
        rows,
        cols,
        ActivationType::LEAKYRELU,
        LayerType::AttentionLayer,
    );

    // Add layers to the network
    let mut layers = transformer_network.layers;

    layers.push(LayerEnum::Embedding(Box::new(embedding_layer)));
    layers.push(LayerEnum::SelfAttention(Box::new(attention_layer)));
    layers.push(LayerEnum::Dense(Box::new(dense_layer)));

    transformer_network.layers = layers;

    let (_tokens, ids) = tokenize("Hallo, wie geht es dir? Как твои дела? В мене справи добре").unwrap();

    let mut output = None;  

    for layer in transformer_network.layers.iter() {
        match layer {
            LayerEnum::Embedding(embedding_layer_box) => {
                let embedding_l = Some(embedding_layer_box).unwrap();
                // Forward pass for the embedding layer
                let embeddings: Vec<Vec<f64>> = embedding_l.forward(&ids);
        
                println!("output embedding layer: {:?}, {:?}", &embeddings.len(), &embeddings[0].len());
                // Store the output for the next layer
                output = Some(convert_to_c_f64_2d(&embeddings));
            }
            LayerEnum::SelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(previous_output) = &output {
                   // let wavelet_output: Vec<Vec<Complex<f64>>> = decompose_in_wavelet_2d_default(previous_output)[0].clone();
                    println!("previous output attention layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let output_attention = attention.forward(&previous_output);
        
                    println!("output attention layer: {:?}, {:?}", &output_attention.len(), &output_attention[0].len());
                    // Store the output for the next layer
                    output = Some(output_attention);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::Dense(dense) => {
                let dense_layer = Some(dense).unwrap();
                // Ensure there's an output from the previous layer before forwarding
                if let Some(previous_output) = &output {
                    println!("Prevous output: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                   // let wavelet_output: Vec<Vec<Complex<f64>>> = decompose_in_wavelet_2d_default(previous_output)[0].clone();
                   let wavelet_output = convert_to_c_f64_2d(previous_output);
    
                    println!("Wavelet output: {:?}, {:?}", &wavelet_output.len(), &wavelet_output[0].len());
    
                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_dense = dense_layer.forward(&wavelet_output);

                    println!("Dense output: {:?}, {:?}", &output_dense.len(), &output_dense[0].len());
        
                    // Store the output for the next layer
                    output = Some(output_dense);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            _ => {
                println!("Other layer type");
            }
        }
    }

    // println!("token ids: {:?}", &ids);
}
