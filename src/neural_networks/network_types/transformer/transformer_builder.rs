use num::Complex;

use crate::neural_networks::{
    network_components::{
        add_rms_norm_layer::RMSNormLayer, embedding_layer::EmbeddingLayer, layer::{create_default_layer, ActivationType, BaseLayer, Layer, LayerEnum, LayerType}, linear_layer::LinearLayer, positional_encoding_layer::PositionalEncodingLayer, softmax_output_layer::SoftmaxLayer
    },
    network_types::{neural_network_generic::{create, NeuralNetwork}, wavelet_network::DECOMPOSITION_LEVELS},
    utils::{matrix::find_highest_index, tokenizer::{detokenize, tokenize}},
};

use super::self_attention_layer::SelfAttentionLayer;

pub fn create_transformer() {
    let number_inputs: usize = 32;
    let number_outputs = 32;
    let number_of_hidden_layers: usize = 1;
    let number_of_hidden_neurons: usize = 32;
    let minibatch_size: usize = 50;
    let learning_rate: f64 = 0.5;

    let mut transformer_network: NeuralNetwork = create(
        number_inputs,
        number_outputs,
        number_of_hidden_layers,
        number_of_hidden_neurons,
        minibatch_size,
        learning_rate,
    );

    let embedding_dim_original: usize = 512;
    let base_2: i32 = 2;
    // embedding_dim_compressed  = 16
    let embedding_dim_compressed = (embedding_dim_original as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize; 
    let vocab_size: usize = 50254;

    let embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim_original, false);

    let positional_encoding_layer = PositionalEncodingLayer {
        embedding_dim: embedding_layer.embedding_dim,
    };

    let num_attention_heads: usize = 4;
    let rows: usize = 16;
    let cols: usize = embedding_dim_compressed;
    let attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, rows, cols, LayerType::AttentionLayer);

    let rows: usize = 16;
    let cols: usize = embedding_dim_compressed;
    let dense_layer: Layer = create_default_layer(rows, cols, &ActivationType::LEAKYRELU, LayerType::AttentionLayer);

    let epsion: f64 = 0.000001;
    let rms_norm_layer1 = RMSNormLayer::new(cols, epsion, learning_rate);
    let rms_norm_layer2 = RMSNormLayer::new(cols, epsion, learning_rate);

    let linear_layer = LinearLayer::new(learning_rate, rows, vocab_size);
    let softmax_layer = SoftmaxLayer::new(learning_rate);

    // Add layers to the network
    let mut layers = transformer_network.layers;

    layers.push(LayerEnum::Embedding(Box::new(embedding_layer)));
    layers.push(LayerEnum::PositionalEncoding(Box::new(positional_encoding_layer)));
    layers.push(LayerEnum::SelfAttention(Box::new(attention_layer)));
    layers.push(LayerEnum::RMSNorm(Box::new(rms_norm_layer1)));
    layers.push(LayerEnum::Dense(Box::new(dense_layer)));
    layers.push(LayerEnum::RMSNorm(Box::new(rms_norm_layer2)));
    layers.push(LayerEnum::Linear(Box::new(linear_layer)));
    layers.push(LayerEnum::Softmax(Box::new(softmax_layer)));

    transformer_network.layers = layers;

    let (_tokens, ids) = tokenize("Hallo, wie geht es dir? Как твои дела? В мене справи добре").unwrap();

    println!("detokenize: {:?}", detokenize(&ids).unwrap());

    let mut output= None;

    for layer in transformer_network.layers.iter() {
        match layer {
            LayerEnum::Embedding(embedding_layer_box) => {
                let embedding_l = Some(embedding_layer_box).unwrap();
                let embeddings: Vec<Vec<Complex<f64>>> = embedding_l.forward(&ids);

                //println!("output embedding layer: {:?}, {:?}", &embeddings.len(), &embeddings[0].len());
                //println!("output embedding: {:?}", &embeddings);
                output = Some(embeddings);
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_output) = &output {
                    println!("previous output embedding layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let positional_encodings = positional_encoding_layer.forward(&previous_output);

                    println!("positional_encodings layer: {:?}, {:?}", &positional_encodings.len(), &positional_encodings[0].len());

                    output = Some(positional_encodings);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(previous_output) = &output {
                    println!("Previous output attention layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

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
                if let Some(previous_output) = &output {
                    println!("Previous output: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_dense = dense_layer.forward(&previous_output);

                    println!("Dense output: {:?}, {:?}", &output_dense.len(), &output_dense[0].len());

                    //println!("dense output: {:?}", &output_dense);
                    output = Some(output_dense);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::RMSNorm(rms_norm_layer) => {
                let rms_norm_layer = Some(rms_norm_layer).unwrap();
                if let Some(previous_output) = &output {
                    println!("Previous output: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_rms = rms_norm_layer.forward(&previous_output);

                    println!("RMS output: {:?}, {:?}", &output_rms.len(), &output_rms[0].len());

                    //println!("RMS output: {:?}", &output_rms);
                    output = Some(output_rms);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                let linear_layer = Some(linear_layer).unwrap();
                if let Some(previous_output) = &output {
                    println!("Previous output in linear layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_linear = linear_layer.forward(&previous_output);

                    println!("Linear output: {:?}, {:?}", &output_linear.len(), &output_linear[0].len());

                    //println!("dense output: {:?}", &output_dense);
                    output = Some(output_linear);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                let softmax_layer = Some(softmax_layer).unwrap();
                if let Some(previous_output) = &output {
                    println!("Previous output in softmax layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let output_softmax: Vec<Vec<Complex<f64>>> = softmax_layer.forward(&previous_output);
                    let high_token_index: u32 = find_highest_index(&output_softmax).unwrap() as u32;

                    let predicted_token: String = detokenize(&vec![high_token_index]).unwrap();

                    println!("Softmax output: {:?}, {:?}", &output_softmax.len(), &output_softmax[0].len());
                    println!("softmax: {:?}", high_token_index);
                    println!("predicted token: {:?}", predicted_token);

                    output = Some(output_softmax);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
        }
    }

    // println!("token ids: {:?}", &ids);
}
