use crate::neural_networks::{
    network_components::{embedding_layer::EmbeddingLayer, layer::LayerEnum, linear_layer::LinearLayer, multi_linear_layer::MultiLinearLayer, positional_encoding_layer::PositionalEncodingLayer, softmax_output_layer::SoftmaxLayer},
    network_types::{
        feedforward_layer::FeedForwardLayer,
        neural_network_generic::{create, NeuralNetwork, OperationMode},
        wavelet_network::DECOMPOSITION_LEVELS,
    },
};

use super::self_attention_layer::SelfAttentionLayer;

pub fn create_transformer(operation_mode: OperationMode) -> NeuralNetwork {
    let number_inputs: usize = 32;
    let number_outputs = 32;
    let number_of_hidden_layers: usize = 1;
    let number_of_hidden_neurons: usize = 32;
    let minibatch_size: usize = 50;
    let learning_rate: f64 = 0.01;
    let num_lin_layers = 30;

    let mut transformer_network: NeuralNetwork = create(number_inputs, number_outputs, number_of_hidden_layers, number_of_hidden_neurons, minibatch_size, learning_rate);

    //Add layers to the network
    let mut layers = transformer_network.layers;

    let embedding_dim_original: usize = 512;
    let base_2: i32 = 2;
    // embedding_dim_compressed  = 64
    let embedding_dim_compressed = (embedding_dim_original as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;
    let vocab_size: usize = 50280;

    let embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim_original, false);
    let positional_encoding_layer = PositionalEncodingLayer::new(embedding_layer.embedding_dim);

    layers.push(LayerEnum::Embedding(Box::new(embedding_layer)));
    layers.push(LayerEnum::PositionalEncoding(Box::new(positional_encoding_layer)));

    let rows: usize = embedding_dim_compressed;

    // Transformer block start
    let num_self_attention_layer: usize = 2;
    for i in 0..num_self_attention_layer {
        let mut num_attention_heads: usize = 4;

        if i == 0 {
            num_attention_heads = 2;
        }
        // Colums are divided into number of heads
        let cols: usize = embedding_dim_compressed;

        let attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, rows, cols, learning_rate);
        layers.push(LayerEnum::SelfAttention(Box::new(attention_layer)));

        let mut hidden_dim = 512;
        if i > 0 {
            hidden_dim = 1024;
            //layers.push(LayerEnum::Wavelet(Box::new(WaveletLayer::new())));
        }

        let ffn_layer: FeedForwardLayer = FeedForwardLayer::new(rows, hidden_dim, learning_rate);
        layers.push(LayerEnum::FeedForward(Box::new(ffn_layer)));
    }
    // Transformer block end

    let _multi_liniear_layer = MultiLinearLayer::new(learning_rate, rows, vocab_size, num_lin_layers);
    let linear_layer = LinearLayer::new(learning_rate, rows, vocab_size);
    let softmax_layer = SoftmaxLayer::new(learning_rate, operation_mode);

    // layers.push(LayerEnum::Wavelet(Box::new(WaveletLayer::new())));
    layers.push(LayerEnum::Linear(Box::new(linear_layer)));
    layers.push(LayerEnum::Softmax(Box::new(softmax_layer)));

    transformer_network.layers = layers;

    transformer_network
}
