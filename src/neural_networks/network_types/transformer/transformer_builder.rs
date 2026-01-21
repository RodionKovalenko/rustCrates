use crate::neural_networks::{
    network_layers::{
        complex_to_linear_layer::ComplexToLinearLayer,
        embedding_layer::EmbeddingLayer,
        layer::LayerEnum,
        network_layers_rm::{
            feedforward_layer_rm::FeedForwardLayerRm,
            norm_layer_rm::NormalNormLayerRm,
            positional_encoding_layer_rm::PositionalEncodingLayerRm,
            softmax_output_layer_rm::SoftmaxLayerRm,
            sparse_linear_layer_rm::SparseLinearLayerRm,
        },
        wavelet_network::DECOMPOSITION_LEVELS,
    },
    network_types::{
        neural_network_generic::{create, NeuralNetwork, OperationMode},
        transformer::{self_attention_layer::SelfAttentionLayer, sparse_self_attention_layer_rm::SparseSelfAttentionLayerRm},
    },
};

pub const NUM_SELF_ATT_LAYERS: usize = 7;
pub const SPARSE_WINDOW_SIZE: usize = 2;

pub fn create_transformer(operation_mode: OperationMode) -> NeuralNetwork {
    let number_inputs: usize = 32;
    let number_outputs = 32;
    let number_of_hidden_layers: usize = 1;
    let number_of_hidden_neurons: usize = 32;
    let minibatch_size: usize = 50;
    let learning_rate: f64 = 0.001;

    let mut transformer_network: NeuralNetwork = create(number_inputs, number_outputs, number_of_hidden_layers, number_of_hidden_neurons, minibatch_size, learning_rate);

    //Add layers to the network
    let mut layers = transformer_network.layers;

    let embedding_dim_original: usize = 512;
    let base_2: i32 = 2;
    // embedding_dim_compressed  = 64
    let embedding_dim_compressed = (embedding_dim_original as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;
    let vocab_size: usize = 50280;
    let epsilon: f64 = 1e-8;
    let embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim_original, false);
    let positional_encoding_layer = PositionalEncodingLayerRm::new(embedding_layer.embedding_dim);

    layers.push(LayerEnum::Embedding(Box::new(embedding_layer)));
    layers.push(LayerEnum::NormRm(Box::new(NormalNormLayerRm::new(embedding_dim_compressed, epsilon, learning_rate))));
    //layers.push(LayerEnum::Wavelet(Box::new(ComplexWaveletLayer::new())));
    layers.push(LayerEnum::PositionalEncodingRm(Box::new(positional_encoding_layer)));

    let rows: usize = embedding_dim_compressed;
    // Transformer block start
    let num_self_attention_layer: usize = NUM_SELF_ATT_LAYERS;
    // let origin_hidden_dim = 512;
    let hidden_dim = 256;
    for _i in 0..num_self_attention_layer {
        let num_attention_heads: usize = 4;

        // Colums are divided into number of heads
        let cols: usize = embedding_dim_compressed;

        let _attention_layer: SparseSelfAttentionLayerRm = SparseSelfAttentionLayerRm::new(num_attention_heads, rows, cols, SPARSE_WINDOW_SIZE, learning_rate);
        layers.push(LayerEnum::SparseSelfAttentionRm(Box::new(_attention_layer)));

        let _attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, rows, cols, learning_rate);
        //layers.push(LayerEnum::SelfAttention(Box::new(_attention_layer)));

        // let hidden_dim = origin_hidden_dim * (_i + 1);

        let ffn_layer: FeedForwardLayerRm = FeedForwardLayerRm::new(rows, hidden_dim, learning_rate);
        layers.push(LayerEnum::FeedForwardRm(Box::new(ffn_layer)));
    }
    // Transformer block end
    let compressed_hidden = 16;
    let _ctl_layer = ComplexToLinearLayer::new(rows, compressed_hidden, learning_rate);
    layers.push(LayerEnum::ComplexToLinear(Box::new(_ctl_layer)));

    let mut _sparse_linear_layer: SparseLinearLayerRm = SparseLinearLayerRm::new(learning_rate, compressed_hidden, vocab_size);
    layers.push(LayerEnum::SparseLinearRm(Box::new(_sparse_linear_layer)));

    //let mut _linear_layer: LinearLayer = LinearLayer::new(learning_rate, compressed_hidden, vocab_size, false);
    //layers.push(LayerEnum::Linear(Box::new(_linear_layer)));

    let softmax_layer = SoftmaxLayerRm::new(learning_rate, operation_mode, vocab_size);
    layers.push(LayerEnum::SoftmaxRm(Box::new(softmax_layer)));

    transformer_network.layers = layers;

    transformer_network
}
