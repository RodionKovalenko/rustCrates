use crate::neural_networks::{
    network_layers::{
        complex_to_linear_layer::ComplexToLinearLayer, embedding_layer::EmbeddingLayer, feedforward_layer::FeedForwardLayer, layer::LayerEnum, linear_layer::LinearLayer, norm_layer::NormalNormLayer, softmax_output_layer::SoftmaxLayer, sparse_linear_layer::SparseLinearLayer, wavelet_network::DECOMPOSITION_LEVELS
    },
    network_types::{
        neural_network_generic::{NeuralNetwork, OperationMode, create},
        transformer::{self_attention_layer::SelfAttentionLayer, sparse_self_attention_layer::SparseSelfAttentionLayer},
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
    let embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim_original);

    layers.push(LayerEnum::Embedding(Box::new(embedding_layer)));
    layers.push(LayerEnum::Norm(Box::new(NormalNormLayer::new(embedding_dim_compressed, epsilon, learning_rate))));
    //layers.push(LayerEnum::Wavelet(Box::new(ComplexWaveletLayer::new())));

    let rows: usize = embedding_dim_compressed;
    // Transformer block start
    let num_self_attention_layer: usize = NUM_SELF_ATT_LAYERS;
    // let origin_hidden_dim = 512;
    let hidden_dim = 256;
    for _i in 0..num_self_attention_layer {
        let num_attention_heads: usize = 4;

        // Colums are divided into number of heads
        let cols: usize = embedding_dim_compressed;

        let _attention_layer: SparseSelfAttentionLayer = SparseSelfAttentionLayer::new(num_attention_heads, rows, cols, SPARSE_WINDOW_SIZE, learning_rate);
        //layers.push(LayerEnum::SparseSelfAttention(Box::new(_attention_layer)));

        let _attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, rows, cols, learning_rate);
        layers.push(LayerEnum::SelfAttention(Box::new(_attention_layer)));

        let ffn_layer: FeedForwardLayer = FeedForwardLayer::new(rows, hidden_dim, learning_rate);
        layers.push(LayerEnum::FeedForward(Box::new(ffn_layer)));
    }
    // Transformer block end
    let compressed_hidden = embedding_dim_compressed;
    let _ctl_layer = ComplexToLinearLayer::new(rows, compressed_hidden, learning_rate);
    layers.push(LayerEnum::ComplexToLinear(Box::new(_ctl_layer)));

    let mut _sparse_linear_layer: SparseLinearLayer = SparseLinearLayer::new(learning_rate, compressed_hidden, vocab_size);
    //layers.push(LayerEnum::SparseLinear(Box::new(_sparse_linear_layer)));

    let mut _linear_layer: LinearLayer = LinearLayer::new(learning_rate, compressed_hidden, vocab_size, true);
    layers.push(LayerEnum::Linear(Box::new(_linear_layer)));

    let softmax_layer = SoftmaxLayer::new(learning_rate, operation_mode, vocab_size);
    layers.push(LayerEnum::Softmax(Box::new(softmax_layer)));

    transformer_network.layers = layers;

    // Restore runtime-only links (e.g. weight tying).
    transformer_network.post_load_init();

    transformer_network
}
