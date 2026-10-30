use crate::neural_networks::{
    network_layers::{
        layer::LayerEnum,
        network_layers_rm::{
            complex_to_linear_layer_rm::ComplexToLinearLayerRm, embedding_layer_rm::EmbeddingLayerRm, eml_linear_layer_rm::EmlLinearLayerRm,
            feedforward_layer_rm::FeedForwardLayerRm, norm_layer_rm::NormalNormLayerRm,
        },
    },
    network_types::{
        neural_network_generic::{NeuralNetwork, OperationMode},
        transformer::{
            sparse_self_attention_layer_rm::SparseSelfAttentionLayerRm,
            transformer_builder::{
                create_transformer_network, embedding_dim_compressed, EMBEDDING_DIM_ORIGINAL, FFN_HIDDEN_DIM, LEARNING_RATE, NORM_EPSILON, NUM_ATTENTION_HEADS, NUM_SELF_ATT_LAYERS,
                SPARSE_WINDOW_SIZE, VOCAB_SIZE,
            },
        },
    },
};

/// Row-major counterpart of `create_transformer`: the same architecture built from RM layers only, so all
/// matrix multiplications run through `multiply_complex_rm` and therefore on the GPU when the `cuda` feature
/// is enabled.
///
/// Embedding -> Norm -> NUM_SELF_ATT_LAYERS x (sparse self-attention + feed-forward) -> ComplexToLinear -> EML head.
///
/// The trained model is saved to its own file (`SLED_DB_TRANSFORMER_RM_V1`), so training this network never
/// overwrites the Vec model.
pub fn create_transformer_rm(_operation_mode: OperationMode) -> NeuralNetwork {
    let learning_rate: f64 = LEARNING_RATE;

    let mut transformer_network: NeuralNetwork = create_transformer_network();
    let mut layers = transformer_network.layers;

    // embedding_dim_compressed = 128
    let embedding_dim_compressed = embedding_dim_compressed();
    let vocab_size: usize = VOCAB_SIZE;

    let embedding_layer = EmbeddingLayerRm::get_or_create(vocab_size, EMBEDDING_DIM_ORIGINAL);
    layers.push(LayerEnum::EmbeddingRm(Box::new(embedding_layer)));
    layers.push(LayerEnum::NormRm(Box::new(NormalNormLayerRm::new(embedding_dim_compressed, NORM_EPSILON, learning_rate))));

    // Transformer block start
    let rows: usize = embedding_dim_compressed;
    // Columns are divided into the number of heads
    let cols: usize = embedding_dim_compressed;
    for _i in 0..NUM_SELF_ATT_LAYERS {
        let attention_layer = SparseSelfAttentionLayerRm::new(NUM_ATTENTION_HEADS, rows, cols, SPARSE_WINDOW_SIZE, learning_rate);
        layers.push(LayerEnum::SparseSelfAttentionRm(Box::new(attention_layer)));

        let ffn_layer = FeedForwardLayerRm::new(rows, FFN_HIDDEN_DIM, learning_rate);
        layers.push(LayerEnum::FeedForwardRm(Box::new(ffn_layer)));
    }
    // Transformer block end

    let compressed_hidden = embedding_dim_compressed;
    let ctl_layer = ComplexToLinearLayerRm::new(rows, compressed_hidden, learning_rate);
    layers.push(LayerEnum::ComplexToLinearRm(Box::new(ctl_layer)));

    // EML output head: dense projection + two-level softmax + cross-entropy in one terminal layer.
    let eml_linear_layer = EmlLinearLayerRm::new(learning_rate, compressed_hidden, vocab_size);
    layers.push(LayerEnum::EmlRm(Box::new(eml_linear_layer)));

    transformer_network.layers = layers;

    // Restore runtime-only links (e.g. weight tying).
    transformer_network.post_load_init();

    transformer_network
}
