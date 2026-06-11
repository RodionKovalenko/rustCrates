use crate::neural_networks::{
    network_layers::layer::LayerEnum,
    network_types::{neural_network_generic::NeuralNetwork, transformer::transformer_builder::NUM_SELF_ATT_LAYERS},
    utils::dtype::{C, Real, r},
};

pub const VERBOSE: bool = false;
pub const SHOW_MAX_PARAMS: bool = false;

pub fn update_transformer(transformer_network: &mut NeuralNetwork, target_batch_ids: &Vec<Vec<u32>>) {
    // IMPORTANT for weight tying:
    // Embedding layers accumulate per-token gradients into a shared accumulator that is consumed
    // by SparseLinearLayer::update_parameters(). Therefore accumulation must happen BEFORE the
    // SparseLinear optimizer step.
    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::Embedding(embedding_layer) => {
                embedding_layer.update_parameters_inner(target_batch_ids);
            }
            LayerEnum::EmbeddingRm(embedding_layer) => {
                embedding_layer.update_parameters_inner(target_batch_ids);
            }
            _ => {}
        }
    }

    for layer in transformer_network.layers.iter_mut().rev() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_pooling) => {}
            LayerEnum::Embedding(_embedding_layer) => {}
            LayerEnum::EmbeddingRm(_embedding_layer) => {}
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
            LayerEnum::PositionalEncodingRm(_positional_encoding_layer) => {}
            LayerEnum::Norm(_norm_layer) => {
                _norm_layer.update_parameters();
            }
            LayerEnum::NormRm(_norm_layer) => {
                _norm_layer.update_parameters();
            }
            LayerEnum::SelfAttention(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SparseSelfAttention(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SparseSelfAttentionRm(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SelfAttentionApproximation(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SelfAttentionApproximationRm(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::FeedForward(dense_layer) => {
                dense_layer.update_parameters();
            }
            LayerEnum::FeedForwardRm(ffn_layer_rm) => {
                ffn_layer_rm.update_parameters();
            }
            LayerEnum::Linear(linear_layer) => {
                linear_layer.update_parameters();
            }
            LayerEnum::LinearRm(linear_layer) => {
                linear_layer.update_parameters();
            }
            LayerEnum::ClusteringLinear(sparse_linear_layer) => {
                sparse_linear_layer.update_parameters();
            }
            LayerEnum::AdaptiveLinear(adaptive_linear_layer) => {
                adaptive_linear_layer.update_parameters();
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                sparse_linear_layer.update_parameters();
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                multi_linear_layer.update_parameters();
            }
            LayerEnum::Wavelet(_wavelet_layer) => {}
            LayerEnum::WaveletRm(_wavelet_layer) => {}
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                wavelet_layer.update_parameters();
            }
            LayerEnum::ComplexToLinear(ctl_layer) => {
                ctl_layer.update_parameters();
            }
            LayerEnum::ComplexToLinearRm(ctl_layer) => {
                ctl_layer.update_parameters();
            }
            LayerEnum::Softmax(_softmax_layer) => {
                _softmax_layer.update_parameters();
            }
            LayerEnum::SoftmaxRm(_softmax_layer) => {
                _softmax_layer.update_parameters();
            }
            _ => {
                println!("Layer type not supported for backward pass");
            }
        }
    }

    transformer_network.tie_embedding_and_sparse_linear();
}

pub fn update_k_mean_clusters(transformer_network: &mut NeuralNetwork, epoch: usize) {
    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::ClusteringLinear(sparse_linear_layer) => {
                sparse_linear_layer.update_centroids(epoch);
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                sparse_linear_layer.update_centroids(epoch);
            }
            _ => {}
        }
    }
}

pub fn max_weight(weights: &Vec<Vec<C>>) -> f64 {
    let mut max_weight = 0.0;
    for row in weights.iter() {
        for &weight in row.iter() {
            let abs_weight = weight.norm() as f64;
            if abs_weight > max_weight {
                max_weight = abs_weight;
            }
        }
    }

    if VERBOSE && SHOW_MAX_PARAMS {
        println!("Max weight: {}", max_weight);
    }
    max_weight
}

pub fn max_bias(bias: &[C]) -> f64 {
    let mut max_bias = 0.0;
    for &b in bias.iter() {
        let abs_b = b.norm() as f64;
        if abs_b > max_bias {
            max_bias = abs_b;
        }
    }

    if VERBOSE && SHOW_MAX_PARAMS {
        println!("Max bias: {}", max_bias);
    }
    max_bias
}

// for scaling the input
pub fn calculate_alpha() -> Real {
    r((3.0f64 * NUM_SELF_ATT_LAYERS as f64).powf(0.25))
}

// for scaling the residuals
pub fn calculate_beta() -> Real {
    let alpha = calculate_alpha();
    r(1.0) / alpha
}
