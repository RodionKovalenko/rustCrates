use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path};

use crate::{
    database::sled_db::get_storage_path_transformer_db,
    neural_networks::{
        network_layers::layer::LayerEnum,
        network_layers::tied_sparse_embeddings::TiedSparseEmbeddings,
        utils::file::{derialize_bin, serialize_bin},
    },
};

use super::transformer::transformer_builder::create_transformer;

pub const FILE_NAME: &str = "feedforward_network.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OperationMode {
    TRAINING,
    PRODUCTION,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<LayerEnum>,
    pub learning_rate: f64,
    pub number_of_input_neurons: usize,
    pub number_of_output_neurons: usize,
    pub number_of_hidden_layers: usize,
    pub number_of_hidden_neurons: usize,
    pub minibatch_size: usize,
    pub time_step: usize,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,

    /// Runtime-only shared parameter bundle used for weight tying.
    /// Not persisted because pointer identity is not stable across serde/bincode.
    #[serde(skip)]
    pub tied_sparse_embeddings: Option<TiedSparseEmbeddings>,
}

// Provide more flexible methods for getting properties of the network
impl NeuralNetwork {
    pub fn get_number_of_input_neurons(&self) -> usize {
        self.number_of_input_neurons
    }
    pub fn get_number_of_output_neurons(&self) -> usize {
        self.number_of_output_neurons
    }
    pub fn get_number_of_hidden_neurons(&self) -> usize {
        self.number_of_hidden_neurons
    }
    pub fn get_number_of_hidden_layers(&self) -> usize {
        self.number_of_hidden_layers
    }
    pub fn get_time_step(&self) -> usize {
        self.time_step
    }
    pub fn get_minibatch_size(&self) -> usize {
        self.minibatch_size
    }
    pub fn update_layer(&mut self, index: usize, new_layer: LayerEnum) {
        if index < self.layers.len() {
            self.layers[index] = new_layer;
        } else {
            println!("Layer index {} is out of bounds", index);
        }
    }
    pub fn update_step_lr_scheduler(&mut self, epoch: usize, step_size: usize, gamma: f64) {
        let factor = (epoch / step_size) as f64;
        let new_learning_rate = self.learning_rate * gamma.powf(factor);

        update_learning_rate(self, new_learning_rate);

        if epoch % step_size == 0 && epoch > 0 {
            println!("initial learning rate is: {:?}", self.learning_rate);
            println!("new learning rate is: {:?}", new_learning_rate);
        }
    }

    pub fn decay_learning_rate(&mut self, decay_factor: f64) {
        self.learning_rate *= decay_factor;
        update_learning_rate(self, self.learning_rate);
        println!("Manual decay applied. New LR: {:?}", self.learning_rate);
    }

    /// Restore runtime-only links (e.g. weight tying) after creating/loading a model.
    pub fn post_load_init(&mut self) {
        self.tie_embedding_and_sparse_linear();
    }

    /// Tie `EmbeddingLayer` and `SparseLinearLayer` to a single shared weight table.
    pub fn tie_embedding_and_sparse_linear(&mut self) {
        let mut embedding_idx: Option<usize> = None;
        let mut sparse_linear_idx: Option<usize> = None;
        let mut linear_idx: Option<usize> = None;

        for (idx, layer) in self.layers.iter().enumerate() {
            match layer {
                LayerEnum::Embedding(_) => embedding_idx = Some(idx),
                LayerEnum::SparseLinear(_) => sparse_linear_idx = Some(idx),
                LayerEnum::Linear(_) => linear_idx = Some(idx),
                _ => {}
            }
        }

        let Some(emb_i) = embedding_idx else {
            return;
        };

        // Prefer SparseLinear tying when present (existing behavior).
        let output_i = sparse_linear_idx.or(linear_idx);
        let Some(out_i) = output_i else {
            return;
        };
        if emb_i == out_i {
            return;
        }

        let (low, high) = if emb_i < out_i { (emb_i, out_i) } else { (out_i, emb_i) };
        let (left, right) = self.layers.split_at_mut(high);
        let a = &mut left[low];
        let b = &mut right[0];

        match (a, b) {
            (LayerEnum::Embedding(embedding_layer), LayerEnum::SparseLinear(sparse_linear_layer))
            | (LayerEnum::SparseLinear(sparse_linear_layer), LayerEnum::Embedding(embedding_layer)) => {
                let tied = TiedSparseEmbeddings::new(sparse_linear_layer.weights.clone());

                // SparseLinear owns the optimizer update; it needs access to embedding-side accumulated grads.
                sparse_linear_layer.tied_embedding_grad_by_token = Some(tied.grad_by_token.clone());
                embedding_layer.set_tied_weights(tied.weights.clone(), tied.grad_by_token.clone());

                self.tied_sparse_embeddings = Some(tied);
            }
            (LayerEnum::Embedding(embedding_layer), LayerEnum::Linear(linear_layer))
            | (LayerEnum::Linear(linear_layer), LayerEnum::Embedding(embedding_layer)) => {
                // Linear weights are stored as (in_dim x out_dim). For LM projection, out_dim is vocab.
                // Build a shared table in (vocab x in_dim) layout, matching Embedding/SparseLinear.
                if linear_layer.weights.is_empty() || linear_layer.weights[0].is_empty() {
                    return;
                }

                let in_dim = linear_layer.weights.len();
                let vocab = linear_layer.weights[0].len();

                // Transpose into vocab x in_dim f32 matrix (real part only).
                let mut table: Vec<Vec<f32>> = vec![vec![0.0; in_dim]; vocab];
                for i in 0..in_dim {
                    if linear_layer.weights[i].len() != vocab {
                        return;
                    }
                    for v in 0..vocab {
                        table[v][i] = linear_layer.weights[i][v].re as f32;
                    }
                }

                let shared = crate::neural_networks::utils::shared_f32_matrix::SharedF32Matrix::new(table);
                let tied = TiedSparseEmbeddings::new(shared.clone());

                // Linear owns the optimizer update; it needs access to embedding-side accumulated grads.
                embedding_layer.set_tied_weights(tied.weights.clone(), tied.grad_by_token.clone());
                linear_layer.set_tied_weights(tied.weights.clone(), tied.grad_by_token.clone());

                self.tied_sparse_embeddings = Some(tied);
            }
            _ => {}
        }
    }
}

pub fn create(number_inputs: usize, number_outputs: usize, number_of_hidden_layers: usize, number_of_hidden_neurons: usize, minibatch_size: usize, learning_rate: f64) -> NeuralNetwork {
    let feed_net = NeuralNetwork {
        layers: vec![],
        learning_rate,
        number_of_input_neurons: number_inputs,
        number_of_output_neurons: number_outputs,
        number_of_hidden_layers,
        number_of_hidden_neurons,
        minibatch_size,
        time_step: 0,
        smoothing: 0.9,
        ema: 0.0,
        global_norm: 0.0,
        max_norm: 0.0,
        tied_sparse_embeddings: None,
    };

    feed_net
}

pub fn save_to_sled(filename: &str, neural_network: &NeuralNetwork) {
    let model_to_save = neural_network.clone();
    let serialized_embedding = bincode::serialize(&model_to_save).expect("Failed to serialize transformer model");
    let filepath_buf: std::path::PathBuf = get_storage_path_transformer_db(filename);
    let filepath: &str = filepath_buf.to_str().unwrap();

    serialize_bin(&serialized_embedding, filepath).expect("File cannot be serialized");
    println!("✅ Transfomer model is saved in file: {:?}", &filepath);
}

pub fn get_from_db(filename: &str) -> Result<NeuralNetwork, String> {
    let filepath_buf: std::path::PathBuf = get_storage_path_transformer_db(filename);
    let filepath: &str = filepath_buf.to_str().unwrap();

    if !Path::new(filepath).exists() {
        println!("Transfomer model file does not exist, creating new model ....");
        let mut network = create_transformer(OperationMode::TRAINING);
        network.post_load_init();
        return Ok(network);
    }

    println!("✅ Transfomer model is loading from file: {:?}", &filepath);
    let transformer_result: Result<NeuralNetwork, std::io::Error> = derialize_bin::<NeuralNetwork>(filepath);
    let mut transformer = transformer_result.unwrap();

    transformer.post_load_init();

    Ok(transformer)
}

pub fn update_learning_rate(transformer: &mut NeuralNetwork, learning_rate: f64) {
    // transformer.learning_rate = learning_rate;
    for layer in transformer.layers.iter_mut() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_avg_pooling_layer) => {
                // println!("adaptive avg pooling layer with output size: {:?}", &adaptive_avg_pooling_layer.output_size);
            }
            LayerEnum::Embedding(embedding_layer) => {
                embedding_layer.learning_rate = learning_rate;
            }
            LayerEnum::EmbeddingRm(embedding_layer) => {
                embedding_layer.learning_rate = learning_rate;
            }
            LayerEnum::Norm(norm_layer) => {
                norm_layer.learning_rate = learning_rate;
            }
            LayerEnum::NormRm(norm_layer) => {
                norm_layer.learning_rate = learning_rate;
            }
            LayerEnum::RMSNorm(norm_layer) => {
                norm_layer.learning_rate = learning_rate;
            }
            LayerEnum::Dense(dense_layer) => {
                dense_layer.learning_rate = learning_rate;
            }
            LayerEnum::DenseRm(dense_layer_rm) => {
                dense_layer_rm.learning_rate = learning_rate;
            }
            LayerEnum::SelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    attention_head.learning_rate = learning_rate;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            rms_norm_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::Norm(norm_layer) => {
                            norm_layer.learning_rate = learning_rate;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    // attention_head.previous_gradient = None;
                    attention_head.gradient = None;
                    attention_head.batch_size = 0;
                    attention_head.learning_rate = learning_rate;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(_norm_layer) => {
                            //_norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                            _norm_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::Norm(_norm_layer) => {
                            //_norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                            _norm_layer.learning_rate = learning_rate;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttentionRm(self_attention_layer_rm) => {
                for attention_head in self_attention_layer_rm.attention_heads.iter_mut() {
                    attention_head.gradient = None;
                    attention_head.batch_size = 0;
                    attention_head.learning_rate = learning_rate;
                }
                self_attention_layer_rm.norm_layer.learning_rate = learning_rate;
            }
            LayerEnum::SelfAttentionApproximation(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    attention_head.learning_rate = learning_rate;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            rms_norm_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::Norm(norm_layer) => {
                            norm_layer.learning_rate = learning_rate;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SelfAttentionApproximationRm(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    attention_head.learning_rate = learning_rate;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            rms_norm_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::NormRm(norm_layer) => {
                            norm_layer.learning_rate = learning_rate;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForward(ffn_layer) => {
                ffn_layer.learning_rate = learning_rate;

                for layer in ffn_layer.layers.iter_mut() {
                    match layer {
                        LayerEnum::Dense(dense_layer) => {
                            dense_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::Linear(linear_layer) => {
                            linear_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::LinearRm(linear_layer) => {
                            linear_layer.learning_rate = learning_rate;
                        }
                        _ => {}
                    }
                }
                if let Some(norm_layer) = ffn_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            rms_norm_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::Norm(norm_layer) => {
                            norm_layer.learning_rate = learning_rate;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForwardRm(ffn_layer_rm) => {
                ffn_layer_rm.learning_rate = learning_rate;

                for layer in ffn_layer_rm.layers.iter_mut() {
                    match layer {
                        LayerEnum::DenseRm(dense_layer_rm) => {
                            dense_layer_rm.learning_rate = learning_rate;
                        }
                        LayerEnum::Dense(dense_layer) => {
                            dense_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::Linear(linear_layer) => {
                            linear_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::LinearRm(linear_layer) => {
                            linear_layer.learning_rate = learning_rate;
                        }
                        _ => {}
                    }
                }
                if let Some(norm_layer) = ffn_layer_rm.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            rms_norm_layer.learning_rate = learning_rate;
                        }
                        LayerEnum::Norm(norm_layer) => {
                            norm_layer.learning_rate = learning_rate;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::Linear(linear_layer) => {
                linear_layer.learning_rate = learning_rate;
            }
            LayerEnum::LinearRm(linear_layer) => {
                linear_layer.learning_rate = learning_rate;
            }
            LayerEnum::SparseLinear(sparse_linear_layer) => {
                sparse_linear_layer.learning_rate = learning_rate;
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                sparse_linear_layer.learning_rate = learning_rate;
            }
            LayerEnum::MultiLinear(linear_layer) => {
                linear_layer.learning_rate = learning_rate;
            }
            LayerEnum::Wavelet(_wavelet_layer) => {}
            LayerEnum::WaveletRm(_wavelet_layer) => {}
            LayerEnum::DiscreteWavelet(_wavelet_layer) => {}
            LayerEnum::DiscreteWaveletRm(_wavelet_layer) => {}
            LayerEnum::ComplexToLinear(ctl) => {
                ctl.learning_rate = crate::neural_networks::utils::dtype::r(learning_rate);
            }
            LayerEnum::ComplexToLinearRm(ctl) => {
                ctl.learning_rate = crate::neural_networks::utils::dtype::r(learning_rate);
            }
            LayerEnum::Softmax(_softmax_layer) => {}
            LayerEnum::SoftmaxRm(_softmax_layer) => {}
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
            LayerEnum::PositionalEncodingRm(_positional_encoding_layer) => {}
        }
    }
}

pub fn reset_previous_gradient(transformer: &mut NeuralNetwork) {
    // transformer.learning_rate = learning_rate;
    for layer in transformer.layers.iter_mut() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_avg_pooling_layer) => {
                // println!("adaptive avg pooling layer with output size: {:?}", &adaptive_avg_pooling_layer.output_size);
            }
            LayerEnum::Embedding(embedding_layer) => {
                // embedding_layer.previous_gradient = None;
                embedding_layer.gradient = None;
                embedding_layer.batch_size = 0;
            }
            LayerEnum::EmbeddingRm(embedding_layer) => {
                embedding_layer.gradient = None;
                embedding_layer.batch_size = 0;
            }
            LayerEnum::Norm(_norm_layer) => {
                // _norm_layer.previous_gradient = None;
                _norm_layer.gradient = None;
                _norm_layer.batch_size = 0;
            }
            LayerEnum::NormRm(_norm_layer) => {
                _norm_layer.gradient = None;
                _norm_layer.batch_size = 0;
            }
            LayerEnum::RMSNorm(_norm_layer) => {
                // _norm_layer.previous_gradient = None;
                _norm_layer.gradient = None;
                _norm_layer.batch_size = 0;
            }
            LayerEnum::Dense(dense_layer) => {
                // dense_layer.previous_gradient = None;
                dense_layer.gradient = None;
                dense_layer.batch_size = 0;
            }
            LayerEnum::DenseRm(dense_layer_rm) => {
                dense_layer_rm.gradient = None;
                dense_layer_rm.batch_size = 0;
            }
            LayerEnum::SelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    // attention_head.previous_gradient = None;
                    attention_head.gradient = None;
                    attention_head.batch_size = 0;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(_norm_layer) => {
                            //_norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        LayerEnum::Norm(_norm_layer) => {
                            //_norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    // attention_head.previous_gradient = None;
                    attention_head.gradient = None;
                    attention_head.batch_size = 0;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(_norm_layer) => {
                            //_norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        LayerEnum::Norm(_norm_layer) => {
                            //_norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttentionRm(self_attention_layer_rm) => {
                self_attention_layer_rm.gradient = None;
                for attention_head in self_attention_layer_rm.attention_heads.iter_mut() {
                    attention_head.gradient = None;
                    attention_head.batch_size = 0;
                }
                self_attention_layer_rm.norm_layer.gradient = None;
                self_attention_layer_rm.norm_layer.batch_size = 0;
            }
            LayerEnum::SelfAttentionApproximation(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    // attention_head.previous_gradient = None;
                    attention_head.gradient = None;
                    attention_head.batch_size = 0;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(_norm_layer) => {
                            //_norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        LayerEnum::Norm(_norm_layer) => {
                            //_norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SelfAttentionApproximationRm(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    attention_head.gradient = None;
                    attention_head.batch_size = 0;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(_norm_layer) => {
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        LayerEnum::NormRm(_norm_layer) => {
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForward(ffn_layer) => {
                ffn_layer.gradient = None;
                for layer in ffn_layer.layers.iter_mut() {
                    match layer {
                        LayerEnum::Dense(dense_layer) => {
                            // dense_layer.previous_gradient = None;
                            dense_layer.gradient = None;
                            dense_layer.batch_size = 0;
                        }
                        LayerEnum::Linear(linear_layer) => {
                            // linear_layer.previous_gradient = None;
                            linear_layer.gradient = None;
                            linear_layer.batch_size = 0;
                        }
                        LayerEnum::LinearRm(linear_layer) => {
                            linear_layer.gradient = None;
                            linear_layer.batch_size = 0;
                        }
                        _ => {}
                    }
                }
                if let Some(norm_layer) = ffn_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(_rms_norm_layer) => {
                            // _rms_norm_layer.previous_gradient = None;
                            _rms_norm_layer.gradient = None;
                            _rms_norm_layer.batch_size = 0;
                        }
                        LayerEnum::Norm(_norm_layer) => {
                            // _norm_layer.previous_gradient = None;
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForwardRm(ffn_layer_rm) => {
                ffn_layer_rm.gradient = None;
                ffn_layer_rm.batch_size = 0;
                for layer in ffn_layer_rm.layers.iter_mut() {
                    match layer {
                        LayerEnum::DenseRm(dense_layer_rm) => {
                            dense_layer_rm.gradient = None;
                            dense_layer_rm.batch_size = 0;
                        }
                        LayerEnum::Dense(dense_layer) => {
                            dense_layer.gradient = None;
                            dense_layer.batch_size = 0;
                        }
                        LayerEnum::Linear(linear_layer) => {
                            linear_layer.gradient = None;
                            linear_layer.batch_size = 0;
                        }
                        LayerEnum::LinearRm(linear_layer) => {
                            linear_layer.gradient = None;
                            linear_layer.batch_size = 0;
                        }
                        _ => {}
                    }
                }
                if let Some(norm_layer) = ffn_layer_rm.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(_rms_norm_layer) => {
                            _rms_norm_layer.gradient = None;
                            _rms_norm_layer.batch_size = 0;
                        }
                        LayerEnum::Norm(_norm_layer) => {
                            _norm_layer.gradient = None;
                            _norm_layer.batch_size = 0;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::Linear(linear_layer) => {
                // linear_layer.previous_gradient = None;
                linear_layer.gradient = None;
                linear_layer.batch_size = 0;
            }
            LayerEnum::LinearRm(linear_layer) => {
                linear_layer.gradient = None;
                linear_layer.batch_size = 0;
            }
            LayerEnum::SparseLinear(sparse_linear_layer) => {
                // sparse_linear_layer.previous_gradient = None;
                sparse_linear_layer.gradient = None;
                sparse_linear_layer.batch_size = 0;
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                sparse_linear_layer.gradient = None;
                sparse_linear_layer.batch_size = 0;
            }
            LayerEnum::MultiLinear(_linear_layer) => {
                // _linear_layer.previous_gradient = None;
                _linear_layer.gradient = None;
                _linear_layer.batch_size = 0;
            }
            LayerEnum::Wavelet(_wavelet_layer) => {
                _wavelet_layer.gradient = None;
            }
            LayerEnum::WaveletRm(_wavelet_layer) => {
                _wavelet_layer.gradient = None;
            }
            LayerEnum::DiscreteWavelet(_wavelet_layer) => {
                _wavelet_layer.gradient = None;
            }
            LayerEnum::DiscreteWaveletRm(_wavelet_layer) => {
                _wavelet_layer.gradient = None;
            }
            LayerEnum::ComplexToLinear(ctl) => {
                ctl.gradient = None;
            }
            LayerEnum::ComplexToLinearRm(ctl) => {
                ctl.gradient = None;
            }
            LayerEnum::Softmax(_softmax_layer) => {
                _softmax_layer.gradient = None;
            }
            LayerEnum::SoftmaxRm(_softmax_layer) => {
                _softmax_layer.gradient = None;
            }
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {
                _positional_encoding_layer.gradient = None;
            }
            LayerEnum::PositionalEncodingRm(_positional_encoding_layer) => {
                _positional_encoding_layer.gradient = None;
                _positional_encoding_layer.input_batch_rm = None;
            }
        }
    }
}

pub fn print_networt_structure(transformer: &mut NeuralNetwork) {
    // transformer.learning_rate = learning_rate;
    for layer in transformer.layers.iter_mut() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_avg_pooling_layer) => {
                // println!("adaptive avg pooling layer with output size: {:?}", &adaptive_avg_pooling_layer.output_size);
            }
            LayerEnum::Embedding(embedding_layer) => {
                println!("embedding layer: {:?}", &embedding_layer.learning_rate);
            }
            LayerEnum::EmbeddingRm(embedding_layer) => {
                println!("embedding_rm layer: {:?}", &embedding_layer.learning_rate);
            }
            LayerEnum::Norm(norm_layer) => {
                println!("norm layer: {:?}", &norm_layer.learning_rate);
            }
            LayerEnum::NormRm(norm_layer) => {
                println!("norm_rm layer: {:?}", &norm_layer.learning_rate);
            }
            LayerEnum::RMSNorm(norm_layer) => {
                println!("rms norm layer: {:?}", &norm_layer.learning_rate);
            }
            LayerEnum::Dense(dense_layer) => {
                println!("dense_layer layer: {:?}", &dense_layer.learning_rate);
            }
            LayerEnum::DenseRm(dense_layer_rm) => {
                println!("dense_rm layer: {:?} ({}x{})", &dense_layer_rm.learning_rate, dense_layer_rm.weights.rows, dense_layer_rm.weights.cols);
            }
            LayerEnum::SelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    println!("self attention head weigths: {} {}", attention_head.weights_k.len(), attention_head.weights_k[0].len());
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            println!("rms_norm_layer in ffn layer: {:?}", &rms_norm_layer.learning_rate);
                        }
                        LayerEnum::Norm(norm_layer) => {
                            println!("norm_layer in ffn layer: {:?}", &norm_layer.learning_rate);
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    println!("self attention head weigths: {} {}", attention_head.weights_k.len(), attention_head.weights_k[0].len());
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            println!("rms_norm_layer in ffn layer: {:?}", &rms_norm_layer.learning_rate);
                        }
                        LayerEnum::Norm(norm_layer) => {
                            println!("norm_layer in ffn layer: {:?}", &norm_layer.learning_rate);
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SelfAttentionApproximation(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    println!("self attention head weigths: {} {}", attention_head.weights_k.len(), attention_head.weights_k[0].len());
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            println!("rms_norm_layer in ffn layer: {:?}", &rms_norm_layer.learning_rate);
                        }
                        LayerEnum::Norm(norm_layer) => {
                            println!("norm_layer in ffn layer: {:?}", &norm_layer.learning_rate);
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SelfAttentionApproximationRm(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    println!("self attention head weigths: {} {}", attention_head.weights_k.len(), attention_head.weights_k[0].len());
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            println!("rms_norm_layer in ffn layer: {:?}", &rms_norm_layer.learning_rate);
                        }
                        LayerEnum::NormRm(norm_layer) => {
                            println!("norm_rm layer in ffn layer: {:?}", &norm_layer.learning_rate);
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForward(ffn_layer) => {
                for layer in ffn_layer.layers.iter_mut() {
                    match layer {
                        LayerEnum::Dense(dense_layer) => {
                            println!("ffn dense layer weigths: {} {}", dense_layer.weights.len(), dense_layer.weights[0].len());
                        }
                        LayerEnum::Linear(linear_layer) => {
                            println!("ffn linear layer weigths: {} {}", linear_layer.weights.len(), linear_layer.weights[0].len());
                        }
                        LayerEnum::LinearRm(linear_layer) => {
                            println!("ffn linear_rm layer weights: {}x{}", linear_layer.weights.rows, linear_layer.weights.cols);
                        }
                        _ => {}
                    }
                }
                if let Some(norm_layer) = ffn_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            println!("rms norm_layer in ffn layer: {:?}", &rms_norm_layer.learning_rate);
                        }
                        LayerEnum::Norm(norm_layer) => {
                            println!("norm_layer in ffn layer: {:?}", &norm_layer.learning_rate);
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForwardRm(ffn_layer_rm) => {
                for layer in ffn_layer_rm.layers.iter_mut() {
                    match layer {
                        LayerEnum::DenseRm(dense_layer_rm) => {
                            println!(
                                "ffn_rm dense layer weights: {}x{}",
                                dense_layer_rm.weights.rows,
                                dense_layer_rm.weights.cols
                            );
                        }
                        LayerEnum::Dense(dense_layer) => {
                            println!("ffn_rm (vec) dense layer weights: {} {}", dense_layer.weights.len(), dense_layer.weights[0].len());
                        }
                        LayerEnum::Linear(linear_layer) => {
                            println!("ffn_rm linear layer weights: {} {}", linear_layer.weights.len(), linear_layer.weights[0].len());
                        }
                        LayerEnum::LinearRm(linear_layer) => {
                            println!("ffn_rm linear_rm layer weights: {}x{}", linear_layer.weights.rows, linear_layer.weights.cols);
                        }
                        _ => {}
                    }
                }
                if let Some(norm_layer) = ffn_layer_rm.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            println!("rms norm_layer in ffn_rm layer: {:?}", &rms_norm_layer.learning_rate);
                        }
                        LayerEnum::Norm(norm_layer) => {
                            println!("norm_layer in ffn_rm layer: {:?}", &norm_layer.learning_rate);
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::Linear(linear_layer) => {
                println!("linear layer weigths: {} {}", linear_layer.weights.len(), linear_layer.weights[0].len());
            }
            LayerEnum::LinearRm(linear_layer) => {
                println!("linear_rm layer weights: {}x{}", linear_layer.weights.rows, linear_layer.weights.cols);
            }
            LayerEnum::SparseLinear(sparse_linear_layer) => {
                let (r, c) = sparse_linear_layer.weights.dims();
                println!("sparse linear layer weigths: {}x{}", r, c);
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                let (r, c) = sparse_linear_layer.weights.dims();
                println!("sparse linear_rm layer weigths: {}x{}", r, c);
            }
            LayerEnum::MultiLinear(multilinear_layer) => {
                println!("multilinear layers: {} ", multilinear_layer.layers.len());
            }
            LayerEnum::Wavelet(_wavelet_layer) => {
                println!("wavelet layer,  {:?}", &_wavelet_layer.wavelet);
            }
            LayerEnum::WaveletRm(_wavelet_layer) => {
                println!("wavelet_rm layer,  {:?}", &_wavelet_layer.wavelet);
            }
            LayerEnum::ComplexToLinear(ctl) => {
                println!("complex to linear layer,  weights: {} {}", ctl.weights_1.len(), ctl.weights_1.len());
            }
            LayerEnum::Softmax(_softmax_layer) => {
                println!("softmax layer");
            }
            LayerEnum::SoftmaxRm(_softmax_layer) => {
                println!("softmax_rm layer");
            }
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {
                println!("positional encoding layer");
            }
            LayerEnum::PositionalEncodingRm(_positional_encoding_layer) => {
                println!("positional encoding rm layer");
            }
            _ => {}
        }
    }
}
