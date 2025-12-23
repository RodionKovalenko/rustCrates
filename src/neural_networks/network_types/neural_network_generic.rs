use reqwest::ClientBuilder;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path};

use crate::{
    database::sled_db::get_storage_path_transformer_db,
    neural_networks::{
        network_components::layer::LayerEnum,
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
    };

    feed_net
}

pub fn save_to_sled(filename: &str, neural_network: &NeuralNetwork) {
    let serialized_embedding = bincode::serialize(&neural_network).expect("Failed to serialize transformer model");
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
        return Ok(create_transformer(OperationMode::TRAINING));
    }

    println!("✅ Transfomer model is loading from file: {:?}", &filepath);
    let transformer_result: Result<NeuralNetwork, std::io::Error> = derialize_bin::<NeuralNetwork>(filepath);
    let transformer = transformer_result.unwrap();

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
            LayerEnum::Norm(norm_layer) => {
                norm_layer.learning_rate = learning_rate;
            }
            LayerEnum::RMSNorm(norm_layer) => {
                norm_layer.learning_rate = learning_rate;
            }
            LayerEnum::Dense(dense_layer) => {
                dense_layer.learning_rate = learning_rate;
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
            LayerEnum::Linear(linear_layer) => {
                linear_layer.learning_rate = learning_rate;
            }
            LayerEnum::MultiLinear(linear_layer) => {
                linear_layer.learning_rate = learning_rate;
            }
            LayerEnum::Wavelet(_wavelet_layer) => {}
            LayerEnum::DiscreteWavelet(_wavelet_layer) => {}
            LayerEnum::ComplexToLinear(ctl) => {}
            LayerEnum::Softmax(_softmax_layer) => {}
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
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
            LayerEnum::Norm(_norm_layer) => {
                // _norm_layer.previous_gradient = None;
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
            LayerEnum::Linear(linear_layer) => {
                // linear_layer.previous_gradient = None;
                linear_layer.gradient = None;
                linear_layer.batch_size = 0;
            }
            LayerEnum::MultiLinear(_linear_layer) => {
                // _linear_layer.previous_gradient = None;
                _linear_layer.gradient = None;
                _linear_layer.batch_size = 0;
            }
            LayerEnum::Wavelet(_wavelet_layer) => {
                _wavelet_layer.gradient = None;
            }
            LayerEnum::DiscreteWavelet(_wavelet_layer) => {
                _wavelet_layer.gradient = None;
            }
            LayerEnum::ComplexToLinear(ctl) => {
                ctl.gradient = None;
            }
            LayerEnum::Softmax(_softmax_layer) => {
                _softmax_layer.gradient = None;
            }
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {
                _positional_encoding_layer.gradient = None;
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
            LayerEnum::Norm(norm_layer) => {
                println!("norm layer: {:?}", &norm_layer.learning_rate);
            }
            LayerEnum::RMSNorm(norm_layer) => {
                println!("rms norm layer: {:?}", &norm_layer.learning_rate);
            }
            LayerEnum::Dense(dense_layer) => {
                println!("dense_layer layer: {:?}", &dense_layer.learning_rate);
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
            LayerEnum::FeedForward(ffn_layer) => {
                for layer in ffn_layer.layers.iter_mut() {
                    match layer {
                        LayerEnum::Dense(dense_layer) => {
                            println!("ffn dense layer weigths: {} {}", dense_layer.weights.len(), dense_layer.weights[0].len());
                        }
                        LayerEnum::Linear(linear_layer) => {
                            println!("ffn linear layer weigths: {} {}", linear_layer.weights.len(), linear_layer.weights[0].len());
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
            LayerEnum::Linear(linear_layer) => {
                println!("linear layer weigths: {} {}", linear_layer.weights.len(), linear_layer.weights[0].len());
            }
            LayerEnum::MultiLinear(multilinear_layer) => {
                println!("multilinear layers: {} ", multilinear_layer.layers.len());
            }
            LayerEnum::Wavelet(_wavelet_layer) => {
                println!("wavelet layer,  {:?}", &_wavelet_layer.wavelet);
            }
            LayerEnum::DiscreteWavelet(_wavelet_layer) => {
                println!("discrete wavelet layer,  {:?}", &_wavelet_layer);
            }
            LayerEnum::ComplexToLinear(ctl) => {
                println!("complex to linear layer,  weights: {} {}", ctl.weights_1.len(), ctl.weights_1.len());
            }
            LayerEnum::Softmax(_softmax_layer) => {
                println!("softmax layer");
            }
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {
                println!("positional encoding layer");
            }
        }
    }
}
