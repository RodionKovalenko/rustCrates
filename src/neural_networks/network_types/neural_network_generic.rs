use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path};

use crate::{
    database::sled_db::get_storage_path_transformer_db,
    neural_networks::{
        network_components::layer::{ActivationType, LayerEnum},
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

        if epoch % step_size == 0 {
            println!("initial learning rate is: {:?}", self.learning_rate);
            println!("new learning rate is: {:?}", new_learning_rate);
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
            LayerEnum::FeedForward(ffn_layer) => {
                ffn_layer.learning_rate = learning_rate;

                for layer in ffn_layer.layers.iter_mut() {
                    match layer {
                        LayerEnum::Dense(dense_layer) => {
                            dense_layer.learning_rate = learning_rate;
                            dense_layer.activation_type = ActivationType::GELU;
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
            LayerEnum::Softmax(_softmax_layer) => {}
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
        }
    }
}

pub fn print_networt_structure(transformer: &mut NeuralNetwork) {
    // transformer.learning_rate = learning_rate;
    for layer in transformer.layers.iter_mut() {
        match layer {
            LayerEnum::Embedding(embedding_layer) => {
                println!("embedding layer: {:?}", &embedding_layer);
            }
            LayerEnum::Norm(norm_layer) => {
                println!("norm layer: {:?}", &norm_layer);
            }
            LayerEnum::RMSNorm(norm_layer) => {
                println!("rms norm layer: {:?}", &norm_layer);
            }
            LayerEnum::Dense(dense_layer) => {
                println!("dense_layer layer: {:?}", &dense_layer);
            }
            LayerEnum::SelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    println!("self attention head weigths: {} {}", attention_head.weights_k.len(), attention_head.weights_k[0].len());
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(rms_norm_layer) => {
                            println!("rms_norm_layer in ffn layer: {:?}", &rms_norm_layer);
                        }
                        LayerEnum::Norm(norm_layer) => {
                            println!("norm_layer in ffn layer: {:?}", &norm_layer);
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
                            dense_layer.activation_type = ActivationType::GELU;
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
                            println!("rms norm_layer in ffn layer: {:?}", &rms_norm_layer);
                        }
                        LayerEnum::Norm(norm_layer) => {
                            println!("norm_layer in ffn layer: {:?}", &norm_layer);
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::Linear(linear_layer) => {
                println!("linear layer weigths: {} {}", linear_layer.weights.len(), linear_layer.weights[0].len());
            }
            LayerEnum::Softmax(_softmax_layer) => {}
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
        }
    }
}
