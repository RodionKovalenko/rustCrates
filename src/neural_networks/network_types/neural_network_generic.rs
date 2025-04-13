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
            LayerEnum::FeedForward(dense_layer) => {
                dense_layer.learning_rate = learning_rate;

                if let Some(norm_layer) = dense_layer.norm_layer.as_mut() {
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
