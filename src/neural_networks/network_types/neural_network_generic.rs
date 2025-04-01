use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::{database::sled_db::get_db, neural_networks::network_components::layer::LayerEnum};

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

pub fn save_to_sled(filename: &str, neural_network: &NeuralNetwork) {
    let db = get_db();

    // Clear old layers
    for i in 0.. {
        let key = format!("{}_layer_{}", filename, i);
        if db.remove(&key).is_err() {
            break; // Stop when there's nothing left
        }
    }

    for (i, layer) in neural_network.layers.iter().enumerate() {
        match layer {
            LayerEnum::Embedding(embedding_layer) => {
                // Serialize the layer after ensuring it is in the correct state
                let key = format!("{}_layer_{}", filename, i);
                let serialized_layer = bincode::serialize(embedding_layer).expect("Failed to serialize layer");
                db.insert(&key, serialized_layer).expect("Failed to save layer");
                db.flush().expect("Failed to flush DB after saving layer");
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                // Serialize the layer after ensuring it is in the correct state
                let key = format!("{}_layer_{}", filename, i);
                let serialized_layer = bincode::serialize(positional_encoding_layer).expect("Failed to serialize layer");
                db.insert(&key, serialized_layer).expect("Failed to save layer");
                db.flush().expect("Failed to flush DB after saving layer");
            }
            LayerEnum::Dense(dense_layer) => {
                // Serialize the layer after ensuring it is in the correct state
                let key = format!("{}_layer_{}", filename, i);
                let serialized_layer = bincode::serialize(dense_layer).expect("Failed to serialize layer");
                db.insert(&key, serialized_layer).expect("Failed to save layer");
                db.flush().expect("Failed to flush DB after saving layer");
            }
            LayerEnum::SelfAttention(attention_layer) => {
                // Serialize the layer after ensuring it is in the correct state
                let key = format!("{}_layer_{}", filename, i);
                let serialized_layer = bincode::serialize(attention_layer).expect("Failed to serialize layer");
                db.insert(&key, serialized_layer).expect("Failed to save layer");
                db.flush().expect("Failed to flush DB after saving layer");
            }
            LayerEnum::FeedForward(feed_forward_layer) => {
                // Serialize the layer after ensuring it is in the correct state
                let key = format!("{}_layer_{}", filename, i);
                let serialized_layer = bincode::serialize(feed_forward_layer).expect("Failed to serialize layer");
                db.insert(&key, serialized_layer).expect("Failed to save layer");
                db.flush().expect("Failed to flush DB after saving layer");
            }
            LayerEnum::Linear(linear_layer) => {
                // Serialize the layer after ensuring it is in the correct state
                let key = format!("{}_layer_{}", filename, i);
                let serialized_layer = bincode::serialize(linear_layer).expect("Failed to serialize layer");
                db.insert(&key, serialized_layer).expect("Failed to save layer");
                db.flush().expect("Failed to flush DB after saving layer");
            }
            LayerEnum::Softmax(softmax_layer) => {
                // Serialize the layer after ensuring it is in the correct state
                let key = format!("{}_layer_{}", filename, i);
                let serialized_layer = bincode::serialize(softmax_layer).expect("Failed to serialize layer");
                db.insert(&key, serialized_layer).expect("Failed to save layer");
                db.flush().expect("Failed to flush DB after saving layer");
            }
            _ => {} // Other layers you may need to handle
        }
    }

    let serialized_metadata: Vec<u8> = bincode::serialize(&(
        neural_network.learning_rate,
        neural_network.number_of_input_neurons,
        neural_network.number_of_output_neurons,
        neural_network.number_of_hidden_layers,
        neural_network.number_of_hidden_neurons,
        neural_network.minibatch_size,
    ))
    .expect("Failed to serialize metadata");

    db.insert(format!("{}_metadata", filename), serialized_metadata).expect("Failed to save metadata");
    db.flush().expect("Failed to flush DB");
    println!("✅ Network was saved in Sled under key: {}", filename);
}

pub fn get_from_db(filename: &str) -> Result<NeuralNetwork, String> {
    let db = get_db();

    let mut layers = Vec::new();
    let mut i = 0;

    // Load metadata
    let metadata_key = format!("{}_metadata", filename);
    let metadata: (f64, usize, usize, usize, usize, usize) = match db.get(&metadata_key).ok().flatten() {
        Some(raw_metadata) => match bincode::deserialize(&raw_metadata) {
            Ok(metadata) => metadata,
            Err(_) => return Err("Failed to deserialize metadata".to_string()),
        },
        None => return Err("Failed to load metadata".to_string()),
    };

    // Deserialize layers
    loop {
        let key = format!("{}_layer_{}", filename, i);
        match db.get(&key) {
            Ok(Some(ivec)) => {
                // Attempt to deserialize the layer based on its type
                match bincode::deserialize(&ivec) {
                    Ok(layer) => {
                       match layer {
                            LayerEnum::Embedding(embedding_layer) => {
                                layers.push(LayerEnum::Embedding(embedding_layer));
                            }
                            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                                layers.push(LayerEnum::PositionalEncoding(positional_encoding_layer));
                            }
                            LayerEnum::Norm(norm_layer) => {
                                layers.push(LayerEnum::Norm(norm_layer));
                            }
                            LayerEnum::SelfAttention(attention_layer) => {
                                layers.push(LayerEnum::SelfAttention(attention_layer));
                            }
                            LayerEnum::FeedForward(feed_forward_layer) => {
                                layers.push(LayerEnum::FeedForward(feed_forward_layer));
                            }
                            LayerEnum::Linear(linear_layer) => {
                                layers.push(LayerEnum::Linear(linear_layer));
                            }
                            LayerEnum::Softmax(softmax_layer) => {
                                layers.push(LayerEnum::Softmax(softmax_layer));
                            }
                            _ => {} // Handle other layers if necessary
                        }
                    },
                    Err(_) => return Err(format!("Failed to deserialize layer {} from DB", i)),
                }
            }
            Ok(None) => break, // No more layers
            Err(e) => return Err(format!("Failed to fetch layer {}: {}", i, e)),
        }
        i += 1;
    }

    if layers.is_empty() {
        return Err("No layers found in DB".to_string());
    }

    // Return the reconstructed neural network
    Ok(NeuralNetwork {
        layers,
        learning_rate: metadata.0,
        number_of_input_neurons: metadata.1,
        number_of_output_neurons: metadata.2,
        number_of_hidden_layers: metadata.3,
        number_of_hidden_neurons: metadata.4,
        minibatch_size: metadata.5,
    })
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
