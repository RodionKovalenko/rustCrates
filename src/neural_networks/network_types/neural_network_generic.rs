use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::fmt::Debug;

use crate::{
    database::sled_config::{get_storage_path, SLED_DB_TRANSFORMER},
    neural_networks::network_components::layer::LayerEnum,
};

pub const FILE_NAME: &str = "feedforward_network.json";
pub const FILE_NAME_TRANSFORMER: &str = "feedforward_network.json";

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
    fn get_db() -> &'static Db {
        lazy_static! {
            static ref DB: Db = sled::open(get_storage_path(SLED_DB_TRANSFORMER)).expect("Failed to open the database");
        }
        &DB
    }
    pub fn save_to_sled(&self, filename: &str) {
        let db: &Db = Self::get_db();
        let serialized_embedding = bincode::serialize(&self).expect("Failed to serialize transformer model");
        db.insert(filename, serialized_embedding).expect("Failed to save or update transformer model in Sled");

        println!("network was saved to sled database");
    }
    pub fn get_from_db(filename: &str) -> Result<NeuralNetwork, String>  {
        let key = filename.to_string();
        let db: &Db = Self::get_db();
        match db.get(&key) {
            Ok(Some(ivec)) => bincode::deserialize(&ivec).map_err(|_| "Failed to deserialize embedding".to_string()), // Directly return Vec<f64>
            Ok(None) => Err("Token ID not found in DB".to_string()),                                                  // Return an error for missing keys
            Err(_) => Err("Failed to fetch embedding from Sled".to_string()),                                         // Handle Sled errors
        }
    }
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
