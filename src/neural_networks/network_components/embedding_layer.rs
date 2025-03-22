use bincode;
use nalgebra::ComplexField;
use core::fmt::Debug;
use lazy_static::lazy_static;
use num::Complex;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::database::sled_config::{get_storage_path, SLED_DB_TOKENIZER};
use crate::neural_networks::network_types::wavelet_network::{decompose_in_wavelet_2d_default, DECOMPOSITION_LEVELS};
use crate::neural_networks::utils::matrix::is_nan_or_inf;

use super::gradient_struct::Gradient;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub weights: Vec<Vec<f64>>,
    pub gradient: Option<Gradient>,
}

pub const EMBEDDING_PATH: &str = "embedding";
pub const FILE_NAME: &str = "embedding_layer.json";

impl EmbeddingLayer {
    fn get_db() -> &'static Db {
        lazy_static! {
            static ref DB: Db = sled::open(get_storage_path(SLED_DB_TOKENIZER)).unwrap();
        }
        &DB
    }

    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let db: &Db = Self::get_db();

        // Generate random embeddings and store them in the Sled database
        for token_id in 0..vocab_size {
            let embedding: Vec<f64> = (0..embedding_dim).map(|_| rng.random_range(-0.4..0.5)).collect();

            let embedding_wavelet = &decompose_in_wavelet_2d_default(&embedding)[0][0];
            //println!("embedding: {:?}", &embedding_wavelet_1d);

            let token_u32: u32 = token_id as u32;
            Self::update_embedding(db, &token_u32, embedding_wavelet);
        }

        let base_2: i32 = 2;
        let embedding_dim_compressed = (embedding_dim as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;

        Self {
            vocab_size,
            embedding_dim: embedding_dim_compressed,
            weights: vec![],
            gradient: None,
        }
    }

    /// Load an existing embedding layer (metadata only, embeddings are stored in Sled)
    pub fn load(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            vocab_size,
            embedding_dim,
            weights: vec![],
            gradient: None,
        }
    }

    pub fn get_or_create(vocab_size: usize, embedding_dim: usize, force_create: bool) -> Self {
        let mut embedding_path: PathBuf = get_storage_path(EMBEDDING_PATH);
        embedding_path.push(FILE_NAME);

        let embedding_file_path: &Path = Path::new(&embedding_path);

        if !embedding_file_path.exists() || force_create {
            println!("Embedding Layer does not exist. Creating Embedding Layer File");
            let embedding_layer = Self::new(vocab_size, embedding_dim);

            Self::serialize(&embedding_layer, &embedding_file_path);

            embedding_layer
        } else {
            println!("File exists. Deserializing the file");
            Self::deserialize(embedding_file_path)
        }
    }

    pub fn apply_padding_to_batch(&self, token_input_ids_batch: &Vec<Vec<u32>>) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let max_token_len = token_input_ids_batch.iter().map(|seq| seq.len()).max().unwrap_or(0);

        //println!("token input before padding: {:?}", &token_input_ids_batch);
        // Initialize a new vector to hold padded sequences
        let mut token_input_ids_padded: Vec<Vec<u32>> = Vec::new();
        let mut padding_mask_batch: Vec<Vec<u32>> = Vec::new();

        // Iterate through each sequence in the batch
        for token_input_ids in token_input_ids_batch.iter() {
            // Clone the sequence to modify it (since we're not allowed to mutate the original)
            let mut padded_sequence = token_input_ids.clone();
            let mut padding_mask = vec![1; token_input_ids.len()];

            // Pad the sequence with token 1 (PAD token) until it reaches the max length
            while padded_sequence.len() < max_token_len {
                padded_sequence.push(1); // Add PAD token (with id = 1)
            }

            while padding_mask.len() < max_token_len {
                padding_mask.push(0); // Add Padding Mask token (0 -> ignore, 1 consider in the calculations)
            }

            //println!("token id len after padding: {:?}", &padded_sequence.len());

            // Push the padded sequence to the output batch
            token_input_ids_padded.push(padded_sequence);
            padding_mask_batch.push(padding_mask);
        }

        // println!("token input after padding: {:?}", &token_input_ids_padded);
        //println!("padding mask after padding: {:?}", &padding_mask_batch);

        (token_input_ids_padded, padding_mask_batch)
    }

    /// Look up embeddings for a batch of token IDs
    pub fn forward(&self, token_input_ids: &Vec<Vec<u32>>) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<u32>>) {
        let db: &Db = Self::get_db();

        let mut token_ids_output: Vec<Vec<Vec<Complex<f64>>>> = vec![];
        let (token_input_batch_padded, padding_mask) = self.apply_padding_to_batch(token_input_ids);

        let embedding_dim = self.embedding_dim.clone();

        //println!("embdding dim: {}", &embedding_dim);

        for token_ids in token_input_batch_padded {
            let token_output_id = token_ids
                .par_iter()
                .map(|&id| {
                    if id == 1 {
                        vec![Complex::new(0.0, 0.0); embedding_dim]
                    } else {
                        Self::get_embedding(db, id).unwrap_or_else(|err| {
                            panic!("Error retrieving embedding for token {}: {}", id, err);
                        })
                    }
                })
                .collect();

            token_ids_output.push(token_output_id);
        }

        (token_ids_output, padding_mask)
    }

    // Update embeddings using gradients
    pub fn backward(&mut self, previous_gradients: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let mut gradient = Gradient::new_default();

        gradient.set_gradient_input_batch(previous_gradients.clone());
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self, token_id_batches: &[Vec<u32>], learning_rate: f64) {
        let db: &Db = Self::get_db();
        let gradient: &Gradient = self.gradient.as_ref().expect("Output batch is missing in dense layer");
        let previous_gradients: Vec<Vec<Vec<Complex<f64>>>> = gradient.get_gradient_input_batch();

        let batch_size = previous_gradients.len() as f64;

        for (batch_idx, token_ids) in token_id_batches.iter().enumerate() {
            for (i, &token_id) in token_ids.iter().enumerate() {
                let mut token_embedding: Vec<Complex<f64>> = Self::get_embedding(db, token_id).unwrap();

                // Assuming previous_gradients[batch_idx][i] contains a single gradient
                let gradient = &previous_gradients[batch_idx][i];

                for j in 0..self.embedding_dim {
                    if !is_nan_or_inf(&gradient[j]) {
                        token_embedding[j] -= learning_rate * (gradient[j] / batch_size);

                        if is_nan_or_inf(&token_embedding[j]) || token_embedding[j].abs() > 2.0 {
                            token_embedding[j] = Complex::new(0.2, 0.3);
                        }
                    }
                }

                Self::update_embedding(db, &token_id, &token_embedding);
            }
        }
    }
    /// Update an embedding for a given token ID
    pub fn update_embedding(db: &Db, token_id: &u32, embedding: &Vec<Complex<f64>>) {
        let key = token_id.to_string();
        let serialized_embedding = bincode::serialize(embedding).expect("Failed to serialize embedding");
        db.insert(key, serialized_embedding).expect("Failed to save or update embedding in Sled");
    }

    pub fn serialize(embedding_layer: &EmbeddingLayer, embedding_file_path: &Path) {
        println!("embedding layer saved: {:}", &embedding_layer.embedding_dim);
        let embedding_layer_meta = EmbeddingLayer {
            embedding_dim: embedding_layer.embedding_dim,
            vocab_size: embedding_layer.vocab_size,
            weights: vec![],
            gradient: None,
        };
        let serialized: Vec<u8> = bincode::serialize(&embedding_layer_meta).expect("Failed to serialize");
        let mut file = File::create(embedding_file_path).expect("Failed to create file");
        file.write_all(&serialized).expect("Failed to write file");
    }

    /// Load the embedding layer from a binary file
    pub fn deserialize(embedding_file_path: &Path) -> EmbeddingLayer {
        println!("embedding file path: {:?}", &embedding_file_path);
        let mut file = File::open(embedding_file_path).expect("Failed to open file ");
        let mut data = Vec::new();
        file.read_to_end(&mut data).expect("Failed to read file");
        bincode::deserialize(&data).expect("Failed to deserialize")
    }

    pub fn get_embedding(db: &Db, token_id: impl ToString) -> Result<Vec<Complex<f64>>, String> {
        let key = token_id.to_string();
        match db.get(&key) {
            Ok(Some(ivec)) => bincode::deserialize(&ivec).map_err(|_| "Failed to deserialize embedding".to_string()), // Directly return Vec<f64>
            Ok(None) => Err("Token ID not found in DB".to_string()),                                                  // Return an error for missing keys
            Err(_) => Err("Failed to fetch embedding from Sled".to_string()),                                         // Handle Sled errors
        }
    }
}
