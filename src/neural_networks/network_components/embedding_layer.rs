use bincode;
use core::fmt::Debug;
use num::Complex;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::database::sled_db::{get_db_embedding, get_storage_path_embedding_db};
use crate::neural_networks::network_types::wavelet_network::{decompose_in_wavelet_2d_default, DECOMPOSITION_LEVELS};
use crate::neural_networks::utils::matrix::{clip_gradients, is_nan_or_inf};
use crate::utils::normalization::normalize;

use super::gradient_struct::Gradient;
use super::layer_input_struct::LayerInput;
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub learning_rate: f64,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub previous_gradient: Option<Gradient>,
    pub time_step: usize,
    #[serde(skip)] // Skip during serialization
    pub cache: Arc<RwLock<HashMap<u32, Vec<Complex<f64>>>>>,
}

pub const EMBEDDING_PATH: &str = "embedding";
pub const FILE_NAME: &str = "embedding_layer.json";

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let db: &Db = get_db_embedding();

        // Generate random embeddings and store them in the Sled database
        for token_id in 0..vocab_size {
            Self::create_embedding(db, embedding_dim, &mut rng, token_id as u32);
        }

        let base_2: i32 = 2;
        let embedding_dim_compressed = (embedding_dim as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;

        Self {
            vocab_size,
            embedding_dim: embedding_dim_compressed,
            learning_rate: 0.001,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn create_embedding(db: &Db, embedding_dim: usize, rng: &mut rand::prelude::ThreadRng, token_id: u32) -> Vec<Complex<f64>> {
        let embedding: Vec<f64> = (0..embedding_dim).map(|_| rng.random_range(-0.4..0.5)).collect();

        // println!("\n\n\nembedding dimension in create: {:?} \n\n\n", embedding_dim);
        let embedding_wavelet: Vec<Complex<f64>> = decompose_in_wavelet_2d_default(&embedding)[0][0].clone();
        //println!("embedding: {:?}", &embedding_wavelet_1d);

        let token_u32: u32 = token_id as u32;
        Self::update_embedding(db, &token_u32, &embedding_wavelet);

        embedding_wavelet
    }

    /// Load an existing embedding layer (metadata only, embeddings are stored in Sled)
    pub fn load(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            vocab_size,
            embedding_dim,
            gradient: None,
            previous_gradient: None,
            learning_rate: 0.001,
            time_step: 0,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn get_or_create(vocab_size: usize, embedding_dim: usize, force_create: bool) -> Self {
        let mut embedding_path: PathBuf = get_storage_path_embedding_db(EMBEDDING_PATH);
        embedding_path.push(FILE_NAME);

        let embedding_file_path: &Path = Path::new(&embedding_path);

        if !embedding_file_path.exists() || force_create {
            println!("Embedding Layer does not exist. Creating Embedding Layer File");
            let embedding_layer = Self::new(vocab_size, embedding_dim);

            Self::serialize(&embedding_layer, &embedding_file_path);

            embedding_layer
        } else {
            println!("File exists. Deserializing the file");
            let embedding_layer = Self::deserialize(embedding_file_path);

            embedding_layer
        }
    }

    pub fn apply_padding_to_batch(token_input_ids_batch: &Vec<Vec<u32>>, _target_batch_ids: &Vec<Vec<u32>>) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let max_token_len = token_input_ids_batch.iter().map(|seq| seq.len()).max().unwrap_or(0);
        // max_token_len = max_token_len - target_batch_ids.len();

        //println!("token input before padding: {:?}", &token_input_ids_batch);
        // Initialize a new vector to hold padded sequences
        let mut token_input_ids_padded: Vec<Vec<u32>> = Vec::new();
        let mut padding_mask_batch: Vec<Vec<u32>> = Vec::new();

        // Iterate through each sequence in the batch
        for (_batch_ind, token_input_ids) in token_input_ids_batch.iter().enumerate() {
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

            // if !target_batch_ids.is_empty() {
            //     for i in 0..target_batch_ids[batch_ind].len() {
            //         padded_sequence.push(target_batch_ids[batch_ind][i]);
            //         padding_mask.push(1);
            //     }
            // }

            //println!("token id len after padding: {:?}", &padded_sequence.len());

            // Push the padded sequence to the output batch
            token_input_ids_padded.push(padded_sequence);
            padding_mask_batch.push(padding_mask);
        }

        // println!("token input after padding: {:?}", &token_input_ids_padded);
        // println!("max sequence length: {:?}", max_token_len);
        // println!("padding mask after padding: {:?}", &padding_mask_batch);

        (token_input_ids_padded, padding_mask_batch)
    }

    /// Look up embeddings for a batch of token IDs
    pub fn forward(&mut self, layer_input: &LayerInput) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<u32>>) {
        let token_input_ids: Vec<Vec<u32>> = layer_input.get_batch_ids();
        let target_batch_ids = layer_input.get_target_batch_ids();
        let db: &Db = get_db_embedding();
        self.time_step = layer_input.get_time_step();

        let (token_input_batch_padded, _padding_mask) = EmbeddingLayer::apply_padding_to_batch(&token_input_ids, &target_batch_ids);
        let embedding_dim = self.embedding_dim;

        // Get a reference to the cache
        let cache_ref = &self.cache;

        // Parallelize the processing of the token_ids
        let token_ids_output = token_input_batch_padded
            .par_iter()
            .map(|token_ids| {
                token_ids
                    .iter()
                    .map(|&id| {
                        if let Ok(cache) = cache_ref.read() {
                            if let Some(embedding) = cache.get(&id) {
                                return embedding.clone();
                            }
                        }
                        // Embedding retrieval if it's not in the cache
                        let token_embedding = if id == 1 {
                            vec![Complex::new(0.0, 0.0); embedding_dim]
                        } else {
                            let mut token_embedding = Self::get_embedding(&db, id).unwrap_or_else(|err| {
                                panic!("Error retrieving embedding for token {}: {}", id, err);
                            });

                            if token_embedding.len() != self.embedding_dim {
                                let mut rng = rand::rngs::ThreadRng::default();
                                let base_2: i32 = 2;
                                token_embedding = Self::create_embedding(&db, embedding_dim * base_2.pow(DECOMPOSITION_LEVELS) as usize, &mut rng, id);
                            }

                            assert_eq!(token_embedding.len(), self.embedding_dim);

                            if let Ok(mut cache) = cache_ref.write() {
                                cache.insert(id, token_embedding.clone());
                            }
                            token_embedding
                        };

                        // Return the embedding for this token
                        token_embedding
                    })
                    .collect::<Vec<Vec<Complex<f64>>>>() // Collect the result for each token
            })
            .collect::<Vec<Vec<Vec<Complex<f64>>>>>(); // Collect the result for the entire batch

        (token_ids_output, _padding_mask)
    }
    // Update embeddings using gradients
    pub fn backward(&mut self, previous_gradients: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let mut gradient = Gradient::new_default();

        gradient.set_gradient_input_batch(previous_gradients.clone());
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self, token_id_batches: &[Vec<u32>], learning_rate: f64) {
        let db: &Db = get_db_embedding();
        let gradient: &Gradient = self.gradient.as_ref().expect("Output batch is missing in dense layer");
        let mut previous_gradients: Vec<Vec<Vec<Complex<f64>>>> = gradient.get_gradient_input_batch();

        let batch_size = (previous_gradients.len() * self.vocab_size) as f64;

        // let max = previous_gradients.iter().flat_map(|v| v.iter().flat_map(|w| w.iter())).max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Less));
        // println!("max in backward embedding layer gradient batch: {:?}", max);
        // println!("min in backward embedding layer gradient batch: {:?}", min);

        for (batch_idx, token_ids) in token_id_batches.iter().enumerate() {
            clip_gradients(&mut previous_gradients[batch_idx], 1.0);

            for (i, &token_id) in token_ids.iter().enumerate() {
                let mut token_embedding: Vec<Complex<f64>> = Self::get_embedding(&db, token_id).unwrap();

                // Assuming previous_gradients[batch_idx][i] contains a single gradient
                let gradient = &previous_gradients[batch_idx][i];

                // println!("gradient embedding: {:?}", gradient);
                for j in 0..self.embedding_dim {
                    if is_nan_or_inf(&gradient[j]) {
                        panic!("gradient in embedding is invalid, contains NaN or infinity values: {:?}", &gradient[j]);
                    }
                    token_embedding[j] -= learning_rate * (gradient[j] / batch_size);

                    if is_nan_or_inf(&token_embedding[j]) {
                        panic!("embedding is invalid, contains NaN or infinity values: {:?}", &token_embedding[j]);
                    }
                }

                let embedding_normalized = normalize(&token_embedding);

                if let Ok(mut cache) = self.cache.write() {
                    cache.insert(token_id, embedding_normalized.clone());
                }

                Self::update_embedding(db, &token_id, &embedding_normalized);
                // println!("new token embedding: {:?}", token_embedding);
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
            gradient: None,
            previous_gradient: None,
            learning_rate: 0.001,
            time_step: 0,
            cache: Arc::new(RwLock::new(HashMap::new())),
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
        let token_id_u32: u32 = key.parse().unwrap();

        match db.get(&key) {
            Ok(Some(ivec)) => bincode::deserialize(&ivec).map_err(|_| "Failed to deserialize embedding".to_string()),
            Ok(None) => panic!("No token embedding found: {}", token_id_u32), // Return an error for missing keys
            Err(_) => Err("Failed to fetch embedding from Sled".to_string()),
        }
    }
}
