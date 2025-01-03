use bincode;
use num::Complex;
use core::fmt::Debug;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use lazy_static::lazy_static;

use crate::database::sled_config::{get_storage_path, SLED_DB_TOKENIZER};
use crate::neural_networks::network_types::wavelet_network::{decompose_in_wavelet_2d_default, DECOMPOSITION_LEVELS};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub weights: Vec<Vec<f64>>,
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
        let mut rng = rand::thread_rng();
        let db: &Db = Self::get_db();

        // Generate random embeddings and store them in the Sled database
        for token_id in 0..vocab_size {
            let embedding: Vec<f64> = (0..embedding_dim)
                .map(|_| rng.gen_range(-0.4..0.5))
                .collect();

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
        }
    }

    /// Load an existing embedding layer (metadata only, embeddings are stored in Sled)
    pub fn load(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            vocab_size,
            embedding_dim,
            weights: vec![],
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

    /// Look up embeddings for a batch of token IDs
    pub fn forward(&self, token_ids: &[u32]) -> Vec<Vec<Complex<f64>>> {
        let db: &Db = Self::get_db();
        token_ids
            .par_iter()
            .map(|&id| {
                Self::get_embedding(db, id).unwrap_or_else(|err| {
                    panic!("Error retrieving embedding for token {}: {}", id, err);
                })
            })
            .collect()
    }

    /// Update embeddings using gradients
    pub fn backward(&self, token_ids: &[u32], gradients: &Vec<Vec<Complex<f64>>>, learning_rate: f64) {
        let db: &Db = Self::get_db();
        for (i, &token_id) in token_ids.iter().enumerate() {
            let mut token_embedding: Vec<Complex<f64>> =
                Self::get_embedding(db, token_id).unwrap_or_else(|err| {
                    panic!("Error retrieving embedding for token {}: {}", token_id, err);
                });
            for j in 0..self.embedding_dim {
                // Gradient descent update for each weight
                token_embedding[j] -= learning_rate * gradients[i][j];
            }

            Self::update_embedding(&db, &token_id, &token_embedding);
        }
    }
    pub fn serialize(embedding_layer: &EmbeddingLayer, embedding_file_path: &Path) {
        println!("embedding layer saved: {:}", &embedding_layer.embedding_dim);
        let embedding_layer_meta = EmbeddingLayer {
            embedding_dim: embedding_layer.embedding_dim,
            vocab_size: embedding_layer.vocab_size,
            weights: vec![],
        };
        let serialized: Vec<u8> =
            bincode::serialize(&embedding_layer_meta).expect("Failed to serialize");
        let mut file = File::create(embedding_file_path).expect("Failed to create file");
        file.write_all(&serialized).expect("Failed to write file");
    }

    /// Load the embedding layer from a binary file
    pub fn deserialize(embedding_file_path: &Path) -> EmbeddingLayer {
        let mut file = File::open(embedding_file_path).expect("Failed to open file");
        let mut data = Vec::new();
        file.read_to_end(&mut data).expect("Failed to read file");
        bincode::deserialize(&data).expect("Failed to deserialize")
    }

    pub fn get_embedding(db: &Db, token_id: impl ToString) -> Result<Vec<Complex<f64>>, String> {
        let key = token_id.to_string();
        match db.get(&key) {
            Ok(Some(ivec)) => bincode::deserialize(&ivec)
                .map_err(|_| "Failed to deserialize embedding".to_string()), // Directly return Vec<f64>
            Ok(None) => Err("Token ID not found in DB".to_string()), // Return an error for missing keys
            Err(_) => Err("Failed to fetch embedding from Sled".to_string()), // Handle Sled errors
        }
    }
    /// Update an embedding for a given token ID
    pub fn update_embedding(db: &Db, token_id: &u32, embedding: &Vec<Complex<f64>>) {
        let key = token_id.to_string();
        let serialized_embedding =
            bincode::serialize(embedding).expect("Failed to serialize embedding");
        db.insert(key, serialized_embedding)
            .expect("Failed to save or update embedding in Sled");
    }
}
