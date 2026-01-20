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
use crate::neural_networks::network_components::gradient_struct::Gradient;
use crate::neural_networks::network_components::layer_input_struct::LayerInput;
use crate::neural_networks::network_layers::wavelet_network::{DECOMPOSITION_LEVELS, decompose_in_wavelet_2d_default};
use crate::neural_networks::utils::dtype::{c_from_f64, c_to_f64, r, C, ONE, Real, ZERO};
use crate::neural_networks::utils::matrix::{clip_all_gradients_by_global_norm_3d, is_nan_or_inf};
use crate::neural_networks::utils::matrix::RowMajorMatrix;

use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub cache: Arc<RwLock<HashMap<u32, Vec<C>>>>,
    #[serde(skip)]
    pub batch_size: usize,
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
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            cache: Arc::new(RwLock::new(HashMap::new())),
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    pub fn create_embedding(db: &Db, embedding_dim: usize, rng: &mut rand::prelude::ThreadRng, token_id: u32) -> Vec<C> {
        let embedding: Vec<f64> = (0..embedding_dim).map(|_| rng.random_range(-0.4..0.5)).collect();

        // println!("\n\n\nembedding dimension in create: {:?} \n\n\n", embedding_dim);
        let embedding_wavelet_f64: Vec<Complex<f64>> = decompose_in_wavelet_2d_default(&embedding)[0][0].clone();
        //println!("embedding: {:?}", &embedding_wavelet_1d);

        let embedding_wavelet: Vec<C> = embedding_wavelet_f64.into_iter().map(c_from_f64).collect();

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
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
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
    pub fn forward(&mut self, layer_input: &LayerInput) -> (Vec<Vec<Vec<C>>>, Vec<Vec<u32>>) {
        let token_input_ids: Vec<Vec<u32>> = layer_input.get_batch_ids();
        let target_batch_ids = layer_input.get_target_batch_ids();
        let db: &Db = get_db_embedding();
        self.time_step = layer_input.get_time_step();
        self.batch_size = token_input_ids.len();

        let (token_input_batch_padded, _padding_mask) = EmbeddingLayer::apply_padding_to_batch(&token_input_ids, &target_batch_ids);
        let embedding_dim = self.embedding_dim;

        // Get a reference to the cache
        let cache_ref = &self.cache;

        // Parallelize the processing of the token_ids
        let token_ids_output: Vec<Vec<Vec<C>>> = token_input_batch_padded
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
                            vec![C::new(ZERO, ZERO); embedding_dim]
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
                        .collect::<Vec<Vec<C>>>() // Collect the result for each token
            })
                    .collect::<Vec<Vec<Vec<C>>>>(); // Collect the result for the entire batch

        (token_ids_output, _padding_mask)
    }

    /// Row-major version of `forward` that avoids nested Vec allocations.
    pub fn forward_rm(&mut self, layer_input: &LayerInput) -> (Vec<RowMajorMatrix<C>>, Vec<Vec<u32>>) {
        let token_input_ids: Vec<Vec<u32>> = layer_input.get_batch_ids();
        let target_batch_ids = layer_input.get_target_batch_ids();
        let db: &Db = get_db_embedding();
        self.time_step = layer_input.get_time_step();
        self.batch_size = token_input_ids.len();

        let (token_input_batch_padded, padding_mask) = EmbeddingLayer::apply_padding_to_batch(&token_input_ids, &target_batch_ids);
        let embedding_dim = self.embedding_dim;

        let cache_ref = &self.cache;

        let batch_rm: Vec<RowMajorMatrix<C>> = token_input_batch_padded
            .par_iter()
            .map(|token_ids| {
                let seq_len = token_ids.len();
            let mut data: Vec<C> = Vec::with_capacity(seq_len * embedding_dim);

                for &id in token_ids {
                    if let Ok(cache) = cache_ref.read() {
                        if let Some(embedding) = cache.get(&id) {
                            data.extend_from_slice(embedding);
                            continue;
                        }
                    }

                    let token_embedding = if id == 1 {
                        vec![C::new(ZERO, ZERO); embedding_dim]
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

                    data.extend_from_slice(&token_embedding);
                }

                RowMajorMatrix::from_data(seq_len, embedding_dim, data)
            })
            .collect();

        (batch_rm, padding_mask)
    }
    // Update embeddings using gradients
    pub fn backward(&mut self, previous_gradients: &Vec<Vec<Vec<C>>>) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(previous_gradients.clone());

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn backward_rm(&mut self, previous_gradients_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(previous_gradients_rm.to_vec());

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self, token_id_batches: &[Vec<u32>], learning_rate: f64) {
        let db: &Db = get_db_embedding();
        let gradient: &Gradient = self.gradient.as_ref().expect("Output batch is missing in dense layer");

        // Prefer RM gradients when present to avoid conversions.
        let previous_gradients_rm_ref = gradient.get_gradient_input_batch_rm_ref();
        let has_rm = previous_gradients_rm_ref.is_some_and(|g| !g.is_empty());

        let mut batch_size: Real = if self.batch_size > 0 {
            r(self.batch_size as f64)
        } else if has_rm {
            r(previous_gradients_rm_ref.unwrap().len() as f64)
        } else {
            r(gradient.get_gradient_input_batch().len() as f64)
        };

        if batch_size <= ZERO {
            batch_size = ONE;
        }

        let mut previous_gradients: Vec<Vec<Vec<C>>> = Vec::new();
        if !has_rm {
            previous_gradients = gradient.get_gradient_input_batch();
            let mut dummy_bias: Vec<C> = Vec::new();
            clip_all_gradients_by_global_norm_3d(&mut previous_gradients, &mut dummy_bias, self.global_norm, self.max_norm);
        }

        // Mirror clip_all_gradients_by_global_norm_3d behavior for RM gradients (scale by 1/total_norm).
        let clip_scale_rm: Real = if has_rm && self.global_norm > self.max_norm {
            r(1.0 / self.global_norm)
        } else {
            ONE
        };

        // let max = previous_gradients.iter().flat_map(|v| v.iter().flat_map(|w| w.iter())).max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Less));
        // println!("max in backward embedding layer gradient batch: {:?}", max);
        // println!("min in backward embedding layer gradient batch: {:?}", min);

        let max_embedding_norm: Real = ONE;

        for (batch_idx, token_ids) in token_id_batches.iter().enumerate() {
            for (i, &token_id) in token_ids.iter().enumerate() {
                let mut token_embedding: Vec<C> = Self::get_embedding(&db, token_id).unwrap();

                // SGD update
                for j in 0..self.embedding_dim {
                    let grad_val: C = if has_rm {
                        let gr_rm = previous_gradients_rm_ref.unwrap();
                        let row = gr_rm[batch_idx].row_range(i);
                        gr_rm[batch_idx].data[row.start + j] * clip_scale_rm
                    } else {
                        previous_gradients[batch_idx][i][j]
                    };

                    if is_nan_or_inf(&grad_val) {
                        panic!("gradient in embedding is invalid, contains NaN or infinity values: {:?}", &grad_val);
                    }

                    token_embedding[j] -= r(learning_rate) * (grad_val / batch_size);

                    if is_nan_or_inf(&token_embedding[j]) {
                        panic!("embedding is invalid, contains NaN or infinity values: {:?}", &token_embedding[j]);
                    }
                }

                // ---- 🔒 Embedding norm clipping (NOT normalization) ----
                let norm: Real = token_embedding.iter().map(|z| z.norm_sqr()).sum::<Real>().sqrt();

                if norm > max_embedding_norm {
                    let scale = max_embedding_norm / norm;
                    for z in &mut token_embedding {
                        *z *= scale;
                    }
                }
                // -------------------------------------------------------

                if let Ok(mut cache) = self.cache.write() {
                    cache.insert(token_id, token_embedding.clone());
                } else {
                    panic!("embedding was not updated in cache");
                }

                Self::update_embedding(db, &token_id, &token_embedding);
            }
        }
    }
    /// Update an embedding for a given token ID
    pub fn update_embedding(db: &Db, token_id: &u32, embedding: &Vec<C>) {
        let key = token_id.to_string();

        let embedding_f64: Vec<Complex<f64>> = embedding.iter().copied().map(c_to_f64).collect();
        let serialized_embedding = bincode::serialize(&embedding_f64).expect("Failed to serialize embedding");
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
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
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

    pub fn get_embedding(db: &Db, token_id: impl ToString) -> Result<Vec<C>, String> {
        let key = token_id.to_string();
        let token_id_u32: u32 = key.parse().unwrap();

        match db.get(&key) {
            Ok(Some(ivec)) => {
                let emb_f64: Vec<Complex<f64>> = bincode::deserialize(&ivec).map_err(|_| "Failed to deserialize embedding".to_string())?;
                Ok(emb_f64.into_iter().map(c_from_f64).collect())
            }
            Ok(None) => panic!("No token embedding found: {}", token_id_u32), // Return an error for missing keys
            Err(_) => Err("Failed to fetch embedding from Sled".to_string()),
        }
    }
}
