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
use std::sync::{Arc, RwLock};

use crate::database::sled_db::{get_db_embedding, get_storage_path_embedding_db};
use crate::neural_networks::network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput};
use crate::neural_networks::network_layers::wavelet_network::{decompose_in_wavelet_2d_default, DECOMPOSITION_LEVELS};
use crate::neural_networks::utils::dtype::{c_from_f64, c_to_f64, r, C, ONE, Real, ZERO};
use crate::neural_networks::utils::matrix::{is_nan_or_inf, RowMajorMatrix};
use crate::neural_networks::utils::shared_f32_matrix::SharedF32Matrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayerRm {
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
    pub tied_weights: Option<SharedF32Matrix>,
    #[serde(skip)]
    pub tied_grad_by_token: Option<Arc<RwLock<HashMap<usize, Vec<C>>>>>,
    #[serde(skip)]
    pub batch_size: usize,
}

pub const EMBEDDING_PATH: &str = "embedding";
pub const FILE_NAME: &str = "embedding_layer.json";

impl EmbeddingLayerRm {
    fn embedding_fallback(embedding_dim: usize) -> Vec<C> {
        vec![C::new(ZERO, ZERO); embedding_dim]
    }

    fn get_embedding_or_fallback(db: &Db, token_id: u32, embedding_dim: usize, init_if_missing: bool) -> Vec<C> {
        match Self::get_embedding(db, token_id) {
            Ok(v) => v,
            Err(_) => {
                if init_if_missing {
                    let mut rng = rand::rngs::ThreadRng::default();
                    let base_2: i32 = 2;
                    Self::create_embedding(db, embedding_dim * base_2.pow(DECOMPOSITION_LEVELS) as usize, &mut rng, token_id)
                } else {
                    Self::embedding_fallback(embedding_dim)
                }
            }
        }
    }

    pub fn set_tied_weights(&mut self, weights: SharedF32Matrix, grad_by_token: Arc<RwLock<HashMap<usize, Vec<C>>>>) {
        self.tied_weights = Some(weights);
        self.tied_grad_by_token = Some(grad_by_token);
    }

    pub fn is_tied(&self) -> bool {
        self.tied_weights.is_some() && self.tied_grad_by_token.is_some()
    }

    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng: rand::prelude::ThreadRng = rand::rng();
        let db: &Db = get_db_embedding();

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
            tied_weights: None,
            tied_grad_by_token: None,
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

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
            tied_weights: None,
            tied_grad_by_token: None,
        }
    }

    pub fn get_or_create(vocab_size: usize, embedding_dim: usize, force_create: bool) -> Self {
        let mut embedding_path: PathBuf = get_storage_path_embedding_db(EMBEDDING_PATH);
        embedding_path.push(FILE_NAME);

        let embedding_file_path: &Path = Path::new(&embedding_path);

        if !embedding_file_path.exists() || force_create {
            let embedding_layer = Self::new(vocab_size, embedding_dim);
            Self::serialize(&embedding_layer, embedding_file_path);
            embedding_layer
        } else {
            Self::deserialize(embedding_file_path)
        }
    }

    pub fn apply_padding_to_batch(token_input_ids_batch: &Vec<Vec<u32>>, _target_batch_ids: &Vec<Vec<u32>>) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let max_token_len = token_input_ids_batch.iter().map(|seq| seq.len()).max().unwrap_or(0);

        let mut token_input_ids_padded: Vec<Vec<u32>> = Vec::new();
        let mut padding_mask_batch: Vec<Vec<u32>> = Vec::new();

        for token_input_ids in token_input_ids_batch.iter() {
            let mut padded_sequence = token_input_ids.clone();
            let mut padding_mask = vec![1; token_input_ids.len()];

            while padded_sequence.len() < max_token_len {
                padded_sequence.push(1);
            }

            while padding_mask.len() < max_token_len {
                padding_mask.push(0);
            }

            token_input_ids_padded.push(padded_sequence);
            padding_mask_batch.push(padding_mask);
        }

        (token_input_ids_padded, padding_mask_batch)
    }

    /// RM-only embedding lookup.
    pub fn forward(&mut self, layer_input: &LayerInput) -> (Vec<RowMajorMatrix<C>>, Vec<Vec<u32>>) {
        let token_input_ids: Vec<Vec<u32>> = layer_input.get_batch_ids();
        let target_batch_ids = layer_input.get_target_batch_ids();
        self.time_step = layer_input.get_time_step();
        self.batch_size = token_input_ids.len();

        let (token_input_batch_padded, padding_mask) = Self::apply_padding_to_batch(&token_input_ids, &target_batch_ids);
        let embedding_dim = self.embedding_dim;

        if let Some(tied_weights) = &self.tied_weights {
            let weights = tied_weights.clone();
            let batch_rm: Vec<RowMajorMatrix<C>> = token_input_batch_padded
                .par_iter()
                .map(|token_ids| {
                    let weights_guard = weights.read();
                    let seq_len = token_ids.len();
                    let mut data: Vec<C> = Vec::with_capacity(seq_len * embedding_dim);

                    for &id in token_ids {
                        if id == 1 {
                            data.extend((0..embedding_dim).map(|_| C::new(ZERO, ZERO)));
                            continue;
                        }

                        let token_idx = id as usize;
                        if token_idx >= weights_guard.len() {
                            panic!("token id out of range for tied weights: {}", token_idx);
                        }
                        let row = &weights_guard[token_idx];
                        if row.len() < embedding_dim {
                            panic!("tied weights row too small for embedding_dim: row={} emb_dim={}", row.len(), embedding_dim);
                        }

                        data.extend(row.iter().take(embedding_dim).map(|&v| C::new(r(v as f64), ZERO)));
                    }

                    RowMajorMatrix::from_data(seq_len, embedding_dim, data)
                })
                .collect();

            return (batch_rm, padding_mask);
        }

        let db: &Db = get_db_embedding();
        let init_if_missing = !layer_input.get_forward_only() && layer_input.get_calculate_gradient();

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
                        let mut token_embedding = Self::get_embedding_or_fallback(&db, id, embedding_dim, init_if_missing);

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

    pub fn backward(&mut self, previous_gradients_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(previous_gradients_rm.to_vec());
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self, token_id_batches: &[Vec<u32>], learning_rate: f64) {
        let gradient: &Gradient = self.gradient.as_ref().expect("EmbeddingLayerRm missing gradients");

        let grads_rm = gradient
            .get_gradient_input_batch_rm_ref()
            .filter(|g| !g.is_empty())
            .expect("EmbeddingLayerRm::update_parameters expects RM gradients");

        if self.is_tied() {
            let acc = self
                .tied_grad_by_token
                .as_ref()
                .expect("tied accumulator missing")
                .clone();

            let mut acc_lock = acc.write().expect("tied grad accumulator poisoned");
            for (batch_idx, token_ids) in token_id_batches.iter().enumerate() {
                for (i, &token_id) in token_ids.iter().enumerate() {
                    if token_id == 1 {
                        continue;
                    }

                    let token_idx = token_id as usize;
                    let row = grads_rm[batch_idx].row_range(i);

                    let entry = acc_lock.entry(token_idx).or_insert_with(|| vec![C::new(ZERO, ZERO); self.embedding_dim]);
                    for j in 0..self.embedding_dim {
                        entry[j] += grads_rm[batch_idx].data[row.start + j];
                    }
                }
            }

            self.gradient = None;
            return;
        }

        let db: &Db = get_db_embedding();

        let mut batch_size: Real = if self.batch_size > 0 {
            r(self.batch_size as f64)
        } else {
            r(grads_rm.len() as f64)
        };
        if batch_size <= ZERO {
            batch_size = ONE;
        }

        let max_embedding_norm: Real = ONE;

        for (batch_idx, token_ids) in token_id_batches.iter().enumerate() {
            for (i, &token_id) in token_ids.iter().enumerate() {
                let mut token_embedding: Vec<C> = Self::get_embedding_or_fallback(&db, token_id, self.embedding_dim, true);

                for j in 0..self.embedding_dim {
                    let row = grads_rm[batch_idx].row_range(i);
                    let grad_val: C = grads_rm[batch_idx].data[row.start + j];

                    if is_nan_or_inf(&grad_val) {
                        panic!("gradient in embedding is invalid: {:?}", &grad_val);
                    }

                    token_embedding[j] -= r(learning_rate) * (grad_val / batch_size);

                    if is_nan_or_inf(&token_embedding[j]) {
                        panic!("embedding is invalid after update: {:?}", &token_embedding[j]);
                    }
                }

                let norm: Real = token_embedding.iter().map(|z| z.norm_sqr()).sum::<Real>().sqrt();
                if norm > max_embedding_norm {
                    let scale = max_embedding_norm / norm;
                    for z in &mut token_embedding {
                        *z *= scale;
                    }
                }

                if let Ok(mut cache) = self.cache.write() {
                    cache.insert(token_id, token_embedding.clone());
                }

                Self::update_embedding(db, &token_id, &token_embedding);
            }
        }
    }

    pub fn create_embedding(db: &Db, embedding_dim: usize, rng: &mut rand::prelude::ThreadRng, token_id: u32) -> Vec<C> {
        let embedding: Vec<f64> = (0..embedding_dim).map(|_| rng.random_range(-0.4..0.5)).collect();
        let embedding_wavelet_f64: Vec<Complex<f64>> = decompose_in_wavelet_2d_default(&embedding)[0][0].clone();
        let embedding_wavelet: Vec<C> = embedding_wavelet_f64.into_iter().map(c_from_f64).collect();
        Self::update_embedding(db, &token_id, &embedding_wavelet);
        embedding_wavelet
    }

    pub fn update_embedding(db: &Db, token_id: &u32, embedding: &Vec<C>) {
        let key = token_id.to_string();
        let embedding_f64: Vec<Complex<f64>> = embedding.iter().copied().map(c_to_f64).collect();
        let serialized_embedding = bincode::serialize(&embedding_f64).expect("Failed to serialize embedding");
        db.insert(key, serialized_embedding).expect("Failed to save or update embedding in Sled");
    }

    pub fn get_embedding(db: &Db, token_id: u32) -> Result<Vec<C>, String> {
        let key = token_id.to_string();
        let value = db.get(key).map_err(|e| e.to_string())?;
        if let Some(bytes) = value {
            let embedding_f64: Vec<Complex<f64>> = bincode::deserialize(&bytes).map_err(|e| e.to_string())?;
            Ok(embedding_f64.into_iter().map(c_from_f64).collect())
        } else {
            Err("Embedding not found".to_string())
        }
    }

    pub fn serialize(embedding_layer: &EmbeddingLayerRm, embedding_file_path: &Path) {
        let serialized = serde_json::to_string(embedding_layer).expect("Failed to serialize EmbeddingLayerRm");
        let mut file = File::create(embedding_file_path).expect("Failed to create file");
        file.write_all(serialized.as_bytes()).expect("Failed to write file");
    }

    pub fn deserialize(embedding_file_path: &Path) -> EmbeddingLayerRm {
        let mut file = File::open(embedding_file_path).expect("Failed to open file");
        let mut contents = String::new();
        file.read_to_string(&mut contents).expect("Failed to read file");
        let mut layer: EmbeddingLayerRm = serde_json::from_str(&contents).expect("Failed to deserialize EmbeddingLayerRm");
        layer.cache = Arc::new(RwLock::new(HashMap::new()));
        layer
    }
}
