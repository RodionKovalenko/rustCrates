use bincode;
use core::fmt::Debug;
use num::Complex;
use rand::RngExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::neural_networks::network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};
use crate::neural_networks::network_layers::default_layer::LayerInterface;
use crate::neural_networks::network_layers::wavelet_network::{decompose_in_wavelet_2d_default, DECOMPOSITION_LEVELS};
use crate::neural_networks::utils::dtype::{c_from_f64, c_to_f64, r, C, ZERO};
use crate::neural_networks::utils::matrix_rm::normalize_gradients_rm;
use crate::neural_networks::utils::matrix_rm::RowMajorMatrix;
use crate::neural_networks::utils::shared_f32_matrix::SharedF32Matrix;
use crate::neural_networks::utils::weights_initializer::initialize_weights_complex;

/// RM counterpart of `EmbeddingLayer`: the embedding table is a (vocab_size x embedding_dim) matrix in the layer,
/// trained in place. In tied mode the embeddings are read from the shared f32 table of `SparseLinearLayerRm`.
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
    /// Embedding table (vocab_size x embedding_dim), like `EmbeddingLayer::tied_weights`.
    pub weights: RowMajorMatrix<C>,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub tied_weights: Option<SharedF32Matrix>,
    #[serde(skip)]
    pub tied_grad_by_token: Option<Arc<RwLock<HashMap<usize, Vec<C>>>>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub last_batch_ids: Option<Vec<Vec<u32>>>,
}

pub const EMBEDDING_PATH: &str = "embedding";
pub const FILE_NAME: &str = "embedding_layer_rm.json";

impl EmbeddingLayerRm {
    /// Random embedding table, initialized like `EmbeddingLayer`.
    fn initialize_embedding_matrix(vocab_size: usize, embedding_dim: usize) -> RowMajorMatrix<C> {
        let mut weights: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); embedding_dim]; vocab_size];
        initialize_weights_complex(vocab_size, embedding_dim, &mut weights);
        RowMajorMatrix::from_rows(&weights)
    }

    pub fn set_tied_weights(&mut self, weights: SharedF32Matrix, grad_by_token: Arc<RwLock<HashMap<usize, Vec<C>>>>) {
        self.tied_weights = Some(weights);
        self.tied_grad_by_token = Some(grad_by_token);
    }

    pub fn is_tied(&self) -> bool {
        self.tied_weights.is_some() && self.tied_grad_by_token.is_some()
    }

    /// `embedding_dim` is the uncompressed dimension; the layer works with embedding_dim / 2^DECOMPOSITION_LEVELS.
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self::load(vocab_size, embedding_dim)
    }

    /// Same as `EmbeddingLayer::load`: a new layer with a random embedding table.
    pub fn load(vocab_size: usize, embedding_dim: usize) -> Self {
        let base_2: i32 = 2;
        let embedding_dim_compressed = (embedding_dim as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;

        Self {
            vocab_size,
            embedding_dim: embedding_dim_compressed,
            learning_rate: 0.001,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            previous_gradient: None,
            weights: Self::initialize_embedding_matrix(vocab_size, embedding_dim_compressed),
            gradient: None,
            time_step: 0,
            tied_weights: None,
            tied_grad_by_token: None,
            batch_size: 0,
            last_batch_ids: None,
        }
    }

    pub fn get_or_create(vocab_size: usize, embedding_dim: usize) -> Self {
        Self::load(vocab_size, embedding_dim)
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
    pub fn forward_inner(&mut self, layer_input: &LayerInput) -> (Vec<RowMajorMatrix<C>>, Vec<Vec<u32>>) {
        let token_input_ids: Vec<Vec<u32>> = layer_input.get_batch_ids();
        let target_batch_ids = layer_input.get_target_batch_ids();
        self.time_step = layer_input.get_time_step();
        self.batch_size = token_input_ids.len();

        let (token_input_batch_padded, padding_mask) = Self::apply_padding_to_batch(&token_input_ids, &target_batch_ids);
        self.last_batch_ids = Some(token_input_batch_padded.clone());
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

        let weights = &self.weights;
        let batch_rm: Vec<RowMajorMatrix<C>> = token_input_batch_padded
            .par_iter()
            .map(|token_ids| {
                let seq_len = token_ids.len();
                let mut data: Vec<C> = Vec::with_capacity(seq_len * embedding_dim);

                for &id in token_ids {
                    if id == 1 {
                        data.extend((0..embedding_dim).map(|_| C::new(ZERO, ZERO)));
                        continue;
                    }

                    let token_idx = id as usize;
                    if token_idx >= weights.rows {
                        panic!("token id out of range for embedding weights: {}", token_idx);
                    }
                    if weights.cols < embedding_dim {
                        panic!("embedding row too small for embedding_dim: row={} emb_dim={}", weights.cols, embedding_dim);
                    }

                    let row = weights.row_range(token_idx);
                    data.extend_from_slice(&weights.data[row.start..row.start + embedding_dim]);
                }

                RowMajorMatrix::from_data(seq_len, embedding_dim, data)
            })
            .collect();

        (batch_rm, padding_mask)
    }

    pub fn backward_inner(&mut self, previous_gradients_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(previous_gradients_rm.to_vec());
        self.gradient = Some(gradient.clone());
        gradient
    }

    /// Same update as `EmbeddingLayer::update_parameters_inner`: the per-sample gradients are normalized, then
    /// either accumulated for the tied table or applied to the embedding rows (plain SGD step).
    pub fn update_parameters_inner(&mut self, token_id_batches: &[Vec<u32>]) {
        let gradient: &Gradient = self.gradient.as_ref().expect("EmbeddingLayerRm missing gradients");

        let mut grads_rm = gradient.get_gradient_input_batch_rm_cow().into_owned();
        grads_rm.iter_mut().for_each(normalize_gradients_rm);

        if self.is_tied() {
            let acc = self.tied_grad_by_token.as_ref().expect("tied accumulator missing").clone();

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

        for (batch_idx, token_ids) in token_id_batches.iter().enumerate() {
            for (i, &token_id) in token_ids.iter().enumerate() {
                if token_id == 1 {
                    continue;
                }

                let token_idx = token_id as usize;
                if token_idx >= self.weights.rows {
                    panic!("token id out of range during embedding update: {}", token_idx);
                }

                let grads = &grads_rm[batch_idx];
                let grad_row = grads.row_range(i);
                let row = self.weights.row_range(token_idx);
                let n = self.weights.cols.min(grads.cols);
                for j in 0..n {
                    self.weights.data[row.start + j] -= grads.data[grad_row.start + j] * r(self.learning_rate);
                }
            }
        }

        self.gradient = None;
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
        serde_json::from_str(&contents).expect("Failed to deserialize EmbeddingLayerRm")
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let (embeddings_rm, padding_mask) = EmbeddingLayerRm::forward_inner(self, layer_input);
        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(embeddings_rm);
        output.set_padding_mask_batch(padding_mask);
        output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        EmbeddingLayerRm::backward_inner(self, &previous_gradient.get_gradient_input_batch_rm_cow())
    }

    pub fn update_parameters(&mut self) {
        let batch_ids = self.last_batch_ids.clone().unwrap_or_default();
        EmbeddingLayerRm::update_parameters_inner(self, &batch_ids)
    }
}

impl LayerInterface for EmbeddingLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        EmbeddingLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        EmbeddingLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        EmbeddingLayerRm::update_parameters(self)
    }
}
