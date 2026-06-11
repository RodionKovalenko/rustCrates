use bincode;
use core::fmt::Debug;
use num::Complex;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::neural_networks::network_components::gradient_struct::Gradient;
use crate::neural_networks::network_components::layer_input_struct::LayerInput;
use crate::neural_networks::network_components::layer_output_struct::LayerOutput;
use crate::neural_networks::network_layers::default_layer::LayerInterface;
use crate::neural_networks::network_layers::wavelet_network::{decompose_in_wavelet_2d_default, DECOMPOSITION_LEVELS};
use crate::neural_networks::utils::dtype::{c_from_f64, c_to_f64, r, C, ZERO};
use crate::neural_networks::utils::matrix::normalize_gradients_batch;
use crate::neural_networks::utils::weights_initializer::initialize_weights_complex;

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
    pub tied_weights: Vec<Vec<C>>,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub tied_grad_by_token: Option<Arc<RwLock<HashMap<usize, Vec<C>>>>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub last_batch_ids: Option<Vec<Vec<u32>>>,
}

pub const EMBEDDING_PATH: &str = "embedding";
pub const FILE_NAME: &str = "embedding_layer.json";

impl EmbeddingLayer {
    fn initialize_embedding_matrix(vocab_size: usize, embedding_dim: usize) -> Vec<Vec<C>> {
        let mut weights: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); embedding_dim]; vocab_size];
        initialize_weights_complex(vocab_size, embedding_dim, &mut weights);
        weights
    }

    pub fn set_tied_weights(&mut self, weights: Vec<Vec<C>>, grad_by_token: Arc<RwLock<HashMap<usize, Vec<C>>>>) {
        self.tied_weights = weights;
        self.tied_grad_by_token = Some(grad_by_token);
    }

    pub fn is_tied(&self) -> bool {
        self.tied_grad_by_token.is_some()
    }

    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let base_2: i32 = 2;
        let embedding_dim_compressed = (embedding_dim as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;

        Self {
            vocab_size,
            embedding_dim: embedding_dim_compressed,
            learning_rate: 0.001,
            gradient: None,
            previous_gradient: None,
            tied_weights: Self::initialize_embedding_matrix(vocab_size, embedding_dim_compressed),
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            tied_grad_by_token: None,
            global_norm: 0.0,
            max_norm: 0.0,
            last_batch_ids: None,
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
        let base_2: i32 = 2;
        let embedding_dim_compressed = (embedding_dim as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;

        Self {
            vocab_size,
            embedding_dim: embedding_dim_compressed,
            gradient: None,
            previous_gradient: None,
            tied_weights: Self::initialize_embedding_matrix(vocab_size, embedding_dim_compressed),
            learning_rate: 0.001,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            tied_grad_by_token: None,
            last_batch_ids: None,
        }
    }

    pub fn get_or_create(vocab_size: usize, embedding_dim: usize) -> Self {
        Self::load(vocab_size, embedding_dim)
    }

    pub fn apply_padding_to_batch(token_input_ids_batch: &Vec<Vec<u32>>, _target_batch_ids: &Vec<Vec<u32>>) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let max_token_len = token_input_ids_batch.iter().map(|seq| seq.len()).max().unwrap_or(0);
        // max_token_len = max_token_len - target_batch_ids.len();

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
    pub fn forward_inner(&mut self, layer_input: &LayerInput) -> (Vec<Vec<Vec<C>>>, Vec<Vec<u32>>) {
        let token_input_ids: Vec<Vec<u32>> = layer_input.get_batch_ids();
        let target_batch_ids = layer_input.get_target_batch_ids();
        self.time_step = layer_input.get_time_step();
        self.batch_size = token_input_ids.len();
        self.last_batch_ids = Some(token_input_ids.clone());

        let (token_input_batch_padded, _padding_mask) = EmbeddingLayer::apply_padding_to_batch(&token_input_ids, &target_batch_ids);
        let embedding_dim = self.embedding_dim;

        let token_ids_output: Vec<Vec<Vec<C>>> = token_input_batch_padded
            .iter()
            .map(|token_ids| {
                token_ids
                    .iter()
                    .map(|&id| {
                        if id == 1 {
                            return vec![C::new(ZERO, ZERO); embedding_dim];
                        }

                        let token_idx = id as usize;
                        if token_idx >= self.tied_weights.len() {
                            panic!("token id out of range for embedding weights: {}", token_idx);
                        }
                        let row = &self.tied_weights[token_idx];
                        if row.len() < embedding_dim {
                            panic!("embedding row too small for embedding_dim: row={} emb_dim={}", row.len(), embedding_dim);
                        }

                        row.iter().take(embedding_dim).copied().collect::<Vec<C>>()
                    })
                    .collect::<Vec<Vec<C>>>()
            })
            .collect::<Vec<Vec<Vec<C>>>>();

        (token_ids_output, _padding_mask)
    }
    // Update embeddings using gradients
    pub fn backward_inner(&mut self, previous_gradients: &Vec<Vec<Vec<C>>>) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(previous_gradients.clone());

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters_inner(&mut self, token_id_batches: &[Vec<u32>]) {
        let gradient: &Gradient = self.gradient.as_ref().expect("Output batch is missing in dense layer");
        let mut previous_gradients: Vec<Vec<Vec<C>>> = gradient.get_gradient_input_batch();

        normalize_gradients_batch(&mut previous_gradients);

        if self.is_tied() {
            let acc = self.tied_grad_by_token.as_ref().expect("tied accumulator missing").clone();

            let mut acc_lock = acc.write().expect("tied grad accumulator poisoned");
            for (batch_idx, token_ids) in token_id_batches.iter().enumerate() {
                for (i, &token_id) in token_ids.iter().enumerate() {
                    if token_id == 1 {
                        continue;
                    }

                    let token_idx = token_id as usize;
                    let grads = &previous_gradients[batch_idx][i];
                    let entry = acc_lock.entry(token_idx).or_insert_with(|| vec![C::new(ZERO, ZERO); self.embedding_dim]);
                    let n = entry.len().min(grads.len());
                    for j in 0..n {
                        entry[j] += grads[j];
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
                if token_idx >= self.tied_weights.len() {
                    panic!("token id out of range during embedding update: {}", token_idx);
                }

                let row = &mut self.tied_weights[token_idx];
                let grads = &previous_gradients[batch_idx][i];
                let n = row.len().min(grads.len());
                for j in 0..n {
                    row[j] -= grads[j] * r(self.learning_rate);
                }
            }
        }

        self.gradient = None;
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
            tied_weights: embedding_layer.tied_weights.clone(),
            learning_rate: 0.001,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            tied_grad_by_token: None,
            last_batch_ids: None,
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

        match db.get(&key) {
            Ok(Some(ivec)) => {
                let emb_f64: Vec<Complex<f64>> = bincode::deserialize(&ivec).map_err(|_| "Failed to deserialize embedding".to_string())?;
                Ok(emb_f64.into_iter().map(c_from_f64).collect())
            }
            Ok(None) => Err(format!("No token embedding found: {}", key)),
            Err(_) => Err("Failed to fetch embedding from Sled".to_string()),
        }
    }
}

impl EmbeddingLayer {
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let (embeddings, padding_mask) = EmbeddingLayer::forward_inner(self, layer_input);
        let mut output = LayerOutput::new_default();
        output.set_output_batch(embeddings);
        output.set_padding_mask_batch(padding_mask);
        output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let grad_batch = previous_gradient.get_gradient_input_batch();
        EmbeddingLayer::backward_inner(self, &grad_batch)
    }

    pub fn update_parameters(&mut self) {
        let batch_ids = self.last_batch_ids.clone().unwrap_or_default();
        EmbeddingLayer::update_parameters_inner(self, &batch_ids)
    }
}

impl LayerInterface for EmbeddingLayer {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        EmbeddingLayer::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        EmbeddingLayer::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        EmbeddingLayer::update_parameters(self)
    }
}
