use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        clustering_common::{
            accumulate_sparse_row_gradients, calculate_target_offset, compute_drift, get_target_id, update_centroids, update_sparse_parameters, ClusterIndex, SparseParameters, TopKScorer,
        },
        default_layer::LayerInterface,
        layer::LayerEnum,
    },
    network_types::transformer::transformer_updater::VERBOSE,
    optimization::k_means_clustering::kmeans,
    utils::{
        dtype::{C, ZERO},
        shared_f32_matrix::SharedF32Matrix,
        weights_initializer::initialize_weights_f32,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringLinearLayer {
    pub weights: SharedF32Matrix,
    pub previous_weights: Vec<Vec<f32>>,
    pub learning_rate: f64,
    pub bias: Vec<f32>,
    pub smoothing: f64,
    pub ema: f64,
    pub norm_layer: Option<LayerEnum>,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub drift_ema: f32,
    #[serde(skip)]
    pub drift_ema_beta: f32,

    // K-means clustering components
    // clusters of centroids; each centroid is a vector of f32 with embedding dimension size
    pub centroids: Vec<Vec<f32>>,
    // assignments of each token to a cluster; length = vocab size
    pub assignments: Vec<usize>,
    // mapping from cluster index to token IDs assigned to that cluste
    pub cluster_to_tokens: Vec<Vec<usize>>,
    pub n_clusters: usize,

    pub last_cluster_update_step: usize,

    #[serde(skip)]
    pub gradients: Vec<Vec<C>>,
    #[serde(skip)]
    pub gradients_bias: Vec<Vec<C>>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub output_indices: Option<Vec<Vec<Vec<usize>>>>,

    /// When set, this layer will also apply gradients accumulated by a tied embedding layer.
    #[serde(skip)]
    pub tied_embedding_grad_by_token: Option<Arc<RwLock<HashMap<usize, Vec<C>>>>>,
}


impl ClusteringLinearLayer {
    pub fn new(learning_rate: f64, embedding_d: usize, vocab_size: usize) -> Self {
        // Weights are used not as weights matrix, but as embedding matrix N x D
        // whre N is vocab size, D is embedding size
        let mut weights: Vec<Vec<f32>> = vec![vec![0.0; embedding_d]; vocab_size];
        let bias: Vec<f32> = vec![1.0; vocab_size];

        initialize_weights_f32(vocab_size, embedding_d, &mut weights);

        let n_clusters: usize = vocab_size / 16;
        let threshold = 0.001;

        let (centroids, assignments, cluster_to_tokens) = kmeans(&mut weights, n_clusters, 100, threshold);
        let previous_weights: Vec<Vec<f32>> = weights.clone();
        let weights = SharedF32Matrix::new(weights);

        Self {
            weights,
            previous_weights,
            bias,
            learning_rate,
            centroids: centroids,
            assignments: assignments,
            cluster_to_tokens: cluster_to_tokens,
            n_clusters: n_clusters,
            gradients: vec![],
            norm_layer: None,
            gradients_bias: vec![],
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            last_cluster_update_step: 0,
            drift_ema: 0.0,
            drift_ema_beta: 0.95,
            output_indices: None,
            tied_embedding_grad_by_token: None,
        }
    }

    pub fn update_centroids(&mut self, epoch: usize) {
        update_centroids(
            ClusterIndex {
                weights: &self.weights,
                previous_weights: &mut self.previous_weights,
                centroids: &mut self.centroids,
                assignments: &mut self.assignments,
                cluster_to_tokens: &mut self.cluster_to_tokens,
                n_clusters: self.n_clusters,
                last_cluster_update_step: &mut self.last_cluster_update_step,
                drift_ema: &mut self.drift_ema,
                drift_ema_beta: self.drift_ema_beta,
            },
            epoch,
        );
    }

    pub fn needs_cluster_update(&self, w: &Vec<Vec<f32>>, w_prev: &Vec<Vec<f32>>, tau: f32) -> bool {
        compute_drift(w, w_prev) > tau
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<C>>> = input.get_input_batch();

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());

        let start = std::time::Instant::now();

        let weights_guard = self.weights.read();
        let (output_batch, output_indices) = self.mutliply_hightest_k_per_row(&input_batch, input, &*weights_guard);

        if VERBOSE {
            println!("Sparse Linear layer complex matmul time for batch size {}: {}", self.batch_size, start.elapsed().as_secs_f64());
        }
        // println!("Output batch size in linear layer after dwt inverse:  {} {} {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output.set_output_indices(output_indices);

        self.output_indices = Some(layer_output.get_output_indices());

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let mut gradient = Gradient::new_default();

        let previous_gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient.get_gradient_input_batch();
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();

        // Initialize gradients for weights and biases
        // weight_gradients: batch x vocab_size x embedding_d
        let weights_guard = self.weights.read();

        let batch_len = input_batch.len();
        let seq_len = input_batch[0].len();
        let embedding_d = input_batch[0][0].len();

        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); weights_guard[0].len()]; weights_guard.len()]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];
        let mut gradient_input_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); embedding_d]; seq_len]; batch_len];

        // output_indices_batch: batch x seq_len x k (which vocab indices were selected)
        let output_indices_batch: &Vec<Vec<Vec<usize>>> = self.output_indices.as_ref().expect("Output indices missing in linear layer backward pass");

        for batch_idx in 0..input_batch.len() {
            let sparse_grad = &previous_gradient_input_batch[batch_idx]; // seq_len x k
            let indices = &output_indices_batch[batch_idx]; // seq_len x k

            for seq_idx in 0..sparse_grad.len().min(indices.len()) {
                accumulate_sparse_row_gradients(
                    &input_batch[batch_idx][seq_idx],
                    &sparse_grad[seq_idx],
                    &indices[seq_idx],
                    &weights_guard,
                    &mut weight_gradients[batch_idx],
                    &mut bias_gradients[batch_idx],
                    &mut gradient_input_batch[batch_idx][seq_idx],
                );
            }
        }

        gradient.set_gradient_input_batch(gradient_input_batch.clone());
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());

        gradient
    }

    // Multiply and select highest k per row, return k highest values per row and original indices
    pub fn mutliply_hightest_k_per_row(&self, input_batch: &Vec<Vec<Vec<C>>>, layer_input: &LayerInput, weights: &Vec<Vec<f32>>) -> (Vec<Vec<Vec<C>>>, Vec<Vec<Vec<usize>>>) {
        let target_batch = &layer_input.get_target_batch_ids();
        let padding_mask_batch = layer_input.get_padding_mask_batch();
        let scorer = TopKScorer {
            weights,
            bias: &self.bias,
            centroids: &self.centroids,
            cluster_to_tokens: &self.cluster_to_tokens,
            k: layer_input.get_top_k_size(),
            is_training: !target_batch.is_empty(),
        };

        input_batch
            .par_iter()
            .enumerate()
            .map(|(batch_idx, input_sample)| {
                let offset = calculate_target_offset(batch_idx, target_batch, &padding_mask_batch);
                let mut input_re_buf: Vec<f32> = vec![0.0; input_sample.first().map(|r| r.len()).unwrap_or(0)];

                input_sample
                    .iter()
                    .enumerate()
                    .map(|(row_idx, input_row)| scorer.top_k_row(input_row, get_target_id(row_idx, batch_idx, offset, target_batch, &padding_mask_batch), &mut input_re_buf))
                    .unzip()
            })
            .unzip()
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.take().expect("No Gradient found in linear layer");
        let params = SparseParameters {
            weights: &self.weights,
            bias: &mut self.bias,
            output_indices: self.output_indices.as_ref().expect("Output indices missing"),
            tied_embedding_grad_by_token: self.tied_embedding_grad_by_token.as_ref(),
            global_norm: self.global_norm,
            max_norm: self.max_norm,
            learning_rate: self.learning_rate,
            time_step: self.time_step,
        };
        self.previous_gradient = Some(update_sparse_parameters(params, gradient, self.previous_gradient.as_ref()));
    }
}

impl LayerInterface for ClusteringLinearLayer {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        ClusteringLinearLayer::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        ClusteringLinearLayer::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        ClusteringLinearLayer::update_parameters(self)
    }
}
