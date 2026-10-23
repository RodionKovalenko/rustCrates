use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        network_layers_rm::clustering_common_rm::{accumulate_sparse_row_gradients, calculate_target_offset, get_target_id, update_centroids, update_sparse_parameters, ClusterIndex, SparseParameters, TopKScorer},
        default_layer::LayerInterface,
    },
    network_types::transformer::transformer_updater::VERBOSE,
    optimization::k_means_clustering::kmeans,
    utils::{
        dtype::{C, ZERO},
        matrix_rm::RowMajorMatrix,
        shared_f32_matrix::SharedF32Matrix,
        weights_initializer::initialize_weights_f32,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLinearLayerRm {
    pub weights: SharedF32Matrix,
    pub previous_weights: Vec<Vec<f32>>,
    pub learning_rate: f64,
    pub bias: Vec<f32>,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    // K-means clustering components
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    pub cluster_to_tokens: Vec<Vec<usize>>,
    pub n_clusters: usize,

    pub last_cluster_update_step: usize,

    #[serde(skip)]
    pub drift_ema: f32,
    #[serde(skip)]
    pub drift_ema_beta: f32,

    #[serde(skip)]
    pub gradients: Vec<Vec<C>>,
    #[serde(skip)]
    pub gradients_bias: Vec<Vec<C>>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
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

impl SparseLinearLayerRm {
    pub fn new(learning_rate: f64, embedding_d: usize, vocab_size: usize) -> Self {
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
            centroids,
            assignments,
            cluster_to_tokens,
            n_clusters,
            gradients: vec![],
            gradients_bias: vec![],
            input_batch_rm: None,
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

    /// Enable weight tying with an embedding layer.
    pub fn enable_weight_tying(&mut self) -> Arc<RwLock<HashMap<usize, Vec<C>>>> {
        let acc = Arc::new(RwLock::new(HashMap::new()));
        self.tied_embedding_grad_by_token = Some(acc.clone());
        acc
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

    /// RM counterpart of `ClusteringLinearLayer::forward`.
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let input_batch_rm = input.get_input_batch_rm();
        let start = std::time::Instant::now();

        let (output_batch_rm, output_indices) = self.mutliply_hightest_k_per_row_rm(&input_batch_rm, input);

        if VERBOSE {
            println!("SparseLinearRm matmul time for batch size {}: {}", self.batch_size, start.elapsed().as_secs_f64());
        }

        self.input_batch_rm = Some(input_batch_rm);
        self.output_indices = Some(output_indices.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output.set_output_indices(output_indices);
        layer_output
    }

    /// RM counterpart of `ClusteringLinearLayer::backward`.
    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("Input batch is missing in SparseLinearLayerRm");
        let output_indices_batch = self.output_indices.as_ref().expect("Output indices missing in SparseLinearLayerRm backward pass");
        let previous_gradient_rm = previous_gradient.get_gradient_input_batch_rm_cow();

        let weights_guard = self.weights.read();
        let (vocab_size, embedding_d) = (weights_guard.len(), weights_guard.first().map(|w| w.len()).unwrap_or(0));
        let batch_len = input_batch_rm.len();

        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); embedding_d]; vocab_size]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];
        let mut input_gradient_batch_rm: Vec<RowMajorMatrix<C>> = RowMajorMatrix::create_from_batch(input_batch_rm);

        for batch_idx in 0..batch_len {
            let (input_rm, grad_rm) = (&input_batch_rm[batch_idx], &previous_gradient_rm[batch_idx]);
            let indices = &output_indices_batch[batch_idx];

            for seq_idx in 0..grad_rm.rows.min(indices.len()) {
                let in_row = input_rm.row_range(seq_idx);
                accumulate_sparse_row_gradients(
                    &input_rm.data[in_row.clone()],
                    &grad_rm.data[grad_rm.row_range(seq_idx)],
                    &indices[seq_idx],
                    &weights_guard,
                    &mut weight_gradients[batch_idx],
                    &mut bias_gradients[batch_idx],
                    &mut input_gradient_batch_rm[batch_idx].data[in_row],
                );
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        gradient.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());
        self.gradient = Some(gradient.clone());

        gradient
    }

    /// Multiply and select highest k per row, return k highest values per row and original indices.
    pub fn mutliply_hightest_k_per_row_rm(&self, input_batch_rm: &[RowMajorMatrix<C>], layer_input: &LayerInput) -> (Vec<RowMajorMatrix<C>>, Vec<Vec<Vec<usize>>>) {
        let target_batch = &layer_input.get_target_batch_ids();
        let padding_mask_batch = layer_input.get_padding_mask_batch();
        let weights_guard = self.weights.read();
        let k = layer_input.get_top_k_size();
        let scorer = TopKScorer {
            weights: &weights_guard,
            bias: &self.bias,
            centroids: &self.centroids,
            cluster_to_tokens: &self.cluster_to_tokens,
            k,
            is_training: !target_batch.is_empty(),
        };

        input_batch_rm
            .par_iter()
            .enumerate()
            .map(|(batch_idx, input_sample)| {
                let offset = calculate_target_offset(batch_idx, target_batch, &padding_mask_batch);
                let mut input_re_buf: Vec<f32> = vec![0.0; input_sample.cols];
                let mut data: Vec<C> = Vec::with_capacity(input_sample.rows * k);
                let mut sample_indices: Vec<Vec<usize>> = Vec::with_capacity(input_sample.rows);

                for row_idx in 0..input_sample.rows {
                    let target_id = get_target_id(row_idx, batch_idx, offset, target_batch, &padding_mask_batch);
                    let (values, indices) = scorer.top_k_row(&input_sample.data[input_sample.row_range(row_idx)], target_id, &mut input_re_buf);
                    data.extend(values);
                    sample_indices.push(indices);
                }

                (RowMajorMatrix::from_data(input_sample.rows, k, data), sample_indices)
            })
            .unzip()
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.take().expect("No Gradient found in sparse linear layer");
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

impl LayerInterface for SparseLinearLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        SparseLinearLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        SparseLinearLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        SparseLinearLayerRm::update_parameters(self)
    }
}
