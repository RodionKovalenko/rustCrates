use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput}, network_layers::layer::LayerEnum, network_types::transformer::transformer_updater::VERBOSE, optimization::k_means_clustering::{kmeans, query_candidates}, utils::{
        adam_w::{calculate_adam_w_bias_f32_sparse, calculate_adam_w_f32_sparse},
        dtype::{C, Real, ZERO, r},
        matrix::{RowMajorMatrix, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d},
        shared_f32_matrix::SharedF32Matrix,
        weights_initializer::initialize_weights_f32,
    }
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLinearLayer {
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

pub const TOP_K_SELECTION: usize = 300;

impl SparseLinearLayer {
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
            output_indices: None,
            tied_embedding_grad_by_token: None,
        }
    }

    /// Enable weight tying with an embedding layer.
    ///
    /// The returned accumulator should be stored by the embedding layer, which will
    /// accumulate per-token gradients into it during backward.
    pub fn enable_weight_tying(&mut self) -> Arc<RwLock<HashMap<usize, Vec<C>>>> {
        let acc = Arc::new(RwLock::new(HashMap::new()));
        self.tied_embedding_grad_by_token = Some(acc.clone());
        acc
    }

    pub fn update_centroids(&mut self, epoch: usize) {
        let tau = self.tau_schedule(epoch);
        let max_gap = self.max_update_gap(epoch);

        let weights_guard = self.weights.read();

        // Compute drift
        let drifted = self.needs_cluster_update(&weights_guard, &self.previous_weights, tau);

        // Forced update only after first few epochs
        let time_forced = epoch > 5 && epoch.saturating_sub(self.last_cluster_update_step) >= max_gap;

        if drifted || time_forced {
            println!(
                "Updating centroids (epoch={}, drift={}, forced={}, last_update={})",
                epoch, drifted, time_forced, self.last_cluster_update_step
            );

            let threshold = 0.001;
            drop(weights_guard);
            let mut w = self.weights.write();
            let (centroids, assignments, cluster_to_tokens) = kmeans(&mut *w, self.n_clusters, 100, threshold);

            self.centroids = centroids;
            self.assignments = assignments;
            self.cluster_to_tokens = cluster_to_tokens;

            self.previous_weights = self.weights.read().clone();
            self.last_cluster_update_step = epoch;
        }
    }

    // Drift threshold schedule: adaptive to training stage
    fn tau_schedule(&self, epoch: usize) -> f32 {
        match epoch {
            0..=4 => 0.005,  // very sensitive early training
            5..=19 => 0.01,  // normal early
            20..=49 => 0.02, // mid training
            _ => 0.05,       // late training, embeddings stabilize
        }
    }

    // Maximum gap between forced updates: adaptive to training stage
    fn max_update_gap(&self, epoch: usize) -> usize {
        match epoch {
            0..=4 => 5,    // early: only drift-driven, no forced
            5..=19 => 10,  // mid-early: occasionally force
            20..=49 => 20, // mid-late: sparse forced updates
            _ => 50,       // late: force rarely
        }
    }

    // Compute drift as before
    pub fn needs_cluster_update(&self, w: &Vec<Vec<f32>>, w_prev: &Vec<Vec<f32>>, tau: f32) -> bool {
        let mut num_sq = 0.0;
        let mut denom_sq = 0.0;

        for (row, row_prev) in w.iter().zip(w_prev.iter()) {
            for (&x, &x_prev) in row.iter().zip(row_prev.iter()) {
                let diff = x - x_prev;
                num_sq += diff * diff;
                denom_sq += x_prev * x_prev;
            }
        }

        let eps = 1e-8;
        let drift = num_sq.sqrt() / (denom_sq.sqrt() + eps);

        drift > tau
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        if !input.has_non_empty_input_batch() {
            if input.has_non_empty_input_batch_rm() {
                panic!("SparseLinearLayer (Vec) received RM-only input; use SparseLinearLayerRm");
            }

            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch(vec![]);
            layer_output.set_output_indices(vec![]);
            self.input_batch = None;
            self.output_indices = Some(vec![]);
            return layer_output;
        }

        let input_batch: Vec<Vec<Vec<C>>> = input.get_input_batch();
        self.input_batch = Some(input_batch.clone());

        let start = std::time::Instant::now();

        let mut layer_output = LayerOutput::new_default();

        let weights_guard = self.weights.read();
        let (output_batch, output_indices) = self.mutliply_hightest_k_per_row(&input_batch, input, &*weights_guard);
        if VERBOSE {
            println!("SparseLinear matmul time for batch size {}: {}", self.batch_size, start.elapsed().as_secs_f64());
        }

        layer_output.set_output_batch(output_batch);
        layer_output.set_output_indices(output_indices);

        self.output_indices = Some(layer_output.get_output_indices());
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_vec = self.input_batch.as_ref();
        if input_batch_vec.is_none() {
            let total_valid_tokens = previous_gradient.get_total_valid_tokens();
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let previous_gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient.get_gradient_input_batch();

        let batch_len = input_batch_vec.expect("vec batch").len();

        if batch_len == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let output_indices_batch: &Vec<Vec<Vec<usize>>> = self.output_indices.as_ref().expect("Output indices missing in sparse linear layer backward pass");

        let first_input_rows = input_batch_vec.expect("vec batch").get(0).cloned().unwrap_or_default();
        let first_input_rm: RowMajorMatrix<C> = RowMajorMatrix::from_rows(&first_input_rows);
        let (seq_len, embedding_d) = (first_input_rm.rows, first_input_rm.cols);

        let k = previous_gradient_input_batch
            .get(0)
            .map(|rows| RowMajorMatrix::from_rows(rows).cols)
            .unwrap_or(0);

        if k == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let weights_guard = self.weights.read();

        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); weights_guard[0].len()]; weights_guard.len()]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];
        let mut gradient_input_batch_rm: Vec<RowMajorMatrix<C>> = vec![RowMajorMatrix::from_data(seq_len, embedding_d, vec![C::new(ZERO, ZERO); seq_len * embedding_d]); batch_len];

        for batch_idx in 0..batch_len {
            let input_sample = &input_batch_vec.expect("vec batch")[batch_idx];
            let input_rm: RowMajorMatrix<C> = RowMajorMatrix::from_rows(input_sample);

            let grad_rm: RowMajorMatrix<C> = RowMajorMatrix::from_rows(&previous_gradient_input_batch[batch_idx]);

            assert_eq!(input_rm.rows, seq_len);
            assert_eq!(input_rm.cols, embedding_d);
            assert_eq!(grad_rm.rows, seq_len);
            assert_eq!(grad_rm.cols, k);
            assert_eq!(output_indices_batch[batch_idx].len(), seq_len);

            for seq_idx in 0..seq_len {
                let in_row = input_rm.row_range(seq_idx);
                let input_row = &input_rm.data[in_row.start..in_row.start + embedding_d];

                let g_row = grad_rm.row_range(seq_idx);
                let grad_row = &grad_rm.data[g_row.start..g_row.start + k];
                let idx_row = &output_indices_batch[batch_idx][seq_idx];

                for k_idx in 0..grad_row.len().min(idx_row.len()).min(k) {
                    let grad_val = grad_row[k_idx];
                    let vocab_idx = idx_row[k_idx];
                    if vocab_idx >= weights_guard.len() {
                        continue;
                    }

                    for (emb_idx, &input_val) in input_row.iter().enumerate() {
                        weight_gradients[batch_idx][vocab_idx][emb_idx] += input_val * grad_val;
                    }
                    bias_gradients[batch_idx][vocab_idx] += grad_val;

                    // Input gradient accumulation
                    let out_in_row = gradient_input_batch_rm[batch_idx].row_range(seq_idx);
                    for (emb_idx, &w) in weights_guard[vocab_idx].iter().enumerate() {
                        gradient_input_batch_rm[batch_idx].data[out_in_row.start + emb_idx] += C::new(r(w as f64), ZERO) * grad_val;
                    }
                }
            }
        }

        let mut gradient = Gradient::new_default();
        let gradient_input_batch: Vec<Vec<Vec<C>>> = gradient_input_batch_rm.iter().map(|m| m.to_rows()).collect();
        gradient.set_gradient_input_batch(gradient_input_batch);
        gradient.set_gradient_input_batch_rm(vec![]);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());
        gradient
    }

    // Helper: Calculate target offset for a batch
    fn calculate_target_offset(batch_idx: usize, target_batch: &[Vec<u32>], padding_mask_batch: &[Vec<u32>]) -> Option<usize> {
        if batch_idx >= target_batch.len() || target_batch[batch_idx].is_empty() {
            return None;
        }

        let seq_len_unpadded = padding_mask_batch[batch_idx].iter().filter(|&&x| x != 0).count();
        Some(seq_len_unpadded.saturating_sub(target_batch[batch_idx].len()))
    }

    // Helper: Get target token ID for current row
    fn get_target_id(row_idx: usize, batch_idx: usize, offset: Option<usize>, target_batch: &[Vec<u32>], padding_mask_batch: &[Vec<u32>]) -> Option<usize> {
        let offset = offset?;

        if row_idx < offset || padding_mask_batch[batch_idx][row_idx] == 0 {
            return None;
        }

        let target_idx = row_idx - offset;
        target_batch[batch_idx].get(target_idx).map(|&id| id as usize)
    }

    // Helper: Compute dot product with weights and bias
    fn compute_output(input_row: &[C], weights_col: &[f32], bias: f32) -> C {
        let mut sum_re: Real = ZERO;
        for (i, &input_val) in input_row.iter().enumerate() {
            sum_re += input_val.re * r(weights_col[i] as f64);
        }
        C::new(sum_re + r(bias as f64), ZERO)
    }

    // Helper: Ensure target is included in candidate indices
    pub fn ensure_target_in_candidates(selected_indices: &mut Vec<usize>, target_id: Option<usize>, k: usize) {
        if let Some(tid) = target_id {
            if !selected_indices.contains(&tid) {
                if selected_indices.len() < k {
                    selected_indices.push(tid);
                } else if !selected_indices.is_empty() {
                    *selected_indices.last_mut().unwrap() = tid;
                }
            }
        }
    }

    // Helper: Maintain top-k heap with protected target
    fn update_topk(top_k: &mut Vec<(Real, C, usize)>, real_value: Real, sum: C, col_idx: usize, is_target: bool, target_pos: &mut Option<usize>, k: usize) {
        if top_k.len() < k {
            // Still filling up to k elements
            top_k.push((real_value, sum, col_idx));
            if is_target {
                *target_pos = Some(top_k.len() - 1);
            }
        } else if is_target && target_pos.is_none() {
            // Target must be included; replace minimum non-target
            let min_idx = (0..top_k.len())
                .filter(|&i| Some(i) != *target_pos)
                .min_by(|&a, &b| top_k[a].0.partial_cmp(&top_k[b].0).unwrap())
                .unwrap();

            top_k[min_idx] = (real_value, sum, col_idx);
            *target_pos = Some(min_idx);
        } else {
            // Replace minimum non-target if new value is higher
            if let Some(min_idx) = (0..top_k.len()).filter(|&i| Some(i) != *target_pos).min_by(|&a, &b| top_k[a].0.partial_cmp(&top_k[b].0).unwrap()) {
                if real_value > top_k[min_idx].0 {
                    top_k[min_idx] = (real_value, sum, col_idx);
                }
            }
        }
    }

    // Multiply and select highest k per row, return k highest values per row and original indices
    pub fn mutliply_hightest_k_per_row(&self, input_batch: &Vec<Vec<Vec<C>>>, layer_input: &LayerInput, weights: &Vec<Vec<f32>>) -> (Vec<Vec<Vec<C>>>, Vec<Vec<Vec<usize>>>) {
        let target_batch = &layer_input.get_target_batch_ids();
        let k = layer_input.get_top_k_size();
        let padding_mask_batch = layer_input.get_padding_mask_batch();
        let is_training = !target_batch.is_empty(); // Training mode if targets are provided

        let results: Vec<_> = input_batch
            .par_iter()
            .enumerate()
            .map(|(batch_idx, input_sample)| {
                let offset = Self::calculate_target_offset(batch_idx, target_batch, &padding_mask_batch);
                let mut sample_values = Vec::with_capacity(input_sample.len());
                let mut sample_indices = Vec::with_capacity(input_sample.len());

                let embedding_d = input_sample.first().map(|r| r.len()).unwrap_or(0);
                let mut input_re_buf: Vec<f32> = vec![0.0; embedding_d];

                for (row_idx, input_row) in input_sample.iter().enumerate() {
                    let target_id = Self::get_target_id(row_idx, batch_idx, offset, target_batch, &padding_mask_batch);

                    // Get candidate indices using k-means clustering
                    for (dst, src) in input_re_buf.iter_mut().zip(input_row.iter()) {
                        *dst = src.re as f32;
                    }

                    let mut selected_indices = query_candidates(
                        &mut input_re_buf,
                        &self.centroids,
                        &self.cluster_to_tokens,
                        TOP_K_SELECTION, // top 8 clusters - high coverage to naturally include targets
                    );

                    // During TRAINING: Force target inclusion for gradient computation
                    // During INFERENCE: No targets available, rely on clustering alone
                    // With 32 clusters, target SHOULD be naturally included most of the time.
                    // If model learns good embeddings, clustering will find the right tokens.
                    if is_training {
                        Self::ensure_target_in_candidates(&mut selected_indices, target_id, k);
                    }

                    // Compute outputs and maintain top-k
                    let mut top_k = Vec::with_capacity(k + 1);
                    let mut target_pos = None;

                    for &col_idx in &selected_indices {
                        let sum = Self::compute_output(input_row, &weights[col_idx], self.bias[col_idx]);
                        let is_target = target_id == Some(col_idx);

                        Self::update_topk(&mut top_k, sum.re, sum, col_idx, is_target, &mut target_pos, k);
                    }

                    let values: Vec<C> = top_k.iter().map(|(_, val, _)| *val).collect();
                    let indices: Vec<usize> = top_k.iter().map(|(_, _, idx)| *idx).collect();

                    sample_values.push(values);
                    sample_indices.push(indices);
                }

                (sample_values, sample_indices)
            })
            .collect();

        results.into_iter().unzip()
    }

    pub fn update_parameters(&mut self) {
        // Collect all unique indices that were actually used during forward/backward pass
        let mut used_indices = self.collect_used_indices();

        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
        let total_valid_tokens = gradient.get_total_valid_tokens();

        // Merge in embedding-side gradients when weights are tied.
        if let Some(acc) = &self.tied_embedding_grad_by_token {
            let mut acc_lock = acc.write().expect("tied grad accumulator poisoned");
            for (token_idx, grad_vec) in acc_lock.drain() {
                if token_idx >= weight_gradients.len() {
                    continue;
                }
                used_indices.push(token_idx);
                let row = &mut weight_gradients[token_idx];
                let n = row.len().min(grad_vec.len());
                for j in 0..n {
                    row[j] += grad_vec[j];
                }
            }
        }

        used_indices.sort_unstable();
        used_indices.dedup();

        let total_valid_tokens: Real = r(total_valid_tokens.max(1) as f64);
        weight_gradients = average_matrix_by_scalar(&weight_gradients, total_valid_tokens);
        bias_gradients = average_vector_by_scalar(&bias_gradients, total_valid_tokens);

        clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights, mut prev_v_weights_hat, mut prev_v_bias_hat) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_bias(),
                previous_gradient.get_prev_v_bias(),
                previous_gradient.get_prev_m_weights(),
                previous_gradient.get_prev_v_weights(),
                previous_gradient.get_prev_v_weights_hat(),
                previous_gradient.get_prev_v_bias_hat(),
            )
        } else {
            let (rows, cols) = self.weights.dims();
            // Initialize to zeros on first step
            (
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![vec![C::new(ZERO, ZERO); cols]; rows],
                vec![vec![C::new(ZERO, ZERO); cols]; rows],
                vec![vec![C::new(ZERO, ZERO); cols]; rows],
                vec![C::new(ZERO, ZERO); self.bias.len()],
            )
        };

        // Use sparse AdamW optimizers - only update indices that were actually used
        calculate_adam_w_bias_f32_sparse(
            &mut self.bias,
            &bias_gradients,
            &mut prev_m_bias,
            &mut prev_v_bias,
            &mut prev_v_bias_hat,
            learning_rate,
            time_step,
            &used_indices,
        );
        let mut w = self.weights.write();
        calculate_adam_w_f32_sparse(
            &mut *w,
            &weight_gradients,
            &mut prev_m_weights,
            &mut prev_v_weights,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
            &used_indices,
        );

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);
        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }

    pub fn group_gradient_batch(&self, weight_gradients_batch: &Vec<Vec<Vec<C>>>) -> Vec<Vec<C>> {
        let mut weight_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); weight_gradients_batch[0][0].len()]; weight_gradients_batch[0].len()];

        for weight_gradient_batch in weight_gradients_batch {
            for (row, w_gradient) in weight_gradient_batch.iter().enumerate() {
                for (col, gradient_value) in w_gradient.iter().enumerate() {
                    weight_gradients[row][col] += gradient_value;
                }
            }
        }

        weight_gradients
    }

    /// Collect all unique vocab indices that were used during the forward pass
    /// This is used for sparse AdamW updates - we only need to update these indices
    fn collect_used_indices(&self) -> Vec<usize> {
        let output_indices_batch = self.output_indices.as_ref().expect("Output indices missing");

        let mut indices_set = std::collections::HashSet::new();

        // Iterate through all batches, sequences, and k-selected indices
        for batch_indices in output_indices_batch {
            for seq_indices in batch_indices {
                for &idx in seq_indices {
                    indices_set.insert(idx);
                }
            }
        }

        // Convert to sorted vector for deterministic behavior
        let mut indices: Vec<usize> = indices_set.into_iter().collect();
        indices.sort_unstable();

        indices
    }
}
