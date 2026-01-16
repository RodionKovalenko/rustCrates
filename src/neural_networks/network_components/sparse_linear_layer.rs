use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::layer::LayerEnum,
    network_types::{transformer::transformer_updater::VERBOSE, wavelet_discrete_layer::DiscreteWaveletLayer},
    optimization::k_means_clustering::{kmeans, query_candidates},
    utils::{
        adam_w::{calculate_adam_w_bias_f32_sparse, calculate_adam_w_f32_sparse},
        matrix::{average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, RowMajorMatrix},
        weights_initializer::initialize_weights_f32,
    },
};

use super::{
    gradient_struct::Gradient,
    layer_input_struct::{InputRepresentation, LayerInput},
    layer_output_struct::LayerOutput,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLinearLayer {
    pub weights: Vec<Vec<f32>>,
    pub previous_weights: Vec<Vec<f32>>,
    pub learning_rate: f64,
    pub bias: Vec<f32>,
    pub smoothing: f64,
    pub ema: f64,
    pub discrete_wavelet_layer: Option<DiscreteWaveletLayer>,
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
    pub gradients: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub gradients_bias: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub output_indices: Option<Vec<Vec<Vec<usize>>>>,
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
            discrete_wavelet_layer: None,
            norm_layer: None,
            gradients_bias: vec![],
            input_batch: None,
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
            output_indices: None,
        }
    }

    pub fn update_centroids(&mut self, epoch: usize) {
        let tau = self.tau_schedule(epoch);
        let max_gap = self.max_update_gap(epoch);

        // Compute drift
        let drifted = self.needs_cluster_update(&self.weights, &self.previous_weights, tau);

        // Forced update only after first few epochs
        let time_forced = epoch > 5 && epoch.saturating_sub(self.last_cluster_update_step) >= max_gap;

        if drifted || time_forced {
            println!(
                "Updating centroids (epoch={}, drift={}, forced={}, last_update={})",
                epoch, drifted, time_forced, self.last_cluster_update_step
            );

            let threshold = 0.001;
            let (centroids, assignments, cluster_to_tokens) = kmeans(&mut self.weights, self.n_clusters, 100, threshold);

            self.centroids = centroids;
            self.assignments = assignments;
            self.cluster_to_tokens = cluster_to_tokens;

            self.previous_weights = self.weights.clone();
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

        let input_batch_rm_ref = input.get_input_batch_rm_ref();
        let use_rm = input_batch_rm_ref.is_some_and(|rm| !rm.is_empty());
        // In rm_strict mode, never call get_input_batch() if RM activations exist.
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = if use_rm { vec![] } else { input.get_input_batch() };

        // Store whichever representation was provided so backward can avoid reshaping.
        if !input_batch.is_empty() {
            self.input_batch = Some(input_batch.clone());
        } else {
            self.input_batch = None;
        }
        if let Some(rm) = input_batch_rm_ref {
            if !rm.is_empty() {
                self.input_batch_rm = Some(rm.to_vec());
            } else {
                self.input_batch_rm = None;
            }
        } else {
            self.input_batch_rm = None;
        }

        let start = std::time::Instant::now();

        let mut layer_output = LayerOutput::new_default();

        match input.input_representation() {
            InputRepresentation::RowMajor => {
                let input_batch_rm = input_batch_rm_ref.expect("rm");
                let (output_batch_rm, output_indices) = self.mutliply_hightest_k_per_row_rm(input_batch_rm, input);
                if VERBOSE {
                    println!("SparseLinear RM matmul time for batch size {}: {}", self.batch_size, start.elapsed().as_secs_f64());
                }

                // Preserve legacy output only when legacy input is used.
                let need_legacy_output = !input_batch.is_empty();
                if need_legacy_output {
                    let output_batch: Vec<Vec<Vec<Complex<f64>>>> = output_batch_rm.iter().map(|m| m.to_rows()).collect();
                    layer_output.set_output_batch(output_batch);
                }

                layer_output.set_output_batch_rm(output_batch_rm);
                layer_output.set_output_indices(output_indices);
            }
            InputRepresentation::Vec => {
                let (output_batch, output_indices) = self.mutliply_hightest_k_per_row(&input_batch, input);
                if VERBOSE {
                    println!("SparseLinear matmul time for batch size {}: {}", self.batch_size, start.elapsed().as_secs_f64());
                }

                // This output is rectangular (seq_len x k), so provide RM output too.
                let output_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = output_batch.iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();

                layer_output.set_output_batch(output_batch);
                layer_output.set_output_batch_rm(output_batch_rm);
                layer_output.set_output_indices(output_indices);
            }
            InputRepresentation::Empty => {
                layer_output.set_output_batch(vec![]);
                layer_output.set_output_batch_rm(vec![]);
                layer_output.set_output_indices(vec![]);
            }
        }

        self.output_indices = Some(layer_output.get_output_indices());
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_vec = self.input_batch.as_ref();
        let input_batch_rm = self.input_batch_rm.as_ref();
        if input_batch_vec.is_none() && input_batch_rm.is_none() {
            let total_valid_tokens = previous_gradient.get_total_valid_tokens();
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let previous_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();
        let previous_gradient_rm_ref = previous_gradient.get_gradient_input_batch_rm_ref();

        let batch_len = if let Some(b) = input_batch_vec {
            b.len()
        } else {
            input_batch_rm.expect("rm batch").len()
        };

        if batch_len == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let output_indices_batch: &Vec<Vec<Vec<usize>>> = self
            .output_indices
            .as_ref()
            .expect("Output indices missing in sparse linear layer backward pass");

        // Determine (seq_len, embedding_d) from the first available non-empty input.
        let first_input_rm: Option<RowMajorMatrix<Complex<f64>>> = if let Some(rm_batch) = input_batch_rm {
            rm_batch.get(0).cloned()
        } else {
            input_batch_vec
                .and_then(|b| b.get(0))
                .map(|rows| RowMajorMatrix::from_rows(rows))
        };

        let (seq_len, embedding_d) = if let Some(rm) = &first_input_rm {
            (rm.rows, rm.cols)
        } else {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        };

        let k = if let Some(grm) = previous_gradient_rm_ref {
            grm.get(0).map(|m| m.cols).unwrap_or(0)
        } else {
            previous_gradient_input_batch
                .get(0)
                .map(|rows| RowMajorMatrix::from_rows(rows).cols)
                .unwrap_or(0)
        };

        if k == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let mut weight_gradients: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()]; batch_len];
        let mut bias_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.bias.len()]; batch_len];
        let mut gradient_input_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = vec![
            RowMajorMatrix::from_data(seq_len, embedding_d, vec![Complex::new(0.0, 0.0); seq_len * embedding_d]);
            batch_len
        ];

        for batch_idx in 0..batch_len {
            let input_rm: RowMajorMatrix<Complex<f64>> = if let Some(rm_batch) = input_batch_rm {
                rm_batch[batch_idx].clone()
            } else {
                let input_sample = &input_batch_vec.expect("vec batch")[batch_idx];
                RowMajorMatrix::from_rows(input_sample)
            };

            let grad_rm: RowMajorMatrix<Complex<f64>> = if let Some(grads_rm) = previous_gradient_rm_ref {
                grads_rm[batch_idx].clone()
            } else {
                RowMajorMatrix::from_rows(&previous_gradient_input_batch[batch_idx])
            };

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
                    if vocab_idx >= self.weights.len() {
                        continue;
                    }

                    for (emb_idx, &input_val) in input_row.iter().enumerate() {
                        weight_gradients[batch_idx][vocab_idx][emb_idx] += input_val * grad_val;
                    }
                    bias_gradients[batch_idx][vocab_idx] += grad_val;

                    // Input gradient accumulation
                    let out_in_row = gradient_input_batch_rm[batch_idx].row_range(seq_idx);
                    for (emb_idx, &w) in self.weights[vocab_idx].iter().enumerate() {
                        gradient_input_batch_rm[batch_idx].data[out_in_row.start + emb_idx] += Complex::new(w as f64, 0.0) * grad_val;
                    }
                }
            }
        }

        let mut gradient = Gradient::new_default();
        let legacy_mode = self.input_batch.is_some();
        if legacy_mode {
            let gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = gradient_input_batch_rm.iter().map(|m| m.to_rows()).collect();
            gradient.set_gradient_input_batch(gradient_input_batch);
        }
        gradient.set_gradient_input_batch_rm(gradient_input_batch_rm);
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
    fn compute_output(input_row: &[Complex<f64>], weights_col: &[f32], bias: f32) -> Complex<f64> {
        let mut sum = Complex::new(0.0, 0.0);
        for (i, &input_val) in input_row.iter().enumerate() {
            sum += input_val.re * weights_col[i] as f64;
        }
        sum + bias as f64
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
    fn update_topk(top_k: &mut Vec<(f64, Complex<f64>, usize)>, real_value: f64, sum: Complex<f64>, col_idx: usize, is_target: bool, target_pos: &mut Option<usize>, k: usize) {
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
    pub fn mutliply_hightest_k_per_row(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, layer_input: &LayerInput) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<Vec<usize>>>) {
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
                        let sum = Self::compute_output(input_row, &self.weights[col_idx], self.bias[col_idx]);
                        let is_target = target_id == Some(col_idx);

                        Self::update_topk(&mut top_k, sum.re, sum, col_idx, is_target, &mut target_pos, k);
                    }

                    let values: Vec<Complex<f64>> = top_k.iter().map(|(_, val, _)| *val).collect();
                    let indices: Vec<usize> = top_k.iter().map(|(_, _, idx)| *idx).collect();

                    sample_values.push(values);
                    sample_indices.push(indices);
                }

                (sample_values, sample_indices)
            })
            .collect();

        results.into_iter().unzip()
    }

    // RM variant: returns a dense (seq_len x k) row-major matrix per batch.
    pub fn mutliply_hightest_k_per_row_rm(&mut self, input_batch_rm: &[RowMajorMatrix<Complex<f64>>], layer_input: &LayerInput) -> (Vec<RowMajorMatrix<Complex<f64>>>, Vec<Vec<Vec<usize>>>) {
        let target_batch = &layer_input.get_target_batch_ids();
        let k = layer_input.get_top_k_size();
        let padding_mask_batch = layer_input.get_padding_mask_batch();
        let is_training = !target_batch.is_empty();

        let results: Vec<_> = input_batch_rm
            .par_iter()
            .enumerate()
            .map(|(batch_idx, input_sample)| {
                let offset = Self::calculate_target_offset(batch_idx, target_batch, &padding_mask_batch);
                let seq_len = input_sample.rows;
                let embedding_d = input_sample.cols;

                let mut data: Vec<Complex<f64>> = Vec::with_capacity(seq_len * k);
                let mut sample_indices: Vec<Vec<usize>> = Vec::with_capacity(seq_len);

                let mut input_re_buf: Vec<f32> = vec![0.0; embedding_d];

                for row_idx in 0..seq_len {
                    let row = input_sample.row_range(row_idx);
                    let input_row = &input_sample.data[row.start..row.start + embedding_d];
                    let target_id = Self::get_target_id(row_idx, batch_idx, offset, target_batch, &padding_mask_batch);

                    for (dst, src) in input_re_buf.iter_mut().zip(input_row.iter()) {
                        *dst = src.re as f32;
                    }

                    let mut selected_indices = query_candidates(
                        &mut input_re_buf,
                        &self.centroids,
                        &self.cluster_to_tokens,
                        TOP_K_SELECTION,
                    );

                    if is_training {
                        Self::ensure_target_in_candidates(&mut selected_indices, target_id, k);
                    }

                    let mut top_k = Vec::with_capacity(k + 1);
                    let mut target_pos = None;
                    for &col_idx in &selected_indices {
                        let sum = Self::compute_output(input_row, &self.weights[col_idx], self.bias[col_idx]);
                        let is_target = target_id == Some(col_idx);
                        Self::update_topk(&mut top_k, sum.re, sum, col_idx, is_target, &mut target_pos, k);
                    }

                    // Pad/truncate to exactly k
                    top_k.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    if top_k.len() > k {
                        top_k.truncate(k);
                    }
                    while top_k.len() < k {
                        top_k.push((f64::NEG_INFINITY, Complex::new(0.0, 0.0), 0));
                    }

                    for p in 0..k {
                        data.push(top_k[p].1);
                    }
                    sample_indices.push(top_k.iter().map(|(_, _, idx)| *idx).collect());
                }

                (RowMajorMatrix::from_data(seq_len, k, data), sample_indices)
            })
            .collect();

        results.into_iter().unzip()
    }

    pub fn update_parameters(&mut self) {
        // Collect all unique indices that were actually used during forward/backward pass
        let used_indices = self.collect_used_indices();

        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
        let total_valid_tokens = gradient.get_total_valid_tokens();

        weight_gradients = average_matrix_by_scalar(&weight_gradients, total_valid_tokens as f64);
        bias_gradients = average_vector_by_scalar(&bias_gradients, total_valid_tokens as f64);

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
            // Initialize to zeros on first step
            (
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()],
                vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()],
                vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()],
                vec![Complex::new(0.0, 0.0); self.bias.len()],
            )
        };

        // Use sparse AdamW optimizers - only update indices that were actually used
        calculate_adam_w_bias_f32_sparse(
            &mut self.bias,
            &gradient.get_gradient_bias(),
            &mut prev_m_bias,
            &mut prev_v_bias,
            &mut prev_v_bias_hat,
            learning_rate,
            time_step,
            &used_indices,
        );
        calculate_adam_w_f32_sparse(
            &mut self.weights,
            &gradient.get_gradient_weights(),
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

    pub fn group_gradient_batch(&self, weight_gradients_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Complex<f64>>> {
        let mut weight_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); weight_gradients_batch[0][0].len()]; weight_gradients_batch[0].len()];

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
