//! Logic shared by `ClusteringLinearLayer` (Vec) and `SparseLinearLayerRm` (row-major):
//! k-means candidate selection, top-k output computation, sparse gradients and the sparse AdamW update.
//!
//! The weights are an embedding-like matrix N x D (N = vocab size). For each input row only the
//! candidates of the nearest clusters are scored and the k best logits are kept.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::neural_networks::{
    network_components::gradient_struct::Gradient,
    optimization::k_means_clustering::{kmeans, query_candidates},
    utils::{
        adam_w::{calculate_adam_w_bias_f32_sparse, calculate_adam_w_f32_sparse},
        dtype::{r, Real, C, ZERO},
        matrix::{average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, normalize_bias, normalize_gradients},
        shared_f32_matrix::SharedF32Matrix,
    },
};

/// Number of candidate tokens requested from the cluster index per row.
pub use crate::neural_networks::network_types::transformer::transformer_network::TOP_K_SIZE as CANDIDATE_POOL_SIZE;

/// Mutable view of the k-means index of a clustering layer.
pub struct ClusterIndex<'a> {
    pub weights: &'a SharedF32Matrix,
    pub previous_weights: &'a mut Vec<Vec<f32>>,
    pub centroids: &'a mut Vec<Vec<f32>>,
    pub assignments: &'a mut Vec<usize>,
    pub cluster_to_tokens: &'a mut Vec<Vec<usize>>,
    pub n_clusters: usize,
    pub last_cluster_update_step: &'a mut usize,
    pub drift_ema: &'a mut f32,
    pub drift_ema_beta: f32,
}

/// Re-runs k-means when the weights drifted enough since the last clustering (smoothed by an EMA and
/// rate-limited by a cooldown) or when the last clustering is too old.
pub fn update_centroids(index: ClusterIndex<'_>, epoch: usize) {
    let tau = tau_schedule(epoch);
    let max_gap = max_update_gap(epoch);
    let min_gap = min_update_gap(epoch);

    let weights_guard = index.weights.read();

    // Compute drift, but smooth it to avoid triggering updates on noisy steps.
    let drift = compute_drift(&weights_guard, index.previous_weights);
    *index.drift_ema = index.drift_ema_beta * *index.drift_ema + (1.0 - index.drift_ema_beta) * drift;

    // Cooldown: don't allow drift-driven updates too frequently.
    let cooldown_ok = epoch.saturating_sub(*index.last_cluster_update_step) >= min_gap;
    let drifted = cooldown_ok && *index.drift_ema > tau;

    // Forced update only after first few epochs
    let time_forced = epoch > 5 && epoch.saturating_sub(*index.last_cluster_update_step) >= max_gap;

    if drifted || time_forced {
        println!(
            "Updating centroids (epoch={}, drifted={}, forced={}, last_update={}, drift={:.6}, drift_ema={:.6}, tau={:.6})",
            epoch, drifted, time_forced, index.last_cluster_update_step, drift, index.drift_ema, tau
        );

        let threshold = 0.001;
        drop(weights_guard);
        let mut w = index.weights.write();
        let (centroids, assignments, cluster_to_tokens) = kmeans(&mut *w, index.n_clusters, 100, threshold);

        *index.centroids = centroids;
        *index.assignments = assignments;
        *index.cluster_to_tokens = cluster_to_tokens;

        // IMPORTANT: avoid deadlock by not acquiring a read-lock while holding the write-lock.
        *index.previous_weights = (*w).clone();
        *index.last_cluster_update_step = epoch;
    }
}

// Drift threshold schedule: adaptive to training stage
pub fn tau_schedule(epoch: usize) -> f32 {
    match epoch {
        0..=4 => 0.005,  // very sensitive early training
        5..=19 => 0.01,  // normal early
        20..=49 => 0.02, // mid training
        _ => 0.05,       // late training, embeddings stabilize
    }
}

// Maximum gap between forced updates: adaptive to training stage
pub fn max_update_gap(epoch: usize) -> usize {
    match epoch {
        0..=4 => 5,    // early: only drift-driven, no forced
        5..=19 => 10,  // mid-early: occasionally force
        20..=49 => 20, // mid-late: sparse forced updates
        _ => 50,       // late: force rarely
    }
}

// Minimum gap between drift-driven updates: prevents k-means from running every epoch early on.
pub fn min_update_gap(epoch: usize) -> usize {
    match epoch {
        0..=4 => 3,    // very early: allow a few steps before re-clustering
        5..=19 => 5,   // early: moderate cooldown
        20..=49 => 10, // mid: longer cooldown
        _ => 25,       // late: rarely re-cluster unless forced
    }
}

/// Relative change ||W - W_prev|| / ||W_prev||.
pub fn compute_drift(w: &Vec<Vec<f32>>, w_prev: &Vec<Vec<f32>>) -> f32 {
    let mut num_sq: f32 = 0.0;
    let mut denom_sq: f32 = 0.0;

    for (row, row_prev) in w.iter().zip(w_prev.iter()) {
        for (&x, &x_prev) in row.iter().zip(row_prev.iter()) {
            let diff = x - x_prev;
            num_sq += diff * diff;
            denom_sq += x_prev * x_prev;
        }
    }

    let eps: f32 = 1e-8;
    num_sq.sqrt() / (denom_sq.sqrt() + eps)
}

// Calculate target offset for a batch
pub fn calculate_target_offset(batch_idx: usize, target_batch: &[Vec<u32>], padding_mask_batch: &[Vec<u32>]) -> Option<usize> {
    if batch_idx >= target_batch.len() || target_batch[batch_idx].is_empty() {
        return None;
    }

    let seq_len_unpadded = padding_mask_batch[batch_idx].iter().filter(|&&x| x != 0).count();
    Some(seq_len_unpadded.saturating_sub(target_batch[batch_idx].len()))
}

// Get target token ID for current row
pub fn get_target_id(row_idx: usize, batch_idx: usize, offset: Option<usize>, target_batch: &[Vec<u32>], padding_mask_batch: &[Vec<u32>]) -> Option<usize> {
    let offset = offset?;

    if row_idx < offset || padding_mask_batch[batch_idx][row_idx] == 0 {
        return None;
    }

    let target_idx = row_idx - offset;
    target_batch[batch_idx].get(target_idx).map(|&id| id as usize)
}

// Dot product of the real input part with one weight row plus bias
pub fn compute_output(input_row: &[C], weights_col: &[f32], bias: f32) -> C {
    let mut sum_re: Real = ZERO;
    for (i, &input_val) in input_row.iter().enumerate() {
        sum_re += input_val.re * r(weights_col[i] as f64);
    }
    C::new(sum_re + r(bias as f64), ZERO)
}

// Ensure target is included in candidate indices
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

// Maintain top-k with protected target
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

/// Read-only parameters needed to score one input row.
pub struct TopKScorer<'a> {
    pub weights: &'a [Vec<f32>],
    pub bias: &'a [f32],
    pub centroids: &'a [Vec<f32>],
    pub cluster_to_tokens: &'a [Vec<usize>],
    pub k: usize,
    pub is_training: bool,
}

impl TopKScorer<'_> {
    /// Scores the cluster candidates of one input row and returns the k highest logits (sorted, padded to k)
    /// and their vocab indices. During training the target is always kept.
    pub fn top_k_row(&self, input_row: &[C], target_id: Option<usize>, input_re_buf: &mut [f32]) -> (Vec<C>, Vec<usize>) {
        let k = self.k;

        // Get candidate indices using k-means clustering
        for (dst, src) in input_re_buf.iter_mut().zip(input_row.iter()) {
            *dst = src.re as f32;
        }
        let mut selected_indices = query_candidates(input_re_buf, self.centroids, self.cluster_to_tokens, CANDIDATE_POOL_SIZE);

        // During TRAINING: Force target inclusion for gradient computation
        // During INFERENCE: No targets available, rely on clustering alone
        if self.is_training {
            ensure_target_in_candidates(&mut selected_indices, target_id, k);
        }

        // Compute outputs and maintain top-k
        let mut top_k = Vec::with_capacity(k + 1);
        let mut target_pos = None;

        for &col_idx in &selected_indices {
            let sum = compute_output(input_row, &self.weights[col_idx], self.bias[col_idx]);
            let is_target = target_id == Some(col_idx);
            update_topk(&mut top_k, sum.re, sum, col_idx, is_target, &mut target_pos, k);
        }

        top_k.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        top_k.truncate(k);
        while top_k.len() < k {
            top_k.push((Real::NEG_INFINITY, C::new(ZERO, ZERO), 0));
        }

        (top_k.iter().map(|(_, val, _)| *val).collect(), top_k.iter().map(|(_, _, idx)| *idx).collect())
    }
}

/// Backward of the top-k projection for one row: accumulates the weight, bias and input gradients
/// of the k selected vocab entries.
pub fn accumulate_sparse_row_gradients(
    input_row: &[C],
    grad_row: &[C],
    idx_row: &[usize],
    weights: &[Vec<f32>],
    weight_gradients: &mut [Vec<C>],
    bias_gradients: &mut [C],
    input_gradient_row: &mut [C],
) {
    for (k_idx, &grad_val) in grad_row.iter().enumerate() {
        let Some(&vocab_idx) = idx_row.get(k_idx) else {
            break;
        };
        if vocab_idx >= weights.len() {
            continue;
        }

        // Weight gradient: grad_weight[vocab_idx] += input_row * grad_val
        for (emb_idx, &input_val) in input_row.iter().enumerate() {
            weight_gradients[vocab_idx][emb_idx] += input_val * grad_val;
        }

        // Bias gradient: grad_bias[vocab_idx] += grad_val
        bias_gradients[vocab_idx] += grad_val;

        // Input gradient: accumulate from all k selected vocab tokens
        for (emb_idx, &weight_val) in weights[vocab_idx].iter().enumerate() {
            input_gradient_row[emb_idx] += C::new(weight_val as Real, 0.0) * grad_val;
        }
    }
}

/// All unique vocab indices selected in the forward pass, sorted.
pub fn collect_used_indices(output_indices_batch: &[Vec<Vec<usize>>]) -> Vec<usize> {
    let indices_set: HashSet<usize> = output_indices_batch.iter().flatten().flatten().copied().collect();
    let mut indices: Vec<usize> = indices_set.into_iter().collect();
    indices.sort_unstable();
    indices
}

/// Mutable parameters of a clustering layer for `update_sparse_parameters`.
pub struct SparseParameters<'a> {
    pub weights: &'a SharedF32Matrix,
    pub bias: &'a mut Vec<f32>,
    pub output_indices: &'a [Vec<Vec<usize>>],
    pub tied_embedding_grad_by_token: Option<&'a Arc<RwLock<HashMap<usize, Vec<C>>>>>,
    pub global_norm: f64,
    pub max_norm: f64,
    pub learning_rate: f64,
    pub time_step: usize,
}

/// Sparse AdamW update: only the vocab rows selected in the forward pass (plus the rows that received
/// gradients from a tied embedding) are updated. Returns the gradient to keep as `previous_gradient`.
pub fn update_sparse_parameters(params: SparseParameters<'_>, mut gradient: Gradient, previous_gradient: Option<&Gradient>) -> Gradient {
    // Collect all unique indices that were actually used during forward/backward pass
    let mut used_indices = collect_used_indices(params.output_indices);

    let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
    let total_valid_tokens = gradient.get_total_valid_tokens();

    // Merge in embedding-side gradients when weights are tied.
    if let Some(acc) = params.tied_embedding_grad_by_token {
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

    clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, params.global_norm, params.max_norm);

    normalize_gradients(&mut weight_gradients);
    normalize_bias(&mut bias_gradients);

    let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights, mut prev_v_weights_hat, mut prev_v_bias_hat) = if let Some(previous_gradient) = previous_gradient {
        (
            previous_gradient.get_prev_m_bias(),
            previous_gradient.get_prev_v_bias(),
            previous_gradient.get_prev_m_weights(),
            previous_gradient.get_prev_v_weights(),
            previous_gradient.get_prev_v_weights_hat(),
            previous_gradient.get_prev_v_bias_hat(),
        )
    } else {
        let (rows, cols) = params.weights.dims();
        // Initialize to zeros on first step
        (
            vec![C::new(ZERO, ZERO); params.bias.len()],
            vec![C::new(ZERO, ZERO); params.bias.len()],
            vec![vec![C::new(ZERO, ZERO); cols]; rows],
            vec![vec![C::new(ZERO, ZERO); cols]; rows],
            vec![vec![C::new(ZERO, ZERO); cols]; rows],
            vec![C::new(ZERO, ZERO); params.bias.len()],
        )
    };

    // Use sparse AdamW optimizers - only update indices that were actually used
    calculate_adam_w_bias_f32_sparse(params.bias, &bias_gradients, &mut prev_m_bias, &mut prev_v_bias, &mut prev_v_bias_hat, params.learning_rate, params.time_step, &used_indices);
    let mut w = params.weights.write();
    calculate_adam_w_f32_sparse(&mut *w, &weight_gradients, &mut prev_m_weights, &mut prev_v_weights, &mut prev_v_weights_hat, params.learning_rate, params.time_step, &used_indices);

    gradient.set_prev_m_bias(prev_m_bias);
    gradient.set_prev_v_bias(prev_v_bias);
    gradient.set_prev_m_weights(prev_m_weights);
    gradient.set_prev_v_weights(prev_v_weights);
    gradient.set_prev_v_weights_hat(prev_v_weights_hat);
    gradient.set_prev_v_bias_hat(prev_v_bias_hat);
    gradient.set_gradient_weights(weight_gradients);
    gradient.set_gradient_bias(bias_gradients);
    gradient
}
