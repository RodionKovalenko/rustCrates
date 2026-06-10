use std::cmp::Reverse;
use std::collections::HashSet;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{
        gradient_struct::Gradient,
        input::{DataTrait, Dataset, load_data_xquad_de_as_dataset, load_data_xquad_en_as_dataset, load_data_xquad_ru_as_dataset},
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
    },
    network_layers::layer::LayerEnum,
    utils::{
        activation::softmax_backward_real_with_gradient,
        adam_w::{calculate_adam_w_bias_f32_sparse, calculate_adam_w_f32_sparse},
        dtype::{C, Real, ZERO, r},
        matrix::{average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, normalize_bias, normalize_gradients},
        tokenizer::tokenize,
        weights_initializer::initialize_weights_f32,
    },
};

const DEFAULT_HEAD_SIZE: usize = 2048;
const DEFAULT_TAIL_CLUSTER_COUNT: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLinearLayer {
    pub cluster_word_weights: Vec<Vec<Vec<f32>>>,
    pub cluster_word_bias: Vec<Vec<f32>>,
    pub frequency_clusters: Vec<Vec<usize>>,
    pub token_frequencies: Vec<u64>,
    pub token_rank_by_id: Vec<usize>,
    pub token_cluster_by_id: Vec<usize>,
    pub token_index_in_cluster: Vec<usize>,
    pub head_size: usize,
    pub tail_cluster_count: usize,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub norm_layer: Option<LayerEnum>,
    pub global_norm: f64,
    pub max_norm: f64,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_indices_batch: Vec<Vec<Vec<usize>>>,
    /// Tail cluster routing weights: [tail_cluster_count, hidden_size]
    pub weights: Vec<Vec<f32>>,
    pub previous_weights: Vec<Vec<f32>>,
    /// Tail cluster routing biases: [tail_cluster_count]
    pub bias: Vec<f32>,
    pub gradient: Option<Gradient>,
    pub previous_gradient: Option<Gradient>,
    pub time_step: usize,
    pub batch_size: usize,
    pub selected_tail_cluster_batch: Vec<Vec<Option<usize>>>,
    pub target_batch_ids: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub cross_entropy_loss_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    cluster_word_weight_gradients: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    cluster_word_bias_gradients: Option<Vec<Vec<C>>>,
}

impl AdaptiveLinearLayer {
    pub fn new(learning_rate: f64, hidden_size: usize, vocab_size: usize) -> Self {
        let head_size = DEFAULT_HEAD_SIZE.min(vocab_size.max(1));
        let tail_cluster_count = DEFAULT_TAIL_CLUSTER_COUNT;
        let (token_frequencies, frequency_clusters, token_rank_by_id, token_cluster_by_id) =
            Self::build_frequency_clusters(vocab_size, head_size, tail_cluster_count);

        let mut token_index_in_cluster = vec![0; vocab_size];
        for tokens in &frequency_clusters {
            for (pos, &token_id) in tokens.iter().enumerate() {
                if token_id < vocab_size {
                    token_index_in_cluster[token_id] = pos;
                }
            }
        }

        let num_clusters = frequency_clusters.len();

        let mut cluster_word_weights = Vec::with_capacity(num_clusters);
        let mut cluster_word_bias = Vec::with_capacity(num_clusters);
        for tokens_in_cluster in &frequency_clusters {
            let cluster_size = tokens_in_cluster.len().max(1);
            let mut w = vec![vec![0.0f32; hidden_size]; cluster_size];
            let b = vec![0.0f32; cluster_size];
            initialize_weights_f32(cluster_size, hidden_size, &mut w);
            cluster_word_weights.push(w);
            cluster_word_bias.push(b);
        }

        // Routing weights: one row per tail cluster (clusters 1..num_clusters)
        let actual_tail_count = num_clusters.saturating_sub(1);
        let mut routing_weights = vec![vec![0.0f32; hidden_size]; actual_tail_count];
        let routing_bias = vec![0.0f32; actual_tail_count];
        if actual_tail_count > 0 {
            initialize_weights_f32(actual_tail_count, hidden_size, &mut routing_weights);
        }
        let previous_weights = routing_weights.clone();

        Self {
            cluster_word_weights,
            cluster_word_bias,
            frequency_clusters,
            token_frequencies,
            token_rank_by_id,
            token_cluster_by_id,
            token_index_in_cluster,
            head_size,
            tail_cluster_count: actual_tail_count,
            learning_rate,
            smoothing: 0.99,
            ema: 0.0,
            norm_layer: None,
            global_norm: 0.0,
            max_norm: 0.0,
            input_batch: None,
            output_indices_batch: vec![],
            weights: routing_weights,
            previous_weights,
            bias: routing_bias,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            selected_tail_cluster_batch: vec![],
            target_batch_ids: None,
            cross_entropy_loss_batch: None,
            cluster_word_weight_gradients: None,
            cluster_word_bias_gradients: None,
        }
    }

    /// Forward pass.
    ///
    /// During training (`target_batch_ids` is set), also computes cross-entropy loss
    /// and all gradients so that the backward pass can start from the stored `self.gradient`.
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = input.get_input_batch();
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());

        let target_tokens = input.get_target_batch_ids();
        let padding_mask_batch = input.get_padding_mask_batch();
        let top_k = input.get_top_k_size().max(1);
        let total_valid_tokens = input.get_total_valid_tokens();
        let is_training = !target_tokens.is_empty() && input.get_calculate_gradient();

        self.target_batch_ids = Some(target_tokens.clone());
        self.gradient = None;
        self.cross_entropy_loss_batch = None;

        // ── Stage 1: compute sparse logits for every batch item ──────────────────

        let cluster_word_weights = &self.cluster_word_weights;
        let cluster_word_bias = &self.cluster_word_bias;
        let routing_weights = &self.weights;
        let routing_bias = &self.bias;
        let token_cluster_by_id = &self.token_cluster_by_id;
        let token_index_in_cluster = &self.token_index_in_cluster;
        let frequency_clusters = &self.frequency_clusters;

        let sparse_results: Vec<(Vec<Vec<C>>, Vec<Vec<usize>>, Vec<Option<usize>>)> = input_batch
            .par_iter()
            .enumerate()
            .map(|(batch_idx, input_seq)| {
                let offset = Self::calculate_target_offset(batch_idx, &target_tokens, &padding_mask_batch);
                let mut sample_values: Vec<Vec<C>> = Vec::with_capacity(input_seq.len());
                let mut sample_indices: Vec<Vec<usize>> = Vec::with_capacity(input_seq.len());
                let mut sample_tails: Vec<Option<usize>> = Vec::with_capacity(input_seq.len());

                for (row_idx, input_row) in input_seq.iter().enumerate() {
                    let target_id = Self::get_target_id(row_idx, batch_idx, offset, &target_tokens, &padding_mask_batch);

                    let selected_tail: Option<usize> = if is_training {
                        target_id
                            .filter(|&tid| tid < token_cluster_by_id.len())
                            .map(|tid| token_cluster_by_id[tid])
                            .filter(|&c| c > 0)
                    } else {
                        if !routing_weights.is_empty() {
                            routing_weights
                                .iter()
                                .enumerate()
                                .map(|(c, rw)| {
                                    let rb = r(routing_bias.get(c).copied().unwrap_or(0.0) as f64);
                                    let score: Real = rw.iter().zip(input_row.iter()).map(|(&w, x)| x.re * r(w as f64)).sum::<Real>() + rb;
                                    (c + 1, score)
                                })
                                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                                .map(|(c, _)| c)
                        } else {
                            None
                        }
                    };

                    // Candidates: head tokens + selected tail cluster tokens
                    let mut candidates: Vec<usize> = frequency_clusters[0].clone();
                    if let Some(tc) = selected_tail {
                        if tc < frequency_clusters.len() {
                            candidates.extend_from_slice(&frequency_clusters[tc]);
                        }
                    }
                    if is_training {
                        if let Some(tid) = target_id {
                            if !candidates.contains(&tid) {
                                candidates.push(tid);
                            }
                        }
                    }

                    // Score each candidate
                    let mut scored: Vec<(Real, usize)> = candidates
                        .iter()
                        .filter_map(|&token_id| {
                            let &cluster = token_cluster_by_id.get(token_id)?;
                            let &idx = token_index_in_cluster.get(token_id)?;
                            let cw = cluster_word_weights.get(cluster)?.get(idx)?;
                            let cb = *cluster_word_bias.get(cluster)?.get(idx)?;

                            let mut logit: Real = r(cb as f64);
                            for (d, &w) in cw.iter().enumerate() {
                                if d < input_row.len() {
                                    logit += input_row[d].re * r(w as f64);
                                }
                            }
                            // Only add routing score during training: at inference the routing
                            // selects which tail-cluster tokens to *include* as candidates, but
                            // should not bias the final logit comparison between head and tail
                            // tokens. During training the correct cluster is forced, so the
                            // routing score is part of the supervised signal; at inference the
                            // cluster is predicted and adding the score would unfairly boost
                            // tail tokens over head-cluster answers.
                            if cluster > 0 && is_training {
                                let rc = cluster - 1;
                                if let Some(rw) = routing_weights.get(rc) {
                                    let rb = r(routing_bias.get(rc).copied().unwrap_or(0.0) as f64);
                                    logit += rw.iter().zip(input_row.iter()).map(|(&w, x)| x.re * r(w as f64)).sum::<Real>() + rb;
                                }
                            }
                            Some((logit, token_id))
                        })
                        .collect();

                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                    if is_training {
                        if let Some(tid) = target_id {
                            if scored.len() > top_k && !scored[..top_k].iter().any(|(_, id)| *id == tid) {
                                if let Some(tgt_pos) = scored.iter().position(|(_, id)| *id == tid) {
                                    scored.swap(top_k - 1, tgt_pos);
                                }
                            }
                        }
                    }

                    scored.truncate(top_k);
                    while scored.len() < top_k {
                        scored.push((Real::NEG_INFINITY, 0));
                    }

                    let values: Vec<C> = scored.iter().map(|&(logit, _)| C::new(logit, ZERO)).collect();
                    let indices: Vec<usize> = scored.iter().map(|&(_, id)| id).collect();

                    sample_values.push(values);
                    sample_indices.push(indices);
                    sample_tails.push(selected_tail);
                }

                (sample_values, sample_indices, sample_tails)
            })
            .collect();

        let mut output_batch: Vec<Vec<Vec<C>>> = Vec::with_capacity(input_batch.len());
        let mut output_indices: Vec<Vec<Vec<usize>>> = Vec::with_capacity(input_batch.len());
        let mut selected_tails: Vec<Vec<Option<usize>>> = Vec::with_capacity(input_batch.len());

        for (vals, idxs, tails) in sparse_results {
            output_batch.push(vals);
            output_indices.push(idxs);
            selected_tails.push(tails);
        }

        self.output_indices_batch = output_indices.clone();
        self.selected_tail_cluster_batch = selected_tails;

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch.clone());
        layer_output.set_output_indices(output_indices.clone());

        // ── Stage 2 (training only): CE loss + full gradient computation ─────────

        if is_training && !padding_mask_batch.is_empty() {
            let batch_len = input_batch.len();
            let seq_len = if batch_len > 0 { input_batch[0].len() } else { 0 };
            let hidden_size = if batch_len > 0 && seq_len > 0 { input_batch[0][0].len() } else { 0 };
            let routing_count = self.weights.len();

            // (ce_losses_batch, sparse_grad_batch) via softmax_backward_real_with_gradient
            let ce_and_grad: Vec<(Vec<Vec<C>>, Vec<Vec<C>>)> = (0..batch_len)
                .into_par_iter()
                .map(|batch_idx| {
                    if batch_idx >= padding_mask_batch.len() || batch_idx >= target_tokens.len() {
                        return (vec![], vec![]);
                    }
                    let logits = &output_batch[batch_idx]; // [seq_len][k]
                    let targets = &target_tokens[batch_idx];
                    let pmask = &padding_mask_batch[batch_idx];
                    let logit_indices = &output_indices[batch_idx]; // [seq_len][k]

                    // padding_mask must match seq_len
                    if pmask.len() != logits.len() {
                        return (vec![], vec![]);
                    }

                    softmax_backward_real_with_gradient(logits, targets, pmask, total_valid_tokens, logit_indices)
                })
                .collect();

            // Separate into ce_losses and sparse_logit_grads
            let mut ce_losses_batch: Vec<Vec<Vec<C>>> = Vec::with_capacity(batch_len);
            let mut sparse_grad_batch: Vec<Vec<Vec<C>>> = Vec::with_capacity(batch_len);
            for (losses, grads) in ce_and_grad {
                ce_losses_batch.push(losses);
                sparse_grad_batch.push(grads);
            }

            layer_output.set_cross_entropy_loss_batch(ce_losses_batch.clone());
            self.cross_entropy_loss_batch = Some(ce_losses_batch);

            // Propagate sparse gradient → gradient_input_batch + weight gradients
            let zero_c = C::new(ZERO, ZERO);
            let mut grad_input = vec![vec![vec![zero_c; hidden_size]; seq_len]; batch_len];
            let mut routing_w_grads_batch = vec![vec![vec![zero_c; hidden_size]; routing_count]; batch_len];
            let mut routing_b_grads_batch = vec![vec![zero_c; routing_count]; batch_len];
            let mut word_w_grads = self.zero_cluster_word_weight_grads(hidden_size);
            let mut word_b_grads = self.zero_cluster_word_bias_grads();

            for batch_idx in 0..batch_len {
                if batch_idx >= sparse_grad_batch.len() {
                    continue;
                }
                let input_seq = &input_batch[batch_idx];
                let grad_seq = &sparse_grad_batch[batch_idx];
                let idx_seq = &output_indices[batch_idx];

                for seq_idx in 0..grad_seq.len().min(idx_seq.len()) {
                    if seq_idx >= input_seq.len() {
                        continue;
                    }
                    let input_row = &input_seq[seq_idx];
                    let grad_row = &grad_seq[seq_idx];
                    let idx_row = &idx_seq[seq_idx];

                    for k in 0..grad_row.len().min(idx_row.len()) {
                        let grad_val = grad_row[k];
                        let token_id = idx_row[k];

                        let cluster = match self.token_cluster_by_id.get(token_id) {
                            Some(&c) => c,
                            None => continue,
                        };
                        let idx_in_cluster = match self.token_index_in_cluster.get(token_id) {
                            Some(&i) => i,
                            None => continue,
                        };
                        if cluster >= self.cluster_word_weights.len() {
                            continue;
                        }
                        if idx_in_cluster >= self.cluster_word_weights[cluster].len() {
                            continue;
                        }

                        let token_w = &self.cluster_word_weights[cluster][idx_in_cluster];

                        // grad_x from token weights
                        for (d, &w) in token_w.iter().enumerate() {
                            if d < hidden_size {
                                grad_input[batch_idx][seq_idx][d] += grad_val * C::new(r(w as f64), ZERO);
                            }
                        }

                        // grad w.r.t. cluster word weights and bias
                        if idx_in_cluster < word_w_grads[cluster].len() {
                            for (d, &x_val) in input_row.iter().enumerate() {
                                if d < word_w_grads[cluster][idx_in_cluster].len() {
                                    word_w_grads[cluster][idx_in_cluster][d] += x_val * grad_val;
                                }
                            }
                            word_b_grads[cluster][idx_in_cluster] += grad_val;
                        }

                        // grad w.r.t. routing weights for tail tokens
                        if cluster > 0 {
                            let rc = cluster - 1;
                            if rc < routing_count {
                                for (d, &x_val) in input_row.iter().enumerate() {
                                    if d < hidden_size {
                                        routing_w_grads_batch[batch_idx][rc][d] += x_val * grad_val;
                                        // routing weight also contributes to grad_x
                                        grad_input[batch_idx][seq_idx][d] +=
                                            grad_val * C::new(r(self.weights[rc][d] as f64), ZERO);
                                    }
                                }
                                routing_b_grads_batch[batch_idx][rc] += grad_val;
                            }
                        }
                    }
                }
            }

            self.cluster_word_weight_gradients = Some(word_w_grads);
            self.cluster_word_bias_gradients = Some(word_b_grads);

            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(grad_input);
            gradient.set_gradient_weight_batch(routing_w_grads_batch);
            gradient.set_gradient_bias_batch(routing_b_grads_batch);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient);
        }

        layer_output
    }

    /// Backward pass — called only when a gradient from a later layer is present.
    /// In the standard architecture (AdaptiveLinear as the last layer), the gradient
    /// computed in `forward()` is used directly; this method handles the rare case where
    /// an additional layer is stacked after this one.
    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("No input batch in adaptive linear layer backward");
        let prev_grad_batch = previous_gradient.get_gradient_input_batch();
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();

        let batch_len = input_batch.len();
        let seq_len = if batch_len > 0 { input_batch[0].len() } else { 0 };
        let hidden_size = if batch_len > 0 && seq_len > 0 { input_batch[0][0].len() } else { 0 };
        let routing_count = self.weights.len();

        let zero_c = C::new(ZERO, ZERO);
        let mut gradient_input_batch = vec![vec![vec![zero_c; hidden_size]; seq_len]; batch_len];
        let mut routing_w_grads_batch = vec![vec![vec![zero_c; hidden_size]; routing_count]; batch_len];
        let mut routing_b_grads_batch = vec![vec![zero_c; routing_count]; batch_len];
        let mut word_w_grads = self.zero_cluster_word_weight_grads(hidden_size);
        let mut word_b_grads = self.zero_cluster_word_bias_grads();

        for batch_idx in 0..batch_len {
            if batch_idx >= prev_grad_batch.len() || batch_idx >= self.output_indices_batch.len() {
                continue;
            }
            let input_seq = &input_batch[batch_idx];
            let grad_seq = &prev_grad_batch[batch_idx];
            let idx_seq = &self.output_indices_batch[batch_idx];

            for seq_idx in 0..grad_seq.len().min(idx_seq.len()) {
                if seq_idx >= input_seq.len() {
                    continue;
                }
                let input_row = &input_seq[seq_idx];
                let grad_row = &grad_seq[seq_idx];
                let idx_row = &idx_seq[seq_idx];

                for k in 0..grad_row.len().min(idx_row.len()) {
                    let grad_val = grad_row[k];
                    let token_id = idx_row[k];

                    let cluster = match self.token_cluster_by_id.get(token_id) {
                        Some(&c) => c,
                        None => continue,
                    };
                    let idx_in_cluster = match self.token_index_in_cluster.get(token_id) {
                        Some(&i) => i,
                        None => continue,
                    };
                    if cluster >= self.cluster_word_weights.len()
                        || idx_in_cluster >= self.cluster_word_weights[cluster].len()
                    {
                        continue;
                    }

                    let token_w = &self.cluster_word_weights[cluster][idx_in_cluster];
                    for (d, &w) in token_w.iter().enumerate() {
                        if d < hidden_size {
                            gradient_input_batch[batch_idx][seq_idx][d] += grad_val * C::new(r(w as f64), ZERO);
                        }
                    }
                    if idx_in_cluster < word_w_grads[cluster].len() {
                        for (d, &x_val) in input_row.iter().enumerate() {
                            if d < word_w_grads[cluster][idx_in_cluster].len() {
                                word_w_grads[cluster][idx_in_cluster][d] += x_val * grad_val;
                            }
                        }
                        word_b_grads[cluster][idx_in_cluster] += grad_val;
                    }
                    if cluster > 0 {
                        let rc = cluster - 1;
                        if rc < routing_count {
                            for (d, &x_val) in input_row.iter().enumerate() {
                                if d < hidden_size {
                                    routing_w_grads_batch[batch_idx][rc][d] += x_val * grad_val;
                                    gradient_input_batch[batch_idx][seq_idx][d] +=
                                        grad_val * C::new(r(self.weights[rc][d] as f64), ZERO);
                                }
                            }
                            routing_b_grads_batch[batch_idx][rc] += grad_val;
                        }
                    }
                }
            }
        }

        self.cluster_word_weight_gradients = Some(word_w_grads);
        self.cluster_word_bias_gradients = Some(word_b_grads);

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(gradient_input_batch);
        gradient.set_gradient_weight_batch(routing_w_grads_batch);
        gradient.set_gradient_bias_batch(routing_b_grads_batch);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.as_mut().expect("No gradient in adaptive linear layer update_parameters");
        let (mut routing_w_grads, mut routing_b_grads) =
            (gradient.get_gradient_weights(), gradient.get_gradient_bias());
        let total_valid_tokens = gradient.get_total_valid_tokens();

        let norm = r(total_valid_tokens.max(1) as f64);
        routing_w_grads = average_matrix_by_scalar(&routing_w_grads, norm);
        routing_b_grads = average_vector_by_scalar(&routing_b_grads, norm);

        clip_all_gradients_by_global_norm_2d(&mut routing_w_grads, &mut routing_b_grads, self.global_norm, self.max_norm);
        normalize_gradients(&mut routing_w_grads);
        normalize_bias(&mut routing_b_grads);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let routing_count = self.weights.len();
        let hidden_size = if routing_count > 0 { self.weights[0].len() } else { 0 };

        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_w, mut prev_v_w, mut prev_v_w_hat, mut prev_v_b_hat) =
            if let Some(prev) = &mut self.previous_gradient {
                (
                    prev.get_prev_m_bias(),
                    prev.get_prev_v_bias(),
                    prev.get_prev_m_weights(),
                    prev.get_prev_v_weights(),
                    prev.get_prev_v_weights_hat(),
                    prev.get_prev_v_bias_hat(),
                )
            } else {
                (
                    vec![C::new(ZERO, ZERO); routing_count],
                    vec![C::new(ZERO, ZERO); routing_count],
                    vec![vec![C::new(ZERO, ZERO); hidden_size]; routing_count],
                    vec![vec![C::new(ZERO, ZERO); hidden_size]; routing_count],
                    vec![vec![C::new(ZERO, ZERO); hidden_size]; routing_count],
                    vec![C::new(ZERO, ZERO); routing_count],
                )
            };

        let all_routing_indices: Vec<usize> = (0..routing_count).collect();

        calculate_adam_w_bias_f32_sparse(
            &mut self.bias,
            &routing_b_grads,
            &mut prev_m_bias,
            &mut prev_v_bias,
            &mut prev_v_b_hat,
            learning_rate,
            time_step,
            &all_routing_indices,
        );
        calculate_adam_w_f32_sparse(
            &mut self.weights,
            &routing_w_grads,
            &mut prev_m_w,
            &mut prev_v_w,
            &mut prev_v_w_hat,
            learning_rate,
            time_step,
            &all_routing_indices,
        );

        // Sparse SGD update for cluster word weights
        let word_w_grads = self.cluster_word_weight_gradients.take();
        let word_b_grads = self.cluster_word_bias_gradients.take();

        if let (Some(ww_grads), Some(wb_grads)) = (word_w_grads, word_b_grads) {
            let used_tokens = self.collect_used_token_indices();
            let lr = learning_rate as f32;

            let updates: Vec<(usize, usize, Vec<C>, C)> = used_tokens
                .iter()
                .filter_map(|&token_id| {
                    let &cluster = self.token_cluster_by_id.get(token_id)?;
                    let &idx = self.token_index_in_cluster.get(token_id)?;
                    let g_row = ww_grads.get(cluster)?.get(idx)?.clone();
                    let g_bias = *wb_grads.get(cluster)?.get(idx)?;
                    Some((cluster, idx, g_row, g_bias))
                })
                .collect();

            for (cluster, idx, g_row, g_bias) in updates {
                if cluster < self.cluster_word_weights.len()
                    && idx < self.cluster_word_weights[cluster].len()
                {
                    let w_row = &mut self.cluster_word_weights[cluster][idx];
                    for (d, w) in w_row.iter_mut().enumerate() {
                        if d < g_row.len() {
                            *w -= lr * g_row[d].re as f32;
                        }
                    }
                }
                if cluster < self.cluster_word_bias.len() && idx < self.cluster_word_bias[cluster].len() {
                    self.cluster_word_bias[cluster][idx] -= lr * g_bias.re as f32;
                }
            }
        }

        let gradient = self.gradient.as_mut().unwrap();
        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_w);
        gradient.set_prev_v_weights(prev_v_w);
        gradient.set_prev_v_weights_hat(prev_v_w_hat);
        gradient.set_prev_v_bias_hat(prev_v_b_hat);
        gradient.set_gradient_weights(routing_w_grads);
        gradient.set_gradient_bias(routing_b_grads);
        self.previous_gradient = Some(gradient.clone());
        self.gradient = None;
    }

    fn collect_used_token_indices(&self) -> Vec<usize> {
        let mut set = HashSet::new();
        for batch in &self.output_indices_batch {
            for seq in batch {
                for &token_id in seq {
                    set.insert(token_id);
                }
            }
        }
        let mut v: Vec<usize> = set.into_iter().collect();
        v.sort_unstable();
        v
    }

    fn get_target_id(
        row_idx: usize,
        batch_idx: usize,
        offset: Option<usize>,
        target_batch: &[Vec<u32>],
        padding_mask_batch: &[Vec<u32>],
    ) -> Option<usize> {
        let offset = offset?;
        if row_idx < offset {
            return None;
        }
        let mask = padding_mask_batch.get(batch_idx)?;
        if row_idx >= mask.len() || mask[row_idx] == 0 {
            return None;
        }
        let target_idx = row_idx - offset;
        target_batch.get(batch_idx)?.get(target_idx).map(|&id| id as usize)
    }

    fn zero_cluster_word_weight_grads(&self, hidden_size: usize) -> Vec<Vec<Vec<C>>> {
        self.cluster_word_weights
            .iter()
            .map(|cluster| vec![vec![C::new(ZERO, ZERO); hidden_size]; cluster.len()])
            .collect()
    }

    fn zero_cluster_word_bias_grads(&self) -> Vec<Vec<C>> {
        self.cluster_word_bias
            .iter()
            .map(|cluster| vec![C::new(ZERO, ZERO); cluster.len()])
            .collect()
    }

    fn calculate_target_offset(
        batch_idx: usize,
        target_batch: &[Vec<u32>],
        padding_mask_batch: &[Vec<u32>],
    ) -> Option<usize> {
        if batch_idx >= target_batch.len()
            || target_batch[batch_idx].is_empty()
            || batch_idx >= padding_mask_batch.len()
        {
            return None;
        }
        let seq_len_unpadded = padding_mask_batch[batch_idx].iter().filter(|&&m| m != 0).count();
        Some(seq_len_unpadded.saturating_sub(target_batch[batch_idx].len()))
    }

    fn build_frequency_clusters(
        vocab_size: usize,
        head_size: usize,
        tail_cluster_count: usize,
    ) -> (Vec<u64>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>) {
        let mut token_frequencies = vec![0_u64; vocab_size];

        if let Ok(dataset) = load_data_xquad_de_as_dataset() {
            Self::accumulate_dataset_frequencies(&dataset, &mut token_frequencies);
        }
        if let Ok(dataset) = load_data_xquad_en_as_dataset() {
            Self::accumulate_dataset_frequencies(&dataset, &mut token_frequencies);
        }
        if let Ok(dataset) = load_data_xquad_ru_as_dataset() {
            Self::accumulate_dataset_frequencies(&dataset, &mut token_frequencies);
        }

        if token_frequencies.iter().all(|&freq| freq == 0) {
            for (token_id, freq) in token_frequencies.iter_mut().enumerate() {
                *freq = (vocab_size.saturating_sub(token_id)) as u64;
            }
        }

        let mut ranked_token_ids: Vec<usize> = (0..vocab_size).collect();
        ranked_token_ids.sort_by_key(|&token_id| Reverse(token_frequencies[token_id]));

        let mut frequency_clusters = Vec::new();
        frequency_clusters.push(ranked_token_ids.iter().take(head_size).copied().collect::<Vec<_>>());

        let tail_tokens = &ranked_token_ids[head_size.min(vocab_size)..];
        if !tail_tokens.is_empty() {
            let chunk_size = (tail_tokens.len() / tail_cluster_count.max(1)).max(1);
            for chunk in tail_tokens.chunks(chunk_size) {
                frequency_clusters.push(chunk.to_vec());
            }
        }

        let mut token_rank_by_id = vec![0_usize; vocab_size];
        for (rank, token_id) in ranked_token_ids.into_iter().enumerate() {
            token_rank_by_id[token_id] = rank;
        }

        let mut token_cluster_by_id = vec![0_usize; vocab_size];
        for (cluster_id, cluster_tokens) in frequency_clusters.iter().enumerate() {
            for &token_id in cluster_tokens {
                token_cluster_by_id[token_id] = cluster_id;
            }
        }

        (token_frequencies, frequency_clusters, token_rank_by_id, token_cluster_by_id)
    }

    fn accumulate_dataset_frequencies(dataset: &Dataset<String, String>, token_frequencies: &mut [u64]) {
        for text in dataset.get_input().iter().chain(dataset.get_target().iter()) {
            if let Ok((_, tokens)) = tokenize(text) {
                for tok in tokens {
                    let idx = tok as usize;
                    if idx < token_frequencies.len() {
                        token_frequencies[idx] += 1;
                    }
                }
            }
        }
    }
}
