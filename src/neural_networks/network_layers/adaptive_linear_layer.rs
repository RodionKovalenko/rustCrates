use serde::{Deserialize, Serialize};
use std::cmp::Reverse;

use crate::neural_networks::utils::tokenizer::tokenize;
use crate::neural_networks::{
    network_components::{
        gradient_struct::Gradient,
        input::{load_data_xquad_de_as_dataset, load_data_xquad_en_as_dataset, load_data_xquad_ru_as_dataset, DataTrait, Dataset},
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
    },
    network_layers::layer::LayerEnum,
    utils::{
        dtype::{r, Real, C, ZERO},
        matrix::clip_all_gradients_by_global_norm_2d,
        weights_initializer::initialize_weights_f32,
    },
};

const DEFAULT_HEAD_SIZE: usize = 2048;
const DEFAULT_TAIL_CLUSTER_COUNT: usize = 16;
const MAX_TAIL_CLUSTERS_PER_ROW: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLinearLayer {
    pub cluster_weights: Vec<Vec<f32>>,
    pub cluster_bias: Vec<f32>,
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
    pub weights: Vec<Vec<f32>>,
    pub previous_weights: Vec<Vec<f32>>,
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
    cluster_weight_gradients: Option<Vec<Vec<C>>>,
    #[serde(skip)]
    cluster_bias_gradients: Option<Vec<C>>,
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
        let mut cluster_weights = vec![vec![0.0; hidden_size]; num_clusters];
        let cluster_bias = vec![0.0; num_clusters];
        initialize_weights_f32(num_clusters, hidden_size, &mut cluster_weights);

        let mut cluster_word_weights = Vec::with_capacity(num_clusters);
        let mut cluster_word_bias = Vec::with_capacity(num_clusters);
        for tokens_in_cluster in &frequency_clusters {
            let cluster_size = tokens_in_cluster.len().max(1);
            let mut weights = vec![vec![0.0; hidden_size]; cluster_size];
            let bias = vec![0.0; cluster_size];
            initialize_weights_f32(cluster_size, hidden_size, &mut weights);
            cluster_word_weights.push(weights);
            cluster_word_bias.push(bias);
        }

        let weights = cluster_word_weights.first().cloned().unwrap_or_else(|| vec![vec![0.0; hidden_size]]);
        let previous_weights = weights.clone();
        let bias = cluster_word_bias.first().cloned().unwrap_or_else(|| vec![0.0]);

        Self {
            cluster_weights,
            cluster_bias,
            cluster_word_weights,
            cluster_word_bias,
            frequency_clusters,
            token_frequencies,
            token_rank_by_id,
            token_cluster_by_id,
            token_index_in_cluster,
            head_size,
            tail_cluster_count,
            learning_rate,
            smoothing: 0.99,
            ema: 0.0,
            norm_layer: None,
            global_norm: 0.0,
            max_norm: 0.0,
            input_batch: None,
            output_indices_batch: vec![],
            weights,
            previous_weights,
            bias,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            selected_tail_cluster_batch: vec![],
            target_batch_ids: None,
            cross_entropy_loss_batch: None,
            cluster_weight_gradients: None,
            cluster_bias_gradients: None,
            cluster_word_weight_gradients: None,
            cluster_word_bias_gradients: None,
        }
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = input.get_input_batch();
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());

        let target_tokens = input.get_target_batch_ids();
        let padding_mask_batch = input.get_padding_mask_batch();
        let top_k = input.get_top_k_size().max(1);
        let is_training = !target_tokens.is_empty() && !padding_mask_batch.is_empty();

        self.target_batch_ids = Some(target_tokens.clone());
        self.cross_entropy_loss_batch = None;
        self.gradient = None;

        let (output_batch, output_indices_batch, selected_tail_cluster_batch) = if is_training {
            let batch_len = input_batch.len();
            let seq_len = input_batch.first().map(|sample| sample.len()).unwrap_or(0);
            (
                vec![vec![vec![C::new(ZERO, ZERO)]; seq_len]; batch_len],
                vec![vec![Vec::new(); seq_len]; batch_len],
                vec![vec![None; seq_len]; batch_len],
            )
        } else {
            self.build_inference_outputs(&input_batch, top_k)
        };

        self.output_indices_batch = output_indices_batch.clone();
        self.selected_tail_cluster_batch = selected_tail_cluster_batch;

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output.set_output_indices(output_indices_batch);

        if is_training {
            let (losses, gradient) = self.compute_training_loss_and_gradient(&input_batch, &target_tokens, &padding_mask_batch);
            self.cross_entropy_loss_batch = Some(losses.clone());
            self.gradient = Some(gradient);
            layer_output.set_cross_entropy_loss_batch(losses);
        }

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        if previous_gradient.get_gradient_input_batch_ref().is_none() && self.gradient.is_some() {
            return self.gradient.as_ref().unwrap().clone();
        }

        let input_batch = self.input_batch.as_ref().expect("AdaptiveLinearLayer input batch missing");
        let sparse_grad_batch = previous_gradient.get_gradient_input_batch();
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();

        if input_batch.is_empty() {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let hidden_size = input_batch[0][0].len();
        let mut grad_input_batch = vec![vec![vec![C::new(ZERO, ZERO); hidden_size]; input_batch[0].len()]; input_batch.len()];
        let mut cluster_weight_grads = vec![vec![C::new(ZERO, ZERO); hidden_size]; self.cluster_weights.len()];
        let mut cluster_bias_grads = vec![C::new(ZERO, ZERO); self.cluster_bias.len()];
        let mut cluster_word_weight_grads = self.zero_cluster_word_weight_grads(hidden_size);
        let mut cluster_word_bias_grads = self.zero_cluster_word_bias_grads();

        for batch_idx in 0..input_batch.len() {
            for row_idx in 0..input_batch[batch_idx].len() {
                let grad_row = sparse_grad_batch.get(batch_idx).and_then(|sample| sample.get(row_idx)).cloned().unwrap_or_default();
                let idx_row = self.output_indices_batch.get(batch_idx).and_then(|sample| sample.get(row_idx)).cloned().unwrap_or_default();
                let input_row = &input_batch[batch_idx][row_idx];

                for (col_idx, grad_val) in grad_row.iter().enumerate() {
                    if col_idx >= idx_row.len() {
                        continue;
                    }
                    self.accumulate_token_gradient(
                        idx_row[col_idx],
                        *grad_val,
                        input_row,
                        &mut grad_input_batch[batch_idx][row_idx],
                        &mut cluster_weight_grads,
                        &mut cluster_bias_grads,
                        &mut cluster_word_weight_grads,
                        &mut cluster_word_bias_grads,
                    );
                }
            }
        }

        self.cluster_weight_gradients = Some(cluster_weight_grads);
        self.cluster_bias_gradients = Some(cluster_bias_grads);
        self.cluster_word_weight_gradients = Some(cluster_word_weight_grads);
        self.cluster_word_bias_gradients = Some(cluster_word_bias_grads);

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(grad_input_batch);
        gradient.set_total_valid_tokens(total_valid_tokens);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let lr = self.learning_rate as f32;

        if let (Some(mut cluster_weight_grads), Some(mut cluster_bias_grads)) =
            (self.cluster_weight_gradients.take(), self.cluster_bias_gradients.take())
        {
            clip_all_gradients_by_global_norm_2d(&mut cluster_weight_grads, &mut cluster_bias_grads, self.global_norm, self.max_norm);
            for (cluster_id, grad_row) in cluster_weight_grads.iter().enumerate() {
                for (emb_idx, grad_val) in grad_row.iter().enumerate() {
                    self.cluster_weights[cluster_id][emb_idx] -= grad_val.re as f32 * lr;
                }
            }
            for (cluster_id, grad_val) in cluster_bias_grads.iter().enumerate() {
                self.cluster_bias[cluster_id] -= grad_val.re as f32 * lr;
            }
        }

        if let (Some(cluster_word_weight_grads), Some(cluster_word_bias_grads)) =
            (self.cluster_word_weight_gradients.take(), self.cluster_word_bias_gradients.take())
        {
            for (cluster_id, cluster_grad) in cluster_word_weight_grads.iter().enumerate() {
                for (token_pos, token_grad) in cluster_grad.iter().enumerate() {
                    for (emb_idx, grad_val) in token_grad.iter().enumerate() {
                        self.cluster_word_weights[cluster_id][token_pos][emb_idx] -= grad_val.re as f32 * lr;
                    }
                }
            }
            for (cluster_id, cluster_grad) in cluster_word_bias_grads.iter().enumerate() {
                for (token_pos, grad_val) in cluster_grad.iter().enumerate() {
                    self.cluster_word_bias[cluster_id][token_pos] -= grad_val.re as f32 * lr;
                }
            }
        }

        self.previous_weights = self.weights.clone();
        self.weights = self.cluster_word_weights.first().cloned().unwrap_or_default();
        self.bias = self.cluster_word_bias.first().cloned().unwrap_or_default();
        self.gradient = None;
    }

    fn build_inference_outputs(&self, input_batch: &[Vec<Vec<C>>], top_k: usize) -> (Vec<Vec<Vec<C>>>, Vec<Vec<Vec<usize>>>, Vec<Vec<Option<usize>>>) {
        let mut output_batch = Vec::with_capacity(input_batch.len());
        let mut output_indices_batch = Vec::with_capacity(input_batch.len());
        let mut selected_tail_cluster_batch = Vec::with_capacity(input_batch.len());

        for sample in input_batch {
            let mut sample_values = Vec::with_capacity(sample.len());
            let mut sample_indices = Vec::with_capacity(sample.len());
            let mut sample_tail_clusters = Vec::with_capacity(sample.len());

            for row in sample {
                let (values, indices, selected_tail_cluster) = self.build_inference_row(row, top_k);
                sample_values.push(values.into_iter().map(|v| C::new(v, ZERO)).collect());
                sample_indices.push(indices);
                sample_tail_clusters.push(selected_tail_cluster);
            }

            output_batch.push(sample_values);
            output_indices_batch.push(sample_indices);
            selected_tail_cluster_batch.push(sample_tail_clusters);
        }

        (output_batch, output_indices_batch, selected_tail_cluster_batch)
    }

    fn build_inference_row(&self, input_row: &[C], top_k: usize) -> (Vec<Real>, Vec<usize>, Option<usize>) {
        let head_token_count = self.head_token_count();
        let head_logits = self.compute_head_logits(input_row);
        let head_probs = Self::softmax(&head_logits);

        let mut candidates: Vec<(Real, usize)> = self
            .top_n_indices(&head_probs[..head_token_count], top_k)
            .into_iter()
            .map(|idx| (head_probs[idx], self.frequency_clusters[0][idx]))
            .collect();

        let mut selected_tail_cluster = None;
        let tail_route_probs = &head_probs[head_token_count..];
        let top_clusters = self
            .top_n_indices(tail_route_probs, MAX_TAIL_CLUSTERS_PER_ROW.min(self.tail_cluster_count.max(1)))
            .into_iter()
            .map(|route_idx| route_idx + 1)
            .collect::<Vec<_>>();

        for cluster_id in top_clusters {
            selected_tail_cluster = Some(cluster_id);
            let tail_logits = self.compute_tail_logits(input_row, cluster_id);
            let tail_probs = Self::softmax(&tail_logits);
            let route_prob = head_probs[head_token_count + cluster_id - 1];
            for tail_idx in self.top_n_indices(&tail_probs, top_k) {
                let token_id = self.frequency_clusters[cluster_id][tail_idx];
                candidates.push((route_prob * tail_probs[tail_idx], token_id));
            }
        }

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by(|a, b| a.1 == b.1);
        candidates.truncate(top_k);

        (
            candidates.iter().map(|(prob, _)| *prob).collect(),
            candidates.iter().map(|(_, token_id)| *token_id).collect(),
            selected_tail_cluster,
        )
    }

    fn compute_training_loss_and_gradient(
        &mut self,
        input_batch: &[Vec<Vec<C>>],
        target_tokens: &[Vec<u32>],
        padding_mask_batch: &[Vec<u32>],
    ) -> (Vec<Vec<Vec<C>>>, Gradient) {
        let total_valid_tokens = Self::calculate_total_valid_tokens(target_tokens, padding_mask_batch);
        let hidden_size = input_batch[0][0].len();
        let scale = r(1.0) / r(total_valid_tokens as f64);

        let mut losses = vec![vec![vec![C::new(ZERO, ZERO)]; input_batch[0].len()]; input_batch.len()];
        let mut grad_input_batch = vec![vec![vec![C::new(ZERO, ZERO); hidden_size]; input_batch[0].len()]; input_batch.len()];
        let mut cluster_weight_grads = vec![vec![C::new(ZERO, ZERO); hidden_size]; self.cluster_weights.len()];
        let mut cluster_bias_grads = vec![C::new(ZERO, ZERO); self.cluster_bias.len()];
        let mut cluster_word_weight_grads = self.zero_cluster_word_weight_grads(hidden_size);
        let mut cluster_word_bias_grads = self.zero_cluster_word_bias_grads();

        for batch_idx in 0..input_batch.len() {
            let offset = Self::calculate_target_offset(batch_idx, target_tokens, padding_mask_batch).unwrap_or(usize::MAX);
            for row_idx in 0..input_batch[batch_idx].len() {
                if row_idx < offset || row_idx >= padding_mask_batch[batch_idx].len() || padding_mask_batch[batch_idx][row_idx] == 0 {
                    continue;
                }

                let target_pos = row_idx - offset;
                if target_pos >= target_tokens[batch_idx].len() {
                    continue;
                }

                let target_token = target_tokens[batch_idx][target_pos] as usize;
                if target_token == 1 || target_token >= self.token_cluster_by_id.len() {
                    continue;
                }

                let (loss, selected_tail_cluster) = self.accumulate_exact_row_gradient(
                    &input_batch[batch_idx][row_idx],
                    target_token,
                    scale,
                    &mut grad_input_batch[batch_idx][row_idx],
                    &mut cluster_weight_grads,
                    &mut cluster_bias_grads,
                    &mut cluster_word_weight_grads,
                    &mut cluster_word_bias_grads,
                );

                losses[batch_idx][row_idx][0] = C::new(loss, ZERO);
                if self.selected_tail_cluster_batch.len() > batch_idx && self.selected_tail_cluster_batch[batch_idx].len() > row_idx {
                    self.selected_tail_cluster_batch[batch_idx][row_idx] = selected_tail_cluster;
                }
            }
        }

        self.cluster_weight_gradients = Some(cluster_weight_grads);
        self.cluster_bias_gradients = Some(cluster_bias_grads);
        self.cluster_word_weight_gradients = Some(cluster_word_weight_grads);
        self.cluster_word_bias_gradients = Some(cluster_word_bias_grads);

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(grad_input_batch);
        gradient.set_total_valid_tokens(total_valid_tokens);
        (losses, gradient)
    }

    fn accumulate_exact_row_gradient(
        &self,
        input_row: &[C],
        target_token: usize,
        scale: Real,
        grad_input_row: &mut [C],
        cluster_weight_grads: &mut [Vec<C>],
        cluster_bias_grads: &mut [C],
        cluster_word_weight_grads: &mut [Vec<Vec<C>>],
        cluster_word_bias_grads: &mut [Vec<C>],
    ) -> (Real, Option<usize>) {
        let head_token_count = self.head_token_count();
        let target_cluster = self.token_cluster_by_id[target_token];
        let target_index_in_cluster = self.token_index_in_cluster[target_token];

        let head_logits = self.compute_head_logits(input_row);
        let (head_loss, head_grad) = if target_cluster == 0 {
            Self::cross_entropy_with_gradient(&head_logits, target_index_in_cluster)
        } else {
            Self::cross_entropy_with_gradient(&head_logits, head_token_count + target_cluster - 1)
        };

        for (head_idx, grad_val) in head_grad.iter().enumerate().take(head_token_count) {
            self.accumulate_word_gradient(0, head_idx, *grad_val * scale, input_row, grad_input_row, cluster_word_weight_grads, cluster_word_bias_grads);
        }

        for cluster_id in 1..self.frequency_clusters.len() {
            let route_idx = head_token_count + cluster_id - 1;
            self.accumulate_cluster_route_gradient(cluster_id, head_grad[route_idx] * scale, input_row, grad_input_row, cluster_weight_grads, cluster_bias_grads);
        }

        if target_cluster == 0 {
            return (head_loss * scale, None);
        }

        let tail_logits = self.compute_tail_logits(input_row, target_cluster);
        let (tail_loss, tail_grad) = Self::cross_entropy_with_gradient(&tail_logits, target_index_in_cluster);
        for (tail_idx, grad_val) in tail_grad.iter().enumerate() {
            self.accumulate_word_gradient(target_cluster, tail_idx, *grad_val * scale, input_row, grad_input_row, cluster_word_weight_grads, cluster_word_bias_grads);
        }

        ((head_loss + tail_loss) * scale, Some(target_cluster))
    }

    fn accumulate_word_gradient(
        &self,
        cluster_id: usize,
        token_pos: usize,
        grad_val: Real,
        input_row: &[C],
        grad_input_row: &mut [C],
        cluster_word_weight_grads: &mut [Vec<Vec<C>>],
        cluster_word_bias_grads: &mut [Vec<C>],
    ) {
        let grad_c = C::new(grad_val, ZERO);
        for emb_idx in 0..input_row.len() {
            cluster_word_weight_grads[cluster_id][token_pos][emb_idx] += input_row[emb_idx] * grad_c;
            grad_input_row[emb_idx] += C::new(r(self.cluster_word_weights[cluster_id][token_pos][emb_idx] as f64), ZERO) * grad_c;
        }
        cluster_word_bias_grads[cluster_id][token_pos] += grad_c;
    }

    fn accumulate_cluster_route_gradient(
        &self,
        cluster_id: usize,
        grad_val: Real,
        input_row: &[C],
        grad_input_row: &mut [C],
        cluster_weight_grads: &mut [Vec<C>],
        cluster_bias_grads: &mut [C],
    ) {
        let grad_c = C::new(grad_val, ZERO);
        for emb_idx in 0..input_row.len() {
            cluster_weight_grads[cluster_id][emb_idx] += input_row[emb_idx] * grad_c;
            grad_input_row[emb_idx] += C::new(r(self.cluster_weights[cluster_id][emb_idx] as f64), ZERO) * grad_c;
        }
        cluster_bias_grads[cluster_id] += grad_c;
    }

    fn accumulate_token_gradient(
        &self,
        token_id: usize,
        grad_val: C,
        input_row: &[C],
        grad_input_row: &mut [C],
        cluster_weight_grads: &mut [Vec<C>],
        cluster_bias_grads: &mut [C],
        cluster_word_weight_grads: &mut [Vec<Vec<C>>],
        cluster_word_bias_grads: &mut [Vec<C>],
    ) {
        let cluster_id = self.token_cluster_by_id[token_id];
        let token_pos = self.token_index_in_cluster[token_id];

        for emb_idx in 0..input_row.len() {
            cluster_word_weight_grads[cluster_id][token_pos][emb_idx] += input_row[emb_idx] * grad_val;
            grad_input_row[emb_idx] += C::new(r(self.cluster_word_weights[cluster_id][token_pos][emb_idx] as f64), ZERO) * grad_val;
        }
        cluster_word_bias_grads[cluster_id][token_pos] += grad_val;

        if cluster_id > 0 {
            for emb_idx in 0..input_row.len() {
                cluster_weight_grads[cluster_id][emb_idx] += input_row[emb_idx] * grad_val;
                grad_input_row[emb_idx] += C::new(r(self.cluster_weights[cluster_id][emb_idx] as f64), ZERO) * grad_val;
            }
            cluster_bias_grads[cluster_id] += grad_val;
        }
    }

    fn compute_head_logits(&self, input_row: &[C]) -> Vec<Real> {
        let head_token_count = self.head_token_count();
        let mut logits = Vec::with_capacity(head_token_count + self.frequency_clusters.len().saturating_sub(1));
        for token_pos in 0..head_token_count {
            logits.push(self.compute_word_score(input_row, 0, token_pos));
        }
        for cluster_id in 1..self.frequency_clusters.len() {
            logits.push(self.compute_cluster_route_score(input_row, cluster_id));
        }
        logits
    }

    fn compute_tail_logits(&self, input_row: &[C], cluster_id: usize) -> Vec<Real> {
        (0..self.frequency_clusters[cluster_id].len())
            .map(|token_pos| self.compute_word_score(input_row, cluster_id, token_pos))
            .collect()
    }

    fn compute_cluster_route_score(&self, input_row: &[C], cluster_id: usize) -> Real {
        self.compute_real_dot(input_row, &self.cluster_weights[cluster_id], self.cluster_bias[cluster_id])
    }

    fn compute_word_score(&self, input_row: &[C], cluster_id: usize, token_pos: usize) -> Real {
        self.compute_real_dot(input_row, &self.cluster_word_weights[cluster_id][token_pos], self.cluster_word_bias[cluster_id][token_pos])
    }

    fn compute_real_dot(&self, input_row: &[C], weights: &[f32], bias: f32) -> Real {
        let mut sum = r(bias as f64);
        for (input_val, &weight) in input_row.iter().zip(weights.iter()) {
            sum += input_val.re * r(weight as f64);
        }
        sum
    }

    fn softmax(logits: &[Real]) -> Vec<Real> {
        if logits.is_empty() {
            return vec![];
        }
        let max_logit = logits.iter().copied().fold(Real::NEG_INFINITY, Real::max);
        let exps = logits.iter().map(|&z| (z - max_logit).exp()).collect::<Vec<_>>();
        let sum = exps.iter().sum::<Real>().max(r(1e-30));
        exps.into_iter().map(|e| e / sum).collect()
    }

    fn cross_entropy_with_gradient(logits: &[Real], target_idx: usize) -> (Real, Vec<Real>) {
        let probs = Self::softmax(logits);
        let p_target = probs[target_idx].max(r(1e-30));
        let loss = -p_target.ln();
        let mut grad = probs;
        grad[target_idx] -= r(1.0);
        (loss, grad)
    }

    fn top_n_indices(&self, values: &[Real], n: usize) -> Vec<usize> {
        let mut idx = (0..values.len()).collect::<Vec<_>>();
        idx.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(n.min(idx.len()));
        idx
    }

    fn head_token_count(&self) -> usize {
        self.frequency_clusters.first().map(|tokens| tokens.len()).unwrap_or(0)
    }

    fn zero_cluster_word_weight_grads(&self, hidden_size: usize) -> Vec<Vec<Vec<C>>> {
        self.cluster_word_weights.iter().map(|cluster| vec![vec![C::new(ZERO, ZERO); hidden_size]; cluster.len()]).collect()
    }

    fn zero_cluster_word_bias_grads(&self) -> Vec<Vec<C>> {
        self.cluster_word_bias.iter().map(|cluster| vec![C::new(ZERO, ZERO); cluster.len()]).collect()
    }

    fn calculate_target_offset(batch_idx: usize, target_batch: &[Vec<u32>], padding_mask_batch: &[Vec<u32>]) -> Option<usize> {
        if batch_idx >= target_batch.len() || target_batch[batch_idx].is_empty() || batch_idx >= padding_mask_batch.len() {
            return None;
        }
        let seq_len_unpadded = padding_mask_batch[batch_idx].iter().filter(|&&m| m != 0).count();
        Some(seq_len_unpadded.saturating_sub(target_batch[batch_idx].len()))
    }

    fn calculate_total_valid_tokens(target_batch: &[Vec<u32>], padding_mask_batch: &[Vec<u32>]) -> usize {
        target_batch
            .iter()
            .zip(padding_mask_batch.iter())
            .map(|(targets, mask)| {
                let target_len = targets.len();
                let valid_seq_len = mask.iter().filter(|&&m| m != 0).count();
                let offset = valid_seq_len.saturating_sub(target_len);
                targets
                    .iter()
                    .enumerate()
                    .filter(|(i, &target_id)| target_id != 1 && offset + i < mask.len() && mask[offset + i] != 0)
                    .count()
            })
            .sum::<usize>()
            .max(1)
    }

    fn build_frequency_clusters(vocab_size: usize, head_size: usize, tail_cluster_count: usize) -> (Vec<u64>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>) {
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

#[cfg(test)]
mod tests {
    use super::AdaptiveLinearLayer;
    use crate::neural_networks::network_components::layer_input_struct::LayerInput;
    use crate::neural_networks::utils::dtype::C;

    #[test]
    fn adaptive_layer_produces_finite_training_loss() {
        let mut layer = AdaptiveLinearLayer::new(0.001, 4, 64);
        let mut input = LayerInput::new_default();

        input.set_input_batch(vec![vec![vec![C::new(0.5, 0.0); 4]; 3]]);
        input.set_batch_size(1);
        input.set_time_step(1);
        input.set_top_k_size(8);
        input.set_padding_mask_batch(vec![vec![1, 1, 1]]);
        input.set_target_batch_ids(vec![vec![10, 11]]);

        let output = layer.forward(&input);
        let loss = output.get_cross_entropy_loss_batch();

        assert_eq!(loss.len(), 1);
        assert_eq!(loss[0].len(), 3);
        assert!(loss[0][1][0].re.is_finite());
        assert!(loss[0][2][0].re.is_finite());
        assert!(layer.gradient.is_some());
    }
}
