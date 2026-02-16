use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::layer::LayerEnum,
    network_types::transformer::transformer_updater::VERBOSE,
    utils::{
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        dtype::{r, C, ONE, ZERO},
        matrix::{add_vector, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose, multiply_complex, normalize_bias, normalize_gradients},
        shared_f32_matrix::SharedF32Matrix,
        weights_initializer::{initialize_weights_complex, initialize_weights_complex_only_real},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weights: Vec<Vec<C>>,
    pub learning_rate: f64,
    pub bias: Vec<C>,
    pub smoothing: f64,
    pub ema: f64,
    pub norm_layer: Option<LayerEnum>,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub is_complex: bool,

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

    /// Optional shared (tied) embedding table. When set, this layer becomes an output projection
    /// over the tied table with transpose semantics: logits[token] = dot(input, tied[token]).
    ///
    /// Uses `SharedF32Matrix` to match `EmbeddingLayer` / `SparseLinearLayer` tying.
    #[serde(skip)]
    pub tied_weights: Option<SharedF32Matrix>,

    /// When set, this layer will also apply gradients accumulated by a tied embedding layer.
    #[serde(skip)]
    pub tied_embedding_grad_by_token: Option<Arc<RwLock<HashMap<usize, Vec<C>>>>>,
}

impl LinearLayer {
    pub fn new(learning_rate: f64, rows: usize, cols: usize, is_complex: bool) -> Self {
        let mut weights: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let bias: Vec<C> = vec![C::new(ONE, ZERO); cols];

        if is_complex {
            initialize_weights_complex(rows, cols, &mut weights);
        } else {
            initialize_weights_complex_only_real(rows, cols, &mut weights);
        }

        Self {
            weights,
            bias,
            learning_rate,
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
            is_complex,
            output_indices: None,
            tied_weights: None,
            tied_embedding_grad_by_token: None,
        }
    }

    /// Enable weight tying with an embedding layer via a shared table.
    ///
    /// The returned accumulator should be stored by the embedding layer, which will accumulate
    /// per-token gradients into it during backward/update.
    pub fn enable_weight_tying(&mut self, weights: SharedF32Matrix) -> Arc<RwLock<HashMap<usize, Vec<C>>>> {
        let acc = Arc::new(RwLock::new(HashMap::new()));
        self.tied_weights = Some(weights);
        self.tied_embedding_grad_by_token = Some(acc.clone());
        acc
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<C>>> = input.get_input_batch();
        if input_batch.is_empty() && input.get_input_batch_rm_ref().is_some_and(|rm| !rm.is_empty()) {
            panic!("LinearLayer received RM input; use LinearLayerRm");
        }

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        self.input_batch = (!input_batch.is_empty()).then_some(input_batch.clone());

        let output_indices: Vec<Vec<Vec<usize>>> = vec![];

        let start = std::time::Instant::now();

        if self.tied_weights.is_some() {
            if self.is_complex {
                panic!("LinearLayer: tied weights are only supported for real (is_complex=false) mode");
            }

            let (output_batch, output_indices) = self.multiply_highest_k_per_row_tied(&input_batch, input);
            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch(output_batch);
            layer_output.set_output_indices(output_indices);
            self.output_indices = Some(layer_output.get_output_indices());
            return layer_output;
        }

        if self.is_complex {
            let output_batch: Vec<Vec<Vec<C>>> = input_batch
                .par_iter()
                .map(|input_rows| {
                    let mut out = multiply_complex(input_rows, &self.weights);
                    add_vector(&mut out, &self.bias);
                    out
                })
                .collect();

            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch(output_batch);
            layer_output.set_output_indices(output_indices);
            self.output_indices = Some(layer_output.get_output_indices());
            return layer_output;
        } else {
            let mut layer_input = input.clone();
            layer_input.set_input_batch(input_batch.clone());
            let (output_batch, output_indices) = self.mutliply_hightest_k_per_row(&input_batch, &layer_input);

            if VERBOSE && !self.is_complex {
                println!("Linear layer complex matmul time for batch size {}: {}", self.batch_size, start.elapsed().as_secs_f64());
            }

            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch(output_batch);
            layer_output.set_output_indices(output_indices);

            self.output_indices = Some(layer_output.get_output_indices());
            return layer_output;
        }
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_vec = self.input_batch.as_ref().filter(|b| !b.is_empty());
        let mut gradient = Gradient::new_default();

        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let previous_gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient.get_gradient_input_batch();
        if previous_gradient_input_batch.is_empty() && previous_gradient.get_gradient_input_batch_rm_ref().is_some_and(|rm| !rm.is_empty()) {
            panic!("LinearLayer received RM gradient; use LinearLayerRm");
        }

        if input_batch_vec.is_none() {
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        // Fast path for tied projection: gradients are accumulated in vocab x embedding_dim layout.
        if let Some(tied) = &self.tied_weights {
            let weights_guard = tied.read();
            let vocab_size = weights_guard.len();
            let embedding_d = weights_guard.first().map(|r| r.len()).unwrap_or(0);

            let input_batch = input_batch_vec.expect("Vec input batch missing");
            let output_indices_batch = self.output_indices.as_ref().expect("Output indices missing in linear layer backward pass");

            let batch_len = input_batch.len();
            let seq_len = input_batch.first().map(|s| s.len()).unwrap_or(0);

            let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); embedding_d]; vocab_size]; batch_len];
            let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); vocab_size]; batch_len];
            let mut gradient_input_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); embedding_d]; seq_len]; batch_len];

            for batch_idx in 0..batch_len {
                let input_sample = &input_batch[batch_idx];
                let sparse_grad = &previous_gradient_input_batch[batch_idx];
                let indices = &output_indices_batch[batch_idx];

                for seq_idx in 0..sparse_grad.len().min(indices.len()).min(input_sample.len()) {
                    let input_row = &input_sample[seq_idx];
                    let grad_row = &sparse_grad[seq_idx];
                    let idx_row = &indices[seq_idx];

                    for (k_idx, &grad_val) in grad_row.iter().enumerate() {
                        if k_idx >= idx_row.len() {
                            continue;
                        }
                        let vocab_idx = idx_row[k_idx];
                        if vocab_idx >= vocab_size {
                            continue;
                        }

                        // Weight grads in vocab x emb layout
                        let n = embedding_d.min(input_row.len());
                        for emb_idx in 0..n {
                            weight_gradients[batch_idx][vocab_idx][emb_idx] += input_row[emb_idx] * grad_val;
                        }

                        // Bias grads
                        bias_gradients[batch_idx][vocab_idx] += grad_val;

                        // Input grads
                        let w_row = &weights_guard[vocab_idx];
                        let m = embedding_d.min(w_row.len()).min(gradient_input_batch[batch_idx][seq_idx].len());
                        for emb_idx in 0..m {
                            gradient_input_batch[batch_idx][seq_idx][emb_idx] += C::new(r(w_row[emb_idx] as f64), ZERO) * grad_val;
                        }
                    }
                }
            }

            gradient.set_gradient_input_batch(gradient_input_batch);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(weight_gradients);
            gradient.set_gradient_bias_batch(bias_gradients);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        // Initialize gradients for weights and biases
        let batch_len = input_batch_vec.expect("Vec input batch missing in LinearLayer::backward").len();

        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];
        if self.is_complex {
            let input_batch = input_batch_vec.expect("Vec input batch missing");
            let weights_h = conjugate_transpose(&self.weights);

            let mut gradient_input_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

            for batch_idx in 0..batch_len {
                let input_sample = &input_batch[batch_idx];
                let grad = &previous_gradient_input_batch[batch_idx];

                weight_gradients[batch_idx] = multiply_complex(&conjugate_transpose(input_sample), grad);

                for grad_row in grad.iter() {
                    for (k, grad_val) in grad_row.iter().enumerate() {
                        bias_gradients[batch_idx][k] += grad_val;
                    }
                }

                gradient_input_batch[batch_idx] = multiply_complex(grad, &weights_h);
            }

            gradient.set_gradient_input_batch(gradient_input_batch);
        } else {
            // rows are sparse with only top k values
            let output_indices_batch = self.output_indices.as_ref().expect("Output indices missing in linear layer backward pass");

            let input_batch = input_batch_vec.expect("Input batch is missing in sparse linear layer backward");
            let mut gradient_input_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

            for batch_idx in 0..input_batch.len() {
                let input_sample = &input_batch[batch_idx];
                let sparse_grad = &previous_gradient_input_batch[batch_idx];
                let indices = &output_indices_batch[batch_idx];

                // Reconstruct full gradient from sparse gradients using indices
                let num_output_cols = self.weights[0].len();
                let mut full_gradient: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); num_output_cols]; input_sample.len()];

                for row_idx in 0..sparse_grad.len().min(indices.len()) {
                    let grad_row = &sparse_grad[row_idx];
                    let idx_row = &indices[row_idx];

                    for (grad_idx, &grad_val) in grad_row.iter().enumerate() {
                        if grad_idx < idx_row.len() {
                            let col_idx = idx_row[grad_idx];
                            if col_idx < num_output_cols {
                                full_gradient[row_idx][col_idx] = grad_val;
                            }
                        }
                    }
                }

                // Compute weight gradients: input^H * gradient
                weight_gradients[batch_idx] = multiply_complex(&conjugate_transpose(&input_sample), &full_gradient);

                // Accumulate gradients for biases
                for grad_row in full_gradient.iter() {
                    for (k, grad_val) in grad_row.iter().enumerate() {
                        bias_gradients[batch_idx][k] += grad_val;
                    }
                }

                // Compute input gradients: gradient * weights^H
                gradient_input_batch[batch_idx] = multiply_complex(&full_gradient, &conjugate_transpose(&self.weights));
            }

            gradient.set_gradient_input_batch(gradient_input_batch);
        }

        // if self.gradient.is_some() {
        //     let previous_gradient = self.gradient.as_ref().expect("");
        //     weight_gradients = add_matrix_3d(&weight_gradients, &previous_gradient.get_gradient_weight_batch());
        //     bias_gradients = add_matrix_2d_c(&bias_gradients, &previous_gradient.get_gradient_bias_batch());
        // }
        //  println!("batch size in linear layer: {}", self.batch_size);

        gradient.set_gradient_input_batch_rm(vec![]);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());

        gradient
    }

    // multiply normally, select highest k per row, return k highest values per row and original indices
    pub fn mutliply_hightest_k_per_row(&mut self, input_batch: &Vec<Vec<Vec<C>>>, layer_input: &LayerInput) -> (Vec<Vec<Vec<C>>>, Vec<Vec<Vec<usize>>>) {
        let target_batch: &Vec<Vec<u32>> = &layer_input.get_target_batch_ids();
        let k: usize = layer_input.get_top_k_size();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();

        let num_output_cols = self.weights[0].len();

        // Parallelize batch processing
        let results: Vec<_> = input_batch
            .par_iter()
            .enumerate()
            .map(|(batch_idx, input_sample)| {
                let mut sample_values: Vec<Vec<C>> = vec![];
                let mut sample_indices: Vec<Vec<usize>> = vec![];

                // Calculate offset only if target_batch has data for this batch
                let offset = if batch_idx < target_batch.len() && !target_batch[batch_idx].is_empty() {
                    let seq_len_unpadded = padding_mask_batch[batch_idx].iter().filter(|&&x| x != 0).count();
                    seq_len_unpadded.saturating_sub(target_batch[batch_idx].len())
                } else {
                    usize::MAX // Set to max to ensure no target tokens are selected during inference
                };

                for (row_idx, input_row) in input_sample.iter().enumerate() {
                    // Get target token id for this row if it exists
                    let target_id = if batch_idx < target_batch.len() && !target_batch[batch_idx].is_empty() && offset != usize::MAX && padding_mask_batch[batch_idx][row_idx] != 0 && row_idx >= offset
                    {
                        let target_idx = row_idx - offset;
                        if target_idx < target_batch[batch_idx].len() {
                            Some(target_batch[batch_idx][target_idx] as usize)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Track top k values: (real_value, value, index)
                    let mut top_k: Vec<(f64, C, usize)> = Vec::with_capacity(k + 1);
                    let mut min_value = f64::NEG_INFINITY;
                    let mut target_idx_in_topk: Option<usize> = None; // Track position of target in top_k

                    // Compute each output element on the fly
                    for col_idx in 0..num_output_cols {
                        // Compute dot product: input_row · weights[:, col_idx]
                        let mut sum = C::new(ZERO, ZERO);
                        for (i, &input_val) in input_row.iter().enumerate() {
                            sum += input_val * self.weights[i][col_idx];
                        }
                        // Add bias
                        sum += self.bias[col_idx];

                        let real_value = sum.re as f64;
                        let is_target = target_id.map_or(false, |tid| col_idx == tid);

                        // Maintain top k elements without sorting until the end
                        if top_k.len() < k {
                            top_k.push((real_value, sum, col_idx));
                            if is_target {
                                target_idx_in_topk = Some(top_k.len() - 1);
                            }
                            if real_value < min_value || top_k.len() == 1 {
                                min_value = real_value;
                            }
                        } else if is_target {
                            // Target token must be included even if score is low
                            if target_idx_in_topk.is_none() {
                                // Find and replace the minimum element with target
                                min_value = top_k.iter().map(|(v, _, _)| *v).fold(f64::INFINITY, f64::min);
                                if let Some(min_pos) = top_k.iter().position(|(v, _, _)| *v == min_value) {
                                    top_k[min_pos] = (real_value, sum, col_idx);
                                    target_idx_in_topk = Some(min_pos);
                                    // Recalculate min_value
                                    min_value = top_k.iter().map(|(v, _, _)| *v).fold(f64::INFINITY, f64::min);
                                }
                            }
                        } else if real_value > min_value {
                            // Find the minimum element that is not the target
                            min_value = f64::INFINITY;
                            let mut min_pos_candidate = None;

                            for (pos, (v, _, _)) in top_k.iter().enumerate() {
                                if Some(pos) != target_idx_in_topk && *v < min_value {
                                    min_value = *v;
                                    min_pos_candidate = Some(pos);
                                }
                            }

                            if let Some(min_pos) = min_pos_candidate {
                                if real_value > min_value {
                                    top_k[min_pos] = (real_value, sum, col_idx);
                                    // Update min_value excluding target position
                                    min_value = top_k
                                        .iter()
                                        .enumerate()
                                        .filter(|(pos, _)| Some(*pos) != target_idx_in_topk)
                                        .map(|(_, (v, _, _))| *v)
                                        .fold(f64::INFINITY, f64::min);
                                }
                            }
                        }
                    }

                    // Extract values and indices (no need to sort for correctness)
                    let values: Vec<C> = top_k.iter().map(|(_, val, _)| *val).collect();
                    let indices: Vec<usize> = top_k.iter().map(|(_, _, idx)| *idx).collect();

                    // println!("Batch {}, Row {}: Top k indices: {}", batch_idx, row_idx, &indices.len());

                    sample_values.push(values);
                    sample_indices.push(indices);
                }

                (sample_values, sample_indices)
            })
            .collect();

        // Unzip results
        let (values_batch, indices_batch): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        (values_batch, indices_batch)
    }

    pub fn update_parameters(&mut self) {
        // Tied projection update: update the shared table sparsely (only used vocab indices)
        if self.tied_weights.is_some() {
            use crate::neural_networks::utils::adam_w::{calculate_adam_w_f32_sparse, B_1, B_2, EPSILON, WEIGHT_DECAY};

            let mut used_indices = self.collect_used_indices();

            let (mut weight_gradients, mut bias_gradients, total_valid_tokens) = {
                let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
                (gradient.get_gradient_weights(), gradient.get_gradient_bias(), gradient.get_total_valid_tokens())
            };

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

            let denom = r(total_valid_tokens.max(1) as f64);
            weight_gradients = average_matrix_by_scalar(&weight_gradients, denom);
            bias_gradients = average_vector_by_scalar(&bias_gradients, denom);

            clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);
            normalize_gradients(&mut weight_gradients);
            normalize_bias(&mut bias_gradients);

            let learning_rate = self.learning_rate;
            let time_step = self.time_step;

            let tied = self.tied_weights.as_ref().expect("tied weights missing").clone();
            let (rows, cols) = tied.dims();

            let mut prev_m_bias;
            let mut prev_v_bias;
            let mut prev_v_bias_hat;
            let mut prev_m_weights;
            let mut prev_v_weights;
            let mut prev_v_weights_hat;

            if let Some(previous_gradient) = &mut self.previous_gradient {
                prev_m_bias = previous_gradient.get_prev_m_bias();
                prev_v_bias = previous_gradient.get_prev_v_bias();
                prev_v_bias_hat = previous_gradient.get_prev_v_bias_hat();
                prev_m_weights = previous_gradient.get_prev_m_weights();
                prev_v_weights = previous_gradient.get_prev_v_weights();
                prev_v_weights_hat = previous_gradient.get_prev_v_weights_hat();

                let ok_bias = prev_m_bias.len() == self.bias.len() && prev_v_bias.len() == self.bias.len() && prev_v_bias_hat.len() == self.bias.len();
                let ok_w = prev_m_weights.len() == rows && prev_m_weights.first().map(|r| r.len()).unwrap_or(0) == cols;
                if !ok_bias || !ok_w {
                    prev_m_bias = vec![C::new(ZERO, ZERO); self.bias.len()];
                    prev_v_bias = vec![C::new(ZERO, ZERO); self.bias.len()];
                    prev_v_bias_hat = vec![C::new(ZERO, ZERO); self.bias.len()];
                    prev_m_weights = vec![vec![C::new(ZERO, ZERO); cols]; rows];
                    prev_v_weights = vec![vec![C::new(ZERO, ZERO); cols]; rows];
                    prev_v_weights_hat = vec![vec![C::new(ZERO, ZERO); cols]; rows];
                }
            } else {
                prev_m_bias = vec![C::new(ZERO, ZERO); self.bias.len()];
                prev_v_bias = vec![C::new(ZERO, ZERO); self.bias.len()];
                prev_v_bias_hat = vec![C::new(ZERO, ZERO); self.bias.len()];
                prev_m_weights = vec![vec![C::new(ZERO, ZERO); cols]; rows];
                prev_v_weights = vec![vec![C::new(ZERO, ZERO); cols]; rows];
                prev_v_weights_hat = vec![vec![C::new(ZERO, ZERO); cols]; rows];
            }

            // Sparse AdamW for complex bias (only used indices)
            {
                let t = time_step.max(1) as i32;
                let current_lr = r(crate::neural_networks::utils::adam_w::get_current_learning_rate(learning_rate, t as usize));
                let b1 = r(B_1);
                let b2 = r(B_2);
                let one_minus_b1 = r(1.0 - B_1);
                let one_minus_b2 = r(1.0 - B_2);

                for &i in &used_indices {
                    if i >= self.bias.len() || i >= bias_gradients.len() {
                        continue;
                    }
                    let g = bias_gradients[i];

                    prev_m_bias[i] = prev_m_bias[i] * b1 + g * one_minus_b1;
                    let g2 = g.norm_sqr();
                    prev_v_bias[i] = num::Complex::new(prev_v_bias[i].re * b2 + one_minus_b2 * g2, r(0.0));

                    if prev_v_bias_hat[i].re < prev_v_bias[i].re {
                        prev_v_bias_hat[i] = prev_v_bias[i];
                    }

                    let m_hat = prev_m_bias[i] / r(1.0 - B_1.powi(t));
                    let v_hat_re = prev_v_bias_hat[i].re / r(1.0 - B_2.powi(t));
                    let denom = v_hat_re.sqrt() + r(EPSILON);
                    let adaptive_step = (m_hat * current_lr) / denom;
                    self.bias[i] = self.bias[i] - self.bias[i] * (current_lr * r(WEIGHT_DECAY)) - adaptive_step;
                }
            }

            // Sparse AdamW for tied f32 weights
            {
                let mut w = tied.write();
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
            }

            {
                let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
                gradient.set_prev_m_bias(prev_m_bias);
                gradient.set_prev_v_bias(prev_v_bias);
                gradient.set_prev_v_bias_hat(prev_v_bias_hat);
                gradient.set_prev_m_weights(prev_m_weights);
                gradient.set_prev_v_weights(prev_v_weights);
                gradient.set_prev_v_weights_hat(prev_v_weights_hat);
                gradient.set_gradient_weights(weight_gradients.clone());
                gradient.set_gradient_bias(bias_gradients.clone());
                self.previous_gradient = Some(gradient.clone());
            }

            self.gradient = None;
            return;
        }

        {
            let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
            let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
            let total_valid_tokens = gradient.get_total_valid_tokens();

            weight_gradients = average_matrix_by_scalar(&weight_gradients, r(total_valid_tokens as f64));
            bias_gradients = average_vector_by_scalar(&bias_gradients, r(total_valid_tokens as f64));

            clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

            normalize_gradients(&mut weight_gradients);
            normalize_bias(&mut bias_gradients);

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
                    vec![C::new(ZERO, ZERO); self.bias.len()],
                    vec![C::new(ZERO, ZERO); self.bias.len()],
                    vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                    vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                    vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                    vec![C::new(ZERO, ZERO); self.bias.len()],
                )
            };

            calculate_adam_w_bias(
                &mut self.bias,
                &bias_gradients,
                &mut prev_m_bias,
                &mut prev_v_bias,
                &mut prev_v_bias_hat,
                learning_rate,
                time_step,
            );
            calculate_adam_w(
                &mut self.weights,
                &weight_gradients,
                &mut prev_m_weights,
                &mut prev_v_weights,
                &mut prev_v_weights_hat,
                learning_rate,
                time_step,
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
        }

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

    fn collect_used_indices(&self) -> Vec<usize> {
        let Some(output_indices_batch) = &self.output_indices else {
            return vec![];
        };

        let mut indices_set = std::collections::HashSet::new();
        for batch_indices in output_indices_batch {
            for seq_indices in batch_indices {
                for &idx in seq_indices {
                    indices_set.insert(idx);
                }
            }
        }
        let mut indices: Vec<usize> = indices_set.into_iter().collect();
        indices.sort_unstable();
        indices
    }

    fn multiply_highest_k_per_row_tied(&self, input_batch: &Vec<Vec<Vec<C>>>, layer_input: &LayerInput) -> (Vec<Vec<Vec<C>>>, Vec<Vec<Vec<usize>>>) {
        let tied = self.tied_weights.as_ref().expect("tied weights missing");
        let weights_guard = tied.read();

        let target_batch: &Vec<Vec<u32>> = &layer_input.get_target_batch_ids();
        let k: usize = layer_input.get_top_k_size();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();

        let num_output_cols = weights_guard.len();

        let results: Vec<_> = input_batch
            .par_iter()
            .enumerate()
            .map(|(batch_idx, input_sample)| {
                let mut sample_values: Vec<Vec<C>> = vec![];
                let mut sample_indices: Vec<Vec<usize>> = vec![];

                let offset = if batch_idx < target_batch.len() && !target_batch[batch_idx].is_empty() {
                    let seq_len_unpadded = padding_mask_batch[batch_idx].iter().filter(|&&x| x != 0).count();
                    seq_len_unpadded.saturating_sub(target_batch[batch_idx].len())
                } else {
                    usize::MAX
                };

                for (row_idx, input_row) in input_sample.iter().enumerate() {
                    let target_id = if batch_idx < target_batch.len() && !target_batch[batch_idx].is_empty() && offset != usize::MAX && padding_mask_batch[batch_idx][row_idx] != 0 && row_idx >= offset
                    {
                        let target_idx = row_idx - offset;
                        if target_idx < target_batch[batch_idx].len() {
                            Some(target_batch[batch_idx][target_idx] as usize)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let mut top_k: Vec<(f64, C, usize)> = Vec::with_capacity(k + 1);
                    let mut min_value = f64::NEG_INFINITY;
                    let mut target_idx_in_topk: Option<usize> = None;

                    for col_idx in 0..num_output_cols {
                        let w_row = &weights_guard[col_idx];
                        let mut sum = C::new(ZERO, ZERO);
                        let n = input_row.len().min(w_row.len());
                        for i in 0..n {
                            sum += input_row[i] * C::new(r(w_row[i] as f64), ZERO);
                        }
                        if col_idx < self.bias.len() {
                            sum += self.bias[col_idx];
                        }

                        let real_value = sum.re as f64;
                        let is_target = target_id.map_or(false, |tid| col_idx == tid);

                        if top_k.len() < k {
                            top_k.push((real_value, sum, col_idx));
                            if is_target {
                                target_idx_in_topk = Some(top_k.len() - 1);
                            }
                            if real_value < min_value || top_k.len() == 1 {
                                min_value = real_value;
                            }
                        } else if is_target {
                            if target_idx_in_topk.is_none() {
                                min_value = top_k.iter().map(|(v, _, _)| *v).fold(f64::INFINITY, f64::min);
                                if let Some(min_pos) = top_k.iter().position(|(v, _, _)| *v == min_value) {
                                    top_k[min_pos] = (real_value, sum, col_idx);
                                    target_idx_in_topk = Some(min_pos);
                                    min_value = top_k.iter().map(|(v, _, _)| *v).fold(f64::INFINITY, f64::min);
                                }
                            }
                        } else if real_value > min_value {
                            min_value = f64::INFINITY;
                            let mut min_pos_candidate = None;
                            for (pos, (v, _, _)) in top_k.iter().enumerate() {
                                if Some(pos) != target_idx_in_topk && *v < min_value {
                                    min_value = *v;
                                    min_pos_candidate = Some(pos);
                                }
                            }
                            if let Some(min_pos) = min_pos_candidate {
                                if real_value > min_value {
                                    top_k[min_pos] = (real_value, sum, col_idx);
                                    min_value = top_k
                                        .iter()
                                        .enumerate()
                                        .filter(|(pos, _)| Some(*pos) != target_idx_in_topk)
                                        .map(|(_, (v, _, _))| *v)
                                        .fold(f64::INFINITY, f64::min);
                                }
                            }
                        }
                    }

                    let values: Vec<C> = top_k.iter().map(|(_, val, _)| *val).collect();
                    let indices: Vec<usize> = top_k.iter().map(|(_, _, idx)| *idx).collect();
                    sample_values.push(values);
                    sample_indices.push(indices);
                }

                (sample_values, sample_indices)
            })
            .collect();

        let (values_batch, indices_batch): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        (values_batch, indices_batch)
    }
}
