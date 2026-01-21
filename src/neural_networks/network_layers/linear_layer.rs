use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::layer::LayerEnum,
    network_types::transformer::transformer_updater::VERBOSE,
    utils::{
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        dtype::{r, C, ONE, ZERO},
        matrix::{
            add_vector_rm, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose, conjugate_transpose_rm, conjugate_transpose_to_rm,
            multiply_complex, multiply_complex_rm, RowMajorMatrix,
        },
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
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub output_indices: Option<Vec<Vec<Vec<usize>>>>,

    weights_rm_cache: Option<RowMajorMatrix<C>>,
    weights_h_rm_cache: Option<RowMajorMatrix<C>>,
    #[serde(skip)]
    weights_cache_outer_ptr: usize,
    #[serde(skip)]
    weights_cache_row0_ptr: usize,
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
            input_batch_rm: None,
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

            weights_rm_cache: None,
            weights_h_rm_cache: None,
            weights_cache_outer_ptr: 0,
            weights_cache_row0_ptr: 0,
        }
    }

    fn current_weights_ptrs(&self) -> (usize, usize) {
        let outer = self.weights.as_ptr() as usize;
        let row0 = self.weights.get(0).map(|r| r.as_ptr() as usize).unwrap_or(0);
        (outer, row0)
    }

    fn invalidate_weights_cache(&mut self) {
        self.weights_rm_cache = None;
        self.weights_h_rm_cache = None;
        self.weights_cache_outer_ptr = 0;
        self.weights_cache_row0_ptr = 0;
    }

    fn ensure_weights_cache(&mut self) {
        let (outer, row0) = self.current_weights_ptrs();

        let needs_rebuild = self.weights_rm_cache.is_none()
            || self.weights_h_rm_cache.is_none()
            || self.weights_cache_outer_ptr != outer
            || self.weights_cache_row0_ptr != row0
            || self.weights_rm_cache.as_ref().is_some_and(|w| w.rows != self.weights.len() || w.cols != self.weights[0].len());

        if !needs_rebuild {
            return;
        }

        let w_rm = RowMajorMatrix::from_rows(&self.weights);
        let w_h_rm = conjugate_transpose_to_rm(&self.weights);

        self.weights_rm_cache = Some(w_rm);
        self.weights_h_rm_cache = Some(w_h_rm);
        self.weights_cache_outer_ptr = outer;
        self.weights_cache_row0_ptr = row0;
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<C>>> = input.get_input_batch();
        let input_batch_rm_ref = input.get_input_batch_rm_ref();

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        // Store whichever representation was provided so backward can avoid reshaping.
        if !input_batch.is_empty() {
            self.input_batch = Some(input_batch.clone());
        } else {
            self.input_batch = None;
        }
        if let Some(rm) = input_batch_rm_ref {
            if !rm.is_empty() {
                self.input_batch_rm = Some(rm.to_vec());
            }
        } else {
            self.input_batch_rm = None;
        }

        let mut output_indices: Vec<Vec<Vec<usize>>> = vec![];

        let start = std::time::Instant::now();

        let mut output_batch: Vec<Vec<Vec<C>>> = vec![];

        if self.is_complex {
            self.ensure_weights_cache();
            let weights_rm = self.weights_rm_cache.as_ref().expect("weights cache missing");

            let output_batch_rm: Vec<RowMajorMatrix<C>> = if let Some(input_batch_rm) = input_batch_rm_ref {
                input_batch_rm
                    .par_iter()
                    .map(|input_rm| {
                        let mut out_rm = multiply_complex_rm(input_rm, weights_rm);
                        add_vector_rm(&mut out_rm, &self.bias);
                        out_rm
                    })
                    .collect()
            } else {
                input_batch
                    .par_iter()
                    .map(|input_rows| {
                        let input_rm = RowMajorMatrix::from_rows(input_rows);
                        let mut out_rm = multiply_complex_rm(&input_rm, weights_rm);
                        add_vector_rm(&mut out_rm, &self.bias);
                        out_rm
                    })
                    .collect()
            };

            // Preserve legacy output when the legacy input representation is used.
            // If the caller provides only row-major inputs, we avoid allocating a nested Vec<Vec<...>> output.
            let need_legacy_output = !input_batch.is_empty();
            if need_legacy_output {
                output_batch = output_batch_rm.iter().map(|m| m.to_rows()).collect();
            }

            let mut layer_output = LayerOutput::new_default();
            if need_legacy_output {
                layer_output.set_output_batch(output_batch);
            }
            layer_output.set_output_batch_rm(output_batch_rm);
            layer_output.set_output_indices(output_indices);
            self.output_indices = Some(layer_output.get_output_indices());
            return layer_output;
        } else {
            let mut layer_input = input.clone();
            layer_input.set_input_batch(input_batch.clone());
            (output_batch, output_indices) = self.mutliply_hightest_k_per_row(&input_batch, &layer_input);
        }

        if VERBOSE && !self.is_complex {
            println!("Linear layer complex matmul time for batch size {}: {}", self.batch_size, start.elapsed().as_secs_f64());
        }
        // println!("Output batch size in linear layer after dwt inverse:  {} {} {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output.set_output_indices(output_indices);

        self.output_indices = Some(layer_output.get_output_indices());

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        if self.is_complex {
            self.ensure_weights_cache();
        }

        let input_batch_vec = self.input_batch.as_ref();
        let input_batch_rm = self.input_batch_rm.as_ref();
        let mut gradient = Gradient::new_default();

        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let previous_gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient.get_gradient_input_batch();
        let previous_gradient_rm_ref = previous_gradient.get_gradient_input_batch_rm_ref();

        if input_batch_vec.is_none() && input_batch_rm.is_none() {
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        // Initialize gradients for weights and biases
        let batch_len = if let Some(b) = input_batch_vec { b.len() } else { input_batch_rm.expect("rm batch").len() };

        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];
        let mut gradient_input_batch: Vec<Vec<Vec<C>>> = vec![];

        let mut gradient_input_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_len);

        if self.is_complex {
            let weights_h_rm = self.weights_h_rm_cache.as_ref().expect("weights^H cache missing");

            for batch_ind in 0..batch_len {
                let input_rm: RowMajorMatrix<C> = if let Some(rm_batch) = input_batch_rm {
                    rm_batch[batch_ind].clone()
                } else {
                    let input_sample = &input_batch_vec.expect("vec batch")[batch_ind];
                    RowMajorMatrix::from_rows(input_sample)
                };

                let grad_rm: RowMajorMatrix<C> = if let Some(grads_rm) = previous_gradient_rm_ref {
                    grads_rm[batch_ind].clone()
                } else {
                    RowMajorMatrix::from_rows(&previous_gradient_input_batch[batch_ind])
                };

                let input_h_rm = conjugate_transpose_rm(&input_rm);
                let wgrad_rm = multiply_complex_rm(&input_h_rm, &grad_rm);
                weight_gradients[batch_ind] = wgrad_rm.to_rows();

                // Bias gradients: sum over rows of grad
                for r in 0..grad_rm.rows {
                    let row = grad_rm.row_range(r);
                    for c in 0..grad_rm.cols {
                        bias_gradients[batch_ind][c] += grad_rm.data[row.start + c];
                    }
                }

                let gx_rm = multiply_complex_rm(&grad_rm, weights_h_rm);
                gradient_input_batch_rm.push(gx_rm);
            }
        } else {
            // rows are sparse with only top k values
            let output_indices_batch = self.output_indices.as_ref().expect("Output indices missing in linear layer backward pass");

            let input_batch = input_batch_vec.expect("Input batch is missing in sparse linear layer backward");
            gradient_input_batch = vec![vec![vec![C::new(ZERO, ZERO); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

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
        }

        // if self.gradient.is_some() {
        //     let previous_gradient = self.gradient.as_ref().expect("");
        //     weight_gradients = add_matrix_3d(&weight_gradients, &previous_gradient.get_gradient_weight_batch());
        //     bias_gradients = add_matrix_2d_c(&bias_gradients, &previous_gradient.get_gradient_bias_batch());
        // }
        //  println!("batch size in linear layer: {}", self.batch_size);

        // If this layer ran in legacy Vec<Vec<...>> mode, keep emitting legacy gradients.
        // If it ran in RM-only mode (input_batch_rm set and input_batch absent), emit RM gradients only.
        let legacy_mode = self.input_batch.is_some();
        if legacy_mode {
            if self.is_complex {
                gradient_input_batch = gradient_input_batch_rm.iter().map(|m| m.to_rows()).collect();
            }
            gradient.set_gradient_input_batch(gradient_input_batch.clone());
        }
        gradient.set_gradient_input_batch_rm(gradient_input_batch_rm);
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
        {
            let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
            let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
            let total_valid_tokens = gradient.get_total_valid_tokens();

            weight_gradients = average_matrix_by_scalar(&weight_gradients, r(total_valid_tokens as f64));
            bias_gradients = average_vector_by_scalar(&bias_gradients, r(total_valid_tokens as f64));

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
                &gradient.get_gradient_bias(),
                &mut prev_m_bias,
                &mut prev_v_bias,
                &mut prev_v_bias_hat,
                learning_rate,
                time_step,
            );
            calculate_adam_w(
                &mut self.weights,
                &gradient.get_gradient_weights(),
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

        // We updated weights in-place; cached row-major views are now stale.
        self.invalidate_weights_cache();
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
}
