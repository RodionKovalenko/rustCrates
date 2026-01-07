use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::layer::LayerEnum,
    network_types::{transformer::transformer_updater::VERBOSE, wavelet_discrete_layer::DiscreteWaveletLayer},
    optimization::k_means_clustering::{kmeans, query_candidates},
    utils::{
        adam_w::{calculate_adam_w_bias_f32, calculate_adam_w_f32},
        low_rank_approx::transpose,
        matrix::{add_matrix_2d_c, add_matrix_3d, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, multiply_complex, multiply_complex_with_f32},
        weights_initializer::initialize_weights_f32,
    },
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLinearLayer {
    pub weights: Vec<Vec<f32>>,
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

    #[serde(skip)]
    pub gradients: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub gradients_bias: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub output_indices: Option<Vec<Vec<Vec<usize>>>>,
}

impl SparseLinearLayer {
    pub fn new(learning_rate: f64, embedding_d: usize, vocab_size: usize) -> Self {
        // Weights are used not as weights matrix, but as embedding matrix N x D
        // whre N is vocab size, D is embedding size
        let mut weights: Vec<Vec<f32>> = vec![vec![0.0; embedding_d]; vocab_size];
        let bias: Vec<f32> = vec![1.0; vocab_size];

        initialize_weights_f32(vocab_size, embedding_d, &mut weights);

        let n_clusters: usize = 50280 / 16;
        let threshold = 0.001;

        let (centroids, assignments, cluster_to_tokens) = kmeans(&mut weights, n_clusters, 100, threshold);

        Self {
            weights,
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
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            output_indices: None,
        }
    }
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = input.get_input_batch();

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());

        let start = std::time::Instant::now();

        let (output_batch, output_indices) = self.mutliply_hightest_k_per_row(&input_batch, &input);

        if VERBOSE {
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
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let mut gradient = Gradient::new_default();

        let previous_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.bias.len()]; input_batch.len()];
        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        // rows are sparse with only top k values
        let output_indices_batch = self.output_indices.as_ref().expect("Output indices missing in linear layer backward pass");

        for batch_idx in 0..input_batch.len() {
            let input_sample = &input_batch[batch_idx];
            let sparse_grad = &previous_gradient_input_batch[batch_idx];
            let indices = &output_indices_batch[batch_idx];

            // Reconstruct full gradient from sparse gradients using indices
            let num_output_cols = self.weights[0].len();
            let mut full_gradient: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); num_output_cols]; input_sample.len()];

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
            weight_gradients[batch_idx] = multiply_complex(&transpose(&input_sample), &full_gradient);

            // Accumulate gradients for biases
            for grad_row in full_gradient.iter() {
                for (k, grad_val) in grad_row.iter().enumerate() {
                    bias_gradients[batch_idx][k] += grad_val;
                }
            }

            // Compute input gradients: gradient * weights^H
            gradient_input_batch[batch_idx] = multiply_complex_with_f32(&full_gradient, &self.weights);
        }

        gradient.set_gradient_input_batch(gradient_input_batch.clone());

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            weight_gradients = add_matrix_3d(&weight_gradients, &previous_gradient.get_gradient_weight_batch());
            bias_gradients = add_matrix_2d_c(&bias_gradients, &previous_gradient.get_gradient_bias_batch());
        }
        //  println!("batch size in linear layer: {}", self.batch_size);

        gradient.set_gradient_input_batch(gradient_input_batch.clone());
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());

        gradient
    }

    // multiply normally, select highest k per row, return k highest values per row and original indices
    pub fn mutliply_hightest_k_per_row(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, layer_input: &LayerInput) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<Vec<usize>>>) {
        let target_batch: &Vec<Vec<u32>> = &layer_input.get_target_batch_ids();
        let k: usize = layer_input.get_top_k_size();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();

        // Parallelize batch processing
        let results: Vec<_> = input_batch
            .par_iter()
            .enumerate()
            .map(|(batch_idx, input_sample)| {
                let mut sample_values: Vec<Vec<Complex<f64>>> = vec![];
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
                    let mut top_k: Vec<(f64, Complex<f64>, usize)> = Vec::with_capacity(k + 1);
                    let mut min_value = f64::NEG_INFINITY;
                    let mut target_idx_in_topk: Option<usize> = None; // Track position of target in top_k

                    let mut input_f32 = input_row.iter().map(|c| c.re as f32).collect::<Vec<f32>>();

                    let mut selected_indices = query_candidates(
                        &mut input_f32,
                        &self.centroids,
                        &self.cluster_to_tokens,
                        4, // select top 4 clusters
                    );

                    // include target index if not already included
                    if let Some(tid) = target_id {
                        if !selected_indices.contains(&tid) {
                            // Replace the last index with target id
                            if selected_indices.len() < k {
                                // If we have space, just add it
                                selected_indices.push(tid);
                            } else {
                                let selected_indices_len: usize = selected_indices.len();
                                // Replace the last one
                                selected_indices[selected_indices_len - 1] = tid;
                            }
                        }
                    }

                    // Compute each output element on the fly
                    for &col_idx in selected_indices.iter() {
                        // Compute dot product: input_row · weights[:, col_idx]
                        let mut sum = Complex::new(0.0, 0.0);
                        for (i, &input_val) in input_row.iter().enumerate() {
                            sum += input_val.re * self.weights[col_idx][i] as f64;
                        }
                        // Add bias
                        sum += self.bias[col_idx] as f64;

                        let real_value = sum.re;
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
                    let values: Vec<Complex<f64>> = top_k.iter().map(|(_, val, _)| *val).collect();
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
        calculate_adam_w_bias_f32(
            &mut self.bias,
            &gradient.get_gradient_bias(),
            &mut prev_m_bias,
            &mut prev_v_bias,
            &mut prev_v_bias_hat,
            learning_rate,
            time_step,
        );
        calculate_adam_w_f32(
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
}
