use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::layer::LayerEnum,
    network_types::{transformer::transformer_updater::VERBOSE, wavelet_discrete_layer::DiscreteWaveletLayer},
    utils::{
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        matrix::{add_matrix_2d_c, add_matrix_3d, add_vector, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose, multiply_complex},
        weights_initializer::{initialize_weights_complex, initialize_weights_complex_only_real},
    },
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub learning_rate: f64,
    pub bias: Vec<Complex<f64>>,
    pub smoothing: f64,
    pub ema: f64,
    pub discrete_wavelet_layer: Option<DiscreteWaveletLayer>,
    pub norm_layer: Option<LayerEnum>,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub is_complex: bool,

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

impl LinearLayer {
    pub fn new(learning_rate: f64, rows: usize, cols: usize, is_complex: bool) -> Self {
        let mut weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

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
            is_complex,
            output_indices: None,
        }
    }
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = input.get_input_batch();

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());
        let mut output_indices: Vec<Vec<Vec<usize>>> = vec![];

        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = input_batch.clone();
        let mut layer_input = input.clone();
        layer_input.set_input_batch(output_batch.clone());

        let start = std::time::Instant::now();

        if self.is_complex {
            output_batch = output_batch
                .par_iter() // Use a parallel iterator to process inputs in parallel
                .map(|input| {
                    let mut output = multiply_complex(input, &self.weights);

                    // Add the bias vector
                    add_vector(&mut output, &self.bias);
                    output
                })
                .collect();
        } else {
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
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let mut gradient = Gradient::new_default();

        let previous_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.bias.len()]; input_batch.len()];
        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        if self.is_complex {
            for (batch_ind, (input_sample, previous_gradient)) in input_batch.iter().zip(previous_gradient_input_batch).enumerate() {
                weight_gradients[batch_ind] = multiply_complex(&conjugate_transpose(&input_sample), &previous_gradient);
                //Accumulate gradients for biases
                for grad_row in previous_gradient.iter() {
                    for (k, grad_val) in grad_row.iter().enumerate() {
                        bias_gradients[batch_ind][k] += grad_val;
                    }
                }

                gradient_input_batch[batch_ind] = multiply_complex(&previous_gradient, &conjugate_transpose(&self.weights));
            }
        } else {
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

        let num_output_cols = self.weights[0].len();

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
                    let mut target_value: Option<(f64, Complex<f64>, usize)> = None;

                    // Compute each output element on the fly
                    for col_idx in 0..num_output_cols {
                        // Compute dot product: input_row · weights[:, col_idx]
                        let mut sum = Complex::new(0.0, 0.0);
                        for (i, &input_val) in input_row.iter().enumerate() {
                            sum += input_val * self.weights[i][col_idx];
                        }
                        // Add bias
                        sum += self.bias[col_idx];

                        let real_value = sum.re;

                        // Check if this is the target token - save it but still process normally
                        if let Some(target_token_id) = target_id {
                            if col_idx == target_token_id {
                                target_value = Some((real_value, sum, col_idx));
                            }
                        }

                        // Maintain top k elements without sorting until the end
                        if top_k.len() < k {
                            top_k.push((real_value, sum, col_idx));
                            if real_value < min_value || top_k.len() == 1 {
                                min_value = real_value;
                            }
                        } else if real_value > min_value {
                            // Find and replace the minimum element
                            if let Some(min_pos) = top_k.iter().position(|(v, _, _)| *v == min_value) {
                                top_k[min_pos] = (real_value, sum, col_idx);
                                // Update min_value
                                min_value = top_k.iter().map(|(v, _, _)| *v).fold(f64::INFINITY, f64::min);
                            }
                        }
                    }

                    // Always include target token if it exists
                    if let Some(target) = target_value {
                        let target_in_topk = top_k.iter().any(|(_, _, idx)| *idx == target.2);
                        if !target_in_topk {
                            // Add target token, replacing the smallest element if top_k is full
                            if top_k.len() >= k {
                                // Find and replace the minimum element
                                min_value = top_k.iter().map(|(v, _, _)| *v).fold(f64::INFINITY, f64::min);
                                if let Some(min_pos) = top_k.iter().position(|(v, _, _)| *v == min_value) {
                                    top_k[min_pos] = target;
                                }
                            } else {
                                // Room available, just add
                                top_k.push(target);
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
        // prev_m_bias = average_gradient_polar_1d(&previous_gradient.get_prev_m_bias(), batch_size);
        // prev_v_bias = average_gradient_polar_1d(&previous_gradient.get_prev_v_bias(), batch_size);
        // prev_m_weights = average_gradient_polar(&previous_gradient.get_prev_m_weights(), batch_size);
        // prev_v_weights = average_gradient_polar(&previous_gradient.get_prev_v_weights(), batch_size);
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
