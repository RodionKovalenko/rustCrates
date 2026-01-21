use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::linear_layer::LinearLayer,
    utils::{
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        array_splitting::split_sizes,
        dtype::{r, C},
        matrix::{average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, normalize_bias, normalize_gradients},
        weights_initializer::initialize_weights_complex,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLinearLayer {
    pub layers: Vec<LinearLayer>,
    pub learning_rate: f64,
    pub col_ranges: Vec<(usize, usize)>,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch: Option<Arc<Vec<Vec<Vec<C>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl MultiLinearLayer {
    pub fn new(learning_rate: f64, rows: usize, cols: usize, num_layers: usize) -> Self {
        let mut layers: Vec<LinearLayer> = Vec::with_capacity(num_layers);
        let col_chunks: Vec<usize> = split_sizes(cols, num_layers);
        let mut col_ranges: Vec<(usize, usize)> = Vec::with_capacity(num_layers);

        // Initialize weights for the entire matrix first
        let mut weights = vec![vec![Complex::new(r(0.0), r(0.0)); cols]; rows];
        initialize_weights_complex(rows, cols, &mut weights);

        // Initialize bias for the entire output
        let bias = vec![Complex::new(r(0.0), r(0.0)); cols];

        let mut start_col = 0;
        for &chunk_size in &col_chunks {
            let end_col = start_col + chunk_size;
            col_ranges.push((start_col, end_col));

            // Create linear layer with proper initialization
            let mut linear_layer = LinearLayer::new(learning_rate, rows, chunk_size, true);

            // Extract weights and bias for this chunk
            linear_layer.weights = weights.iter().map(|row| row[start_col..end_col].to_vec()).collect();
            linear_layer.bias = bias[start_col..end_col].to_vec();

            layers.push(linear_layer);
            start_col = end_col;
        }

        Self {
            layers,
            learning_rate,
            col_ranges,
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            smoothing: 0.99,
            ema: 0.0,
            time_step: 0,
            batch_size: 0,
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    pub fn get_combined_weights(&self) -> (Vec<Vec<C>>, Vec<C>) {
        let rows = self.layers[0].weights.len();
        let total_cols: usize = self.layers.iter().map(|layer| layer.weights[0].len()).sum();

        let mut combined_weights = vec![vec![Complex::new(r(0.0), r(0.0)); total_cols]; rows];
        let mut combined_bias = vec![Complex::new(r(0.0), r(0.0)); total_cols];

        let mut col_offset = 0;
        for layer in &self.layers {
            let chunk_size = layer.weights[0].len();

            // Copy weights
            for (row_idx, row) in layer.weights.iter().enumerate() {
                combined_weights[row_idx][col_offset..col_offset + chunk_size].copy_from_slice(row);
            }

            // Copy bias
            combined_bias[col_offset..col_offset + chunk_size].copy_from_slice(&layer.bias);

            col_offset += chunk_size;
        }

        (combined_weights, combined_bias)
    }

    pub fn get_combined_gradients(&self) -> (Vec<Vec<C>>, Vec<C>) {
        let rows = self.layers[0].weights.len();
        let total_cols: usize = self.layers.iter().map(|layer| layer.weights[0].len()).sum();

        let mut combined_weights_gradients = vec![vec![Complex::new(r(0.0), r(0.0)); total_cols]; rows];
        let mut combined_bias_gradients = vec![Complex::new(r(0.0), r(0.0)); total_cols];

        let mut col_offset = 0;
        for layer in &self.layers {
            let chunk_size = layer.weights[0].len();

            let layer_weight_gradients = layer
                .gradient
                .as_ref()
                .map(|g| g.get_gradient_weights())
                .unwrap_or_else(|| vec![vec![Complex::new(r(0.0), r(0.0)); chunk_size]; rows]);
            let layer_bias_gradients = layer.gradient.as_ref().map(|g| g.get_gradient_bias()).unwrap_or_else(|| vec![Complex::new(r(0.0), r(0.0)); chunk_size]);

            // Copy weight gradients
            for (row_idx, row) in layer_weight_gradients.iter().enumerate() {
                combined_weights_gradients[row_idx][col_offset..col_offset + chunk_size].copy_from_slice(row);
            }

            // Copy bias gradients
            combined_bias_gradients[col_offset..col_offset + chunk_size].copy_from_slice(&layer_bias_gradients);

            col_offset += chunk_size;
        }

        normalize_gradients(&mut combined_weights_gradients);
        normalize_bias(&mut combined_bias_gradients);

        (combined_weights_gradients, combined_bias_gradients)
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch_vec = input.get_input_batch();
        let rm_ref = input.get_input_batch_rm_ref().filter(|rm| !rm.is_empty());
        if input_batch_vec.is_empty() && rm_ref.is_some() {
            panic!("MultiLinearLayer received RM-only input; use a RM-specific MultiLinear layer");
        }

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let input_batch = Arc::new(input_batch_vec);
        self.input_batch = Some(input_batch.clone());

        let batch_size = input_batch.len();
        let seq_len = input_batch.first().map(|b| b.len()).unwrap_or(0);

        let total_cols: usize = self.layers.iter().map(|layer| layer.weights[0].len()).sum();

        let mut sub_input = input.clone();
        sub_input.set_input_batch(input_batch.as_ref().clone());

        let output_chunks: Vec<Vec<Vec<Vec<C>>>> = self.layers.iter_mut().map(|lin_layer| lin_layer.forward(&sub_input).get_output_batch()).collect();

        let mut output_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(r(0.0), r(0.0)); total_cols]; seq_len]; batch_size];
        for b in 0..batch_size {
            for t in 0..seq_len {
                let mut offset = 0;
                for chunk in &output_chunks {
                    let row = &chunk[b][t];
                    let chunk_cols = row.len();
                    output_batch[b][t][offset..offset + chunk_cols].copy_from_slice(row);
                    offset += chunk_cols;
                }
            }
        }

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output.set_output_batch_rm(vec![]);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let previous_input_gradient = previous_gradient.get_gradient_input_batch();
        let prev_rm_ref = previous_gradient.get_gradient_input_batch_rm_ref().filter(|rm| !rm.is_empty());
        if previous_input_gradient.is_empty() && prev_rm_ref.is_some() {
            panic!("MultiLinearLayer received RM-only gradient; use a RM-specific MultiLinear layer");
        }

        let input_batch = match self.input_batch.as_ref() {
            Some(b) if !b.is_empty() => b,
            _ => {
                let mut gradient = Gradient::new_default();
                gradient.set_gradient_input_batch(vec![]);
                gradient.set_gradient_input_batch_rm(vec![]);
                gradient.set_gradient_weight_batch(vec![]);
                gradient.set_gradient_weight_2_batch(vec![]);
                gradient.set_total_valid_tokens(total_valid_tokens);
                self.gradient = Some(gradient.clone());
                return gradient;
            }
        };

        let batch_size = input_batch.len();
        let seq_len = input_batch.first().map(|b| b.len()).unwrap_or(0);
        let feature_dim = input_batch.first().map(|b| b.first().map(|r| r.len()).unwrap_or(0)).unwrap_or(0);

        let mut summed_dx: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(r(0.0), r(0.0)); feature_dim]; seq_len]; batch_size];

        for (layer_ind, lin_layer) in self.layers.iter_mut().enumerate() {
            let (start_col, end_col) = self.col_ranges[layer_ind];

            let prev_chunk: Vec<Vec<Vec<C>>> = previous_input_gradient
                .iter()
                .map(|batch_rows| batch_rows.iter().map(|row| row[start_col..end_col].to_vec()).collect())
                .collect();

            let mut grad_sliced = Gradient::new_default();
            grad_sliced.set_gradient_input_batch(prev_chunk);
            grad_sliced.set_gradient_input_batch_rm(vec![]);
            grad_sliced.set_total_valid_tokens(total_valid_tokens);

            let dx_chunk = lin_layer.backward(&grad_sliced).get_gradient_input_batch();
            for b in 0..batch_size {
                for t in 0..seq_len {
                    for f in 0..feature_dim {
                        summed_dx[b][t][f] += dx_chunk[b][t][f];
                    }
                }
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(summed_dx);
        gradient.set_gradient_input_batch_rm(vec![]);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        // First, ensure all sublayers have computed their gradients
        for layer in &mut self.layers {
            if layer.gradient.is_none() {
                // Skip update if no gradient computed
                return;
            }
        }

        let (mut weight_gradients, mut bias_gradients) = self.get_combined_gradients();
        let (mut combined_weights, mut combined_bias) = self.get_combined_weights();

        // Get total valid tokens from gradient
        let gradient = self.gradient.as_mut().expect("No gradients available");
        let total_valid_tokens = gradient.get_total_valid_tokens().max(1);

        weight_gradients = average_matrix_by_scalar(&weight_gradients, r(total_valid_tokens as f64));
        bias_gradients = average_vector_by_scalar(&bias_gradients, r(total_valid_tokens as f64));

        // Clip gradients
        clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        // Get previous optimizer states
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
            (
                vec![Complex::new(r(0.0), r(0.0)); bias_gradients.len()],
                vec![Complex::new(r(0.0), r(0.0)); bias_gradients.len()],
                vec![vec![Complex::new(r(0.0), r(0.0)); weight_gradients[0].len()]; weight_gradients.len()],
                vec![vec![Complex::new(r(0.0), r(0.0)); weight_gradients[0].len()]; weight_gradients.len()],
                vec![vec![Complex::new(r(0.0), r(0.0)); weight_gradients[0].len()]; weight_gradients.len()],
                vec![Complex::new(r(0.0), r(0.0)); bias_gradients.len()],
            )
        };

        // Apply AdamW updates
        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        calculate_adam_w_bias(&mut combined_bias, &bias_gradients, &mut prev_m_bias, &mut prev_v_bias, &mut prev_v_bias_hat, learning_rate, time_step);
        calculate_adam_w(
            &mut combined_weights,
            &weight_gradients,
            &mut prev_m_weights,
            &mut prev_v_weights,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
        );

        // Store updated optimizer states
        if let Some(gradient) = &mut self.gradient {
            gradient.set_prev_m_bias(prev_m_bias.clone());
            gradient.set_prev_v_bias(prev_v_bias.clone());
            gradient.set_prev_m_weights(prev_m_weights.clone());
            gradient.set_prev_v_weights(prev_v_weights.clone());
            gradient.set_prev_v_weights_hat(prev_v_weights_hat.clone());
            gradient.set_prev_v_bias_hat(prev_v_bias_hat.clone());
            gradient.set_gradient_weights(weight_gradients.clone());
            gradient.set_gradient_bias(bias_gradients.clone());
            self.previous_gradient = Some(gradient.clone());
        }

        // Distribute updated weights and biases back to sublayers
        let mut col_offset = 0;
        for layer in &mut self.layers {
            let chunk_size = layer.weights[0].len();

            // Update weights
            for (row_idx, row) in layer.weights.iter_mut().enumerate() {
                row.copy_from_slice(&combined_weights[row_idx][col_offset..col_offset + chunk_size]);
            }

            // Update bias
            layer.bias.copy_from_slice(&combined_bias[col_offset..col_offset + chunk_size]);

            // Update sublayer optimizer states if needed
            if let Some(layer_grad) = &mut layer.gradient {
                // Extract the relevant portion of gradients for this sublayer
                let layer_weight_grads: Vec<Vec<C>> = weight_gradients.iter().map(|row| row[col_offset..col_offset + chunk_size].to_vec()).collect();
                let layer_bias_grads = bias_gradients[col_offset..col_offset + chunk_size].to_vec();

                layer_grad.set_gradient_weights(layer_weight_grads);
                layer_grad.set_gradient_bias(layer_bias_grads);

                // Set optimizer states for sublayer
                let layer_prev_m_weights: Vec<Vec<C>> = prev_m_weights.iter().map(|row| row[col_offset..col_offset + chunk_size].to_vec()).collect();
                let layer_prev_v_weights: Vec<Vec<C>> = prev_v_weights.iter().map(|row| row[col_offset..col_offset + chunk_size].to_vec()).collect();
                let layer_prev_m_bias = prev_m_bias[col_offset..col_offset + chunk_size].to_vec();
                let layer_prev_v_bias = prev_v_bias[col_offset..col_offset + chunk_size].to_vec();

                layer_grad.set_prev_m_weights(layer_prev_m_weights);
                layer_grad.set_prev_v_weights(layer_prev_v_weights);
                layer_grad.set_prev_m_bias(layer_prev_m_bias);
                layer_grad.set_prev_v_bias(layer_prev_v_bias);

                layer.previous_gradient = Some(layer_grad.clone());
            }

            col_offset += chunk_size;
        }
    }
}
