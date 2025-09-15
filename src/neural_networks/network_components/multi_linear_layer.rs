use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::neural_networks::{
    network_components::linear_layer::LinearLayer,
    network_types::transformer::transformer_network::EMA_SCALER,
    utils::{
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        array_splitting::split_sizes,
        matrix::{average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, compute_global_norm},
        weights_initializer::initialize_weights_complex,
    },
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLinearLayer {
    pub layers: Vec<LinearLayer>,
    pub learning_rate: f64,
    pub col_ranges: Vec<(usize, usize)>,
    pub smoothing: f64,
    pub ema: f64,
    #[serde(skip)]
    pub input_batch: Option<Arc<Vec<Vec<Vec<Complex<f64>>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub previous_gradient: Option<Gradient>,
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

        let mut weights = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        initialize_weights_complex(rows, cols, &mut weights);

        let mut start_col = 0;
        for &chunk_size in &col_chunks {
            let end_col = start_col + chunk_size;
            col_ranges.push((start_col, end_col));

            let mut linear_layer = LinearLayer::new(learning_rate, rows, chunk_size);
            linear_layer.weights = weights.iter().map(|row| row[start_col..end_col].to_vec()).collect();
            layers.push(linear_layer);
            start_col = end_col;
        }

        println!("chunk sizes: {:?}", col_chunks);

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
        }
    }

    pub fn get_combined_weights(&self) -> (Vec<Vec<Complex<f64>>>, Vec<Complex<f64>>) {
        let mut combined_weights = Vec::with_capacity(self.layers[0].weights.len());
        let mut combined_bias = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_weights = &layer.weights;
            let layer_bias = &layer.bias;
            if i == 0 {
                combined_weights = layer_weights.clone();
                combined_bias = layer_bias.clone();
            } else {
                for (row_ind, row) in layer_weights.iter().enumerate() {
                    combined_weights[row_ind].extend_from_slice(row);
                }
                combined_bias.extend(layer_bias.iter().cloned());
            }
        }

        (combined_weights, combined_bias)
    }

    pub fn get_combined_gradients(&self) -> (Vec<Vec<Complex<f64>>>, Vec<Complex<f64>>) {
        let mut combined_weights_gradients = Vec::with_capacity(self.layers[0].weights.len());
        let mut combined_bias_gradients = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_weights = &layer.weights;

            let layer_weight_gradients = layer.gradient
                .as_ref()
                .map(|g| g.get_gradient_weights())
                .unwrap_or_else(|| vec![vec![Complex::new(0.0, 0.0); layer_weights[0].len()]; layer_weights.len()]);
            let layer_bias_gradients = layer.gradient
                .as_ref()
                .map(|g| g.get_gradient_bias())
                .unwrap_or_else(|| vec![Complex::new(0.0, 0.0); layer_weights.len()]);

            if i == 0 {
                combined_weights_gradients = layer_weight_gradients;
                combined_bias_gradients = layer_bias_gradients;
            } else {
                for (row_ind, row) in layer_weight_gradients.iter().enumerate() {
                    combined_weights_gradients[row_ind].extend_from_slice(row);
                }
                combined_bias_gradients.extend(layer_bias_gradients);
            }
        }

        (combined_weights_gradients, combined_bias_gradients)
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = Arc::new(input.get_input_batch());
        self.input_batch = Some(input_batch.clone());
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        let lin_layer_output_chunks: Vec<_> = self.layers.iter_mut()
            .map(|lin_layer| lin_layer.forward(input).get_output_batch())
            .collect();

        let output_feature_size: usize = lin_layer_output_chunks.iter().map(|chunk| chunk[0][0].len()).sum();

        // Efficient output allocation
        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = (0..batch_size)
            .map(|_| vec![vec![Complex::new(0.0, 0.0); output_feature_size]; seq_len])
            .collect();

        // Parallelizing batch assembly
        output_batch.iter_mut().enumerate().for_each(|(b, batch_row)| {
            for i in 0..seq_len {
                let mut offset = 0;
                for lin_layer_output_chunk in &lin_layer_output_chunks {
                    batch_row[i][offset..offset + lin_layer_output_chunk[b][i].len()]
                        .copy_from_slice(&lin_layer_output_chunk[b][i]);
                    offset += lin_layer_output_chunk[b][i].len();
                }
            }
        });

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let previous_input_gradient = previous_gradient.get_gradient_input_batch();

        // Compute gradient chunks for each sublayer by slicing previous gradient input batch columns
        let gradient_input_batch_chunks: Vec<_> = self.layers.iter_mut()
            .enumerate()
            .map(|(layer_ind, lin_layer)| {
                let (start_col, end_col) = self.col_ranges[layer_ind];
                let prev_chunk: Vec<Vec<Vec<Complex<f64>>>> = previous_input_gradient.iter()
                    .map(|batch_row| batch_row.iter()
                        .map(|row| row[start_col..end_col].to_vec())
                        .collect())
                    .collect();

                // Create a sliced Gradient with only needed input batch slice to avoid cloning entire previous_gradient
                let mut grad_sliced = Gradient::new_default();
                grad_sliced.set_gradient_input_batch(prev_chunk);

                lin_layer.backward(&grad_sliced).get_gradient_input_batch()
            }).collect();

        // Infer feature dimension from first chunk
        let feature_dim = gradient_input_batch_chunks[0][0][0].len();

        // Prepare empty gradient input batch to accumulate summed gradient inputs from chunks
        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = (0..batch_size)
            .map(|_| (0..seq_len)
                .map(|_| vec![Complex::new(0.0, 0.0); feature_dim])
                .collect())
            .collect();

        // Parallel safe accumulation of gradient chunks into final input gradient batch
        gradient_input_batch.iter_mut().enumerate().for_each(|(b, batch_row)| {
            batch_row.iter_mut().enumerate().for_each(|(i, feature_vec)| {
                for chunk in &gradient_input_batch_chunks {
                    for (dst, src) in feature_vec.iter_mut().zip(&chunk[b][i]) {
                        *dst += *src;
                    }
                }
            });
        });

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(gradient_input_batch);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let (weight_gradients, mut bias_gradients) = self.get_combined_gradients();
        let (mut combined_weights, mut combined_bias) = self.get_combined_weights();
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");

        let input_batch = gradient.get_gradient_input_batch();
        let mut batch_size_num = input_batch.len() as f64;
        let mut all_gradients = vec![weight_gradients];
        let global_norm = compute_global_norm(&all_gradients, &bias_gradients);
        self.ema = self.smoothing * self.ema + (1.0 - self.smoothing) * global_norm;
        let max_norm = self.ema * EMA_SCALER;
        clip_all_gradients_by_global_norm_2d(&mut all_gradients, &mut bias_gradients, global_norm, max_norm);
        if self.batch_size > 0 {
            batch_size_num = self.batch_size as f64;
        }
        let mut weight_gradients: Vec<Vec<Complex<f64>>> = all_gradients[0].clone();
        weight_gradients = average_matrix_by_scalar(&weight_gradients, batch_size_num);
        bias_gradients = average_vector_by_scalar(&bias_gradients, batch_size_num);
        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());
        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_bias(),
                previous_gradient.get_prev_v_bias(),
                previous_gradient.get_prev_m_weights(),
                previous_gradient.get_prev_v_weights(),
            )
        } else {
            (
                vec![Complex::new(0.0, 0.0); bias_gradients.len()],
                vec![Complex::new(0.0, 0.0); bias_gradients.len()],
                vec![vec![Complex::new(0.0, 0.0); weight_gradients[0].len()]; weight_gradients.len()],
                vec![vec![Complex::new(0.0, 0.0); weight_gradients[0].len()]; weight_gradients.len()],
            )
        };
        calculate_adam_w_bias(&mut combined_bias, &bias_gradients, &mut prev_m_bias, &mut prev_v_bias, learning_rate, time_step);
        calculate_adam_w(&mut combined_weights, &weight_gradients, &mut prev_m_weights, &mut prev_v_weights, learning_rate, time_step);

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        self.previous_gradient = Some(gradient.clone());

        // Update weights and biases back into sublayers with correct slicing
        self.layers.iter_mut().enumerate().for_each(|(layer_ind, lin_layer)| {
            let (start_col, end_col) = self.col_ranges[layer_ind];
            lin_layer.weights = combined_weights.iter().map(|row| row[start_col..end_col].to_vec()).collect();
            lin_layer.bias = combined_bias[start_col..end_col].to_vec();
        });
    }
}
