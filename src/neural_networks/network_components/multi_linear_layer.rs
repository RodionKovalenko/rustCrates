use core::fmt::Debug;
use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{network_components::linear_layer::LinearLayer, utils::array_splitting::split_sizes};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLinearLayer {
    pub layers: Vec<LinearLayer>,
    pub learning_rate: f64,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
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
        let mut layers: Vec<LinearLayer> = vec![];
        let col_chunks: Vec<usize> = split_sizes(cols, num_layers);

        for i in 0..col_chunks.len() {
            let linear_layer: LinearLayer = LinearLayer::new(learning_rate, rows, col_chunks[i]);
            // println!("Linear layer {}: weights shape: ({}, {})", i, linear_layer.weights.len(), linear_layer.weights[0].len());
            layers.push(linear_layer);
        }

        Self {
            layers: layers,
            learning_rate: learning_rate,
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
        }
    }
    pub fn get_combined_weights(&self) -> Vec<Vec<Complex<f64>>> {
        let mut combined_weights: Vec<Vec<Complex<f64>>> = vec![];

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_weights = layer.weights.clone();
            if i == 0 {
                combined_weights = layer_weights;
            } else {
                for (row_ind, row) in layer_weights.iter().enumerate() {
                    combined_weights[row_ind].extend_from_slice(row);
                }
            }
        }

        combined_weights
    }
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        // Avoid full clone, just store reference
        let input_batch = input.get_input_batch();
        self.input_batch = Some(input_batch.clone()); // ideally use Arc<...> to avoid clone
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        // Run all layers in parallel (expensive step)
        let lin_layer_output_chunks: Vec<_> = self.layers.par_iter_mut().map(|lin_layer| lin_layer.forward(&input).get_output_batch()).collect();

        // Prepare output container
        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = (0..batch_size).map(|_| vec![Vec::new(); seq_len]).collect();

        // Parallelize over batches
        output_batch.par_iter_mut().enumerate().for_each(|(b, batch_row)| {
            for i in 0..seq_len {
                // Precompute total size for allocation
                let total_len: usize = lin_layer_output_chunks.iter().map(|c| c[b][i].len()).sum();
                let mut combined_output = Vec::with_capacity(total_len);

                for lin_layer_output_chunk in &lin_layer_output_chunks {
                    combined_output.extend_from_slice(&lin_layer_output_chunk[b][i]);
                }

                batch_row[i] = combined_output;
            }
        });

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }
    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");

        let mut gradient = Gradient::new_default();
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let normalizer = self.layers.len();
        let previous_input_gradient = previous_gradient.get_gradient_input_batch();

        // 1. Compute per-layer gradients in parallel
        let gradient_input_batch_chunks: Vec<_> = self
            .layers
            .par_iter_mut()
            .enumerate()
            .map(|(layer_ind, lin_layer)| {
                let cols_per_layer = previous_input_gradient[0][0].len() / normalizer;
                let start_col = layer_ind * cols_per_layer;
                let end_col = start_col + cols_per_layer;

                // Slice gradient input for this layer
                let prev_chunk: Vec<Vec<Vec<Complex<f64>>>> = previous_input_gradient.iter().map(|batch_row| batch_row.iter().map(|row| row[start_col..end_col].to_vec()).collect()).collect();

                let mut grad_copy = previous_gradient.clone(); // keep metadata
                grad_copy.set_gradient_input_batch(prev_chunk);

                let grad_out = lin_layer.backward(&grad_copy);

                grad_out.get_gradient_input_batch()
            })
            .collect();

        // 2. Initialize accumulator with zeros of correct size
        let feature_dim = gradient_input_batch_chunks[0][0][0].len();
        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = (0..batch_size).map(|_| (0..seq_len).map(|_| vec![Complex::new(0.0, 0.0); feature_dim]).collect()).collect();

        // 3. Accumulate all chunks in-place
        for chunk in &gradient_input_batch_chunks {
            for b in 0..batch_size {
                for i in 0..seq_len {
                    for (dst, src) in gradient_input_batch[b][i].iter_mut().zip(&chunk[b][i]) {
                        *dst += src;
                    }
                }
            }
        }

        // 4. Add stored gradient if exists
        if let Some(prev_grad) = &self.gradient {
            let prev_batch = prev_grad.get_gradient_input_batch();
            for b in 0..batch_size {
                for i in 0..seq_len {
                    for (dst, src) in gradient_input_batch[b][i].iter_mut().zip(&prev_batch[b][i]) {
                        *dst += src;
                    }
                }
            }
        }

        gradient.set_gradient_input_batch(gradient_input_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        self.layers.par_iter_mut().for_each(|lin_layer| lin_layer.update_parameters());
    }
}
