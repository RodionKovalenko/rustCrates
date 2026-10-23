use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        adaptive_pooling::adaptive_avg_pool1d_layer::{average_pool_window, causal_pooling_windows, CompressionMetadata, CompressionType},
        default_layer::LayerInterface,
    },
    utils::{
        dtype::{r, C},
        matrix::RowMajorMatrix,
    },
};

/// RM counterpart of `AdaptiveAvgPool1dLayer`: causal average pooling of the sequence to `output_size` tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveAvgPool1dLayerRm {
    pub output_size: usize,
    pub metadata: CompressionMetadata,
    pub hidden_dim: usize,
}

impl AdaptiveAvgPool1dLayerRm {
    pub fn new(output_size: usize) -> Self {
        Self {
            output_size,
            metadata: CompressionMetadata::new(),
            hidden_dim: 0,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        self.compress(layer_input)
    }

    /// RM counterpart of `AdaptiveAvgPool1dLayer::compress`.
    pub fn compress(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input = layer_input.get_input_batch_rm();
        let seq_len = input[0].rows;
        for (i, seq) in input.iter().enumerate() {
            if seq.rows != seq_len {
                panic!("Batch {} sequence length {} does not match expected {}", i, seq.rows, seq_len);
            }
        }
        self.hidden_dim = if seq_len > 0 { input[0].cols } else { 0 };

        let (output, window_mappings, compressed_length) = if seq_len <= self.output_size {
            (input, (0..seq_len).map(|i| vec![i]).collect(), seq_len)
        } else {
            let windows = causal_pooling_windows(seq_len, self.output_size);
            let output = input
                .iter()
                .map(|seq| {
                    let mut pooled = RowMajorMatrix::create_empty(windows.len(), self.hidden_dim);
                    for (w, window) in windows.iter().enumerate() {
                        let range = pooled.row_range(w);
                        average_pool_window(|pos| &seq.data[seq.row_range(pos)], seq.rows, window, &mut pooled.data[range]);
                    }
                    pooled
                })
                .collect();
            (output, windows, self.output_size)
        };

        self.metadata = CompressionMetadata {
            original_length: seq_len,
            compressed_length,
            window_mappings,
            compression_type: CompressionType::Adaptive,
        };

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output);
        layer_output.set_pooling_metadata(self.metadata.clone());
        layer_output
    }

    pub fn update_parameters(&mut self) {}

    /// RM counterpart of `AdaptiveAvgPool1dLayer::backward`: spreads each pooled gradient evenly over its window.
    pub fn backward(&mut self, gradient: &Gradient) -> Gradient {
        let grad_in = gradient.get_gradient_input_batch_rm_cow();
        let metadata = &self.metadata;

        if grad_in[0].rows == metadata.original_length {
            return self.backward_through_decompress(gradient);
        }

        let (compressed_len, original_length) = (metadata.compressed_length, metadata.original_length);
        if grad_in.is_empty() || compressed_len == 0 || original_length == 0 {
            return Gradient::new_default();
        }
        if original_length <= compressed_len {
            return gradient.clone();
        }

        let hidden_dim = self.hidden_dim;
        let grad_out: Vec<RowMajorMatrix<C>> = grad_in
            .iter()
            .map(|g| {
                let mut out = RowMajorMatrix::create_empty(original_length, hidden_dim);
                for (win_idx, window) in metadata.window_mappings.iter().enumerate().take(compressed_len.min(g.rows)) {
                    let n = window.len();
                    if n == 0 {
                        continue;
                    }
                    let grad_vec = &g.data[g.row_range(win_idx)];
                    for &pos in window.iter().filter(|&&pos| pos < original_length) {
                        let row = out.row_range(pos);
                        for dim in 0..hidden_dim {
                            out.data[row.start + dim] += grad_vec[dim] / r(n as f64);
                        }
                    }
                }
                out
            })
            .collect();

        let mut grad = Gradient::new_default();
        grad.set_gradient_input_batch_rm(grad_out);
        grad.set_pooling_metadata(metadata.clone());
        grad
    }

    /// RM counterpart of `AdaptiveAvgPool1dLayer::backward_through_decompress`: sums the gradients of each window.
    pub fn backward_through_decompress(&mut self, gradient: &Gradient) -> Gradient {
        let grad_in = gradient.get_gradient_input_batch_rm_cow();
        let metadata = &self.metadata;
        let (compressed_len, original_length) = (metadata.compressed_length, metadata.original_length);

        if grad_in.is_empty() || original_length == 0 || compressed_len == 0 {
            return Gradient::new_default();
        }

        let hidden_dim = grad_in[0].cols;
        let sample_len = grad_in[0].rows;

        let grad_compressed: Vec<RowMajorMatrix<C>> = grad_in
            .iter()
            .map(|g| {
                let mut out = RowMajorMatrix::create_empty(compressed_len, hidden_dim);
                for win_idx in 0..compressed_len {
                    let dst = out.row_range(win_idx);
                    if sample_len == compressed_len {
                        // already compressed gradient, just copy
                        if win_idx >= g.rows {
                            break;
                        }
                        let src = &g.data[g.row_range(win_idx)];
                        let n = src.len().min(hidden_dim);
                        out.data[dst.start..dst.start + n].copy_from_slice(&src[..n]);
                    } else {
                        // gradient w.r.t. decompressed outputs (or unexpected shape): accumulate per window
                        if sample_len == original_length && win_idx >= g.rows {
                            continue;
                        }
                        for &pos in metadata.window_mappings[win_idx].iter().filter(|&&pos| pos < g.rows) {
                            let src = &g.data[g.row_range(pos)];
                            for dim in 0..hidden_dim {
                                out.data[dst.start + dim] += src[dim];
                            }
                        }
                    }
                }
                out
            })
            .collect();

        let mut grad = Gradient::new_default();
        grad.set_gradient_input_batch_rm(grad_compressed);
        grad.set_pooling_metadata(metadata.clone());
        grad
    }

    /// RM counterpart of `AdaptiveAvgPool1dLayer::decompress`: every position gets the token of its window.
    pub fn decompress(&self, compressed: &[RowMajorMatrix<C>]) -> Vec<RowMajorMatrix<C>> {
        let metadata = &self.metadata;
        if compressed.is_empty() || metadata.original_length == 0 {
            return Vec::new();
        }

        compressed
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                if seq.rows != metadata.compressed_length {
                    panic!("Batch {} compressed length {} != metadata {}", i, seq.rows, metadata.compressed_length);
                }
                let mut out = RowMajorMatrix::create_empty(metadata.original_length, seq.cols);
                for (comp_idx, window) in metadata.window_mappings.iter().enumerate().take(seq.rows) {
                    let token = &seq.data[seq.row_range(comp_idx)];
                    for &pos in window.iter().filter(|&&pos| pos < metadata.original_length) {
                        let dst = out.row_range(pos);
                        out.data[dst].copy_from_slice(token);
                    }
                }
                out
            })
            .collect()
    }
}

impl LayerInterface for AdaptiveAvgPool1dLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        AdaptiveAvgPool1dLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        AdaptiveAvgPool1dLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        AdaptiveAvgPool1dLayerRm::update_parameters(self)
    }
}
