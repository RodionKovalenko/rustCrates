use serde::{Deserialize, Serialize};

use crate::neural_networks::network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};
use crate::neural_networks::utils::dtype::{r, Real, C, ZERO};

// Compression metadata for decompression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    pub original_length: usize,
    pub compressed_length: usize,
    pub window_mappings: Vec<Vec<usize>>,
    pub compression_type: CompressionType,
}

impl CompressionMetadata {
    pub fn new() -> Self {
        CompressionMetadata {
            original_length: 0,
            compressed_length: 0,
            window_mappings: Vec::new(),
            compression_type: CompressionType::Adaptive,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    Adaptive,
    Strided { stride: usize, kernel_size: usize },
    ContentAware,
    MultiStage,
}

// AdaptiveAvgPool1d struct and implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveAvgPool1dLayer {
    pub output_size: usize,
    pub metadata: CompressionMetadata,
}

impl AdaptiveAvgPool1dLayer {
    pub fn new(output_size: usize) -> Self {
        Self {
            output_size,
            metadata: CompressionMetadata::new(),
        }
    }

    // Forward: compress input sequences
    pub fn forward(&mut self, input_layer: &LayerInput) -> LayerOutput {
        self.compress(input_layer)
    }

    // Compression implementation
    pub fn compress(&mut self, input_layer: &LayerInput) -> LayerOutput {
        let input = input_layer.get_input_batch();
        let mut layer_output = LayerOutput::new_default();

        let seq_len = input[0].len();
        for (i, seq) in input.iter().enumerate() {
            if seq.len() != seq_len {
                panic!("Batch {} sequence length {} does not match expected {}", i, seq.len(), seq_len);
            }
        }
        let hidden_dim = if seq_len > 0 && !input[0].is_empty() { input[0][0].len() } else { 0 };

        if seq_len <= self.output_size {
            let clone_input = input.to_vec();
            let window_mappings = (0..seq_len).map(|i| vec![i]).collect();

            let metadata: CompressionMetadata = CompressionMetadata {
                original_length: seq_len,
                compressed_length: seq_len,
                window_mappings,
                compression_type: CompressionType::Adaptive,
            };
            self.metadata = metadata.clone();
            layer_output.set_output_batch(clone_input.clone());
            layer_output.set_pooling_metadata(metadata);

            return layer_output;
        }

        let windows = self.calculate_causal_pooling_windows(seq_len);
        let mut output = Vec::with_capacity(input.len());

        for seq in input.iter() {
            let compressed_seq = Self::compress_sequence_with_windows(seq, &windows, hidden_dim);
            output.push(compressed_seq);
        }

        let metadata = CompressionMetadata {
            original_length: seq_len,
            compressed_length: self.output_size,
            window_mappings: windows,
            compression_type: CompressionType::Adaptive,
        };

        self.metadata = metadata.clone();
        layer_output.set_output_batch(output);
        layer_output.set_pooling_metadata(metadata);

        layer_output
    }

    // Helper compress function
    fn compress_sequence_with_windows(sequence: &Vec<Vec<C>>, windows: &[Vec<usize>], hidden_dim: usize) -> Vec<Vec<C>> {
        let mut compressed = Vec::with_capacity(windows.len());
        for window in windows.iter() {
            let mut pooled_token = vec![C::new(ZERO, ZERO); hidden_dim];
            let window_size = window.len();
            if window_size > 0 {
                for &pos in window.iter() {
                    if pos < sequence.len() {
                        let token = &sequence[pos];
                        let use_dim = usize::min(token.len(), hidden_dim);
                        for dim in 0..use_dim {
                            pooled_token[dim] += token[dim];
                        }
                    }
                }
                let divisor: Real = r(window_size as f64);
                for dim in 0..hidden_dim {
                    pooled_token[dim] /= divisor;
                }
            }
            compressed.push(pooled_token);
        }
        compressed
    }

    pub fn backward(&self, gradient: &Gradient) -> Gradient {
        let grad_in = gradient.get_gradient_input_batch();
        let metadata = &self.metadata;

        if grad_in[0].len() == metadata.original_length {
            return self.backward_through_decompress(gradient);
        }

        let batch_size = grad_in.len();
        let compressed_len = metadata.compressed_length;
        let original_length = metadata.original_length;

        if batch_size == 0 || compressed_len == 0 || original_length == 0 {
            return Gradient::new_default();
        }

        let hidden_dim = if !grad_in[0].is_empty() { grad_in[0][0].len() } else { 0 };

        // gradient for *original* sequence length
        let mut grad_out = vec![vec![vec![C::new(ZERO, ZERO); hidden_dim]; original_length]; batch_size];

        for batch_idx in 0..batch_size {
            for win_idx in 0..compressed_len {
                if win_idx >= grad_in[batch_idx].len() {
                    continue;
                }
                let grad_vec = &grad_in[batch_idx][win_idx];
                let n = metadata.window_mappings[win_idx].len();

                if n == 0 {
                    continue;
                }

                for &pos in metadata.window_mappings[win_idx].iter() {
                    if pos >= original_length {
                        continue;
                    }
                    for dim in 0..hidden_dim {
                        // distribute gradient equally (avg pooling backward)
                        grad_out[batch_idx][pos][dim] += grad_vec[dim] / r(n as f64);
                    }
                }
            }
        }

        // println!("Backward");
        // println!("gradient in dim in backward: {} {} {}", grad_in.len(), grad_in[0].len(), grad_in[0][0].len());
        // println!("gradient dim in backward: {} {} {}", grad_out.len(), grad_out[0].len(), grad_out[0][0].len());

        let mut grad = Gradient::new_default();
        grad.set_gradient_input_batch(grad_out);
        grad.set_pooling_metadata(metadata.clone());
        grad
    }

    pub fn backward_through_decompress(&self, gradient: &Gradient) -> Gradient {
        let grad_in = gradient.get_gradient_input_batch();
        let metadata = &self.metadata;

        let batch_size = grad_in.len();
        let compressed_len = metadata.compressed_length;
        let original_length = metadata.original_length;

        if batch_size == 0 || original_length == 0 || compressed_len == 0 {
            return Gradient::new_default();
        }

        // determine hidden_dim
        let hidden_dim = if !grad_in[0].is_empty() { grad_in[0][0].len() } else { 0 };

        // zero-initialized gradient for compressed tokens
        let mut grad_compressed = vec![vec![vec![C::new(ZERO, ZERO); hidden_dim]; compressed_len]; batch_size];

        let sample_len = grad_in[0].len();

        if sample_len == original_length {
            // gradient is w.r.t. decompressed outputs → accumulate per window
            for batch_idx in 0..batch_size {
                for win_idx in 0..compressed_len {
                    if win_idx >= grad_in[batch_idx].len() {
                        continue;
                    }
                    for &pos in metadata.window_mappings[win_idx].iter() {
                        if pos >= sample_len {
                            continue;
                        }
                        for dim in 0..hidden_dim {
                            grad_compressed[batch_idx][win_idx][dim] += grad_in[batch_idx][pos][dim];
                        }
                    }
                }
            }
        } else if sample_len == compressed_len {
            // already compressed gradient, just copy
            for batch_idx in 0..batch_size {
                for win_idx in 0..compressed_len {
                    if win_idx >= grad_in[batch_idx].len() {
                        break;
                    }
                    let src = &grad_in[batch_idx][win_idx];
                    let n = usize::min(src.len(), hidden_dim);
                    for dim in 0..n {
                        grad_compressed[batch_idx][win_idx][dim] = src[dim];
                    }
                }
            }
        } else {
            // unexpected shape, best-effort accumulation
            for batch_idx in 0..batch_size {
                for win_idx in 0..compressed_len {
                    for &pos in metadata.window_mappings[win_idx].iter() {
                        if pos >= grad_in[batch_idx].len() {
                            continue;
                        }
                        for dim in 0..hidden_dim {
                            grad_compressed[batch_idx][win_idx][dim] += grad_in[batch_idx][pos][dim];
                        }
                    }
                }
            }
        }

        // println!("Backward through decompress");
        // println!("gradient in dim in backward_through_decompress: {} {} {}", grad_in.len(), grad_in[0].len(), grad_in[0][0].len());
        // println!("gradient out dim in backward_through_decompress: {} {} {}", grad_compressed.len(), grad_compressed[0].len(), grad_compressed[0][0].len());

        let mut grad = Gradient::new_default();
        grad.set_gradient_input_batch(grad_compressed);
        grad.set_pooling_metadata(metadata.clone());

        grad
    }

    // Decompression implementation
    pub fn decompress(&self, compressed: &[Vec<Vec<C>>]) -> Vec<Vec<Vec<C>>> {
        let metadata = &self.metadata;
        if compressed.is_empty() || metadata.original_length == 0 {
            return Vec::new();
        }
        for (i, seq) in compressed.iter().enumerate() {
            if seq.len() != metadata.compressed_length {
                panic!("Batch {} compressed length {} != metadata {}", i, seq.len(), metadata.compressed_length);
            }
        }
        let hidden_dim = if !compressed[0].is_empty() { compressed[0][0].len() } else { 0 };
        let mut decompressed = Vec::with_capacity(compressed.len());

        for batch_seq in compressed.iter() {
            decompressed.push(Self::decompress_sequence(batch_seq, metadata.original_length, &metadata.window_mappings, hidden_dim));
        }
        decompressed
    }

    // Helper decompress function
    fn decompress_sequence(compressed: &Vec<Vec<C>>, original_length: usize, window_mappings: &[Vec<usize>], hidden_dim: usize) -> Vec<Vec<C>> {
        let mut decompressed = vec![vec![C::new(ZERO, ZERO); hidden_dim]; original_length];
        for (comp_idx, window) in window_mappings.iter().enumerate() {
            if comp_idx >= compressed.len() {
                break;
            }
            let token = &compressed[comp_idx];
            for &pos in window.iter() {
                if pos < original_length {
                    if token.len() == hidden_dim {
                        decompressed[pos] = token.clone();
                    } else {
                        let mut tmp = vec![C::new(ZERO, ZERO); hidden_dim];
                        let n = usize::min(token.len(), hidden_dim);
                        for i in 0..n {
                            tmp[i] = token[i];
                        }
                        decompressed[pos] = tmp;
                    }
                }
            }
        }
        decompressed
    }

    //e.g. Sequence length is 36, output size is 15:
    //Calculated windows: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19],
    //                     [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35]]
    // Calculates pooling windows for adaptive average pooling
    pub fn calculate_pooling_windows(&self, seq_len: usize) -> Vec<Vec<usize>> {
        let mut windows = Vec::with_capacity(self.output_size);
        let base_size = seq_len / self.output_size;
        let remainder = seq_len % self.output_size;
        let mut current_pos = 0;

        for i in 0..self.output_size {
            let window_size = if i < remainder { base_size + 1 } else { base_size };
            let mut window = Vec::with_capacity(window_size);
            for j in 0..window_size {
                if current_pos + j < seq_len {
                    window.push(current_pos + j);
                }
            }
            windows.push(window);
            current_pos += window_size;
        }

        // println!("Calculated windows: {:?}", windows);
        windows
    }

    pub fn calculate_causal_pooling_windows(&self, seq_len: usize) -> Vec<Vec<usize>> {
        let mut windows: Vec<Vec<usize>> = Vec::with_capacity(self.output_size);
        let base_size = seq_len / self.output_size;
        let remainder = seq_len % self.output_size;
        let mut current_pos = 0;

        for i in 0..self.output_size {
            let window_size = if i < remainder { base_size + 1 } else { base_size };
            let mut window = Vec::with_capacity(window_size);

            for j in 0..window_size {
                let pos = current_pos + j;
                if pos < seq_len {
                    window.push(pos);
                }
            }

            // 🔒 CAUSALITY ASSERTION
            if i > 0 {
                let prev_max = *windows[i - 1].last().unwrap();
                let curr_min = *window.first().unwrap();
                assert!(prev_max < curr_min, "Causal pooling violated: window {} overlaps or looks ahead", i);
            }

            windows.push(window);
            current_pos += window_size;
        }

        windows
    }
}
