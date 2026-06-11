use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{adaptive_pooling::adaptive_avg_pool1d_layer::{CompressionMetadata, CompressionType}, default_layer::LayerInterface},
    utils::dtype::{C, Real, ZERO, r},
};

// StridedPooling implementation with forward/backward/backward_gradient
pub struct StridedPoolingLayer {
    pub stride: usize,
    pub kernel_size: usize,
}

impl StridedPoolingLayer {
    pub fn new(stride: usize, kernel_size: Option<usize>) -> Self {
        Self { stride, kernel_size: kernel_size.unwrap_or(stride) }
    }

    pub fn forward_inner(&self, input: &Vec<Vec<Vec<C>>>) -> (Vec<Vec<Vec<C>>>, CompressionMetadata) {
        if input.is_empty() {
            return (
                Vec::new(),
                CompressionMetadata {
                    original_length: 0,
                    compressed_length: 0,
                    window_mappings: Vec::new(),
                    compression_type: CompressionType::Strided { stride: self.stride, kernel_size: self.kernel_size },
                },
            );
        }

        let batch_size = input.len();
        let seq_len = input[0].len();
        let hidden_dim = if seq_len > 0 { input[0][0].len() } else { 0 };
        let output_len = if seq_len >= self.kernel_size { (seq_len - self.kernel_size) / self.stride + 1 } else { 0 };

        let mut window_mappings = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let start_pos = i * self.stride;
            let end_pos = (start_pos + self.kernel_size).min(seq_len);
            window_mappings.push((start_pos..end_pos).collect());
        }

        let mut output = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let batch_input = &input[batch_idx];
            let pooled = self.apply_strided_pooling(batch_input, output_len, hidden_dim);
            output.push(pooled);
        }

        (
            output,
            CompressionMetadata {
                original_length: seq_len,
                compressed_length: output_len,
                window_mappings,
                compression_type: CompressionType::Strided { stride: self.stride, kernel_size: self.kernel_size },
            },
        )
    }

    pub fn decompress(&self, compressed: &Vec<Vec<Vec<C>>>, metadata: &CompressionMetadata) -> Vec<Vec<Vec<C>>> {
        if compressed.is_empty() || metadata.original_length == 0 {
            return Vec::new();
        }
        let batch_size = compressed.len();
        let hidden_dim = if !compressed[0].is_empty() { compressed[0][0].len() } else { 0 };
        let mut decompressed = Vec::with_capacity(batch_size);
        for batch_seq in compressed.iter() {
            decompressed.push(self.decompress_sequence_strided(batch_seq, metadata.original_length, &metadata.window_mappings, hidden_dim));
        }
        decompressed
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input = layer_input.get_input_batch();
        let (output, metadata) = self.forward_inner(&input);
        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output);
        layer_output.set_pooling_metadata(metadata);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let metadata = previous_gradient.get_pooling_metadata();
        if let Some(metadata) = metadata {
            let grad_input = self.backward_inner(&previous_gradient.get_gradient_input_batch(), &metadata);
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(grad_input);
            gradient
        } else {
            previous_gradient.clone()
        }
    }

    pub fn update_parameters(&mut self) {}

    pub fn backward_inner(&self, grad_output: &[Vec<Vec<C>>], metadata: &CompressionMetadata) -> Vec<Vec<Vec<C>>> {
        // Similar to AdaptiveAvgPool1d.backward_gradient but adjusted for overlapping windows
        if grad_output.is_empty() || metadata.original_length == 0 {
            return Vec::new();
        }
        let batch_size = grad_output.len();
        let original_length = metadata.original_length;
        let hidden_dim = if batch_size > 0 && !grad_output[0].is_empty() { grad_output[0][0].len() } else { 0 };
        let mut grad_input = vec![vec![vec![C::new(ZERO, ZERO); hidden_dim]; original_length]; batch_size];

        for batch_idx in 0..batch_size {
            for (comp_idx, window) in metadata.window_mappings.iter().enumerate() {
                if comp_idx >= grad_output[batch_idx].len() {
                    continue;
                }
                let window_len: Real = r(window.len() as f64);
                for &pos in window {
                    if pos < original_length {
                        for dim in 0..hidden_dim {
                            grad_input[batch_idx][pos][dim] += grad_output[batch_idx][comp_idx][dim] / window_len;
                        }
                    }
                }
            }
        }
        grad_input
    }

    fn apply_strided_pooling(&self, sequence: &Vec<Vec<C>>, output_len: usize, hidden_dim: usize) -> Vec<Vec<C>> {
        let mut output = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let start_pos = i * self.stride;
            let end_pos = (start_pos + self.kernel_size).min(sequence.len());
            let mut pooled_token = vec![C::new(ZERO, ZERO); hidden_dim];
            let window_size = end_pos - start_pos;

            for pos in start_pos..end_pos {
                for dim in 0..hidden_dim {
                    pooled_token[dim] += sequence[pos][dim];
                }
            }
            if window_size > 0 {
                let window_size_real: Real = r(window_size as f64);
                for dim in 0..hidden_dim {
                    pooled_token[dim] /= window_size_real;
                }
            }
            output.push(pooled_token);
        }
        output
    }

    fn decompress_sequence_strided(&self, compressed: &Vec<Vec<C>>, original_length: usize, window_mappings: &[Vec<usize>], hidden_dim: usize) -> Vec<Vec<C>> {
        let mut decompressed = vec![vec![C::new(ZERO, ZERO); hidden_dim]; original_length];
        let mut position_counts = vec![0usize; original_length];
        for (comp_idx, window) in window_mappings.iter().enumerate() {
            if comp_idx >= compressed.len() {
                continue;
            }
            let compressed_token = &compressed[comp_idx];
            for &pos in window {
                if pos < original_length {
                    for dim in 0..hidden_dim {
                        decompressed[pos][dim] += compressed_token[dim];
                    }
                    position_counts[pos] += 1;
                }
            }
        }
        for pos in 0..original_length {
            if position_counts[pos] > 1 {
                let count_real: Real = r(position_counts[pos] as f64);
                for dim in 0..hidden_dim {
                    decompressed[pos][dim] /= count_real;
                }
            }
        }
        decompressed
    }
}

impl LayerInterface for StridedPoolingLayer {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        StridedPoolingLayer::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        StridedPoolingLayer::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        StridedPoolingLayer::update_parameters(self)
    }
}
