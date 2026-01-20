use crate::neural_networks::network_layers::adaptive_pooling::adaptive_avg_pool1d_layer::{AdaptiveAvgPool1dLayer, CompressionMetadata, CompressionType};
use crate::neural_networks::utils::dtype::C;

use crate::neural_networks::network_components::{
    layer_input_struct::LayerInput,
    layer_output_struct::LayerOutput,
};

// Dynamic Sequence Compressor (uses AdaptiveAvgPool1d internally)
pub struct DynamicSequenceCompressorLayer {
    target_min: usize,
    target_max: usize,
    pool_layer: AdaptiveAvgPool1dLayer,
}

impl DynamicSequenceCompressorLayer {
    pub fn new(target_min: usize, target_max: usize) -> Self {
        Self {
            target_min,
            target_max,
            pool_layer: AdaptiveAvgPool1dLayer::new(target_max),
        }
    }

    pub fn forward(&mut self, input: &Vec<Vec<Vec<C>>>) -> LayerOutput {
        self.compress(input)
    }

    pub fn decompress(&self, compressed: &Vec<Vec<Vec<C>>>) -> Vec<Vec<Vec<C>>> {
        self.pool_layer.decompress(compressed)
    }

    pub fn compress(&mut self, input: &Vec<Vec<Vec<C>>>) -> LayerOutput {
        let metadata = CompressionMetadata {
            original_length: 0,
            compressed_length: 0,
            window_mappings: Vec::new(),
            compression_type: CompressionType::Adaptive,
        };
        let mut layer_output = LayerOutput::new_default();
        layer_output.set_pooling_metadata(metadata);

        if input.is_empty() {
            return layer_output;
        }

        let seq_len = input[0].len();

        if seq_len <= self.target_max {
            let metadata = CompressionMetadata {
                original_length: seq_len,
                compressed_length: seq_len,
                window_mappings: (0..seq_len).map(|i| vec![i]).collect(),
                compression_type: CompressionType::Adaptive,
            };
            layer_output.set_output_batch(input.clone());
            layer_output.set_pooling_metadata(metadata);
            return layer_output;
        }

        let target_size = self.calculate_target_size(seq_len);
        let mut pool = AdaptiveAvgPool1dLayer::new(target_size);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input.clone());
        layer_output = pool.compress(&layer_input);
        self.pool_layer = pool.clone();
        
        layer_output
    }

    fn calculate_target_size(&self, input_len: usize) -> usize {
        let target_middle = (self.target_min + self.target_max) / 2;
        if input_len / target_middle > 0 {
            let ideal_ratio = input_len / target_middle;
            let target_size = input_len / ideal_ratio;
            target_size.max(self.target_min).min(self.target_max)
        } else {
            target_middle
        }
    }
}
