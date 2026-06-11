use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::default_layer::LayerInterface,
    utils::dtype::{r, C, Real, ZERO},
};

// Interpolation-based decompression methods
pub struct InterpolationDecompressorLayer;

impl InterpolationDecompressorLayer {
    /// Linear interpolation decompression
    pub fn linear_interpolate(compressed: &Vec<Vec<Vec<C>>>, target_length: usize) -> Vec<Vec<Vec<C>>> {
        if compressed.is_empty() || target_length == 0 {
            return Vec::new();
        }

        let batch_size = compressed.len();
        let compressed_length = compressed[0].len();
        let hidden_dim = if compressed_length > 0 { compressed[0][0].len() } else { 0 };

        let mut output = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let batch_compressed = &compressed[batch_idx];
            let interpolated = Self::interpolate_sequence(batch_compressed, target_length, hidden_dim);
            output.push(interpolated);
        }

        output
    }

    fn interpolate_sequence(compressed: &Vec<Vec<C>>, target_length: usize, hidden_dim: usize) -> Vec<Vec<C>> {
        if compressed.is_empty() || target_length == 0 {
            return Vec::new();
        }

        let compressed_length = compressed.len();
        if compressed_length == target_length {
            return compressed.clone();
        }

        let mut interpolated = Vec::with_capacity(target_length);
        let scale: f64 = (compressed_length - 1) as f64 / (target_length - 1) as f64;

        for i in 0..target_length {
            let float_idx = i as f64 * scale;
            let left_idx = float_idx.floor() as usize;
            let right_idx = (left_idx + 1).min(compressed_length - 1);
            let weight = float_idx - left_idx as f64;
            let w: Real = r(weight);
            let one_minus_w: Real = r(1.0 - weight);

            let mut interpolated_token = vec![C::new(ZERO, ZERO); hidden_dim];

            if left_idx == right_idx {
                // No interpolation needed
                interpolated_token = compressed[left_idx].clone();
            } else {
                // Linear interpolation between two tokens
                for dim in 0..hidden_dim {
                    let left_val = compressed[left_idx][dim];
                    let right_val = compressed[right_idx][dim];
                    interpolated_token[dim] = left_val * one_minus_w + right_val * w;
                }
            }

            interpolated.push(interpolated_token);
        }

        interpolated
    }

    pub fn forward(&mut self, _layer_input: &LayerInput) -> LayerOutput {
        LayerOutput::new_default()
    }

    pub fn backward(&mut self, _previous_gradient: &Gradient) -> Gradient {
        Gradient::new_default()
    }

    pub fn update_parameters(&mut self) {}
}

impl LayerInterface for InterpolationDecompressorLayer {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        InterpolationDecompressorLayer::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        InterpolationDecompressorLayer::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        InterpolationDecompressorLayer::update_parameters(self)
    }
}
