use crate::neural_networks::{
    network_layers::adaptive_pooling::{adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer, interpalation_decompressor_layer::InterpolationDecompressorLayer},
    utils::dtype::{r, Real, C, ZERO},
};

// Helper function to calculate reconstruction error (Mean Squared Error)
pub fn calculate_reconstruction_error(original: &Vec<Vec<Vec<C>>>, reconstructed: &Vec<Vec<Vec<C>>>) -> Real {
    if original.len() != reconstructed.len() || original.is_empty() {
        return Real::INFINITY;
    }

    let mut total_error: Real = ZERO;
    let mut total_elements = 0;

    for (batch_orig, batch_recon) in original.iter().zip(reconstructed.iter()) {
        if batch_orig.len() != batch_recon.len() {
            return Real::INFINITY;
        }

        for (seq_orig, seq_recon) in batch_orig.iter().zip(batch_recon.iter()) {
            if seq_orig.len() != seq_recon.len() {
                return Real::INFINITY;
            }

            for (orig_val, recon_val) in seq_orig.iter().zip(seq_recon.iter()) {
                let diff = orig_val - recon_val;
                total_error += diff.norm_sqr();
                total_elements += 1;
            }
        }
    }

    if total_elements > 0 {
        total_error / r(total_elements as f64)
    } else {
        ZERO
    }
}

// Test different decompression methods
pub fn test_decompression_methods(original: &Vec<Vec<Vec<C>>>, compressed: &Vec<Vec<Vec<C>>>, adaptive_pool: &AdaptiveAvgPool1dLayer) {
    // Method 1: Window-based decompression
    let decompressed_windows = adaptive_pool.decompress(compressed);
    let window_error = calculate_reconstruction_error(original, &decompressed_windows);

    // Method 2: Linear interpolation
    let decompressed_interpolation: Vec<Vec<Vec<C>>> = InterpolationDecompressorLayer::linear_interpolate(compressed, adaptive_pool.output_size);
    let interpolation_error = calculate_reconstruction_error(original, &decompressed_interpolation);

    println!("Window-based decompression error: {:.6}", window_error);
    println!("Interpolation decompression error: {:.6}", interpolation_error);

    if window_error < interpolation_error {
        println!("→ Window-based decompression is better for this case");
    } else {
        println!("→ Linear interpolation is better for this case");
    }
}

// Helper function to create random complex input (reused from previous version)
pub fn create_random_input(batch_size: usize, seq_len: usize, hidden_dim: usize) -> Vec<Vec<Vec<C>>> {
    use std::f64::consts::PI;

    let mut input = Vec::with_capacity(batch_size);

    for batch in 0..batch_size {
        let mut sequence = Vec::with_capacity(seq_len);

        for seq in 0..seq_len {
            let mut features = Vec::with_capacity(hidden_dim);

            for dim in 0..hidden_dim {
                // Create some realistic complex values
                let phase = 2.0 * PI * (seq as f64) / (seq_len as f64) + (dim as f64) * 0.1;
                let magnitude = 1.0 / (1.0 + (seq as f64) * 0.001); // Decay over sequence

                let real: Real = r(magnitude * phase.cos() + (batch as f64) * 0.01);
                let imag: Real = r(magnitude * phase.sin() + (dim as f64) * 0.001);

                features.push(C::new(real, imag));
            }

            sequence.push(features);
        }

        input.push(sequence);
    }

    input
}

#[cfg(test)]
mod adaptive_pooling_complex_tests {
    use crate::neural_networks::{
        network_components::layer_input_struct::LayerInput,
        network_layers::adaptive_pooling::{
            adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer, dynamic_sequence_compressor_layer::DynamicSequenceCompressorLayer, interpalation_decompressor_layer::InterpolationDecompressorLayer,
            strided_pooling_layer::StridedPoolingLayer,
        },
    };

    use super::*;

    #[test]
    fn test_compression_decompression_roundtrip() {
        let input = create_random_input(1, 100, 16);
        let mut pool = AdaptiveAvgPool1dLayer::new(25);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input.clone());
        let pooling_output = pool.compress(&layer_input);
        let compressed = pooling_output.get_output_batch();
        let decompressed = pool.decompress(&compressed);

        assert_eq!(input.len(), decompressed.len());
        assert_eq!(input[0].len(), decompressed[0].len());
        assert_eq!(input[0][0].len(), decompressed[0][0].len());

        // Error should be finite (not perfect reconstruction, but reasonable)
        let error = calculate_reconstruction_error(&input, &decompressed);
        println!("Reconstruction error: {:?}", error);
        assert!(error.is_finite() && error >= ZERO);
    }

    #[test]
    fn test_metadata_consistency() {
        let input = create_random_input(2, 200, 32);
        let mut compressor = DynamicSequenceCompressorLayer::new(40, 50);

        let pooling_output = compressor.compress(&input);
        let (compressed, metadata) = (pooling_output.get_output_batch(), pooling_output.get_pooling_metadata().unwrap());

        assert_eq!(metadata.original_length, 200);
        assert!(metadata.compressed_length >= 40 && metadata.compressed_length <= 50);
        assert_eq!(compressed[0].len(), metadata.compressed_length);
    }

    #[test]
    fn test_strided_pooling_decompression() {
        let input = create_random_input(1, 50, 8);
        let pool = StridedPoolingLayer::new(5, Some(5));

        let (compressed, metadata) = pool.forward_inner(&input);
        let decompressed = pool.decompress(&compressed, &metadata);

        assert_eq!(decompressed[0].len(), 50);

        // Test that overlapping regions are handled correctly
        let error = calculate_reconstruction_error(&input, &decompressed);
        println!("Strided Pooling Reconstruction error: {:?}", error);
        assert!(error.is_finite());
    }

    #[test]
    fn test_interpolation_decompression() {
        let compressed = create_random_input(1, 10, 4);
        let interpolated = InterpolationDecompressorLayer::linear_interpolate(&compressed, 25);

        assert_eq!(interpolated[0].len(), 25);
        assert_eq!(interpolated[0][0].len(), 4);
    }
}
