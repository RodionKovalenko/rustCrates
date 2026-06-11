#[cfg(test)]
mod test_embedding_layer {
    use num::Complex;

    use crate::neural_networks::{
        network_components::layer_input_struct::LayerInput,
        network_layers::{embedding_layer::EmbeddingLayer, wavelet_network::DECOMPOSITION_LEVELS},
    };

    #[test]
    fn embedding_layer_forward_reads_from_in_memory_weights() {
        let vocab_size = 8;
        let embedding_dim_original = 16;
        let base_2: i32 = 2;
        let embedding_dim = (embedding_dim_original as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;

        let mut embedding_layer = EmbeddingLayer::new(vocab_size, embedding_dim_original);
        embedding_layer.tied_weights[3] = (0..embedding_dim)
            .map(|idx| Complex::new(idx as f64 + 0.5, -(idx as f64)))
            .collect();

        let mut layer_input = LayerInput::new_default();
        layer_input.set_batch_ids(vec![vec![3]]);

        let (output, padding_mask) = embedding_layer.forward_inner(&layer_input);

        assert_eq!(padding_mask, vec![vec![1]]);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 1);
        assert_eq!(output[0][0].len(), embedding_dim);

        for idx in 0..embedding_dim {
            assert_eq!(output[0][0][idx], Complex::new(idx as f64 + 0.5, -(idx as f64)));
        }
    }
}
