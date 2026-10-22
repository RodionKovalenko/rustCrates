//! Every `*Rm` layer must produce the same outputs, gradients and parameter updates as its Vec counterpart.
#[cfg(test)]
mod test_layers_rm {
    use crate::neural_networks::{
        network_layers::{
            feedforward_layer::FeedForwardLayer,
            layer::LayerEnum,
            linear_layer::LinearLayer,
            network_layers_rm::{feedforward_layer_rm::FeedForwardLayerRm, linear_layer_rm::LinearLayerRm, norm_layer_rm::NormalNormLayerRm, positional_encoding_layer_rm::PositionalEncodingLayerRm},
            norm_layer::NormalNormLayer,
            positional_encoding_layer::PositionalEncodingLayer,
        },
        utils::matrix::RowMajorMatrix,
    };

    use super::super::rm_test_utils::{assert_close_1d, assert_close_2d, compare_training_steps};

    const STEPS: usize = 3;

    #[test]
    fn test_linear_layer_rm_matches_linear_layer() {
        let (input_dim, output_dim) = (8, 6);
        let mut layer = LinearLayer::new(0.01, input_dim, output_dim, true);
        let mut layer_rm = LinearLayerRm::new(0.01, input_dim, output_dim);
        layer_rm.weights = RowMajorMatrix::from_rows(&layer.weights);
        layer_rm.bias = layer.bias.clone();

        compare_training_steps(
            "LinearLayerRm",
            &mut layer,
            &mut layer_rm,
            (3, 5, input_dim, output_dim),
            None,
            STEPS,
            |gradient, gradient_rm, name| {
                assert_close_2d(&format!("{} weight gradient", name), &gradient.get_gradient_weights(), &gradient_rm.get_gradient_weights_rm().unwrap());
                assert_close_1d(&format!("{} bias gradient", name), &gradient.get_gradient_bias(), &gradient_rm.get_gradient_bias());
            },
            |layer, layer_rm, name| {
                assert_close_2d(&format!("{} weights", name), &layer.weights, &layer_rm.weights);
                assert_close_1d(&format!("{} bias", name), &layer.bias, &layer_rm.bias);
            },
        );
    }

    #[test]
    fn test_norm_layer_rm_matches_norm_layer() {
        let feature_dim = 8;
        let mut layer = NormalNormLayer::new(feature_dim, 1e-8, 0.01);
        let mut layer_rm = NormalNormLayerRm::new(feature_dim, 1e-8, 0.01);

        // The last position of the second sequence is padding.
        let padding_mask_batch = vec![vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 0], vec![1, 1, 1, 1, 1]];

        compare_training_steps(
            "NormalNormLayerRm",
            &mut layer,
            &mut layer_rm,
            (3, 5, feature_dim, feature_dim),
            Some(padding_mask_batch),
            STEPS,
            |gradient, gradient_rm, name| {
                assert_close_1d(&format!("{} gamma gradient", name), &gradient.get_gradient_gamma(), &gradient_rm.get_gradient_gamma());
                assert_close_1d(&format!("{} beta gradient", name), &gradient.get_gradient_beta(), &gradient_rm.get_gradient_beta());
            },
            |layer, layer_rm, name| {
                assert_close_1d(&format!("{} gamma", name), &layer.gamma, &layer_rm.gamma);
                assert_close_1d(&format!("{} beta", name), &layer.beta, &layer_rm.beta);
            },
        );
    }

    #[test]
    fn test_positional_encoding_layer_rm_matches_positional_encoding_layer() {
        let embedding_dim = 8;
        let mut layer = PositionalEncodingLayer::new(embedding_dim);
        let mut layer_rm = PositionalEncodingLayerRm::new(embedding_dim);

        compare_training_steps("PositionalEncodingLayerRm", &mut layer, &mut layer_rm, (2, 6, embedding_dim, embedding_dim), None, STEPS, |_, _, _| {}, |_, _, _| {});
    }

    #[test]
    fn test_feedforward_layer_rm_matches_feedforward_layer() {
        let (rows, cols) = (8, 6);
        let mut layer = FeedForwardLayer::new(rows, cols, 0.01);
        let mut layer_rm = FeedForwardLayerRm::new(rows, cols, 0.01);

        // Same parameters for the inner dense layers (SWiGLU -> LEAKYRELU).
        for (dense, dense_rm) in layer.layers.iter().zip(layer_rm.layers.iter_mut()) {
            match (dense, dense_rm) {
                (LayerEnum::Dense(dense), LayerEnum::DenseRm(dense_rm)) => {
                    assert_eq!(dense.activation_type, dense_rm.activation_type);
                    dense_rm.weights = RowMajorMatrix::from_rows(&dense.weights);
                    dense_rm.bias = dense.bias.clone();
                }
                _ => panic!("unexpected FFN layer layout"),
            }
        }

        compare_training_steps(
            "FeedForwardLayerRm",
            &mut layer,
            &mut layer_rm,
            (2, 5, rows, rows),
            None,
            STEPS,
            |_, _, _| {},
            |layer, layer_rm, name| {
                for (i, (dense, dense_rm)) in layer.layers.iter().zip(layer_rm.layers.iter()).enumerate() {
                    if let (LayerEnum::Dense(dense), LayerEnum::DenseRm(dense_rm)) = (dense, dense_rm) {
                        assert_close_2d(&format!("{} dense {} weights", name, i), &dense.weights, &dense_rm.weights);
                        assert_close_1d(&format!("{} dense {} bias", name, i), &dense.bias, &dense_rm.bias);
                    }
                }
                if let (Some(LayerEnum::Norm(norm)), Some(LayerEnum::NormRm(norm_rm))) = (&layer.norm_layer, &layer_rm.norm_layer) {
                    assert_close_1d(&format!("{} norm gamma", name), &norm.gamma, &norm_rm.gamma);
                    assert_close_1d(&format!("{} norm beta", name), &norm.beta, &norm_rm.beta);
                } else {
                    panic!("both FFN layers must have a norm layer");
                }
            },
        );
    }
}
