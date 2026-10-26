//! Every `*Rm` layer must produce the same outputs, gradients and parameter updates as its Vec counterpart.
#[cfg(test)]
mod test_layers_rm {
    use std::{
        collections::HashMap,
        sync::{Arc, RwLock},
    };

    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
        network_layers::{
            clustering_linear_layer::ClusteringLinearLayer,
            complex_to_linear_layer::ComplexToLinearLayer,
            eml_linear_layer::EmlLinearLayer,
            embedding_layer::EmbeddingLayer,
            feedforward_layer::FeedForwardLayer,
            layer::LayerEnum,
            linear_layer::LinearLayer,
            multi_linear_layer::MultiLinearLayer,
            network_layers_rm::{
                adaptive_avg_pool1d_layer_rm::AdaptiveAvgPool1dLayerRm,
                adaptive_linear_layer_rm::AdaptiveLinearLayerRm,
                eml_linear_layer_rm::EmlLinearLayerRm,
                complex_to_linear_layer_rm::ComplexToLinearLayerRm, embedding_layer_rm::EmbeddingLayerRm, feedforward_layer_rm::FeedForwardLayerRm, linear_layer_rm::LinearLayerRm, multi_linear_layer_rm::MultiLinearLayerRm,
                norm_layer_rm::NormalNormLayerRm, positional_encoding_layer_rm::PositionalEncodingLayerRm, rms_norm_layer_rm::RMSNormLayerRm, softmax_output_layer_rm::SoftmaxLayerRm, sparse_linear_layer_rm::SparseLinearLayerRm, wavelet_complex_layer_rm::ComplexWaveletLayerRm,
                wavelet_discrete_layer_rm::DiscreteWaveletLayerRm,
            },
            adaptive_linear_layer::AdaptiveLinearLayer,
            adaptive_pooling::adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer,
            add_rms_norm_layer::RMSNormLayer,
            norm_layer::NormalNormLayer,
            positional_encoding_layer::PositionalEncodingLayer,
            softmax_output_layer::SoftmaxLayer,
            wavelet_complex_layer::ComplexWaveletLayer,
            wavelet_discrete_layer::DiscreteWaveletLayer,
            wavelet_network::DECOMPOSITION_LEVELS,
        },
        network_types::neural_network_generic::OperationMode,
        utils::{
            dtype::C,
            matrix_rm::RowMajorMatrix,
            random_arrays::{generate_random_complex_2d, generate_random_complex_3d},
            shared_f32_matrix::SharedF32Matrix,
        },
    };

    use super::super::rm_test_utils::{
        assert_close_1d, assert_close_2d, assert_close_3d, assert_close_real_2d, assert_dense_gradients_close, assert_input_gradients_close, assert_linear_gradients_close, assert_norm_gradients_close, compare_training_steps, copy_parameters, layer_inputs, layer_inputs_with_mask, to_rm_batch, upstream_gradients,
    };

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
            (3, 5, input_dim),
            None,
            STEPS,
            |_, _, gradient, gradient_rm, name| {
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
            (3, 5, feature_dim),
            Some(padding_mask_batch),
            STEPS,
            |_, _, gradient, gradient_rm, name| {
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

        compare_training_steps("PositionalEncodingLayerRm", &mut layer, &mut layer_rm, (2, 6, embedding_dim), None, STEPS, |_, _, _, _, _| {}, |_, _, _| {});
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
            (2, 5, rows),
            None,
            STEPS,
            |layer, layer_rm, _, _, name| {
                for (i, (dense, dense_rm)) in layer.layers.iter().zip(layer_rm.layers.iter()).enumerate() {
                    match (dense, dense_rm) {
                        (LayerEnum::Dense(dense), LayerEnum::DenseRm(dense_rm)) => assert_dense_gradients_close(&format!("{} dense {}", name, i), dense, dense_rm),
                        _ => panic!("unexpected FFN layer layout"),
                    }
                }
                match (&layer.norm_layer, &layer_rm.norm_layer) {
                    (Some(LayerEnum::Norm(norm)), Some(LayerEnum::NormRm(norm_rm))) => assert_norm_gradients_close(&format!("{} norm", name), norm, norm_rm),
                    _ => panic!("both FFN layers must have a norm layer"),
                }
            },
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

    #[test]
    fn test_complex_to_linear_layer_rm_matches_complex_to_linear_layer() {
        let (input_dim, output_dim) = (8, 6);
        let mut layer = ComplexToLinearLayer::new(input_dim, output_dim, 0.01);
        let mut layer_rm = ComplexToLinearLayerRm::new(input_dim, output_dim, 0.01);
        layer_rm.weights_1 = RowMajorMatrix::from_rows(&layer.weights_1);
        layer_rm.weights_2 = RowMajorMatrix::from_rows(&layer.weights_2);

        compare_training_steps(
            "ComplexToLinearLayerRm",
            &mut layer,
            &mut layer_rm,
            (3, 5, input_dim),
            None,
            STEPS,
            |_, _, gradient, gradient_rm, name| {
                assert_close_2d(&format!("{} weights_1 gradient", name), &gradient.get_gradient_weights(), &gradient_rm.get_weight_gradient_rm("weights").unwrap());
                assert_close_2d(&format!("{} weights_2 gradient", name), &gradient.get_gradient_weights_2(), &gradient_rm.get_weight_gradient_rm("weights_2").unwrap());
            },
            |layer, layer_rm, name| {
                assert_close_2d(&format!("{} weights_1", name), &layer.weights_1, &layer_rm.weights_1);
                assert_close_2d(&format!("{} weights_2", name), &layer.weights_2, &layer_rm.weights_2);
            },
        );
    }

    #[test]
    fn test_complex_wavelet_layer_rm_matches_complex_wavelet_layer() {
        for is_full_mode in [false, true] {
            let mut layer = ComplexWaveletLayer::new();
            let mut layer_rm = ComplexWaveletLayerRm::new();
            layer.is_full_mode = is_full_mode;
            layer_rm.is_full_mode = is_full_mode;

            compare_training_steps(&format!("ComplexWaveletLayerRm full={}", is_full_mode), &mut layer, &mut layer_rm, (2, 6, 8), None, 1, |_, _, _, _, _| {}, |_, _, _| {});
        }
    }

    #[test]
    fn test_discrete_wavelet_layer_rm_matches_discrete_wavelet_layer() {
        for add_details in [false, true] {
            let mut layer = DiscreteWaveletLayer::new();
            let mut layer_rm = DiscreteWaveletLayerRm::new();
            layer.add_details = add_details;
            layer_rm.add_details = add_details;

            compare_training_steps(&format!("DiscreteWaveletLayerRm add_details={}", add_details), &mut layer, &mut layer_rm, (2, 32, 8), None, 1, |_, _, _, _, _| {}, |_, _, _| {});
        }
    }

    #[test]
    fn test_softmax_layer_rm_matches_softmax_layer() {
        let (batch_size, seq_len, vocab_dim) = (3, 6, 10);
        let logits = generate_random_complex_3d(batch_size, seq_len, vocab_dim);
        // Second sequence: last position is padding. Targets cover the last unpadded positions.
        let padding_mask_batch = vec![vec![1; seq_len], vec![1, 1, 1, 1, 1, 0], vec![1; seq_len]];
        let target_batch = vec![vec![2, 3, 4], vec![5, 6, 7], vec![8, 9, 0]];

        let (input, input_rm) = layer_inputs_with_mask(&logits, &padding_mask_batch);

        // Training: CE loss and logit gradients
        let mut layer = SoftmaxLayer::new(0.01, OperationMode::TRAINING, vocab_dim);
        let mut layer_rm = SoftmaxLayerRm::new(0.01, OperationMode::TRAINING, vocab_dim);
        layer.forward_inner(&input, Some(padding_mask_batch.clone()), Some(target_batch.clone()));
        layer_rm.forward_inner(&input_rm, Some(padding_mask_batch.clone()), Some(target_batch.clone()));

        let losses = layer.cross_entropy_loss_batch.as_ref().unwrap();
        let losses_rm = layer_rm.cross_entropy_loss_batch.as_ref().unwrap();
        for (b, (l, l_rm)) in losses.iter().zip(losses_rm.iter()).enumerate() {
            let flat: Vec<_> = l.iter().flatten().copied().collect();
            let flat_rm: Vec<_> = l_rm.iter().flatten().copied().collect();
            assert_close_1d(&format!("softmax loss [batch {}]", b), &flat, &flat_rm);
        }

        let gradient = layer.backward(&Gradient::new_default());
        let gradient_rm = layer_rm.backward(&Gradient::new_default());
        assert_eq!(gradient.get_total_valid_tokens(), gradient_rm.get_total_valid_tokens());
        assert_input_gradients_close("softmax", &gradient, &gradient_rm);

        // Inference: softmax of the last row
        let mut layer = SoftmaxLayer::new(0.01, OperationMode::PRODUCTION, vocab_dim);
        let mut layer_rm = SoftmaxLayerRm::new(0.01, OperationMode::PRODUCTION, vocab_dim);
        let output = layer.forward(&input).get_output_batch();
        let output_rm = layer_rm.forward(&input_rm);
        assert_close_3d("softmax production output", &output, output_rm.get_output_batch_rm_ref().unwrap());
    }

    #[test]
    fn test_embedding_layer_rm_matches_embedding_layer_tied() {
        let (vocab_size, embedding_dim) = (12, 8);

        // The RM layer stores tied weights as f32, so use real-valued embeddings for both.
        let weights_f32: Vec<Vec<f32>> = (0..vocab_size).map(|v| (0..embedding_dim).map(|d| ((v * embedding_dim + d) as f32 * 0.37).sin()).collect()).collect();

        let mut layer = EmbeddingLayer::new(vocab_size, embedding_dim * 2usize.pow(DECOMPOSITION_LEVELS));
        assert_eq!(layer.embedding_dim, embedding_dim);
        let accumulator = Arc::new(RwLock::new(HashMap::new()));
        layer.set_tied_weights(weights_f32.iter().map(|row| row.iter().map(|&w| C::new(w as f64, 0.0)).collect()).collect(), accumulator.clone());

        let mut layer_rm = EmbeddingLayerRm::load(vocab_size, embedding_dim * 2usize.pow(DECOMPOSITION_LEVELS));
        let accumulator_rm = Arc::new(RwLock::new(HashMap::new()));
        layer_rm.set_tied_weights(SharedF32Matrix::new(weights_f32), accumulator_rm.clone());

        // Ragged batch -> padded with token id 1
        let batch_ids: Vec<Vec<u32>> = vec![vec![2, 5, 7, 11], vec![3, 3, 9]];
        let mut input = LayerInput::new_default();
        input.set_batch_ids(batch_ids.clone());

        let output = layer.forward(&input);
        let output_rm = layer_rm.forward(&input);
        assert_close_3d("embedding output", &output.get_output_batch(), output_rm.get_output_batch_rm_ref().unwrap());
        assert_eq!(output.get_padding_mask_batch(), output_rm.get_padding_mask_batch());

        let upstream_batch: Vec<Vec<Vec<C>>> = output.get_output_batch().iter().map(|m| generate_random_complex_2d(m.len(), embedding_dim)).collect();
        let (upstream, upstream_rm) = upstream_gradients(&upstream_batch);
        let gradient = layer.backward(&upstream);
        let gradient_rm = layer_rm.backward(&upstream_rm);
        assert_input_gradients_close("embedding", &gradient, &gradient_rm);

        layer.update_parameters();
        layer_rm.update_parameters();

        let accumulated = accumulator.read().unwrap();
        let accumulated_rm = accumulator_rm.read().unwrap();
        assert_eq!(accumulated.len(), accumulated_rm.len(), "embedding: different tokens accumulated");
        for (token, grad) in accumulated.iter() {
            assert_close_1d(&format!("embedding tied gradient token {}", token), grad, &accumulated_rm[token]);
        }
    }

    /// Untied: both layers keep the embedding table in the layer and train it in place.
    #[test]
    fn test_embedding_layer_rm_matches_embedding_layer() {
        let (vocab_size, embedding_dim) = (12, 8);
        let mut layer = EmbeddingLayer::new(vocab_size, embedding_dim * 2usize.pow(DECOMPOSITION_LEVELS));
        let mut layer_rm = EmbeddingLayerRm::new(vocab_size, embedding_dim * 2usize.pow(DECOMPOSITION_LEVELS));
        assert_eq!((layer.embedding_dim, layer_rm.embedding_dim), (embedding_dim, embedding_dim));
        layer_rm.weights = RowMajorMatrix::from_rows(&layer.tied_weights);

        // Ragged batch -> padded with token id 1; token 3 appears twice.
        let batch_ids: Vec<Vec<u32>> = vec![vec![2, 5, 7, 11], vec![3, 3, 9]];
        for step in 1..=STEPS {
            let name = format!("EmbeddingLayerRm step {}", step);
            let mut input = LayerInput::new_default();
            input.set_batch_ids(batch_ids.clone());

            let output = layer.forward(&input);
            let output_rm = layer_rm.forward(&input);
            assert_close_3d(&format!("{} output", name), &output.get_output_batch(), output_rm.get_output_batch_rm_ref().unwrap());
            assert_eq!(output.get_padding_mask_batch(), output_rm.get_padding_mask_batch());

            let upstream_batch: Vec<Vec<Vec<C>>> = output.get_output_batch().iter().map(|m| generate_random_complex_2d(m.len(), embedding_dim)).collect();
            let (upstream, upstream_rm) = upstream_gradients(&upstream_batch);
            let gradient = layer.backward(&upstream);
            let gradient_rm = layer_rm.backward(&upstream_rm);
            assert_input_gradients_close(&name, &gradient, &gradient_rm);

            layer.update_parameters();
            layer_rm.update_parameters();
            assert_close_2d(&format!("{} embedding table", name), &layer.tied_weights, &layer_rm.weights);
        }
    }

    #[test]
    fn test_sparse_linear_layer_rm_matches_clustering_linear_layer() {
        let (vocab_size, embedding_dim, batch_size, seq_len, k) = (64, 8, 2, 5, 10);
        let mut layer = ClusteringLinearLayer::new(0.01, embedding_dim, vocab_size);

        // Same weights and k-means index for the RM layer.
        let mut layer_rm = SparseLinearLayerRm::new(0.01, embedding_dim, vocab_size);
        layer_rm.weights = SharedF32Matrix::new(layer.weights.read().clone());
        layer_rm.previous_weights = layer.previous_weights.clone();
        layer_rm.bias = layer.bias.clone();
        layer_rm.centroids = layer.centroids.clone();
        layer_rm.assignments = layer.assignments.clone();
        layer_rm.cluster_to_tokens = layer.cluster_to_tokens.clone();

        let padding_mask_batch = vec![vec![1; seq_len], vec![1, 1, 1, 1, 0]];
        let target_batch: Vec<Vec<u32>> = vec![vec![3, 17, 42], vec![5, 60, 9]];

        for step in 1..=2 {
            let name = format!("SparseLinearLayerRm step {}", step);
            let input_batch = generate_random_complex_3d(batch_size, seq_len, embedding_dim);
            let (mut input, mut input_rm) = layer_inputs_with_mask(&input_batch, &padding_mask_batch);
            for li in [&mut input, &mut input_rm] {
                li.set_target_batch_ids(target_batch.clone());
                li.set_top_k_size(k);
                li.set_time_step(step);
            }

            let output = layer.forward(&input);
            let output_rm = layer_rm.forward(&input_rm);
            assert_eq!(output.get_output_indices(), output_rm.get_output_indices(), "{}: selected vocab indices differ", name);
            assert_close_3d(&format!("{} output", name), &output.get_output_batch(), output_rm.get_output_batch_rm_ref().unwrap());

            let (upstream, upstream_rm) = upstream_gradients(&generate_random_complex_3d(batch_size, seq_len, k));
            let gradient = layer.backward(&upstream);
            let gradient_rm = layer_rm.backward(&upstream_rm);
            assert_input_gradients_close(&name, &gradient, &gradient_rm);
            assert_close_1d(&format!("{} bias gradient", name), &gradient.get_gradient_bias(), &gradient_rm.get_gradient_bias());
            for (row, (g, g_rm)) in gradient.get_gradient_weights().iter().zip(gradient_rm.get_gradient_weights().iter()).enumerate() {
                assert_close_1d(&format!("{} weight gradient row {}", name, row), g, g_rm);
            }

            layer.update_parameters();
            layer_rm.update_parameters();
            assert_eq!(*layer.weights.read(), *layer_rm.weights.read(), "{}: weights differ after update", name);
            assert_eq!(layer.bias, layer_rm.bias, "{}: bias differs after update", name);
        }
    }

    #[test]
    fn test_rms_norm_layer_rm_matches_rms_norm_layer() {
        let (batch_size, seq_len, dim) = (2, 4, 6);
        let mut layer = RMSNormLayer::new(dim, 1e-8, 0.01);
        let mut layer_rm = RMSNormLayerRm::new(dim, 1e-8, 0.01);

        for step in 1..=3 {
            let name = format!("RMSNormLayerRm step {}", step);
            let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let residual_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let (mut input, mut input_rm) = layer_inputs(&input_batch);
            input.set_input_batch_before(residual_batch.clone());
            input_rm.set_input_batch_before_rm(to_rm_batch(&residual_batch));

            let output = layer.forward(&input).get_output_batch();
            let output_rm = layer_rm.forward(&input_rm);
            assert_close_3d(&format!("{} output", name), &output, output_rm.get_output_batch_rm_ref().unwrap());

            let (upstream, upstream_rm) = upstream_gradients(&generate_random_complex_3d(batch_size, seq_len, dim));
            let gradient = layer.backward(&upstream);
            let gradient_rm = layer_rm.backward(&upstream_rm);
            assert_input_gradients_close(&name, &gradient, &gradient_rm);
            assert_close_1d(&format!("{} gamma gradient", name), &gradient.get_gradient_gamma(), &gradient_rm.get_gradient_gamma());

            layer.update_parameters();
            layer_rm.update_parameters();
            assert_close_1d(&format!("{} gamma", name), &layer.gamma, &layer_rm.gamma);
        }
    }

    #[test]
    fn test_multi_linear_layer_rm_matches_multi_linear_layer() {
        let (input_dim, output_dim, num_layers) = (8, 7, 3);
        let mut layer = MultiLinearLayer::new(0.01, input_dim, output_dim, num_layers);
        let mut layer_rm = MultiLinearLayerRm::new(0.01, input_dim, output_dim, num_layers);
        assert_eq!(layer.col_ranges, layer_rm.col_ranges);
        for (sub, sub_rm) in layer.layers.iter().zip(layer_rm.layers.iter_mut()) {
            sub_rm.weights = RowMajorMatrix::from_rows(&sub.weights);
            sub_rm.bias = sub.bias.clone();
        }

        compare_training_steps(
            "MultiLinearLayerRm",
            &mut layer,
            &mut layer_rm,
            (3, 5, input_dim),
            None,
            STEPS,
            |layer, layer_rm, _, _, name| {
                for (i, (sub, sub_rm)) in layer.layers.iter().zip(layer_rm.layers.iter()).enumerate() {
                    assert_linear_gradients_close(&format!("{} sub-layer {}", name, i), sub, sub_rm);
                }
            },
            |layer, layer_rm, name| {
                let (weights, bias) = layer.get_combined_weights();
                let (weights_rm, bias_rm) = layer_rm.get_combined_weights();
                assert_close_2d(&format!("{} weights", name), &weights, &weights_rm);
                assert_close_1d(&format!("{} bias", name), &bias, &bias_rm);
            },
        );
    }

    #[test]
    fn test_adaptive_avg_pool1d_layer_rm_matches_adaptive_avg_pool1d_layer() {
        // seq_len 10 -> 4 windows (pooling) and seq_len 3 -> passthrough
        for (seq_len, output_size) in [(10, 4), (3, 4)] {
            let name = format!("AdaptiveAvgPool1dLayerRm {}->{}", seq_len, output_size);
            let mut layer = AdaptiveAvgPool1dLayer::new(output_size);
            let mut layer_rm = AdaptiveAvgPool1dLayerRm::new(output_size);
            compare_training_steps(&name, &mut layer, &mut layer_rm, (2, seq_len, 6), None, 1, |_, _, _, _, _| {}, |_, _, _| {});
            assert_eq!(layer.metadata.window_mappings, layer_rm.metadata.window_mappings);

            // Gradient w.r.t. the decompressed sequence (original length)
            let (upstream, upstream_rm) = upstream_gradients(&generate_random_complex_3d(2, seq_len, 6));
            let gradient = layer.backward_through_decompress(&upstream);
            let gradient_rm = layer_rm.backward_through_decompress(&upstream_rm);
            assert_input_gradients_close(&format!("{} through decompress", name), &gradient, &gradient_rm);

            let compressed = generate_random_complex_3d(2, layer.metadata.compressed_length, 6);
            assert_close_3d(&format!("{} decompress", name), &layer.decompress(&compressed), &layer_rm.decompress(&to_rm_batch(&compressed)));
        }
    }

    /// Training input for the output heads: targets cover the last unpadded positions.
    const OUTPUT_HEAD_TARGETS: [[u32; 3]; 2] = [[3, 17, 25], [5, 30, 9]];

    fn output_head_inputs(input_batch: &Vec<Vec<Vec<C>>>, step: usize) -> (LayerInput, LayerInput) {
        output_head_inputs_with_targets(input_batch, step, OUTPUT_HEAD_TARGETS.iter().map(|t| t.to_vec()).collect())
    }

    fn output_head_inputs_with_targets(input_batch: &Vec<Vec<Vec<C>>>, step: usize, target_batch_ids: Vec<Vec<u32>>) -> (LayerInput, LayerInput) {
        let padding_mask_batch = vec![vec![1; 5], vec![1, 1, 1, 1, 0]];
        let (mut input, mut input_rm) = layer_inputs_with_mask(input_batch, &padding_mask_batch);
        for li in [&mut input, &mut input_rm] {
            li.set_target_batch_ids(target_batch_ids.clone());
            li.set_top_k_size(6);
            li.set_time_step(step);
            li.set_total_valid_tokens(6);
        }
        (input, input_rm)
    }

    fn flatten(rows: &Vec<Vec<C>>) -> Vec<C> {
        rows.iter().flatten().copied().collect()
    }

    fn assert_losses_close(name: &str, losses: &Vec<Vec<Vec<C>>>, losses_rm: &Vec<Vec<Vec<C>>>) {
        assert_eq!(losses.len(), losses_rm.len(), "{}: loss batch size", name);
        for (b, (l, l_rm)) in losses.iter().zip(losses_rm.iter()).enumerate() {
            let flat: Vec<C> = l.iter().flatten().copied().collect();
            let flat_rm: Vec<C> = l_rm.iter().flatten().copied().collect();
            assert_close_1d(&format!("{} loss [batch {}]", name, b), &flat, &flat_rm);
        }
    }

    #[test]
    fn test_adaptive_linear_layer_rm_matches_adaptive_linear_layer() {
        // vocab 40: every token is in the head cluster. vocab 2048 + 64: tail clusters exist, so the
        // routing weights get gradients (the targets mix head and tail tokens).
        for vocab in [40, 2048 + 64] {
            check_adaptive_linear_layer_rm(8, vocab);
        }
    }

    fn check_adaptive_linear_layer_rm(hidden: usize, vocab: usize) {
        let mut layer = AdaptiveLinearLayer::new(0.01, hidden, vocab);
        let mut layer_rm: AdaptiveLinearLayerRm = copy_parameters(&layer);
        let target_batch_ids: Vec<Vec<u32>> = if layer.frequency_clusters.len() > 1 {
            let tail = |cluster: usize, i: usize| layer.frequency_clusters[cluster][i] as u32;
            vec![vec![3, tail(1, 0), tail(2, 1)], vec![tail(1, 2), 30, tail(3, 0)]]
        } else {
            OUTPUT_HEAD_TARGETS.iter().map(|t| t.to_vec()).collect()
        };
        assert!(layer.frequency_clusters.len() == 1 || !layer.weights.is_empty(), "vocab {}: expected routing weights", vocab);

        for step in 1..=3 {
            let name = format!("AdaptiveLinearLayerRm vocab {} step {}", vocab, step);
            let (input, input_rm) = output_head_inputs_with_targets(&generate_random_complex_3d(2, 5, hidden), step, target_batch_ids.clone());

            let output = layer.forward(&input);
            let output_rm = layer_rm.forward(&input_rm);
            assert_close_3d(&format!("{} logits", name), &output.get_output_batch(), output_rm.get_output_batch_rm_ref().unwrap());
            assert_eq!(output.get_output_indices(), output_rm.get_output_indices(), "{}: indices", name);
            assert_losses_close(&name, &output.get_cross_entropy_loss_batch(), &output_rm.get_cross_entropy_loss_batch());
            assert_input_gradients_close(&name, layer.gradient.as_ref().unwrap(), layer_rm.gradient().unwrap());
            let (gradient, gradient_rm) = (layer.gradient.as_ref().unwrap(), layer_rm.gradient().unwrap());
            assert_close_1d(&format!("{} routing weight gradient", name), &flatten(&gradient.get_gradient_weights()), &flatten(&gradient_rm.get_gradient_weights()));
            assert_close_1d(&format!("{} routing bias gradient", name), &gradient.get_gradient_bias(), &gradient_rm.get_gradient_bias());
            let (word_w, word_w_rm) = (layer.cluster_word_weight_gradients().unwrap(), layer_rm.cluster_word_weight_gradients().unwrap());
            assert_eq!(word_w.len(), word_w_rm.len(), "{}: cluster count", name);
            for (cluster, (g, g_rm)) in word_w.iter().zip(word_w_rm.iter()).enumerate() {
                assert_close_1d(&format!("{} word weight gradient [cluster {}]", name, cluster), &flatten(g), &flatten(g_rm));
            }
            assert_close_1d(&format!("{} word bias gradient", name), &flatten(layer.cluster_word_bias_gradients().unwrap()), &flatten(layer_rm.cluster_word_bias_gradients().unwrap()));

            layer.update_parameters();
            layer_rm.update_parameters();
            assert_eq!(layer.weights, layer_rm.weights, "{}: routing weights", name);
            assert_eq!(layer.cluster_word_weights, layer_rm.cluster_word_weights, "{}: word table", name);
        }

        // Inference (no targets)
        let mut input = LayerInput::new_default();
        input.set_top_k_size(6);
        let mut input_rm = input.clone();
        let input_batch = generate_random_complex_3d(2, 5, hidden);
        input.set_input_batch(input_batch.clone());
        input_rm.set_input_batch_rm(to_rm_batch(&input_batch));
        let (output, output_rm) = (layer.forward(&input), layer_rm.forward(&input_rm));
        assert_close_3d("AdaptiveLinearLayerRm inference", &output.get_output_batch(), output_rm.get_output_batch_rm_ref().unwrap());
        assert_eq!(output.get_output_indices(), output_rm.get_output_indices());
    }

    #[test]
    fn test_eml_linear_layer_rm_matches_eml_linear_layer() {
        let (d_model, vocab) = (8, 40);
        let mut layer = EmlLinearLayer::new(0.01, d_model, vocab);
        let mut layer_rm: EmlLinearLayerRm = copy_parameters(&layer);

        for step in 1..=3 {
            let name = format!("EmlLinearLayerRm step {}", step);
            let (input, input_rm) = output_head_inputs(&generate_random_complex_3d(2, 5, d_model), step);

            let output = layer.forward(&input);
            let output_rm = layer_rm.forward(&input_rm);
            assert_losses_close(&name, &output.get_cross_entropy_loss_batch(), &output_rm.get_cross_entropy_loss_batch());
            assert_input_gradients_close(&name, layer.gradient.as_ref().unwrap(), layer_rm.gradient().unwrap());
            for (which, g, g_rm) in [
                ("w_uc", layer.grad_w_uc(), layer_rm.grad_w_uc()),
                ("w_vc", layer.grad_w_vc(), layer_rm.grad_w_vc()),
                ("w_uw", layer.grad_w_uw(), layer_rm.grad_w_uw()),
                ("w_vw", layer.grad_w_vw(), layer_rm.grad_w_vw()),
            ] {
                assert_close_real_2d(&format!("{} {} gradient", name, which), g.unwrap(), g_rm.unwrap());
            }
            let (scalars, scalars_rm) = (layer.grad_scalars().unwrap(), layer_rm.grad_scalars().unwrap());
            assert_close_real_2d(&format!("{} alpha/beta gradients", name), &vec![vec![scalars.0, scalars.1, scalars.2, scalars.3]], &vec![vec![scalars_rm.0, scalars_rm.1, scalars_rm.2, scalars_rm.3]]);

            layer.update_parameters();
            layer_rm.update_parameters();
            assert_eq!(bincode::serialize(&layer).unwrap(), bincode::serialize(&layer_rm).unwrap(), "{}: parameters", name);
        }

        // Inference (no targets)
        let mut input = LayerInput::new_default();
        input.set_top_k_size(4);
        let mut input_rm = input.clone();
        let input_batch = generate_random_complex_3d(2, 5, d_model);
        input.set_input_batch(input_batch.clone());
        input_rm.set_input_batch_rm(to_rm_batch(&input_batch));
        let (output, output_rm) = (layer.forward(&input), layer_rm.forward(&input_rm));
        assert_close_3d("EmlLinearLayerRm inference", &output.get_output_batch(), output_rm.get_output_batch_rm_ref().unwrap());
        assert_eq!(output.get_output_indices(), output_rm.get_output_indices());
    }
}
