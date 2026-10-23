//! The RM attention structs must produce the same outputs, gradients and parameter updates as their Vec counterparts.
#[cfg(test)]
mod test_attention_rm {
    use crate::neural_networks::{
        network_components::layer_input_struct::LayerInput,
        network_layers::{layer::LayerEnum, network_layers_rm::complex_to_linear_layer_rm::ComplexToLinearLayerRm},
        network_types::transformer::{
            masked_attention_head::MaskedAttentionHead,
            masked_attention_head_approximation::MaskedAttentionHeadApproximation,
            masked_attention_head_approximation_rm::MaskedAttentionHeadApproximationRm,
            masked_attention_head_rm::MaskedAttentionHeadRm,
            self_attention_layer::SelfAttentionLayer,
            self_attention_layer_rm::SelfAttentionLayerRm,
            self_attention_layer_approximation::SelfAttentionLayerApproximation,
            self_attention_layer_approximation_rm::SelfAttentionLayerApproximationRm,
            sparse_masked_attention_head::SparseMaskedAttentionHead, sparse_masked_attention_head_rm::SparseMaskedAttentionHeadRm, sparse_self_attention_layer::SparseSelfAttentionLayer,
            sparse_self_attention_layer_rm::SparseSelfAttentionLayerRm,
        },
        network_layers::complex_to_linear_layer::ComplexToLinearLayer,
        utils::{
            dtype::C,
            matrix_rm::RowMajorMatrix,
            random_arrays::{generate_random_complex_2d, generate_random_complex_3d},
        },
    };

    use super::super::rm_test_utils::{assert_close_1d, assert_close_2d, assert_close_3d, assert_input_gradients_close, copy_parameters, layer_inputs, layer_inputs_with_mask, to_rm_batch, upstream_gradients};

    fn copy_ctl(ctl: &Option<ComplexToLinearLayer>, ctl_rm: &mut Option<ComplexToLinearLayerRm>) {
        match (ctl, ctl_rm) {
            (Some(ctl), Some(ctl_rm)) => {
                ctl_rm.weights_1 = RowMajorMatrix::from_rows(&ctl.weights_1);
                ctl_rm.weights_2 = RowMajorMatrix::from_rows(&ctl.weights_2);
            }
            (None, ctl_rm) => *ctl_rm = None,
            _ => panic!("CTL layout differs"),
        }
    }

    fn copy_sparse_head(head: &SparseMaskedAttentionHead, head_rm: &mut SparseMaskedAttentionHeadRm) {
        head_rm.weights_q_rm = RowMajorMatrix::from_rows(&head.weights_q);
        head_rm.weights_k_rm = RowMajorMatrix::from_rows(&head.weights_k);
        head_rm.weights_v_rm = RowMajorMatrix::from_rows(&head.weights_v);
        head_rm.positional_encoding_layer.base = head.positional_encoding_layer.base;
        copy_ctl(&head.ctl_q, &mut head_rm.ctl_q);
        copy_ctl(&head.ctl_k, &mut head_rm.ctl_k);
    }

    fn assert_sparse_head_parameters_close(name: &str, head: &SparseMaskedAttentionHead, head_rm: &SparseMaskedAttentionHeadRm) {
        assert_close_2d(&format!("{} w_q", name), &head.weights_q, &head_rm.weights_q_rm);
        assert_close_2d(&format!("{} w_k", name), &head.weights_k, &head_rm.weights_k_rm);
        assert_close_2d(&format!("{} w_v", name), &head.weights_v, &head_rm.weights_v_rm);
        for (ctl, ctl_rm, which) in [(&head.ctl_q, &head_rm.ctl_q, "ctl_q"), (&head.ctl_k, &head_rm.ctl_k, "ctl_k")] {
            let (ctl, ctl_rm) = (ctl.as_ref().unwrap(), ctl_rm.as_ref().unwrap());
            assert_close_2d(&format!("{} {} weights_1", name, which), &ctl.weights_1, &ctl_rm.weights_1);
            assert_close_2d(&format!("{} {} weights_2", name, which), &ctl.weights_2, &ctl_rm.weights_2);
        }
    }

    #[test]
    fn test_sparse_attention_head_rm_matches_sparse_attention_head() {
        let (batch_size, seq_len, dim, window_size) = (2, 9, 8, 2);
        let mut head = SparseMaskedAttentionHead::new(dim, dim, window_size, 0.01);
        let mut head_rm = SparseMaskedAttentionHeadRm::new(dim, dim, window_size, 0.01);
        copy_sparse_head(&head, &mut head_rm);

        // The last two positions of the second sequence are padding.
        let padding_mask_batch = vec![vec![1; seq_len], vec![1, 1, 1, 1, 1, 1, 1, 0, 0]];

        for step in 1..=3 {
            let name = format!("SparseMaskedAttentionHeadRm step {}", step);
            let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let (mut input, mut input_rm) = layer_inputs_with_mask(&input_batch, &padding_mask_batch);
            input.set_time_step(step);
            input_rm.set_time_step(step);

            let output = head.forward(&input).get_output_batch();
            let output_rm = head_rm.forward(&input_rm);
            assert_close_3d(&format!("{} output", name), &output, output_rm.get_output_batch_rm_ref().unwrap());

            let upstream_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let gradient = head.backward(&upstream_batch);
            let gradient_rm = head_rm.backward_rm(&to_rm_batch(&upstream_batch));
            assert_input_gradients_close(&name, &gradient, &gradient_rm);
            assert_close_2d(&format!("{} w_q gradient", name), &gradient.get_gradient_weights_q(), &gradient_rm.get_weight_gradient_rm("w_q").unwrap());
            assert_close_2d(&format!("{} w_k gradient", name), &gradient.get_gradient_weights_k(), &gradient_rm.get_weight_gradient_rm("w_k").unwrap());
            assert_close_2d(&format!("{} w_v gradient", name), &gradient.get_gradient_weights_v(), &gradient_rm.get_weight_gradient_rm("w_v").unwrap());

            head.update_parameters();
            head_rm.update_parameters();
            assert_sparse_head_parameters_close(&name, &head, &head_rm);
        }
    }

    #[test]
    fn test_sparse_attention_head_rm_cached_decoding_matches_sparse_attention_head() {
        let (batch_size, prompt_len, dim, window_size) = (2, 6, 8, 2);
        let mut head = SparseMaskedAttentionHead::new(dim, dim, window_size, 0.01);
        let mut head_rm = SparseMaskedAttentionHeadRm::new(dim, dim, window_size, 0.01);
        copy_sparse_head(&head, &mut head_rm);

        // Prompt first, then one token per step.
        let mut chunks = vec![generate_random_complex_3d(batch_size, prompt_len, dim)];
        for _ in 0..3 {
            chunks.push((0..batch_size).map(|_| generate_random_complex_2d(1, dim)).collect());
        }

        for (step, chunk) in chunks.iter().enumerate() {
            let mut input = LayerInput::new_default();
            input.set_forward_only(true);
            input.set_calculate_k_v_cache(true);
            let mut input_rm = input.clone();
            input.set_input_batch(chunk.clone());
            input_rm.set_input_batch_rm(to_rm_batch(chunk));

            let output = head.forward(&input).get_output_batch();
            let output_rm = head_rm.forward(&input_rm);
            assert_close_3d(&format!("cached decoding step {}", step), &output, output_rm.get_output_batch_rm_ref().unwrap());
        }
    }

    #[test]
    fn test_sparse_self_attention_layer_rm_matches_sparse_self_attention_layer() {
        let (batch_size, seq_len, dim, num_heads, window_size) = (2, 7, 8, 2, 2);
        let mut layer = SparseSelfAttentionLayer::new(num_heads, dim, dim, window_size, 0.01);
        let mut layer_rm = SparseSelfAttentionLayerRm::new(num_heads, dim, dim, window_size, 0.01);
        for (head, head_rm) in layer.attention_heads.iter().zip(layer_rm.attention_heads.iter_mut()) {
            copy_sparse_head(head, head_rm);
        }
        let Some(LayerEnum::Norm(norm)) = &layer.norm_layer else { panic!("SparseSelfAttentionLayer is expected to use a Norm layer") };
        assert_eq!(norm.epsilon, layer_rm.norm_layer.epsilon);

        let padding_mask_batch = vec![vec![1; seq_len]; batch_size];

        for step in 1..=3 {
            let name = format!("SparseSelfAttentionLayerRm step {}", step);
            let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let (mut input, mut input_rm) = layer_inputs_with_mask(&input_batch, &padding_mask_batch);
            input.set_time_step(step);
            input_rm.set_time_step(step);

            let output = layer.forward(&input).get_output_batch();
            let output_rm = layer_rm.forward(&input_rm);
            assert_close_3d(&format!("{} output", name), &output, output_rm.get_output_batch_rm_ref().unwrap());

            let upstream_batch: Vec<Vec<Vec<C>>> = generate_random_complex_3d(batch_size, seq_len, dim);
            let (upstream, upstream_rm) = upstream_gradients(&upstream_batch);
            let gradient = layer.backward(&upstream);
            let gradient_rm = layer_rm.backward(&upstream_rm);
            assert_input_gradients_close(&name, &gradient, &gradient_rm);

            layer.update_parameters();
            layer_rm.update_parameters();
            for (h, (head, head_rm)) in layer.attention_heads.iter().zip(layer_rm.attention_heads.iter()).enumerate() {
                assert_sparse_head_parameters_close(&format!("{} head {}", name, h), head, head_rm);
            }
        }
    }

    #[test]
    fn test_attention_head_approximation_rm_input_matches_vec_input() {
        // seq_len and head dim must stay <= 5 (size of bias_pos).
        let (batch_size, seq_len, dim, head_dim) = (2, 5, 6, 4);
        let mut head = MaskedAttentionHeadApproximation::new(dim, head_dim, 0.01);
        let mut head_rm: MaskedAttentionHeadApproximationRm = copy_parameters(&head);
        let padding_mask_batch = vec![vec![1; seq_len], vec![1, 1, 1, 1, 0]];

        for step in 1..=3 {
            let name = format!("MaskedAttentionHeadApproximation RM step {}", step);
            let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let (mut input, mut input_rm) = layer_inputs_with_mask(&input_batch, &padding_mask_batch);
            input.set_time_step(step);
            input_rm.set_time_step(step);

            let output = head.forward(&input).get_output_batch();
            let output_rm = head_rm.forward(&input_rm);
            assert_close_3d(&format!("{} output", name), &output, output_rm.get_output_batch_rm_ref().unwrap());

            let upstream_batch = generate_random_complex_3d(batch_size, seq_len, head_dim);
            let gradient = head.backward(&upstream_batch);
            let gradient_rm = head_rm.backward_rm(&to_rm_batch(&upstream_batch));
            assert_input_gradients_close(&name, &gradient, &gradient_rm);
            for (which, g, g_rm) in [
                ("w_q", gradient.get_gradient_weights_q(), gradient_rm.get_gradient_weights_q()),
                ("w_k", gradient.get_gradient_weights_k(), gradient_rm.get_gradient_weights_k()),
                ("w_v", gradient.get_gradient_weights_v(), gradient_rm.get_gradient_weights_v()),
            ] {
                assert_close_2d(&format!("{} {} gradient", name, which), &g, &RowMajorMatrix::from_rows(&g_rm));
            }

            head.update_parameters();
            head_rm.update_parameters();
            assert_close_2d(&format!("{} w_q", name), &head.weights_q, &RowMajorMatrix::from_rows(&head_rm.weights_q));
            assert_close_2d(&format!("{} w_k", name), &head.weights_k, &RowMajorMatrix::from_rows(&head_rm.weights_k));
            assert_close_2d(&format!("{} w_v", name), &head.weights_v, &RowMajorMatrix::from_rows(&head_rm.weights_v));
        }
    }

    #[test]
    fn test_self_attention_layer_approximation_rm_matches_self_attention_layer_approximation() {
        let (batch_size, seq_len, dim, num_heads) = (2, 5, 8, 2);
        let mut layer = SelfAttentionLayerApproximation::new(num_heads, dim, dim, 0.01);
        let mut layer_rm = SelfAttentionLayerApproximationRm::new(num_heads, dim, dim, 0.01);
        layer_rm.attention_heads = layer.attention_heads.iter().map(copy_parameters).collect();

        let padding_mask_batch = vec![vec![1; seq_len]; batch_size];

        for step in 1..=3 {
            let name = format!("SelfAttentionLayerApproximationRm step {}", step);
            let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let (mut input, mut input_rm) = layer_inputs_with_mask(&input_batch, &padding_mask_batch);
            input.set_time_step(step);
            input_rm.set_time_step(step);

            let output = layer.forward(&input).get_output_batch();
            let output_rm = layer_rm.forward(&input_rm);
            assert_close_3d(&format!("{} output", name), &output, output_rm.get_output_batch_rm_ref().unwrap());

            let upstream_batch: Vec<Vec<Vec<C>>> = generate_random_complex_3d(batch_size, seq_len, dim);
            let gradient = layer.backward(&upstream_batch);
            let gradient_rm = layer_rm.backward_rm(&to_rm_batch(&upstream_batch));
            assert_input_gradients_close(&name, &gradient, &gradient_rm);

            layer.update_parameters();
            layer_rm.update_parameters();
            for (h, (head, head_rm)) in layer.attention_heads.iter().zip(layer_rm.attention_heads.iter()).enumerate() {
                assert_close_2d(&format!("{} head {} w_q", name, h), &head.weights_q, &RowMajorMatrix::from_rows(&head_rm.weights_q));
                assert_close_2d(&format!("{} head {} w_v", name, h), &head.weights_v, &RowMajorMatrix::from_rows(&head_rm.weights_v));
            }
        }
    }

    fn copy_masked_head(head: &MaskedAttentionHead, head_rm: &mut MaskedAttentionHeadRm) {
        head_rm.weights_q = RowMajorMatrix::from_rows(&head.weights_q);
        head_rm.weights_k = RowMajorMatrix::from_rows(&head.weights_k);
        head_rm.weights_v = RowMajorMatrix::from_rows(&head.weights_v);
        head_rm.positional_encoding_layer.base = head.positional_encoding_layer.base;
        copy_ctl(&head.ctl_q, &mut head_rm.ctl_q);
        copy_ctl(&head.ctl_k, &mut head_rm.ctl_k);
        head_rm.linear_v.weights = RowMajorMatrix::from_rows(&head.linear_v.weights);
        head_rm.linear_v.bias = head.linear_v.bias.clone();
        for (norm, norm_rm) in [(&head.norm_layer_k, &mut head_rm.norm_layer_k), (&head.norm_layer_v, &mut head_rm.norm_layer_v)] {
            norm_rm.gamma = norm.gamma.clone();
            norm_rm.beta = norm.beta.clone();
            norm_rm.epsilon = norm.epsilon;
        }
    }

    fn assert_masked_head_parameters_close(name: &str, head: &MaskedAttentionHead, head_rm: &MaskedAttentionHeadRm) {
        assert_close_2d(&format!("{} w_q", name), &head.weights_q, &head_rm.weights_q);
        assert_close_2d(&format!("{} w_k", name), &head.weights_k, &head_rm.weights_k);
        assert_close_2d(&format!("{} w_v", name), &head.weights_v, &head_rm.weights_v);
        assert_close_2d(&format!("{} linear_v", name), &head.linear_v.weights, &head_rm.linear_v.weights);
        assert_close_1d(&format!("{} norm_k gamma", name), &head.norm_layer_k.gamma, &head_rm.norm_layer_k.gamma);
        assert_close_1d(&format!("{} norm_v beta", name), &head.norm_layer_v.beta, &head_rm.norm_layer_v.beta);
        let (ctl, ctl_rm) = (head.ctl_q.as_ref().unwrap(), head_rm.ctl_q.as_ref().unwrap());
        assert_close_2d(&format!("{} ctl_q", name), &ctl.weights_1, &ctl_rm.weights_1);
    }

    #[test]
    fn test_masked_attention_head_rm_matches_masked_attention_head() {
        let (batch_size, seq_len, dim, head_dim) = (2, 6, 8, 4);
        let mut head = MaskedAttentionHead::new(dim, head_dim, 0.01);
        let mut head_rm = MaskedAttentionHeadRm::new(dim, head_dim, 0.01);
        copy_masked_head(&head, &mut head_rm);
        let padding_mask_batch = vec![vec![1; seq_len], vec![1, 1, 1, 1, 1, 0]];

        for step in 1..=3 {
            let name = format!("MaskedAttentionHeadRm step {}", step);
            let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let (mut input, mut input_rm) = layer_inputs_with_mask(&input_batch, &padding_mask_batch);
            input.set_time_step(step);
            input_rm.set_time_step(step);

            let output = head.forward(&input).get_output_batch();
            let output_rm = head_rm.forward(&input_rm);
            assert_close_3d(&format!("{} output", name), &output, output_rm.get_output_batch_rm_ref().unwrap());

            let upstream_batch = generate_random_complex_3d(batch_size, seq_len, head_dim);
            let gradient = head.backward(&upstream_batch);
            let gradient_rm = head_rm.backward_rm(&to_rm_batch(&upstream_batch));
            assert_input_gradients_close(&name, &gradient, &gradient_rm);
            assert_close_2d(&format!("{} w_q gradient", name), &gradient.get_gradient_weights_q(), &gradient_rm.get_weight_gradient_rm("w_q").unwrap());
            assert_close_2d(&format!("{} w_k gradient", name), &gradient.get_gradient_weights_k(), &gradient_rm.get_weight_gradient_rm("w_k").unwrap());
            assert_close_2d(&format!("{} w_v gradient", name), &gradient.get_gradient_weights_v(), &gradient_rm.get_weight_gradient_rm("w_v").unwrap());

            head.update_parameters();
            head_rm.update_parameters();
            assert_masked_head_parameters_close(&name, &head, &head_rm);
        }
    }

    #[test]
    fn test_masked_attention_head_rm_kv_cache_matches_masked_attention_head() {
        let (seq_len, dim, head_dim) = (4, 8, 4);
        let mut head = MaskedAttentionHead::new(dim, head_dim, 0.01);
        let mut head_rm = MaskedAttentionHeadRm::new(dim, head_dim, 0.01);
        copy_masked_head(&head, &mut head_rm);

        // The input always holds the full sequence; with a cache only the last token's K/V are computed.
        let mut sequence = generate_random_complex_2d(seq_len, dim);
        for step in 0..3 {
            if step > 0 {
                sequence.extend(generate_random_complex_2d(1, dim));
            }
            let input_batch = vec![sequence.clone()];
            let (mut input, mut input_rm) = layer_inputs(&input_batch);
            for li in [&mut input, &mut input_rm] {
                li.set_forward_only(true);
                li.set_calculate_k_v_cache(true);
            }

            let output = head.forward(&input).get_output_batch();
            let output_rm = head_rm.forward(&input_rm);
            assert_close_3d(&format!("kv cache step {}", step), &output, output_rm.get_output_batch_rm_ref().unwrap());
        }
    }

    #[test]
    fn test_self_attention_layer_rm_matches_self_attention_layer() {
        let (batch_size, seq_len, dim, num_heads) = (2, 5, 8, 2);
        let mut layer = SelfAttentionLayer::new(num_heads, dim, dim, 0.01);
        let mut layer_rm = SelfAttentionLayerRm::new(num_heads, dim, dim, 0.01);
        for (head, head_rm) in layer.attention_heads.iter().zip(layer_rm.attention_heads.iter_mut()) {
            copy_masked_head(head, head_rm);
        }
        let Some(LayerEnum::Norm(norm)) = &layer.norm_layer else { panic!("SelfAttentionLayer is expected to use a Norm layer") };
        assert_eq!(norm.epsilon, layer_rm.norm_layer.epsilon);

        let padding_mask_batch = vec![vec![1; seq_len]; batch_size];

        for step in 1..=3 {
            let name = format!("SelfAttentionLayerRm step {}", step);
            let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
            let (mut input, mut input_rm) = layer_inputs_with_mask(&input_batch, &padding_mask_batch);
            input.set_time_step(step);
            input_rm.set_time_step(step);

            let output = layer.forward(&input).get_output_batch();
            let output_rm = layer_rm.forward(&input_rm);
            assert_close_3d(&format!("{} output", name), &output, output_rm.get_output_batch_rm_ref().unwrap());

            let (upstream, upstream_rm) = upstream_gradients(&generate_random_complex_3d(batch_size, seq_len, dim));
            let gradient = layer.backward(&upstream);
            let gradient_rm = layer_rm.backward(&upstream_rm);
            assert_input_gradients_close(&name, &gradient, &gradient_rm);

            layer.update_parameters();
            layer_rm.update_parameters();
            for (h, (head, head_rm)) in layer.attention_heads.iter().zip(layer_rm.attention_heads.iter()).enumerate() {
                assert_masked_head_parameters_close(&format!("{} head {}", name, h), head, head_rm);
            }
        }
    }
}
