#[cfg(test)]
mod test_dense_layer_rm {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
        network_layers::{
            layer::{ActivationType, Layer, LayerType},
            network_layers_rm::layer_rm::LayerRm,
        },
        utils::{
            dtype::C,
            matrix::RowMajorMatrix,
            random_arrays::generate_random_complex_3d,
        },
    };

    const TOLERANCE: f64 = 1e-10;

    const ACTIVATIONS: [ActivationType; 11] = [
        ActivationType::LINEAR,
        ActivationType::TANH,
        ActivationType::SIGMOID,
        ActivationType::RELU,
        ActivationType::LEAKYRELU,
        ActivationType::ELU,
        ActivationType::SELU,
        ActivationType::GELU,
        ActivationType::SOFTSIGN,
        ActivationType::SOFTPLUS,
        ActivationType::SWiGLU,
    ];

    // Activations whose analytical derivative is the holomorphic one, so it can be checked against finite differences.
    const HOLOMORPHIC_ACTIVATIONS: [ActivationType; 6] = [
        ActivationType::LINEAR,
        ActivationType::TANH,
        ActivationType::SIGMOID,
        ActivationType::LEAKYRELU,
        ActivationType::GELU,
        ActivationType::SWiGLU,
    ];

    /// Creates a `Layer` and a `LayerRm` sharing the same weights and bias.
    fn create_layers(input_dim: usize, output_dim: usize, activation: &ActivationType) -> (Layer, LayerRm) {
        let learning_rate = 0.01;
        let mut layer = Layer::new(input_dim, output_dim, &learning_rate, activation, LayerType::DenseLayer);
        // Non-trivial bias so that bias gradients go through the whole chain.
        layer.bias = generate_random_complex_3d(1, 1, output_dim).remove(0).remove(0);

        let mut layer_rm = LayerRm::new(input_dim, output_dim, &learning_rate, activation, LayerType::DenseLayer);
        layer_rm.weights = RowMajorMatrix::from_rows(&layer.weights);
        layer_rm.bias = layer.bias.clone();

        (layer, layer_rm)
    }

    fn layer_input(input_batch: &Vec<Vec<Vec<C>>>) -> LayerInput {
        let mut input = LayerInput::new_default();
        input.set_input_batch(input_batch.clone());
        input.set_padding_mask_batch(vec![vec![1; input_batch[0].len()]; input_batch.len()]);
        input.set_batch_size(input_batch.len());
        input
    }

    fn layer_input_rm(input_batch: &Vec<Vec<Vec<C>>>) -> LayerInput {
        let mut input = LayerInput::new_default();
        input.set_input_batch_rm(input_batch.iter().map(|m| RowMajorMatrix::from_rows(m)).collect());
        input.set_padding_mask_batch(vec![vec![1; input_batch[0].len()]; input_batch.len()]);
        input.set_batch_size(input_batch.len());
        input
    }

    fn previous_gradient_rm(previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(previous_gradient_batch.iter().map(|m| RowMajorMatrix::from_rows(m)).collect());
        gradient
    }

    fn output_cols(output_dim: usize, activation: &ActivationType) -> usize {
        match activation {
            ActivationType::SWiGLU => output_dim / 2,
            _ => output_dim,
        }
    }

    fn assert_close_1d(name: &str, expected: &[C], actual: &[C]) {
        assert_eq!(expected.len(), actual.len(), "{}: length mismatch", name);
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).norm();
            let scale = e.norm().max(a.norm()).max(1.0);
            assert!(diff / scale <= TOLERANCE, "{}: mismatch at {}: expected {:?}, actual {:?}", name, i, e, a);
        }
    }

    fn assert_close_2d(name: &str, expected: &Vec<Vec<C>>, actual: &RowMajorMatrix<C>) {
        assert_eq!(expected.len(), actual.rows, "{}: row count mismatch", name);
        assert_eq!(expected[0].len(), actual.cols, "{}: col count mismatch", name);
        let flat: Vec<C> = expected.iter().flatten().copied().collect();
        assert_close_1d(name, &flat, &actual.data);
    }

    fn assert_close_3d(name: &str, expected: &Vec<Vec<Vec<C>>>, actual: &[RowMajorMatrix<C>]) {
        assert_eq!(expected.len(), actual.len(), "{}: batch size mismatch", name);
        for (batch_ind, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert_close_2d(&format!("{} [batch {}]", name, batch_ind), e, a);
        }
    }

    /// Runs forward + backward on `Layer` and `LayerRm` with identical parameters, inputs and upstream gradients
    /// and checks that outputs and all derivatives (input, weights, bias) are the same.
    fn check_layer_rm_matches_layer(activation: &ActivationType) {
        let batch_size = 3;
        let seq_len = 5;
        let input_dim = 8;
        let output_dim = 6;

        let (mut layer, mut layer_rm) = create_layers(input_dim, output_dim, activation);

        let input_batch = generate_random_complex_3d(batch_size, seq_len, input_dim);
        let previous_gradient_batch = generate_random_complex_3d(batch_size, seq_len, output_cols(output_dim, activation));

        // 1) Normal layer
        let output = layer.forward(&layer_input(&input_batch)).get_output_batch();
        let gradient = layer.backward(&previous_gradient_batch);

        // 2) Row-major layer
        let output_rm = layer_rm.forward(&layer_input_rm(&input_batch)).get_output_batch_rm();
        let gradient_rm = layer_rm.backward(&previous_gradient_rm(&previous_gradient_batch));

        let name = format!("{:?}", activation);
        assert_close_3d(&format!("{} output", name), &output, &output_rm);
        assert_close_3d(&format!("{} input gradient", name), &gradient.get_gradient_input_batch(), gradient_rm.get_gradient_input_batch_rm_ref().expect("RM input gradient missing"));
        assert_close_3d(&format!("{} weight gradient batch", name), &gradient.get_gradient_weight_batch(), gradient_rm.get_gradient_weight_batch_rm_ref().expect("RM weight gradient missing"));
        assert_close_2d(&format!("{} weight gradient", name), &gradient.get_gradient_weights(), &gradient_rm.get_gradient_weights_rm().expect("RM weight gradient missing"));
        assert_close_1d(&format!("{} bias gradient", name), &gradient.get_gradient_bias(), &gradient_rm.get_gradient_bias());
    }

    #[test]
    fn test_layer_rm_derivatives_match_layer() {
        for activation in ACTIVATIONS.iter() {
            check_layer_rm_matches_layer(activation);
        }
    }

    #[test]
    fn test_layer_rm_accepts_vec_gradient() {
        // A non-RM upstream layer only provides Vec<Vec<C>> gradients; LayerRm must convert them.
        let activation = ActivationType::GELU;
        let (mut layer, mut layer_rm) = create_layers(8, 6, &activation);
        let input_batch = generate_random_complex_3d(2, 4, 8);
        let previous_gradient_batch = generate_random_complex_3d(2, 4, 6);

        layer.forward(&layer_input(&input_batch));
        let gradient = layer.backward(&previous_gradient_batch);

        layer_rm.forward(&layer_input_rm(&input_batch));
        let mut previous_gradient = Gradient::new_default();
        previous_gradient.set_gradient_input_batch(previous_gradient_batch.clone());
        let gradient_rm = layer_rm.backward(&previous_gradient);

        assert_close_3d("input gradient", &gradient.get_gradient_input_batch(), gradient_rm.get_gradient_input_batch_rm_ref().unwrap());
        assert_close_2d("weight gradient", &gradient.get_gradient_weights(), &gradient_rm.get_gradient_weights_rm().unwrap());
    }

    #[test]
    fn test_layer_rm_update_parameters_matches_layer() {
        for activation in ACTIVATIONS.iter() {
            let (mut layer, mut layer_rm) = create_layers(8, 6, activation);

            // Several steps so the AdamW moment state (prev_m / prev_v / prev_v_hat) is exercised as well.
            for step in 1..=3 {
                let input_batch = generate_random_complex_3d(2, 4, 8);
                let previous_gradient_batch = generate_random_complex_3d(2, 4, output_cols(6, activation));

                let mut input = layer_input(&input_batch);
                input.set_time_step(step);
                layer.forward(&input);
                layer.backward(&previous_gradient_batch);
                layer.update_parameters();

                let mut input_rm = layer_input_rm(&input_batch);
                input_rm.set_time_step(step);
                layer_rm.forward(&input_rm);
                layer_rm.backward(&previous_gradient_rm(&previous_gradient_batch));
                layer_rm.update_parameters();

                let name = format!("{:?} step {}", activation, step);
                assert_close_2d(&format!("{} weights", name), &layer.weights, &layer_rm.weights);
                assert_close_1d(&format!("{} bias", name), &layer.bias, &layer_rm.bias);
            }
        }
    }

    /// Loss L = Re(sum(conj(G) * y)). Its gradient w.r.t. the output is G, so the analytical backward
    /// with upstream gradient G must match the finite differences of L (dL/dRe + i dL/dIm).
    fn loss_rm(layer_rm: &mut LayerRm, input_batch: &Vec<Vec<Vec<C>>>, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> f64 {
        let output = layer_rm.forward(&layer_input_rm(input_batch)).get_output_batch_rm();
        let mut loss = 0.0;
        for (y, g) in output.iter().zip(previous_gradient_batch.iter()) {
            let g_flat: Vec<C> = g.iter().flatten().copied().collect();
            for (y_val, g_val) in y.data.iter().zip(g_flat.iter()) {
                loss += (g_val.conj() * y_val).re;
            }
        }
        loss
    }

    #[test]
    fn test_layer_rm_numerical_gradient() {
        let epsilon = 1e-6;
        let tolerance = 1e-5;

        for activation in HOLOMORPHIC_ACTIVATIONS.iter() {
            let (_, mut layer_rm) = create_layers(4, 6, activation);
            let input_batch = generate_random_complex_3d(2, 3, 4);
            let previous_gradient_batch = generate_random_complex_3d(2, 3, output_cols(6, activation));

            layer_rm.forward(&layer_input_rm(&input_batch));
            let gradient_rm = layer_rm.backward(&previous_gradient_rm(&previous_gradient_batch));
            let analytical_weights = gradient_rm.get_gradient_weights_rm().unwrap();
            let analytical_bias = gradient_rm.get_gradient_bias();

            let weights = layer_rm.weights.clone();
            for i in 0..weights.data.len() {
                let mut numerical = C::new(0.0, 0.0);
                for (delta, is_im) in [(C::new(epsilon, 0.0), false), (C::new(0.0, epsilon), true)] {
                    layer_rm.weights.data[i] = weights.data[i] + delta;
                    let loss_plus = loss_rm(&mut layer_rm, &input_batch, &previous_gradient_batch);
                    layer_rm.weights.data[i] = weights.data[i] - delta;
                    let loss_minus = loss_rm(&mut layer_rm, &input_batch, &previous_gradient_batch);
                    let derivative = (loss_plus - loss_minus) / (2.0 * epsilon);
                    if is_im {
                        numerical.im = derivative;
                    } else {
                        numerical.re = derivative;
                    }
                }
                layer_rm.weights.data[i] = weights.data[i];

                let diff = (numerical - analytical_weights.data[i]).norm();
                assert!(diff <= tolerance * numerical.norm().max(1.0), "{:?} weight {}: numerical {:?}, analytical {:?}", activation, i, numerical, analytical_weights.data[i]);
            }

            let bias = layer_rm.bias.clone();
            for i in 0..bias.len() {
                let mut numerical = C::new(0.0, 0.0);
                for (delta, is_im) in [(C::new(epsilon, 0.0), false), (C::new(0.0, epsilon), true)] {
                    layer_rm.bias[i] = bias[i] + delta;
                    let loss_plus = loss_rm(&mut layer_rm, &input_batch, &previous_gradient_batch);
                    layer_rm.bias[i] = bias[i] - delta;
                    let loss_minus = loss_rm(&mut layer_rm, &input_batch, &previous_gradient_batch);
                    let derivative = (loss_plus - loss_minus) / (2.0 * epsilon);
                    if is_im {
                        numerical.im = derivative;
                    } else {
                        numerical.re = derivative;
                    }
                }
                layer_rm.bias[i] = bias[i];

                let diff = (numerical - analytical_bias[i]).norm();
                assert!(diff <= tolerance * numerical.norm().max(1.0), "{:?} bias {}: numerical {:?}, analytical {:?}", activation, i, numerical, analytical_bias[i]);
            }
        }
    }
}
