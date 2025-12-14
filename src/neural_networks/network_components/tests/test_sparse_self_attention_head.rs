#[cfg(test)]
mod test_sparse_self_attention_head {
    use num::Complex;

    use crate::neural_networks::{
        network_components::{complex_to_linear_layer::ComplexToLinearLayer, gradient_struct::Gradient, layer_input_struct::LayerInput, softmax_output_layer::SoftmaxLayer},
        network_types::{
            neural_network_generic::OperationMode,
            transformer::{
                sparse_masked_attention_head::{calculate_window_tokens, calculate_window_tokens_batch, SparseMaskedAttentionHead},
                transformer_network::cross_entropy_sum_batch,
            },
        },
        utils::{
            derivative::{global_relative_error_2d_l2, numerical_gradient_input, numerical_gradient_weights, test_gradient_error_2d},
            low_rank_approx::transpose,
            matrix::multiply_complex,
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    #[test]
    fn test_sparse_window_size() {
        let seq_len = 16;
        let feature_dim = 4;

        let mut counter = 0;

        let mut seq: Vec<Vec<Complex<f64>>> = Vec::new();
        for _ in 0..seq_len {
            let mut features: Vec<Complex<f64>> = Vec::new();
            for _ in 0..feature_dim {
                features.push(Complex::new(counter as f64, 0.0));
                counter += 1;
            }
            seq.push(features);
        }

        for row in &seq {
            println!("{:?}", row);
        }

        let window_tokens = calculate_window_tokens(&seq, 1);

        print!("\n\n windos len: {}\n", window_tokens.len());

        for window in &window_tokens {
            println!("Window:");
            for token in window {
                println!("{:?}", token);
            }
            println!("---");
        }
    }

    #[test]
    fn test_loss_sparse_attention_head_backward() {
        let batch_size = 1;
        let input_dim = 4;
        let output_dim = 16;
        let epsilon: f64 = 1e-4;

        let learning_rate = 0.0001;

        let mut attention_head_layer: SparseMaskedAttentionHead = SparseMaskedAttentionHead::new(input_dim, input_dim, 1, learning_rate);
        let mut complex_to_linear_layer: ComplexToLinearLayer = ComplexToLinearLayer::new(input_dim, learning_rate);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, OperationMode::TRAINING, input_dim);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; output_dim]; batch_size];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let output = attention_head_layer.forward(&layer_input);

        layer_input.set_input_batch(output.get_output_batch());
        let output_complex_to_linear = complex_to_linear_layer.forward(&layer_input);
        let output_batch = output_complex_to_linear.get_output_batch();

        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, input_dim, input_dim as u32);

        layer_input.set_input_batch(output_batch.clone());
        let _output_softmax = softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        println!(
            "\ninput batch in attention head dim : {:?}, {}, {}",
            &input_batch.len(),
            &input_batch[0].len(),
            &input_batch[0][0].len()
        );
        println!("\ninput batch in attention head :{:?}", &input_batch);

        println!(
            "\noutput_batch in attention head dim : {:?}, {}, {}",
            &output_batch.len(),
            &output_batch[0].len(),
            &output_batch[0][0].len()
        );
        println!("\noutput_batch attention head: {:?}", &output_batch);

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let complex_to_linear_gradient = complex_to_linear_layer.backward(&gradient_softmax);
        let gradient = attention_head_layer.backward(&complex_to_linear_gradient.get_gradient_input_batch());
        let analytical_gradient_weights_v = gradient.get_gradient_weights_v();
        let analytical_gradient_weights_q = gradient.get_gradient_weights_q();
        let analytical_gradient_weights_k = gradient.get_gradient_weights_k();
        let analytical_gradient_input = gradient.get_gradient_input();
        // let analytical_bias_pos_batch = gradient.get_gradient_bias_pos();

        let weights_v = attention_head_layer.weights_v.clone();
        let weights_q = attention_head_layer.weights_q.clone();
        let weights_k = attention_head_layer.weights_k.clone();
        // let bias_pos = attention_head_layer.bias_pos.clone();

        // // Weight V ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            attention_head_layer.weights_v = weights.clone();

            layer_input.set_input_batch(input.clone());
            let attention_head_output = attention_head_layer.forward(&layer_input);

            layer_input.set_input_batch(attention_head_output.get_output_batch());
            let output_complex_to_linear = complex_to_linear_layer.forward(&layer_input);

            layer_input.set_input_batch(output_complex_to_linear.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_weight_v = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_v.clone(), epsilon);

        println!("\n\n numerical gradient weight v attention layer {:?}", numerical_grad_weight_v);
        println!("\n\n analytical gradient weight v attention layer {:?}", analytical_gradient_weights_v);

        for s in 0..numerical_grad_weight_v.len() {
            let analytical_row_sum: Complex<f64> = analytical_gradient_weights_v[s].iter().sum();
            let numerical_row_sum: Complex<f64> = numerical_grad_weight_v[s].iter().sum();

            println!("analytical row weight v sum: {:?}", analytical_row_sum);
            println!("numerical row weight v sum: {:?}", numerical_row_sum);
        }

        let global_error = global_relative_error_2d_l2(&numerical_grad_weight_v, &analytical_gradient_weights_v);
        println!("\n\n global relative gradient error weight v batch: {:?}", &global_error);

        test_gradient_error_2d(&numerical_grad_weight_v, &analytical_gradient_weights_v, epsilon);

        attention_head_layer.weights_v = weights_v.clone();
        // Weight Q ------------------------------------------------------------------------------------------- start

        // // Weight Q ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            attention_head_layer.weights_q = weights.clone();

            layer_input.set_input_batch(input.clone());
            let attention_head_output = attention_head_layer.forward(&layer_input);

            layer_input.set_input_batch(attention_head_output.get_output_batch());
            let output_complex_to_linear = complex_to_linear_layer.forward(&layer_input);

            layer_input.set_input_batch(output_complex_to_linear.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_weight_q = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_q.clone(), epsilon);

        println!("\n numerical gradient weight q attention layer {:?}", numerical_grad_weight_q);
        println!("\n dim numerical gradient weights q {:?}, {}", numerical_grad_weight_q.len(), numerical_grad_weight_q[0].len());

        println!("\n\n analytical gradient weight q attention layer {:?}", analytical_gradient_weights_q);
        println!("\n dim nanalytical gradient {:?}, {} ", analytical_gradient_weights_q.len(), analytical_gradient_weights_q[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad_weight_q, &analytical_gradient_weights_q);
        println!("\n\n global relative gradient error weight q: {:?}", &global_error);

        // For Gelu it can a little more deviation
        test_gradient_error_2d(&numerical_grad_weight_q, &analytical_gradient_weights_q, epsilon);
        attention_head_layer.weights_q = weights_q.clone();
        // Weight Q ------------------------------------------------------------------------------------------- end

        // Weight K ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            attention_head_layer.weights_k = weights.clone();

            layer_input.set_input_batch(input.clone());
            let attention_head_output = attention_head_layer.forward(&layer_input);

            layer_input.set_input_batch(attention_head_output.get_output_batch());
            let output_complex_to_linear = complex_to_linear_layer.forward(&layer_input);

            layer_input.set_input_batch(output_complex_to_linear.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_weight_k = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_k.clone(), epsilon);

        println!("\n numerical gradient weight k attention layer {:?}", numerical_grad_weight_k);
        println!("\n dim numerical gradient weights k {:?}, {}", numerical_grad_weight_k.len(), numerical_grad_weight_k[0].len());

        println!("\n\n analytical gradient weight k attention layer {:?}", analytical_gradient_weights_k);
        println!(
            "\n dim nanalytical gradient weights k {:?}, {} ",
            analytical_gradient_weights_k.len(),
            analytical_gradient_weights_k[0].len()
        );

        let global_error = global_relative_error_2d_l2(&numerical_grad_weight_k, &analytical_gradient_weights_k);
        println!("\n\n global relative gradient error weight k: {:?}", &global_error);

        // For Gelu it can a little more deviation
        test_gradient_error_2d(&numerical_grad_weight_k, &analytical_gradient_weights_k, epsilon);
        attention_head_layer.weights_k = weights_k.clone();
        // Weight K ------------------------------------------------------------------------------------------- end

        // // Bias Positional gradient ------------------------------------------------------------------------------------------- start
        // let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
        //     attention_head_layer.bias_pos = weights.clone();

        //     layer_input.set_input_batch(input.clone());
        //     let attention_head_output = attention_head_layer.forward(&layer_input);
        //     let output_softmax = softmax_layer.forward(&attention_head_output.get_output_batch(), Some(padding_mask_batch.clone()));

        //     let loss = cross_entropy_loss_batch(&output_softmax, &target_token_id_batch, &padding_mask_batch, batch_size);

        //     loss
        // };

        // let small_bias_pos: Vec<Vec<Complex<f64>>> = bias_pos
        //     .iter()
        //     .take(output_dim) // take first 5 rows
        //     .map(|row| row.iter().take(output_dim).cloned().collect()) // take first 5 columns from each row
        //     .collect();

        // let numerical_grad_bias_pos = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &small_bias_pos.clone(), epsilon);

        // println!("\n numerical gradient bias pos attention layer {:?}", numerical_grad_bias_pos);
        // println!("\n dim numerical gradient bias pos {:?}, {}", numerical_grad_bias_pos.len(), numerical_grad_bias_pos[0].len());

        // println!("\n\n analytical gradient bias pos attention layer {:?}", analytical_bias_pos_batch);
        // println!("\n dim nanalytical gradient bias pos {:?}, {} ", analytical_bias_pos_batch.len(), analytical_bias_pos_batch[0].len());

        // let global_error = global_relative_error_2d_l2(&numerical_grad_bias_pos, &analytical_bias_pos_batch);
        // println!("\n\n global relative gradient error bias pos: {:?}", &global_error);

        // for s in 0..analytical_bias_pos_batch.len() {
        //     let analytical_row_sum: Complex<f64> = analytical_bias_pos_batch[s].iter().sum();
        //     let numerical_row_sum: Complex<f64> = numerical_grad_bias_pos[s].iter().sum();

        //     println!("analytical row bias pos sum: {:?}", analytical_row_sum);
        //     println!("numerical row bias pos sum: {:?}", numerical_row_sum);
        // }

        // // For Gelu it can a little more deviation
        // test_gradient_error_2d(&numerical_grad_bias_pos, &analytical_bias_pos_batch, epsilon);
        // attention_head_layer.bias_pos = bias_pos.clone();
        // // Bias Positional gradient ------------------------------------------------------------------------------------------- end

        // Input gradient ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let attention_head_output = attention_head_layer.forward(&layer_input);

            layer_input.set_input_batch(attention_head_output.get_output_batch());
            let output_complex_to_linear = complex_to_linear_layer.forward(&layer_input);

            layer_input.set_input_batch(output_complex_to_linear.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_input_batch: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\n numerical_grad_input_batch attention layer {:?}", numerical_grad_input_batch);
        println!("\n dim numerical_grad_input_batch {:?}, {}", numerical_grad_input_batch.len(), numerical_grad_input_batch[0].len());

        println!("\n analytical gradient input attention layer {:?}", analytical_gradient_input);
        println!("\n dim nanalytical gradient {:?}, {}", analytical_gradient_input.len(), analytical_gradient_input[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad_input_batch, &analytical_gradient_input);
        println!("\n\n global relative gradient error input batch: {:?}", &global_error);

        // For Gelu it can a little more deviation
        test_gradient_error_2d(&numerical_grad_input_batch, &analytical_gradient_input, epsilon);
        //Input gradient ------------------------------------------------------------------------------------------- end
    }

    #[test]
    fn test_sparse_multiplication() {
        let q = vec![vec![
            vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            vec![Complex::new(2.0, 0.0), Complex::new(2.0, 0.0), Complex::new(2.0, 0.0), Complex::new(2.0, 0.0)],
            vec![Complex::new(3.0, 0.0), Complex::new(3.0, 0.0), Complex::new(3.0, 0.0), Complex::new(3.0, 0.0)],
            vec![Complex::new(4.0, 0.0), Complex::new(4.0, 0.0), Complex::new(4.0, 0.0), Complex::new(4.0, 0.0)],
            vec![Complex::new(5.0, 0.0), Complex::new(5.0, 0.0), Complex::new(5.0, 0.0), Complex::new(5.0, 0.0)],
        ]];
        let k = q.clone();
        let window_size = 1;

        let sparse_attention_head: SparseMaskedAttentionHead = SparseMaskedAttentionHead::new(4, 4, 1, 0.001);

        let windows_k: Vec<Vec<Vec<Vec<Complex<f64>>>>> = calculate_window_tokens_batch(&k, window_size);

        let attention_weights = sparse_attention_head.calculate_local_attention::<Complex<f64>>(&q[0], &windows_k[0], false);

        for row in &attention_weights {
            println!("\n sparse attention weights row Q*K: {:?}", row);
        }

        let attention_scores = sparse_attention_head.calculate_local_attention::<Complex<f64>>(&attention_weights, &windows_k[0], false);

        for row in &attention_scores {
            println!("\n sparse attention scores row (Q*K) * V: {:?}", row);
        }

        println!("\n--- Full attention calculation for comparison ---");
        let attention_weights = multiply_complex(&q[0], &transpose(&q[0]));
        for row in &attention_weights {
            println!("\n full matrix row Q*K: {:?}", row);
        }
        let full_attention_scores = multiply_complex(&attention_weights, &q[0]);

        for row in &full_attention_scores {
            println!("\n full attention scores row (Q*K) * V: {:?}", row);
        }
    }
    #[test]
    fn test_sparse_causal_mask_propagation() {
        let batch_size = 1;
        let seq_len = 16;
        let embed_dim = 64;

        let q_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, seq_len, embed_dim);
        let k_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, seq_len, embed_dim);

        let q = &q_batch[0];

        let window_size = 5;

        let sparse_attention_head: SparseMaskedAttentionHead = SparseMaskedAttentionHead::new(embed_dim, embed_dim, window_size, 0.001);

        let k_windows: Vec<Vec<Vec<Vec<Complex<f64>>>>> = calculate_window_tokens_batch(&k_batch, window_size);
        let sparse_attention_weights = sparse_attention_head.calculate_local_attention(&q, &k_windows[0], true);

        let mut sparse_attention_weights_causal_masked: Vec<Vec<Complex<f64>>> = sparse_attention_weights.clone();

        //sparse_attention_head.apply_sparse_causal_mask(&mut sparse_attention_weights_causal_masked);
        sparse_attention_head.apply_unified_mask(&mut sparse_attention_weights_causal_masked, &vec![1; seq_len]);

        for (i, row) in sparse_attention_weights_causal_masked.iter().enumerate() {
            println!("\n sparse original q row {}: {:?}", i, sparse_attention_weights[i]);
            println!("\n sparse causal masked q row {}: {:?}", i, row);
        }
    }
}
