#[cfg(test)]
mod test_self_attention_layer_approx_with_loss {

    use num::Complex;

    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, softmax_output_layer::SoftmaxLayer},
        network_types::{
            neural_network_generic::OperationMode,
            transformer::{masked_attention_head_approximation::MaskedAttentionHeadApproximation, transformer_network::cross_entropy_loss_batch},
        },
        utils::derivative::{global_relative_error_2d_l2, numerical_gradient_input, numerical_gradient_weights, test_gradient_error_2d},
    };

    #[test]
    fn test_loss_attention_head_approx_backward() {
        let batch_size = 2;
        let input_dim = 2;
        let output_dim = 4;
        let epsilon: f64 = 1e-3;

        let learning_rate = 0.0001;

        let mut attention_head_layer: MaskedAttentionHeadApproximation = MaskedAttentionHeadApproximation::new(input_dim, output_dim, learning_rate);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, OperationMode::TRAINING);

        let input_batch = vec![
            vec![
                vec![Complex::new(-0.021967, 0.0), Complex::new(0.035711, 0.0)],
                vec![Complex::new(0.147789, 0.0), Complex::new(-0.051827, 0.0)],
                vec![Complex::new(-0.080849, 0.0), Complex::new(-0.050176, 0.0)],
                vec![Complex::new(0.091540, 0.0), Complex::new(0.032875, 0.0)],
            ],
            vec![
                vec![Complex::new(-0.052976, 0.0), Complex::new(0.051327, 0.0)],
                vec![Complex::new(0.009708, 0.0), Complex::new(0.096864, 0.0)],
                vec![Complex::new(-0.070205, 0.0), Complex::new(-0.032766, 0.0)],
                vec![Complex::new(-0.039211, 0.0), Complex::new(-0.146351, 0.0)],
            ],
        ];

        // Also initialize your attention head weights to these fixed values:
        attention_head_layer.weights_k = vec![
            vec![Complex::new(-0.046947, 0.0), Complex::new(0.054256, -0.3), Complex::new(-0.046342, 0.0), Complex::new(-0.046573, 0.1)],
            vec![Complex::new(0.024196, 0.0), Complex::new(-0.191328, -0.4), Complex::new(-0.172492, 0.0), Complex::new(-0.056229, 0.6)],
        ];
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; output_dim]; batch_size];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let output = attention_head_layer.forward(&layer_input);
        let output_batch = output.get_output_batch();

        let _output_softmax = softmax_layer.forward(&output_batch, Some(padding_mask_batch.clone()));
        let target_token_id_batch = vec![vec![0u32, 0u32, 2u32], vec![2u32, 1u32, 2u32]];

        println!("\ninput batch in attention head dim : {:?}, {}, {}", &input_batch.len(), &input_batch[0].len(), &input_batch[0][0].len());
        println!("\ninput batch in attention head :{:?}", &input_batch);

        println!("\noutput_batch in attention head dim : {:?}, {}, {}", &output_batch.len(), &output_batch[0].len(), &output_batch[0][0].len());
        println!("\noutput_batch attention head: {:?}", &output_batch);

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient = attention_head_layer.backward(&gradient_softmax.get_gradient_input_batch());
        let analytical_gradient_weights_v = gradient.get_gradient_weights_v();
        let analytical_gradient_weights_q = gradient.get_gradient_weights_q();
        let analytical_gradient_weights_k = gradient.get_gradient_weights_k();
        let analytical_gradient_input = gradient.get_gradient_input();
        let analytical_bias_pos_batch = gradient.get_gradient_bias_pos();

        let weights_v = attention_head_layer.weights_v.clone();
        let weights_q = attention_head_layer.weights_q.clone();
        let weights_k = attention_head_layer.weights_k.clone();
        let bias_pos = attention_head_layer.bias_pos.clone();

        // // Weight V ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            attention_head_layer.weights_v = weights.clone();

            layer_input.set_input_batch(input.clone());
            let attention_head_output = attention_head_layer.forward(&layer_input);
            let output_softmax = softmax_layer.forward(&attention_head_output.get_output_batch(), Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&output_softmax, &target_token_id_batch, &padding_mask_batch, batch_size);

            loss
        };

        let numerical_grad_weight_v = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_v.clone(), epsilon);

        println!("\n\nnumerical gradient weight v attention layer {:?}", numerical_grad_weight_v);
        println!("\n\nanalytical gradient weight v attention layer {:?}", analytical_gradient_weights_v);

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
            let output_softmax = softmax_layer.forward(&attention_head_output.get_output_batch(), Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&output_softmax, &target_token_id_batch, &padding_mask_batch, batch_size);

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
            let output_softmax = softmax_layer.forward(&attention_head_output.get_output_batch(), Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&output_softmax, &target_token_id_batch, &padding_mask_batch, batch_size);

            loss
        };

        let numerical_grad_weight_k = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_k.clone(), epsilon);

        println!("\n numerical gradient weight k attention layer {:?}", numerical_grad_weight_k);
        println!("\n dim numerical gradient weights k {:?}, {}", numerical_grad_weight_k.len(), numerical_grad_weight_k[0].len());

        println!("\n\n analytical gradient weight k attention layer {:?}", analytical_gradient_weights_k);
        println!("\n dim nanalytical gradient weights k {:?}, {} ", analytical_gradient_weights_k.len(), analytical_gradient_weights_k[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad_weight_k, &analytical_gradient_weights_k);
        println!("\n\n global relative gradient error weight k: {:?}", &global_error);

        // For Gelu it can a little more deviation
        //test_gradient_error_2d(&numerical_grad_weight_k, &analytical_gradient_weights_k, epsilon);
        attention_head_layer.weights_k = weights_k.clone();
        // Weight K ------------------------------------------------------------------------------------------- end

        // Bias Positional gradient ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            attention_head_layer.bias_pos = weights.clone();

            layer_input.set_input_batch(input.clone());
            let attention_head_output = attention_head_layer.forward(&layer_input);
            let output_softmax = softmax_layer.forward(&attention_head_output.get_output_batch(), Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&output_softmax, &target_token_id_batch, &padding_mask_batch, batch_size);

            loss
        };

        let small_bias_pos: Vec<Vec<Complex<f64>>> = bias_pos
            .iter()
            .take(output_dim) // take first 5 rows
            .map(|row| row.iter().take(output_dim).cloned().collect()) // take first 5 columns from each row
            .collect();

        let numerical_grad_bias_pos = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &small_bias_pos.clone(), epsilon);

        println!("\n numerical gradient bias pos attention layer {:?}", numerical_grad_bias_pos);
        println!("\n dim numerical gradient bias pos {:?}, {}", numerical_grad_bias_pos.len(), numerical_grad_bias_pos[0].len());

        println!("\n\n analytical gradient bias pos attention layer {:?}", analytical_bias_pos_batch);
        println!("\n dim nanalytical gradient bias pos {:?}, {} ", analytical_bias_pos_batch.len(), analytical_bias_pos_batch[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad_bias_pos, &analytical_bias_pos_batch);
        println!("\n\n global relative gradient error bias pos: {:?}", &global_error);

        for s in 0..analytical_bias_pos_batch.len() {
            let analytical_row_sum: Complex<f64> = analytical_bias_pos_batch[s].iter().sum();
            let numerical_row_sum: Complex<f64> = numerical_grad_bias_pos[s].iter().sum();

            println!("analytical row bias pos sum: {:?}", analytical_row_sum);
            println!("numerical row bias pos sum: {:?}", numerical_row_sum);
        }

        // For Gelu it can a little more deviation
        // test_gradient_error_2d(&numerical_grad_bias_pos, &analytical_bias_pos_batch, epsilon);
        attention_head_layer.bias_pos = bias_pos.clone();
        // Bias Positional gradient ------------------------------------------------------------------------------------------- end

        // Input gradient ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let attention_head_output = attention_head_layer.forward(&layer_input);
            let output_softmax = softmax_layer.forward(&attention_head_output.get_output_batch(), Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&output_softmax, &target_token_id_batch, &padding_mask_batch, batch_size);

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
}
