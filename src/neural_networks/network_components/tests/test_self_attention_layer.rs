#[cfg(test)]
mod test_self_attention_layer {
    use num::Complex;

    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{
            feedforward_layer::FeedForwardLayer,
            neural_network_generic::OperationMode,
            transformer::{masked_attention_head::MaskedAttentionHead, self_attention_layer::SelfAttentionLayer, transformer_network::cross_entropy_loss_batch},
        },
        utils::{
            derivative::{global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_input_batch, numerical_gradient_input_batch_without_loss, numerical_gradient_weights, numerical_gradient_weights_multiple_layers_without_loss, test_gradient_batch_error, test_gradient_error_2d},
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    #[test]
    fn test_attention_head_backward() {
        let batch_size = 2;
        let input_dim = 5;
        let output_dim = 5;
        let epsilon: f64 = 1e-6;

        let learning_rate = 0.0001;

        let mut attention_head_layer: MaskedAttentionHead = MaskedAttentionHead::new(input_dim, output_dim, learning_rate);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);

        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; output_dim]; batch_size];

        let output_batch = attention_head_layer.forward(&input_batch, &padding_mask_batch);

        println!("\ninput batch in attention head dim : {:?}, {}, {}", &input_batch.len(), &input_batch[0].len(), &input_batch[0][0].len());
        println!("\ninput batch in attention head :{:?}", &input_batch);

        println!("\noutput_batch in attention head dim : {:?}, {}, {}", &output_batch.len(), &output_batch[0].len(), &output_batch[0][0].len());
        println!("\noutput_batch attention head: {:?}", &output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); output_batch[0][0].len()]; output_batch[0].len()]; output_batch.len()];

        let gradient = attention_head_layer.backward(&previous_gradient);
        let analytical_gradient_weights_v_batch = gradient.get_gradient_weights_v_batch();
        let analytical_gradient_weights_q_batch = gradient.get_gradient_weights_q_batch();
        let analytical_gradient_weights_k_batch = gradient.get_gradient_weights_k_batch();
        let analytical_gradient_input_batch = gradient.get_gradient_input_batch();

        let weights_v = attention_head_layer.weights_v.clone();
        let weights_q = attention_head_layer.weights_q.clone();
        let weights_k = attention_head_layer.weights_k.clone();

        // Weight V ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            attention_head_layer.weights_v = weights.clone();

            attention_head_layer.forward(input, &padding_mask_batch)
        };

        let numerical_grad_weight_v_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_multiple_layers_without_loss(&mut loss_fn, input_batch.clone(), &weights_v.clone(), output_batch.clone(), epsilon);

        println!("\n\nnumerical gradient weight v attention layer {:?}", numerical_grad_weight_v_batch);
        println!("\n\nanalytical gradient weight v attention layer {:?}", analytical_gradient_weights_v_batch);

        let global_error = global_relative_error_l2(&numerical_grad_weight_v_batch, &analytical_gradient_weights_v_batch);
        println!("\n\n global relative gradient error weight v batch: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_weight_v_batch, &analytical_gradient_weights_v_batch, epsilon);

        attention_head_layer.weights_v = weights_v.clone();
        // Weight Q ------------------------------------------------------------------------------------------- start

        // // Weight Q ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            attention_head_layer.weights_q = weights.clone();

            attention_head_layer.forward(input, &padding_mask_batch)
        };

        let numerical_grad_weight_q_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_multiple_layers_without_loss(&mut loss_fn, input_batch.clone(), &weights_q.clone(), output_batch.clone(), epsilon);

        println!("\n\nnumerical gradient weight q attention layer {:?}", numerical_grad_weight_q_batch);
        println!("\n dim numerical gradient {:?}, {}, {}", numerical_grad_weight_q_batch.len(), numerical_grad_weight_q_batch[0].len(), numerical_grad_weight_q_batch[0][0].len());

        println!("\n\nanalytical gradient weight q attention layer {:?}", analytical_gradient_weights_q_batch);
        println!("\n dim nanalytical gradient {:?}, {}, {}", analytical_gradient_weights_q_batch.len(), analytical_gradient_weights_q_batch[0].len(), analytical_gradient_weights_q_batch[0][0].len());

        let global_error = global_relative_error_l2(&numerical_grad_weight_q_batch, &analytical_gradient_weights_q_batch);
        println!("\n\n global relative gradient error weight q batch: {:?}", &global_error);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&numerical_grad_weight_q_batch, &analytical_gradient_weights_q_batch, epsilon);
        attention_head_layer.weights_q = weights_q.clone();
        // Weight Q ------------------------------------------------------------------------------------------- end

        // // Weight K ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            attention_head_layer.weights_k = weights.clone();

            attention_head_layer.forward(input, &padding_mask_batch)
        };

        let numerical_grad_weight_k_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_multiple_layers_without_loss(&mut loss_fn, input_batch.clone(), &weights_k.clone(), output_batch.clone(), epsilon);

        println!("\n\nnumerical gradient weight k attention layer {:?}", numerical_grad_weight_k_batch);
        println!("\n dim numerical gradient {:?}, {}, {}", numerical_grad_weight_k_batch.len(), numerical_grad_weight_k_batch[0].len(), numerical_grad_weight_k_batch[0][0].len());

        println!("\n\nanalytical gradient weight k attention layer {:?}", analytical_gradient_weights_k_batch);
        println!("\n dim nanalytical gradient {:?}, {}, {}", analytical_gradient_weights_k_batch.len(), analytical_gradient_weights_k_batch[0].len(), analytical_gradient_weights_k_batch[0][0].len());

        let global_error = global_relative_error_l2(&numerical_grad_weight_k_batch, &analytical_gradient_weights_k_batch);
        println!("\n\n global relative gradient error weight k batch: {:?}", &global_error);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&numerical_grad_weight_k_batch, &analytical_gradient_weights_k_batch, epsilon);
        attention_head_layer.weights_k = weights_k.clone();
        // Weight K ------------------------------------------------------------------------------------------- end

        // Input gradient ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> { attention_head_layer.forward(input, &padding_mask_batch) };

        let numerical_grad_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\n\nnumerical_grad_input_batch attention layer {:?}", numerical_grad_input_batch);
        println!("\n dim numerical_grad_input_batch {:?}, {}, {}", numerical_grad_input_batch.len(), numerical_grad_input_batch[0].len(), numerical_grad_input_batch[0][0].len());

        println!("\n\nanalytical gradient input attention layer {:?}", analytical_gradient_input_batch);
        println!("\n dim nanalytical gradient {:?}, {}, {}", analytical_gradient_input_batch.len(), analytical_gradient_input_batch[0].len(), analytical_gradient_input_batch[0][0].len());

        let global_error = global_relative_error_l2(&numerical_grad_input_batch, &analytical_gradient_input_batch);
        println!("\n\n global relative gradient error input batch: {:?}", &global_error);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&numerical_grad_input_batch, &analytical_gradient_input_batch, epsilon);
        //Input gradient ------------------------------------------------------------------------------------------- end
    }

    #[test]
    fn test_self_attention_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let input_dim = 16;
        let output_dim = 16;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let num_attention_heads = 4;
        let epsilon = 1e-5;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, input_dim, output_dim, learning_rate);
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(output_dim, input_dim, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, output_dim, input_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim, (output_dim - 1) as u32);

        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; output_dim]; batch_size];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let attention_layer_output = attention_layer.forward(&input_batch, &padding_mask_batch);
        println!("attention layer output: {} {} {}", attention_layer_output.len(), attention_layer_output[0].len(), attention_layer_output[0][0].len());
        let ffn_batch_output = ffn_layer.forward(&attention_layer_output);
        let linear_batch_output = linear_layer.forward(&ffn_batch_output);
        let _softmax_batch_output = softmax_layer.forward(&linear_batch_output, Some(padding_mask_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax.get_gradient_input_batch());
        let gradient_ffn: Gradient = ffn_layer.backward(&gradient_linear.get_gradient_input_batch());
        let gradient_attention_layer: Gradient = attention_layer.backward(&gradient_ffn.get_gradient_input_batch());

        let gradient_input_batch_att_l = gradient_attention_layer.get_gradient_input_batch();

        println!("padding mask batch in test transformer: {:?}", &padding_mask_batch);
        println!("target tokens ids: {:?}", &target_token_id_batch);
        println!("final output dim: {} {} {}", _softmax_batch_output.len(), _softmax_batch_output[0].len(), _softmax_batch_output[0][0].len());

        // // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            let attention_layer_output = attention_layer.forward(&input, &padding_mask_batch);
            let ffn_batch_output = ffn_layer.forward(&attention_layer_output);
            let linear_batch_output = linear_layer.forward(&ffn_batch_output);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output, Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let num_gradient_input_batch = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", gradient_input_batch_att_l);
        println!("\n gradient_input_batch_att_l gradient dim: {} {} {}", gradient_input_batch_att_l.len(), gradient_input_batch_att_l[0].len(), gradient_input_batch_att_l[0][0].len());

        println!("\n numerical grad: {:?}", num_gradient_input_batch);
        println!("\n numerical_gradient_input_batch gradient dim: {} {} {}", num_gradient_input_batch.len(), num_gradient_input_batch[0].len(), num_gradient_input_batch[0][0].len());

        let global_error = global_relative_error_l2(&num_gradient_input_batch, &gradient_input_batch_att_l);

        println!("global relative gradient error input: {:?}", &global_error);

        test_gradient_batch_error(&num_gradient_input_batch, &gradient_input_batch_att_l, 1e-3);

        let attention_head = attention_layer.attention_heads.get(0).unwrap().clone();
        let weight_q = attention_head.weights_q.clone();
        let gradient = attention_head.gradient.as_ref().unwrap();
        let analytical_weight_q_gradient = gradient.get_gradient_weights_q();

        // Test weights q
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            let attention_head = attention_layer.attention_heads.get_mut(0).unwrap();
            attention_head.weights_q = weights.clone();

            let attention_layer_output = attention_layer.forward(&input, &padding_mask_batch);
            let ffn_batch_output = ffn_layer.forward(&attention_layer_output);
            let linear_batch_output = linear_layer.forward(&ffn_batch_output);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output, Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let num_gradient_weights_q: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weight_q.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad weight_q_gradient: {:?}", analytical_weight_q_gradient);
        println!("\n analytical_weight_q_gradient gradient dim: {} {}", analytical_weight_q_gradient.len(), analytical_weight_q_gradient[0].len());

        println!("\n numerical grad: {:?}", num_gradient_weights_q);
        println!("\n  num_gradient_weights_q dim: {} {}", num_gradient_weights_q.len(), num_gradient_weights_q[0].len());

        let global_error = global_relative_error_2d_l2(&num_gradient_weights_q, &analytical_weight_q_gradient);
        println!("global relative gradient error weights: {:?}", &global_error);

        test_gradient_error_2d(&analytical_weight_q_gradient, &num_gradient_weights_q, 1e-3);

        let mut attention_head = attention_layer.attention_heads.get_mut(0).unwrap().clone();
        attention_head.weights_q = weight_q.clone();
    }
}
