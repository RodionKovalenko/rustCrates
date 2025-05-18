#[cfg(test)]
mod test_self_attention_layer {
    use std::time::Instant;

    use num::Complex;

    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{
            feedforward_layer::FeedForwardLayer,
            neural_network_generic::OperationMode,
            transformer::{masked_attention_head::MaskedAttentionHead, self_attention_layer::SelfAttentionLayer, transformer_network::cross_entropy_loss_batch},
        },
        utils::{
            derivative::{global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_input_batch, numerical_gradient_input_batch_sum_without_loss, numerical_gradient_weights, numerical_gradient_weights_multiple_layers_without_loss, test_gradient_batch_error, test_gradient_error_2d},
            random_arrays::{generate_random_complex_3d, generate_u32_batch_from_indices},
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

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let output = attention_head_layer.forward(&layer_input);
        let output_batch = output.get_output_batch();

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
        let analytical_bias_pos_batch = gradient.get_gradient_bias_pos_batch();

        let weights_v = attention_head_layer.weights_v.clone();
        let weights_q = attention_head_layer.weights_q.clone();
        let weights_k = attention_head_layer.weights_k.clone();
        let bias_pos = attention_head_layer.bias_pos.clone();

        // // Weight V ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            attention_head_layer.weights_v = weights.clone();

            layer_input.set_input_batch(input.clone());
            let output = attention_head_layer.forward(&layer_input);

            output.get_output_batch()
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

            layer_input.set_input_batch(input.clone());
            let output = attention_head_layer.forward(&layer_input);

            output.get_output_batch()
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

            layer_input.set_input_batch(input.clone());
            let output = attention_head_layer.forward(&layer_input);

            output.get_output_batch()
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
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            layer_input.set_input_batch(input.clone());
            let output = attention_head_layer.forward(&layer_input);

            output.get_output_batch()
        };

        let numerical_grad_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_sum_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\n numerical_grad_input_batch attention layer {:?}", numerical_grad_input_batch);
        println!("\n dim numerical_grad_input_batch {:?}, {}, {}", numerical_grad_input_batch.len(), numerical_grad_input_batch[0].len(), numerical_grad_input_batch[0][0].len());

        println!("\n analytical gradient input attention layer {:?}", analytical_gradient_input_batch);
        println!("\n dim nanalytical gradient {:?}, {}, {}", analytical_gradient_input_batch.len(), analytical_gradient_input_batch[0].len(), analytical_gradient_input_batch[0][0].len());

        let global_error = global_relative_error_l2(&numerical_grad_input_batch, &analytical_gradient_input_batch);
        println!("\n\n global relative gradient error input batch: {:?}", &global_error);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&numerical_grad_input_batch, &analytical_gradient_input_batch, epsilon);
        //Input gradient ------------------------------------------------------------------------------------------- end

        // Bias Positional gradient ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, bias_pos: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            //println!("bias pos dim: {} {}", bias_pos.len(), bias_pos[0].len());
            attention_head_layer.bias_pos = bias_pos.clone();

            layer_input.set_calculate_gradient(false);
            layer_input.set_input_batch(input.clone());
            let output = attention_head_layer.forward(&layer_input);

            output.get_output_batch()
        };

        let small_bias_pos: Vec<Vec<Complex<f64>>> = bias_pos
            .iter()
            .take(output_dim) // take first 5 rows
            .map(|row| row.iter().take(output_dim).cloned().collect()) // take first 5 columns from each row
            .collect();

        let numerical_bias_pos_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_multiple_layers_without_loss(&mut loss_fn, input_batch.clone(), &small_bias_pos.clone(), output_batch.clone(), epsilon);

        println!("\n numerical gradient bias positonal attention layer {:?}", numerical_bias_pos_batch);
        println!("\n numerical gradient bias positonal dim {:?}, {}, {}", numerical_bias_pos_batch.len(), numerical_bias_pos_batch[0].len(), numerical_bias_pos_batch[0][0].len());

        println!("\n analytical gradient bias positional attention layer {:?}", analytical_bias_pos_batch);
        println!("\n analytical gradient bias positional dim: {:?}, {}, {}", analytical_bias_pos_batch.len(), analytical_bias_pos_batch[0].len(), analytical_bias_pos_batch[0][0].len());

        let global_error = global_relative_error_l2(&numerical_bias_pos_batch, &analytical_bias_pos_batch);
        println!("\n\n global relative gradient error input batch: {:?}", &global_error);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&numerical_bias_pos_batch, &analytical_bias_pos_batch, epsilon);
        // Bias Positional gradient ------------------------------------------------------------------------------------------- end
    }

    #[test]
    fn test_self_attention_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 2;
        let input_dim = 8;
        let output_dim = 8;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let num_attention_heads = 4;
        let epsilon = 1e-8;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, input_dim, output_dim, learning_rate);

        let sequence_len = 8;
        let embedding_dim = 16;
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(sequence_len, embedding_dim, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, sequence_len, embedding_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_u32_batch_from_indices(batch_size, output_dim);

        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; output_dim]; batch_size];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let attention_layer_output = attention_layer.forward(&layer_input);
        layer_input.set_input_batch(attention_layer_output.get_output_batch());

        let ffn_batch_output = ffn_layer.forward(&layer_input);
        layer_input.set_input_batch(ffn_batch_output.get_output_batch());

        let linear_output = linear_layer.forward(&layer_input);
        let _softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), Some(padding_mask_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);
        let gradient_ffn: Gradient = ffn_layer.backward(&gradient_linear.get_gradient_input_batch());
        let gradient_attention_layer: Gradient = attention_layer.backward(&gradient_ffn.get_gradient_input_batch());

        let gradient_input_batch_att_l = gradient_attention_layer.get_gradient_input();
        let gradient_input_batch_att_batch = gradient_attention_layer.get_gradient_input_batch();

        let attention_head = attention_layer.attention_heads.get(0).unwrap().clone();
        let weight_q = attention_head.weights_q.clone();
        let gradient = attention_head.gradient.as_ref().unwrap();
        let analytical_weight_q_gradient = gradient.get_gradient_weights_q();

        //println!("input batch: {:?}", &input_batch);
        println!("padding mask batch in test transformer: {:?}", &padding_mask_batch);
        println!("target tokens ids: {:?}", &target_token_id_batch);
        println!("final output dim: {} {} {}", _softmax_batch_output.len(), _softmax_batch_output[0].len(), _softmax_batch_output[0][0].len());

        let now = Instant::now();

        // TEST INPUT GRADIENT
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            let seconds_elapsed = now.elapsed();
            layer_input.set_calculate_gradient(false);
            layer_input.set_input_batch(input.clone());
            layer_input.set_padding_mask_batch(padding_mask_batch.clone());

            let attention_layer_output = attention_layer.forward(&layer_input);
            layer_input.set_input_batch(attention_layer_output.get_output_batch());

            let ffn_batch_output = ffn_layer.forward(&layer_input);
            layer_input.set_input_batch(ffn_batch_output.get_output_batch());

            let linear_output = linear_layer.forward(&layer_input);
            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

            let seconds_elapsed_end = now.elapsed();
            let duration = seconds_elapsed_end - seconds_elapsed;
            let _seconds = duration.as_secs_f64();
            //println!("time elapsed for forward pass for input gradient in seconds: {:?}", _seconds);
            loss
        };

        let num_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);
        let num_gradient_input_batch_aggregated = gradient_attention_layer.group_gradient_batch(&num_gradient_input_batch);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical grad: {:?}", gradient_input_batch_att_l);
        println!("\n gradient_input_batch_att_l gradient dim: {} {}", gradient_input_batch_att_l.len(), gradient_input_batch_att_l[0].len());

        // println!("\n numerical grad: {:?}", num_gradient_input_batch);
        println!("\n numerical_gradient_input_batch gradient dim: {} {}", num_gradient_input_batch_aggregated.len(), num_gradient_input_batch_aggregated[0].len());

        let global_error = global_relative_error_2d_l2(&num_gradient_input_batch_aggregated, &gradient_input_batch_att_l);

        println!("global relative gradient error input: {:?}", &global_error);

        for b in 0..gradient_input_batch_att_batch.len() {
            for s in 0..gradient_input_batch_att_batch[b].len() {
                let analytical_row_sum: Complex<f64> = gradient_input_batch_att_batch[b][s].iter().sum();
                let numerical_row_sum: Complex<f64> = num_gradient_input_batch[b][s].iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        //test_gradient_batch_error(&num_gradient_input_batch, &gradient_input_batch_att_l, 1e-2);

        // Test weights q
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            let seconds_elapsed = now.elapsed();
            let attention_head = attention_layer.attention_heads.get_mut(0).unwrap();
            attention_head.weights_q = weights.clone();

            layer_input.set_calculate_gradient(false);
            layer_input.set_input_batch(input.clone());
            layer_input.set_padding_mask_batch(padding_mask_batch.clone());

            let attention_layer_output = attention_layer.forward(&layer_input);
            layer_input.set_input_batch(attention_layer_output.get_output_batch());

            let ffn_batch_output = ffn_layer.forward(&layer_input);
            layer_input.set_input_batch(ffn_batch_output.get_output_batch());

            let linear_output = linear_layer.forward(&layer_input);
            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

            let seconds_elapsed_end = now.elapsed();
            let duration = seconds_elapsed_end - seconds_elapsed;
            let _seconds = duration.as_secs_f64();
            //println!("time elapsed for forward pass for weight q gradient in seconds: {:?}", _seconds);

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

        for (i, numeric_gradient) in num_gradient_weights_q.iter().enumerate() {
            let row_sum_numeric: Complex<f64> = numeric_gradient.iter().sum();
            let row_sum_analytic: Complex<f64> = analytical_weight_q_gradient[i].iter().sum();

            println!("row_sum numerical: {:?}, {:?}", row_sum_numeric.re, row_sum_numeric.im);
            println!("row sum analytical: {:?}, {:?}", row_sum_analytic.re, row_sum_analytic.im);
        }

        test_gradient_error_2d(&analytical_weight_q_gradient, &num_gradient_weights_q, 1e-2);

        let mut attention_head = attention_layer.attention_heads.get_mut(0).unwrap().clone();
        attention_head.weights_q = weight_q.clone();
    }
}
