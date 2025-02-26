#[cfg(test)]
mod test_self_attention_layer {
    use num::Complex;

    use crate::neural_networks::{
        network_types::transformer::masked_attention_head::MaskedAttentionHead,
        utils::derivative::{numerical_gradient_input_batch_without_loss, numerical_gradient_weights_multiple_layers_without_loss, test_gradient_batch_error},
    };

    #[test]
    fn test_attention_head_backward() {
        let batch_size = 1;
        let input_dim = 5;
        let output_dim = 4;
        let epsilon: f64 = 1e-6;

        let learning_rate = 0.0001;

        let mut attention_head_layer: MaskedAttentionHead = MaskedAttentionHead::new(input_dim, output_dim, learning_rate);

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![
            vec![Complex::new(0.10, 0.20), Complex::new(0.30, 0.50), Complex::new(0.60, 0.40), Complex::new(0.10, 0.45), Complex::new(0.35, 0.60)],
            vec![Complex::new(0.30, 0.40), Complex::new(0.50, 0.60), Complex::new(0.70, 0.80), Complex::new(0.90, 1.11), Complex::new(2.50, 3.80)],
        ]];

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

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&numerical_grad_weight_k_batch, &analytical_gradient_weights_k_batch, epsilon);
        attention_head_layer.weights_k = weights_k.clone();
        // Weight K ------------------------------------------------------------------------------------------- end

        // Input gradient ------------------------------------------------------------------------------------------- start
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> { attention_head_layer.forward(input, &padding_mask_batch) };

        let numerical_grad_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\n\nnumerical_grad_input_batch attention layer {:?}", numerical_grad_input_batch);
        println!("\n dim numerical_grad_input_batch {:?}, {}, {}", numerical_grad_input_batch.len(), numerical_grad_input_batch[0].len(), numerical_grad_input_batch[0][0].len());

        println!("\n\nanalytical gradient weight k attention layer {:?}", analytical_gradient_input_batch);
        println!("\n dim nanalytical gradient {:?}, {}, {}", analytical_gradient_input_batch.len(), analytical_gradient_input_batch[0].len(), analytical_gradient_input_batch[0][0].len());

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&numerical_grad_input_batch, &analytical_gradient_input_batch, epsilon);
        // Input gradient ------------------------------------------------------------------------------------------- end
    }
}
