#[cfg(test)]
mod test_positional_encoding_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, positional_encoding_layer::PositionalEncodingLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_sum_batch},
        utils::{
            derivative::{
                global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_input, numerical_gradient_input_batch_sum_without_loss, test_gradient_batch_error, test_gradient_error_2d,
            },
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_positional_encoding_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 8; // Match the input dimension with your input batch
        let output_dim = 6; // Match output_dim to your layer's output
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut positonal_encoding_layer: PositionalEncodingLayer = PositionalEncodingLayer::new(input_dim);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        let _positonal_encoding_output = positonal_encoding_layer.forward(&layer_input);
        let positonal_encoding_gradient = positonal_encoding_layer.backward(&previous_gradient_batch);

        let (analytical_grad_batch, _analytical_grad) = (positonal_encoding_gradient.get_gradient_input_batch(), positonal_encoding_gradient.get_gradient_input());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            layer_input.set_input_batch(input.clone());
            positonal_encoding_layer.forward(&layer_input)
        };

        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_sum_without_loss(&mut loss_fn, input_batch.clone(), epsilon);
        //let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad_batch);
        println!("\n numerical grad: {:?}", numerical_grad_batch);

        for b in 0..analytical_grad_batch.len() {
            for s in 0..analytical_grad_batch[b].len() {
                let analytical_row_sum: Complex<f64> = analytical_grad_batch[b][s].iter().sum();
                let numerical_row_sum: Complex<f64> = numerical_grad_batch[b][s].iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        let global_error = global_relative_error_l2(&numerical_grad_batch, &analytical_grad_batch);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_batch, &analytical_grad_batch, epsilon_test);
    }

    #[test]
    fn test_softmax_positional_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1;
        let input_dim = 16;
        let output_dim = 20;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut positional_enc_layer: PositionalEncodingLayer = PositionalEncodingLayer::new(input_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, input_dim);

        // Define a small input batch, [2][2][3]
        // input includes target tokens + padding already !
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, input_dim, input_dim as u32);
        // let target_token_id_batch: Vec<Vec<u32>> = vec![vec![1]; input_batch.len()];
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        // println!("input_batch: {:?}", input_batch);
        println!("target_token id batch: {:?}", target_token_id_batch);
        println!("padding mask batch: {:?}", padding_mask_batch);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_target_batch_ids(target_token_id_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let positonal_encoding = positional_enc_layer.forward(&layer_input);

        layer_input.set_input_batch(positonal_encoding);
        softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let positional_enc_gradient: Gradient = positional_enc_layer.backward(&softmax_gradient.get_gradient_input_batch());

        let analytical_grad = positional_enc_gradient.get_gradient_input();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let positonal_encoding = positional_enc_layer.forward(&layer_input);

            layer_input.set_input_batch(positonal_encoding);
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, input_batch.clone(), epsilon);

        // //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n analytical grad dim : {:?} {}", analytical_grad.len(), analytical_grad[0].len());
        println!("\n numerical grad: {:?}", numerical_grad);
        println!("\n numerical grad dim: {:?} {}", numerical_grad.len(), numerical_grad[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon_test);
    }
}
