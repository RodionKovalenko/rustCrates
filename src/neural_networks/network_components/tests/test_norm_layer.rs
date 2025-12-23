#[cfg(test)]
mod test_norm_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, norm_layer::NormalNormLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_sum_batch},
        utils::{
            derivative::{global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_bias, numerical_gradient_input_batch, test_gradient_batch_error, test_gradient_error_1d},
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_norm_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 5;
        let _seq_len: usize = 5; // Update to match the input structure
        let _input_dim = 16; // Match the input dimension with your input batch
        let _output_dim = 5;
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-7;
        let epsilot_test = 1e-3;

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(_batch_size, _output_dim, _input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(_batch_size, _output_dim - 1, (_output_dim - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut norm_layer = NormalNormLayer::new(input_batch[0][0].len(), 1e-8, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, _input_dim, _output_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, OperationMode::TRAINING, _output_dim);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        let norm_output = norm_layer.forward(&layer_input);

        layer_input.set_input_batch(norm_output.get_output_batch());
        let linear_layer_output = linear_layer.forward(&layer_input);

        layer_input.set_input_batch(linear_layer_output.get_output_batch());
        let _softmax_batch_output = softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);

        println!("input batch :{:?}", &input_batch);

        // norm_layer.previous_gradient_input_batch = Some(vec![vec![vec![Complex::new(1.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()]);

        let linear_gradient = linear_layer.backward(&softmax_gradient);
        let gradient_norm = norm_layer.backward(&linear_gradient);
        let analytical_gradient_input_norm = gradient_norm.get_gradient_input_batch();
        let analytical_beta_gradient = gradient_norm.get_gradient_beta();
        let beta = norm_layer.beta.clone();
        let analytical_gamma_gradient = gradient_norm.get_gradient_gamma();
        let gamma = norm_layer.gamma.clone();

        //TEST 1: input batch itself
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());

            let norm_output = norm_layer.forward(&layer_input);

            layer_input.set_input_batch(norm_output.get_output_batch());
            let linear_layer_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_layer_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_input_norm: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient input batch norm: {:?}", &numerical_grad_input_norm);
        println!("\nanalytical gradient input batch norm: {:?}", &analytical_gradient_input_norm);

        let global_error = global_relative_error_l2(&numerical_grad_input_norm, &analytical_gradient_input_norm);

        println!("\n\nglobal relative gradient input batch error: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_input_norm, &analytical_gradient_input_norm, epsilot_test);

        //TEST 2: gamma gradient
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, gamma: &Vec<Complex<f64>>| -> Complex<f64> {
            norm_layer.gamma = gamma.clone();
            layer_input.set_input_batch(input.clone());
            let norm_output = norm_layer.forward(&layer_input);

            layer_input.set_input_batch(norm_output.get_output_batch());
            let linear_layer_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_layer_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_gamma: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &gamma, epsilon);

        println!("\nnumerical gradient gamma norm: {:?}", &numerical_grad_gamma);
        println!("\nanalytical gradient gamma norm: {:?}", &analytical_gamma_gradient);

        let global_error = global_relative_error_2d_l2(&vec![numerical_grad_gamma.clone()], &vec![analytical_gamma_gradient.clone()]);

        println!("\n\nglobal relative gradient error gamma: {:?}", &global_error);

        test_gradient_error_1d(&numerical_grad_gamma, &analytical_gamma_gradient, epsilot_test);

        //TEST 3: beta gradient
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, beta: &Vec<Complex<f64>>| -> Complex<f64> {
            norm_layer.beta = beta.clone();
            layer_input.set_input_batch(input.clone());
            let norm_output = norm_layer.forward(&layer_input);

            layer_input.set_input_batch(norm_output.get_output_batch());
            let linear_layer_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_layer_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_beta: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &beta, epsilon);

        println!("\nnumerical gradient beta norm: {:?}", &numerical_grad_beta);
        println!("\nanalytical gradient beta norm: {:?}", &analytical_beta_gradient);

        let global_error = global_relative_error_2d_l2(&vec![numerical_grad_beta.clone()], &vec![analytical_beta_gradient.clone()]);

        println!("\n\nglobal relative gradient error beta: {:?}", &global_error);

        test_gradient_error_1d(&numerical_grad_beta, &analytical_beta_gradient, epsilot_test);
    }
}
