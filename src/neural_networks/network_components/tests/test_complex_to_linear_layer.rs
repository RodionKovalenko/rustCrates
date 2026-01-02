#[cfg(test)]
mod test_complex_to_linear_layer {

    use crate::neural_networks::{
        network_components::{complex_to_linear_layer::ComplexToLinearLayer, gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_sum_batch},
        utils::{
            derivative::{global_relative_error_2d_l2, numerical_gradient_bias, numerical_gradient_weights, test_gradient_error_1d, test_gradient_error_2d},
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_softmax_complex_to_linear_linear_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let cols = 5; // Match the input dimension with your input batch
        let rows = 15; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, cols, rows, true);
        let mut complex_to_linear_layer: ComplexToLinearLayer = ComplexToLinearLayer::new(rows, rows, learning_rate);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, cols);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, rows, cols);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, rows - 1, (rows - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        //let target_token_id_batch = vec![vec![0]];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);
        layer_input.set_input_batch(linear_output.get_output_batch());

        let complex_output = complex_to_linear_layer.forward(&layer_input);
        layer_input.set_input_batch(complex_output.get_output_batch());

        softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_complex_to_linear: Gradient = complex_to_linear_layer.backward(&gradient_softmax);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_complex_to_linear);
        let (grouped_linear_gradient, analytical_gradient_bias) = (gradient_linear.get_gradient_weights(), gradient_linear.get_gradient_bias());

        let linear_weights = linear_layer.weights.clone();

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();
            layer_input.set_input_batch(_input.clone());

            let linear_output = linear_layer.forward(&layer_input);
            layer_input.set_input_batch(linear_output.get_output_batch());

            let complex_output = complex_to_linear_layer.forward(&layer_input);
            layer_input.set_input_batch(complex_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-5;
        let numerical_grad_linear: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &linear_weights.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad weights: {:?}", grouped_linear_gradient);
        println!("\nnumerical grad weights: {:?}", numerical_grad_linear);

        // Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad weights dim: {:?}, {}", grouped_linear_gradient.len(), grouped_linear_gradient[0].len());
        println!("numerical grad weights dim: {:?}, {}", numerical_grad_linear.len(), numerical_grad_linear[0].len());

        for b in 0..grouped_linear_gradient.len() {
            let analytical_row_sum: Complex<f64> = grouped_linear_gradient[b].iter().sum();
            let numerical_row_sum: Complex<f64> = numerical_grad_linear[b].iter().sum();

            println!("analytical row sum: {:?}", analytical_row_sum);
            println!("numerical row sum: {:?}", numerical_row_sum);
        }

        let global_error = global_relative_error_2d_l2(&numerical_grad_linear, &grouped_linear_gradient);
        println!("\n\n global relative gradient error weights linear: {:?}", &global_error);
        test_gradient_error_2d(&grouped_linear_gradient, &numerical_grad_linear, 1e-5);

        // TEST BIAS
        let linear_bias = linear_layer.bias.clone();
        linear_layer.weights = linear_weights;

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Complex<f64> {
            linear_layer.bias = bias.clone();
            layer_input.set_input_batch(_input.clone());

            let linear_output = linear_layer.forward(&layer_input);
            layer_input.set_input_batch(linear_output.get_output_batch());

            let complex_output = complex_to_linear_layer.forward(&layer_input);

            layer_input.set_input_batch(complex_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &linear_bias, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad bias: {:?}", analytical_gradient_bias);
        println!("\nnumerical grad bias: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&analytical_gradient_bias, &numerical_grad_linear_bias, 1e-5);
    }

    #[test]
    fn test_complex_softmax_to_linear_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let cols = 5; // Match the input dimension with your input batch
        let rows = 15; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        // Create a simple LinearLayer with the given input and output dimensions

        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, cols, rows, false);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, cols);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, rows, cols);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, rows - 1, (rows - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        //let target_token_id_batch = vec![vec![0]];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);
        layer_input.set_input_batch(linear_output.get_output_batch());

        softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);

        let (grouped_linear_gradient, analytical_gradient_bias) = (gradient_linear.get_gradient_weights(), gradient_linear.get_gradient_bias());

        let linear_weights = linear_layer.weights.clone();

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();
            layer_input.set_input_batch(_input.clone());

            let linear_output = linear_layer.forward(&layer_input);
            layer_input.set_input_batch(linear_output.get_output_batch());

            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-5;
        let numerical_grad_linear: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &linear_weights.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad weights: {:?}", grouped_linear_gradient);
        println!("\nnumerical grad weights: {:?}", numerical_grad_linear);

        // Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad weights dim: {:?}, {}", grouped_linear_gradient.len(), grouped_linear_gradient[0].len());
        println!("numerical grad weights dim: {:?}, {}", numerical_grad_linear.len(), numerical_grad_linear[0].len());

        for b in 0..grouped_linear_gradient.len() {
            let analytical_row_sum: Complex<f64> = grouped_linear_gradient[b].iter().sum();
            let numerical_row_sum: Complex<f64> = numerical_grad_linear[b].iter().sum();

            println!("analytical row sum: {:?}", analytical_row_sum);
            println!("numerical row sum: {:?}", numerical_row_sum);
        }

        let global_error = global_relative_error_2d_l2(&numerical_grad_linear, &grouped_linear_gradient);
        println!("\n\n global relative gradient error weights linear: {:?}", &global_error);
        test_gradient_error_2d(&grouped_linear_gradient, &numerical_grad_linear, 1e-5);

        // TEST BIAS
        let linear_bias = linear_layer.bias.clone();
        linear_layer.weights = linear_weights;

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Complex<f64> {
            linear_layer.bias = bias.clone();
            layer_input.set_input_batch(_input.clone());

            let linear_output = linear_layer.forward(&layer_input);
            layer_input.set_input_batch(linear_output.get_output_batch());

            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &linear_bias, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad bias: {:?}", analytical_gradient_bias);
        println!("\n numerical grad bias: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&analytical_gradient_bias, &numerical_grad_linear_bias, 1e-5);
    }
}
