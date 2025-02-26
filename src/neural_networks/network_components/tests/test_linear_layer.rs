#[cfg(test)]
mod test_linear_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::derivative::{numerical_gradient_bias, numerical_gradient_bias_without_loss, numerical_gradient_input_batch_without_loss, numerical_gradient_weights, numerical_gradient_weights_without_loss, test_gradient_batch_error, test_gradient_error_1d, test_gradient_error_2d},
    };

    use num::Complex;

    #[test]
    fn test_softmax_linear_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let output_dim = 4; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.5, 0.0), Complex::new(0.8, 0.0), Complex::new(0.1, 0.0)]], vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]]];

        let target_token_id_batch = vec![vec![0], vec![1]];
        //let target_token_id_batch = vec![vec![0]];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_batch_output = linear_layer.forward(&input_batch);
        let _softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

        let gradient_linear: Gradient = softmax_layer.backward(&target_token_id_batch);
        let analytical_grad_batch_softmax: Vec<Vec<Vec<Complex<f64>>>> = gradient_linear.get_gradient_input_batch();

        let gradient_softmax: Gradient = linear_layer.backward(&analytical_grad_batch_softmax);
        let (grouped_linear_gradient, analytical_gradient_bias) = (gradient_softmax.get_gradient_weights(), gradient_softmax.get_gradient_bias());

        let linear_weights = linear_layer.weights.clone();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();
            let linear_batch_output = linear_layer.forward(input);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-7;
        let numerical_grad_linear: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &linear_weights, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad: {:?}", grouped_linear_gradient);
        println!("\nnumerical grad: {:?}", numerical_grad_linear);

        test_gradient_error_2d(&grouped_linear_gradient, &numerical_grad_linear, epsilon);

        // TEST BIAS
        let linear_bias = linear_layer.bias.clone();
        linear_layer.weights = linear_weights;

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Complex<f64> {
            linear_layer.bias = bias.clone();
            let linear_batch_output = linear_layer.forward(input);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &linear_bias, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad bias: {:?}", analytical_gradient_bias);
        println!("\nnumerical grad bias: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&analytical_gradient_bias, &numerical_grad_linear_bias, epsilon);
    }

    #[test]
    fn test_linear_backward_batch() {
        let num_row: usize = 2;
        let num_col: usize = 3;

        // Initialize a LinearLayer
        let mut linear_layer = LinearLayer::new(0.01, num_row, num_col);

        // Define a simple input batch (2 samples, 2D vectors)
        let input_batch = vec![vec![vec![Complex::new(1.0, 0.3), Complex::new(2.0, 0.8)]]];

        // Run forward pass (needed before backward)
        let linear_output_batch = linear_layer.forward(&input_batch);

        println!("linear output batch: {} {} {}", linear_output_batch.len(), linear_output_batch[0].len(), linear_output_batch[0][0].len());
        println!("linear weights: {} {} ", linear_layer.weights.len(), linear_layer.weights[0].len());

        // Define previous gradients (as if coming from next layer)
        let previous_gradient_batch = vec![vec![vec![Complex::new(1.0, 0.0); num_col]; input_batch[0].len()]; input_batch.len()];

        let gradient_batch: Gradient = linear_layer.backward(&previous_gradient_batch);

        let gradient_input_batch = gradient_batch.get_gradient_input_batch();
        let (weight_gradients, bias_gradients) = (gradient_batch.get_gradient_weight_batch(), gradient_batch.get_gradient_bias());
        let linear_weights = linear_layer.weights.clone();
        let bias = linear_layer.bias.clone();
        // Run backward pass

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            linear_layer.weights = weights.clone();
            let linear_batch_output = linear_layer.forward(input);

            linear_batch_output
        };

        let epsilon = 1e-7;
        let numerical_grad_linear: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_without_loss(&mut loss_fn, input_batch.clone(), &linear_weights, epsilon);

        println!("\nnumerical gradient linear: {:?}", &numerical_grad_linear);
        println!("\nanalytical gradient linear: {:?}", &weight_gradients);

        test_gradient_batch_error(&numerical_grad_linear, &weight_gradients, epsilon);

        linear_layer.weights = linear_weights;

        // Bias gradient check
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            linear_layer.bias = bias.clone();
            let linear_batch_output = linear_layer.forward(input);

            linear_batch_output
        };

        let nummerical_grad_bias_linear = numerical_gradient_bias_without_loss(&mut loss_fn, input_batch.clone(), &bias, epsilon);

        println!("\nnumerical grad bias: {:?}", &nummerical_grad_bias_linear);
        println!("\nanalytical grad bias: {:?}", &bias_gradients);

        test_gradient_error_1d(&nummerical_grad_bias_linear, &bias_gradients, epsilon);

        linear_layer.bias = bias.clone();

        // Test input gradient check
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            let linear_batch_output = linear_layer.forward(input);

            linear_batch_output
        };

        let numerical_grad_input_linear: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient input batch dense: {:?}", &numerical_grad_input_linear);
        println!("\nanalytical gradient input batch dense: {:?}", &gradient_input_batch);

        test_gradient_batch_error(&numerical_grad_input_linear, &gradient_input_batch, epsilon);
    }
}
