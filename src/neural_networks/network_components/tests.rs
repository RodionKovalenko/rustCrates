#[cfg(test)]
mod tests {
    use crate::neural_networks::{
        network_components::{add_rms_norm_layer::RMSNormLayer, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::derivative::{
            numerical_gradient_bias, numerical_gradient_bias_without_loss, numerical_gradient_input, numerical_gradient_input_batch, numerical_gradient_input_batch_without_loss,
            numerical_gradient_weights, numerical_gradient_weights_without_loss, test_gradient_batch_error, test_gradient_error_1d, test_gradient_error_2d,
        },
    };

    use num::{Complex, Float};

    #[test]
    fn test_softmax_layer_backward() {
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
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![Complex::new(0.5, 0.0), Complex::new(0.8, 0.0), Complex::new(0.1, 0.0)]],
            vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]],
        ];

        let target_token_id_batch = vec![vec![0], vec![1]];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_batch_output = linear_layer.forward(&input_batch);

        println!("linear batch output: {:?}", &linear_batch_output);
        let _softmax_batch_output = softmax_layer.forward(&linear_batch_output);
        let analytical_grad = softmax_layer.backward(&target_token_id_batch);
        let analytical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = softmax_layer.backward_batch(&target_token_id_batch);

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            let softmax_batch_output = softmax_layer.forward(&input);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-9;
        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, linear_batch_output.clone(), epsilon);
        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, linear_batch_output.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("analytical grad: {:?}", analytical_grad);
        println!("numerical grad: {:?}", numerical_grad);

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon);

        println!("analytical grad batch: {:?}", analytical_grad_batch);
        println!("numerical grad batch: {:?}", numerical_grad_batch);
        test_gradient_batch_error(&numerical_grad_batch, &analytical_grad_batch, epsilon);
    }

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
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![Complex::new(0.5, 0.0), Complex::new(0.8, 0.0), Complex::new(0.1, 0.0)]],
            vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]],
        ];

        let target_token_id_batch = vec![vec![0], vec![1]];
        //let target_token_id_batch = vec![vec![0]];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_batch_output = linear_layer.forward(&input_batch);
        let _softmax_batch_output = softmax_layer.forward(&linear_batch_output);

        let analytical_grad_batch_softmax: Vec<Vec<Vec<Complex<f64>>>> = softmax_layer.backward_batch(&target_token_id_batch);
        let (analytical_grad_linear, analytical_gradient_bias) = linear_layer.backward_batch(&analytical_grad_batch_softmax);
        let grouped_linear_gradient = linear_layer.group_gradient_batch(&analytical_grad_linear);

        let linear_weights = linear_layer.weights.clone();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();
            let linear_batch_output = linear_layer.forward(input);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output);

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
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output);

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
        let num_col: usize = 5;

        // Initialize a LinearLayer
        let mut linear_layer = LinearLayer::new(0.01, num_row, num_col);

        // Define a simple input batch (2 samples, 2D vectors)
        let input_batch = vec![
            vec![vec![Complex::new(1.0, 0.3), Complex::new(2.0, 0.8)]],
            vec![vec![Complex::new(3.0, 0.2), Complex::new(4.0, 0.5)]],
        ];

        // Run forward pass (needed before backward)
        let linear_output_batch = linear_layer.forward(&input_batch);

        println!(
            "linear output batch: {} {} {}",
            linear_output_batch.len(),
            linear_output_batch[0].len(),
            linear_output_batch[0][0].len()
        );
        println!("linear weights: {} {} ", linear_layer.weights.len(), linear_layer.weights[0].len());

        // Define previous gradients (as if coming from next layer)
        let previous_gradient_batch = vec![vec![vec![Complex::new(1.0, 0.0); num_col]; input_batch[0].len()]; input_batch.len()];

        let (weight_gradients, bias_gradients) = linear_layer.backward_batch(&previous_gradient_batch);
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

        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            linear_layer.bias = bias.clone();
            let linear_batch_output = linear_layer.forward(input);

            linear_batch_output
        };

        let nummerical_grad_bias_linear = numerical_gradient_bias_without_loss(&mut loss_fn, input_batch.clone(), &bias, epsilon);

        println!("\nnumerical grad bias: {:?}", &nummerical_grad_bias_linear);
        println!("\nanalytical grad bias: {:?}", &bias_gradients);

        test_gradient_error_1d(&nummerical_grad_bias_linear, &bias_gradients, epsilon);
    }

    #[test]
    fn test_rms_norm_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let output_dim = 4; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-7;

        // Create a simple LinearLayer with the given input and output dimensions

        // Define a small input batch, [2][2][3]
        // let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
        //     vec![vec![Complex::new(0.5, 1.0), Complex::new(0.8, 5.0), Complex::new(0.1, 3.0)]],
        //     vec![vec![Complex::new(1.0, 2.0), Complex::new(2.0, 3.0), Complex::new(3.0, 4.0)]],
        // ];

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)]]];

        let target_token_id_batch = vec![vec![0], vec![1]];

        let mut rms_norm_layer = RMSNormLayer::new(output_dim, epsilon, learning_rate);

        let rms_output_batch = rms_norm_layer.forward(&input_batch, &input_batch);

        println!("input batch :{:?}", &input_batch);
        println!("\nrms output_batch: {:?}", &rms_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        let backward = rms_norm_layer.backward(&previous_gradient);

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            let linear_batch_output = rms_norm_layer.forward(input, input);

            linear_batch_output
        };

        let epsilon = 1e-7;
        let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient rms: {:?}", &numerical_grad_rms);
        println!("\nanalytical gradient rms: {:?}", &backward);

        test_gradient_batch_error(&numerical_grad_rms, &backward, epsilon)       
    }
}
