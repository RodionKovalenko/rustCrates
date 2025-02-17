#[cfg(test)]
mod tests {
    use crate::neural_networks::{
        network_components::{
            add_rms_norm_layer::RMSNormLayer,
            gradient_struct::Gradient,
            layer::{ActivationType, Layer, LayerEnum, LayerType},
            linear_layer::LinearLayer,
            softmax_output_layer::SoftmaxLayer,
        },
        network_types::{feedforward_layer::FeedForwardLayer, neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::{
            activation::{gelu_complex, sigmoid_complex, softsign_complex},
            derivative::{
                gelu_derivative_complex, numerical_gradient_bias, numerical_gradient_bias_without_loss, numerical_gradient_input, numerical_gradient_input_batch, numerical_gradient_input_batch_jacobi_without_loss, numerical_gradient_weights, numerical_gradient_weights_multiple_layers_without_loss, numerical_gradient_weights_without_loss, sigmoid_derivative_complex, softsign_derivative_complex, test_gradient_batch_error, test_gradient_error_1d, test_gradient_error_2d
            },
        },
    };

    use num::Complex;

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
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.5, 0.0), Complex::new(0.8, 0.0), Complex::new(0.1, 0.0)]], vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]]];

        let target_token_id_batch = vec![vec![0], vec![1]];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_batch_output = linear_layer.forward(&input_batch);

        println!("linear batch output: {:?}", &linear_batch_output);
        let _softmax_batch_output = softmax_layer.forward(&linear_batch_output);
        let gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let (analytical_grad_batch, analytical_grad) = (gradient.get_gradient_input_batch(), gradient.get_gradient_input());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            let softmax_batch_output = softmax_layer.forward(&input);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-7;
        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, linear_batch_output.clone(), epsilon);
        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, linear_batch_output.clone(), epsilon);

        //Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad: {:?}", analytical_grad);
        println!("\nnumerical grad: {:?}", numerical_grad);

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon);

        println!("\nanalytical grad batch: {:?}", analytical_grad_batch);
        println!("\nnumerical grad batch: {:?}", numerical_grad_batch);
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
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.5, 0.0), Complex::new(0.8, 0.0), Complex::new(0.1, 0.0)]], vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]]];

        let target_token_id_batch = vec![vec![0], vec![1]];
        //let target_token_id_batch = vec![vec![0]];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_batch_output = linear_layer.forward(&input_batch);
        let _softmax_batch_output = softmax_layer.forward(&linear_batch_output);

        let gradient_linear: Gradient = softmax_layer.backward(&target_token_id_batch);
        let analytical_grad_batch_softmax: Vec<Vec<Vec<Complex<f64>>>> = gradient_linear.get_gradient_input_batch();

        let gradient_softmax: Gradient = linear_layer.backward(&analytical_grad_batch_softmax);
        let (grouped_linear_gradient, analytical_gradient_bias) = (gradient_softmax.get_gradient_weights(), gradient_softmax.get_gradient_bias());

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
        let input_batch = vec![vec![vec![Complex::new(1.0, 0.3), Complex::new(2.0, 0.8)]], vec![vec![Complex::new(3.0, 0.2), Complex::new(4.0, 0.5)]]];

        // Run forward pass (needed before backward)
        let linear_output_batch = linear_layer.forward(&input_batch);

        println!("linear output batch: {} {} {}", linear_output_batch.len(), linear_output_batch[0].len(), linear_output_batch[0][0].len());
        println!("linear weights: {} {} ", linear_layer.weights.len(), linear_layer.weights[0].len());

        // Define previous gradients (as if coming from next layer)
        let previous_gradient_batch = vec![vec![vec![Complex::new(1.0, 0.0); num_col]; input_batch[0].len()]; input_batch.len()];

        let gradient_batch: Gradient = linear_layer.backward(&previous_gradient_batch);

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
        let _input_dim = 3; // Match the input dimension with your input batch
        let output_dim = 4; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-7;

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch_before: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.2, 0.3), Complex::new(0.5, 0.7), Complex::new(0.9, 0.13)]]];
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)]]];

        let mut rms_norm_layer = RMSNormLayer::new(output_dim, epsilon, learning_rate);

        let rms_output_batch = rms_norm_layer.forward(&input_batch, &input_batch_before);

        println!("input batch :{:?}", &input_batch);
        println!("\nrms output_batch: {:?}", &rms_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); rms_output_batch[0][0].len()]; rms_output_batch[0].len()]; rms_output_batch.len()];

        let gradient = rms_norm_layer.backward(&previous_gradient);
        let analytical_gradient_rms = gradient.get_gradient_input_batch();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            let linear_batch_output = rms_norm_layer.forward(input, &input_batch_before);

            linear_batch_output
        };

        let epsilon = 1e-7;
        let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_jacobi_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient rms: {:?}", &numerical_grad_rms);
        println!("\nanalytical gradient rms: {:?}", &analytical_gradient_rms);

        test_gradient_batch_error(&numerical_grad_rms, &analytical_gradient_rms, epsilon)
    }

    #[test]
    fn test_ffn_dense_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let output_dim = 4; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon: f64 = 1e-6;

        let mut dense_layer: Layer = Layer::new(input_dim, output_dim, &learning_rate, &ActivationType::GELU, LayerType::DenseLayer);

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0010, 0.20), Complex::new(0.0030, 0.50), Complex::new(0.60, 0.40)]]];

        let dense_output_batch = dense_layer.forward(&input_batch);

        println!("\ninput batch :{:?}", &input_batch);
        println!("\ndense output_batch: {:?}", &dense_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); dense_output_batch[0][0].len()]; dense_output_batch[0].len()]; dense_output_batch.len()];

        let gradient = dense_layer.backward(&previous_gradient);
        let dense_analytical_gradient_batch = gradient.get_gradient_weight_batch();
        let weights = dense_layer.weights.clone();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            dense_layer.weights = weights.clone();
            let dense_batch_output = dense_layer.forward(input);

            dense_batch_output
        };

        let dense_numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_without_loss(&mut loss_fn, input_batch.clone(), &weights, epsilon);

        println!("\nnumerical gradient dense layer {:?}", dense_numerical_grad_batch);
        println!("\nanalytical gradient dense layer {:?}", dense_analytical_gradient_batch);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&dense_numerical_grad_batch, &dense_analytical_gradient_batch, epsilon);
    }

    #[test]
    fn test_sigmoid() {
        let test_array = [
            Complex::new(1.0, 2.0),
            Complex::new(-2.345451523555475, 15.239089157237373),
            Complex::new(0.1013699645231736, 0.10315190720252415),
            Complex::new(3.003158122183436, 4.0090442543494476),
        ];

        let h = 1e-7; // Step size for numerical gradient

        // Iterate over the array and test each element
        for (i, z) in test_array.iter().enumerate() {
            let sigmoid_output = sigmoid_complex(*z);
            let analytical_derivative = sigmoid_derivative_complex(sigmoid_output);
            let numerical_derivative = numerical_gradient(sigmoid_complex, *z, h);

            println!("Sigmoid: Test case {}:", i + 1);
            println!("  Input: {}", z);
            println!("  Analytical derivative: {}", analytical_derivative);
            println!("  Numerical derivative: {}", numerical_derivative);
            println!("  Difference: {}", analytical_derivative - numerical_derivative);
            println!();

            let numerical_vec = vec![numerical_derivative];
            let analytic_vec = vec![analytical_derivative];
            test_gradient_error_1d(&analytic_vec, &numerical_vec, 1e-5);
        }
    }

    #[test]
    fn test_gelu() {
        let test_array = [
            Complex::new(1.0, 2.0),
            Complex::new(-2.345451523555475, 15.239089157237373),
            Complex::new(0.1013699645231736, 0.10315190720252415),
            Complex::new(3.003158122183436, 4.0090442543494476),
        ];

        let h = 1e-7; // Step size for numerical gradient

        // Iterate over the array and test each element
        for (i, z) in test_array.iter().enumerate() {
            let analytical_derivative = gelu_derivative_complex(*z);
            let numerical_derivative = numerical_gradient(gelu_complex, *z, h);

            println!("Test case {}:", i + 1);
            println!("  Input: {}", z);
            println!("  Analytical derivative: {}", analytical_derivative);
            println!("  Numerical derivative: {}", numerical_derivative);
            println!("  Difference: {}", analytical_derivative - numerical_derivative);
            println!();

            let numerical_vec = vec![numerical_derivative];
            let analytic_vec = vec![analytical_derivative];
            test_gradient_error_1d(&analytic_vec, &numerical_vec, 1e-5);
        }
    }
    // Numerical gradient for verification
    fn numerical_gradient<F>(f: F, z: Complex<f64>, h: f64) -> Complex<f64>
    where
        F: Fn(Complex<f64>) -> Complex<f64>,
    {
        let f_z_plus_h = f(z + h);
        let f_z_minus_h = f(z - h);

        let grad = (f_z_plus_h - f_z_minus_h) / (2.0 * h);

        grad
    }

    #[test]
    fn test_softsign() {
        let test_values = vec![Complex::new(1.0, 2.0), Complex::new(-2.345451523555475, 15.239089157237373), Complex::new(0.5, -0.5), Complex::new(1.0, 1.0)];

        let h = 1e-7; // Step size for numerical gradient

        // Iterate over the array and test each element
        for (i, z) in test_values.iter().enumerate() {
            let analytical_derivative = softsign_derivative_complex(*z);
            let numerical_derivative = numerical_gradient(softsign_complex, *z, h);

            println!("Test case {}:", i + 1);
            println!("  Input: {}", z);
            println!("  Analytical derivative: {}", analytical_derivative);
            println!("  Numerical derivative: {}", numerical_derivative);
            println!("  Difference: {}", analytical_derivative - numerical_derivative);
            println!();

            // let numerical_vec = vec![numerical_derivative];
            // let analytic_vec = vec![analytical_derivative];
            //test_gradient_error_1d(&analytic_vec, &numerical_vec, 1e-5);
        }
    }

    #[test]
    fn test_ffn_network_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let output_dim = 5; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon: f64 = 1e-6;

        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(input_dim,  output_dim, learning_rate);

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.10, 0.20), Complex::new(0.30, 0.50), Complex::new(0.60, 0.40)]]];

        let ffn_output_batch = ffn_layer.forward(&input_batch);

        println!("\ninput batch :{:?}", &input_batch);
        println!("\ndense output_batch: {:?}", &ffn_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); ffn_output_batch[0][0].len()]; ffn_output_batch[0].len()]; ffn_output_batch.len()];

        let gradient = ffn_layer.backward(&previous_gradient);
        let dense_analytical_gradient_batch = gradient.get_gradient_weight_batch();

        // let weights_linear: Vec<Vec<Complex<f64>>> = match ffn_layer.layers.get(1) {
        //     Some(LayerEnum::Linear(linear_layer)) => linear_layer.weights.clone(),
        //     _ => vec![],
        // };

        let weights_dense: Vec<Vec<Complex<f64>>> = match ffn_layer.layers.get(0) {
            Some(LayerEnum::Dense(dense_layer)) => dense_layer.weights.clone(),
            _ => vec![],
        };

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            // if let Some(LayerEnum::Linear(linear_layer)) = ffn_layer.layers.get_mut(1) {
            //     linear_layer.weights = weights.clone();
            // } else {
            //     println!("Layer 2 does not exist!");
            // }
            if let Some(LayerEnum::Dense(dense_layer)) = ffn_layer.layers.get_mut(0) {
                dense_layer.weights = weights.clone();
            } else {
                println!("Layer 2 does not exist!");
            }

            let dense_batch_output = ffn_layer.forward(input);
            dense_batch_output
        };

        // println!("linear weights: {:?}", &weights_linear);
        let dense_numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_multiple_layers_without_loss(&mut loss_fn, input_batch.clone(), &weights_dense.clone(), ffn_output_batch.clone(), epsilon);

        println!("\nnumerical gradient ffn layer {:?}", dense_numerical_grad_batch);
        println!("\nanalytical gradient ffn layer {:?}", dense_analytical_gradient_batch);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&dense_numerical_grad_batch, &dense_analytical_gradient_batch, epsilon);
    }
}
