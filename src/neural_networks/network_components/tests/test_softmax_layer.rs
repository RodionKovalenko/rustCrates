#[cfg(test)]
mod test_softmax_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::{
            activation::{gelu_complex, sigmoid_complex, softmax_complex, softsign_complex},
            derivative::{
                gelu_derivative_complex, global_relative_error_2d_l2, global_relative_error_l2, norm_softmax_derivative_complex, numerical_gradient_check_f64, numerical_gradient_input, numerical_gradient_input_batch, sigmoid_derivative_complex, softsign_derivative_complex,
                test_gradient_batch_error, test_gradient_error_1d, test_gradient_error_2d,
            },
            random_arrays::{generate_random_complex_2d, generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_softmax_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 5; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-5;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim, (output_dim - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);
        let _softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), Some(padding_mask_batch.clone()));

        let gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let (analytical_grad_batch, analytical_grad) = (gradient.get_gradient_input_batch(), gradient.get_gradient_input());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            let softmax_batch_output = softmax_layer.forward(&input, None);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

            loss
        };

        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, linear_output.get_output_batch(), epsilon);
        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, linear_output.get_output_batch(), epsilon);

        //Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad: {:?}", analytical_grad);
        println!("\nnumerical grad: {:?}", numerical_grad);

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon);

        // println!("\nanalytical grad batch: {:?}", analytical_grad_batch);
        // println!("\nnumerical grad batch: {:?}", numerical_grad_batch);
        test_gradient_batch_error(&numerical_grad_batch, &analytical_grad_batch, epsilon);

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);
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
    fn test_softmax_gradient() {
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 5; // Match output_dim to your layer's output
        let epsilon = 1e-6;

        // Define a small input batch, [2][2][3]
        let z: Vec<Vec<Complex<f64>>> = generate_random_complex_2d(output_dim, input_dim);

        let softmax_values: Vec<Vec<f64>> = softmax_complex(&z);

        let analytical_gradient: Vec<Vec<Vec<Complex<f64>>>> = norm_softmax_derivative_complex(&z, &softmax_values);
        let numerical_gradient: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_check_f64(softmax_complex, &z, epsilon);

        println!("\n dim analytical gradient {:?}", analytical_gradient);
        println!("\n dim numerical gradient {:?}", numerical_gradient);

        test_gradient_batch_error(&numerical_gradient, &analytical_gradient, epsilon);

        let global_error = global_relative_error_l2(&numerical_gradient, &analytical_gradient);
        println!("\n\n global relative error input gradient: {:?}", &global_error);
    }
}
