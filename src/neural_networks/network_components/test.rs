#[cfg(test)]
mod tests {
    use crate::neural_networks::{
        network_components::{linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::derivative::{numerical_gradient_bias, numerical_gradient_input, numerical_gradient_input_batch, numerical_gradient_weights, test_gradient_batch_error, test_gradient_error_1d, test_gradient_error_2d},
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
}
