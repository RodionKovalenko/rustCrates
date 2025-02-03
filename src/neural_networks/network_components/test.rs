#[cfg(test)]
mod tests {
    use crate::neural_networks::{
        network_components::{linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::derivative::{numerical_gradient, numerical_gradient_batch, test_gradient_batch_error, test_gradient_error},
    };

    use nalgebra::ComplexField;
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

        let epsilon = 1e-7;
        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient(&mut loss_fn, linear_batch_output.clone(), epsilon);
        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_batch(&mut loss_fn, linear_batch_output.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("analytical grad: {:?}", analytical_grad);
        println!("numerical grad: {:?}", numerical_grad);

        test_gradient_error(&numerical_grad, &analytical_grad, epsilon);

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

        let epsilon = 1e-7;
        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient(&mut loss_fn, linear_batch_output.clone(), epsilon);
        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_batch(&mut loss_fn, linear_batch_output.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("analytical grad: {:?}", analytical_grad);
        println!("numerical grad: {:?}", numerical_grad);

        for (row_numerical, row_analytical) in numerical_grad.iter().zip(analytical_grad) {
            for (val_numerical, val_analytical) in row_numerical.iter().zip(row_analytical) {
                let abs_diff = (val_numerical - val_analytical).abs();
                // take the largest value out of (val_numerical, val_analytical, epsilon)
                let max_val = val_numerical.abs().max(val_analytical.abs()).max(epsilon);
                let rel_diff = abs_diff / max_val;

                if rel_diff > epsilon {
                    println!(
                        "Gradient mismatch: numerical = {:.6}, analytical = {:.6}, abs_diff = {:.6}, rel_diff = {:.6}",
                        val_numerical, val_analytical, abs_diff, rel_diff
                    );
                }

                assert!(rel_diff < epsilon);
            }
        }

        println!("analytical grad batch: {:?}", analytical_grad_batch);
        println!("numerical grad batch: {:?}", numerical_grad_batch);

        for (gradient_numerical, gradient_analytical) in numerical_grad_batch.iter().zip(analytical_grad_batch) {
            for (row_numerical, row_analytical) in gradient_numerical.iter().zip(gradient_analytical) {
                for (val_numerical, val_analytical) in row_numerical.iter().zip(row_analytical) {
                    let abs_diff = (val_numerical - val_analytical).abs();
                    // take the largest value out of (val_numerical, val_analytical, epsilon)
                    let max_val = val_numerical.abs().max(val_analytical.abs()).max(epsilon);
                    let rel_diff = abs_diff / max_val;

                    if rel_diff > epsilon {
                        println!(
                            "Gradient mismatch: numerical = {:.6}, analytical = {:.6}, abs_diff = {:.6}, rel_diff = {:.6}",
                            val_numerical, val_analytical, abs_diff, rel_diff
                        );
                    }

                    assert!(rel_diff < epsilon);
                }
            }
        }
    }
}
