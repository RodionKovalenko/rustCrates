#[cfg(test)]
mod test_rms_norm_layer {
    use crate::neural_networks::{
        network_components::add_rms_norm_layer::RMSNormLayer,
        network_types::neural_network_generic::OperationMode,
        utils::derivative::{numerical_gradient_input_batch_jacobi_without_loss, test_gradient_batch_error},
    };

    use num::Complex;

    #[test]
    fn test_rms_norm_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-7;

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch_before: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.2, 0.3), Complex::new(0.5, 0.7), Complex::new(0.9, 0.13)]]];
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)]]];

        let mut rms_norm_layer = RMSNormLayer::new(input_dim, epsilon, learning_rate);

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
}
