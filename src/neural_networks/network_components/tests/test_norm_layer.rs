#[cfg(test)]
mod test_norm_layer {
    use crate::neural_networks::{
        network_components::{layer_input_struct::LayerInput, norm_layer::NormalNormLayer},
        network_types::neural_network_generic::OperationMode,
        utils::{
            derivative::{global_relative_error_l2, numerical_gradient_input_batch_without_loss, test_gradient_batch_error},
            random_arrays::generate_random_complex_3d,
        },
    };

    use num::Complex;

    #[test]
    fn test_norm_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let _input_dim = 15; // Match the input dimension with your input batch
        let _output_dim = 1;
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(5.0, 0.0), Complex::new(4.0, 0.0), Complex::new(3.0, 0.0)]]];

        let mut norm_layer = NormalNormLayer::new(input_batch[0][0].len(), 1e-8, learning_rate);

        let mut layer_input = LayerInput::new_default();

        layer_input.set_input_batch_before(input_batch.clone());
        layer_input.set_input_batch(input_batch.clone());

        let rms_output = norm_layer.forward(&layer_input);
        let rms_output_batch = rms_output.get_output_batch();

        println!("input batch :{:?}", &input_batch);
        println!("\n norm output_batch: {:?}", &rms_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); rms_output_batch[0][0].len()]; rms_output_batch[0].len()]; rms_output_batch.len()];

        let gradient = norm_layer.backward(&previous_gradient);
        let analytical_gradient_norm = gradient.get_gradient_input_batch();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            layer_input.set_input_batch(input.clone());
            let rms_output = norm_layer.forward(&layer_input);
            let rms_output_batch = rms_output.get_output_batch();

            rms_output_batch
        };

        //let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_jacobi_without_loss(&mut loss_fn, input_batch.clone(), epsilon);
        let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient norm: {:?}", &numerical_grad_rms);
        println!("\nanalytical gradient norm: {:?}", &analytical_gradient_norm);

        let global_error = global_relative_error_l2(&numerical_grad_rms, &analytical_gradient_norm);

        println!("\n\nglobal relative gradient error: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_rms, &analytical_gradient_norm, 1e-3);

        // TEST 2
        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]]];

        let mut norm_layer = NormalNormLayer::new(input_batch[0][0].len(), 1e-8, learning_rate);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch_before(input_batch.clone());
        layer_input.set_input_batch(input_batch.clone());

        let rms_output = norm_layer.forward(&layer_input);
        let rms_output_batch = rms_output.get_output_batch();

        println!("input batch :{:?}", &input_batch);
        println!("\n norm output_batch: {:?}", &rms_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); rms_output_batch[0][0].len()]; rms_output_batch[0].len()]; rms_output_batch.len()];

        let gradient = norm_layer.backward(&previous_gradient);
        let analytical_gradient_norm = gradient.get_gradient_input_batch();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            layer_input.set_input_batch(input.clone());
            let rms_output = norm_layer.forward(&layer_input);
            let rms_output_batch = rms_output.get_output_batch();

            rms_output_batch
        };

        //let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_jacobi_without_loss(&mut loss_fn, input_batch.clone(), epsilon);
        let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient norm: {:?}", &numerical_grad_rms);
        println!("\nanalytical gradient norm: {:?}", &analytical_gradient_norm);

        let global_error = global_relative_error_l2(&numerical_grad_rms, &analytical_gradient_norm);

        println!("\n\nglobal relative gradient error: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_rms, &analytical_gradient_norm, 1e-3);

        // TEST 3
        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(_batch_size, _output_dim, _input_dim);

        let mut norm_layer = NormalNormLayer::new(input_batch[0][0].len(), 1e-8, learning_rate);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch_before(input_batch.clone());
        layer_input.set_input_batch(input_batch.clone());

        let rms_output = norm_layer.forward(&layer_input);
        let rms_output_batch = rms_output.get_output_batch();

        println!("input batch :{:?}", &input_batch);
        println!("\n norm output_batch: {:?}", &rms_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); rms_output_batch[0][0].len()]; rms_output_batch[0].len()]; rms_output_batch.len()];

        let gradient = norm_layer.backward(&previous_gradient);
        let analytical_gradient_norm = gradient.get_gradient_input_batch();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            layer_input.set_input_batch(input.clone());
            let rms_output = norm_layer.forward(&layer_input);
            let rms_output_batch = rms_output.get_output_batch();

            rms_output_batch
        };

        //let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_jacobi_without_loss(&mut loss_fn, input_batch.clone(), epsilon);
        let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient norm: {:?}", &numerical_grad_rms);
        println!("\nanalytical gradient norm: {:?}", &analytical_gradient_norm);

        let global_error = global_relative_error_l2(&numerical_grad_rms, &analytical_gradient_norm);

        println!("\n\nglobal relative gradient error: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_rms, &analytical_gradient_norm, 1e-6);
    }
}
