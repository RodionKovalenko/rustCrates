#[cfg(test)]
mod test_wavelet_layer {
    use crate::{
        neural_networks::{
            network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
            network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch, wavelet_layer::WaveletLayer},
            utils::{
                derivative::{global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_input, numerical_gradient_input_batch, numerical_gradient_input_batch_sum_without_loss, numerical_gradient_input_batch_without_loss, test_gradient_batch_error, test_gradient_error_2d},
                random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
            },
        },
        wavelet_transform::{
            cwt_complex::CWTComplex,
            cwt_type_resolver::{cmor_derivative, cmorl},
            cwt_types::ContinuousWaletetType,
        },
    };

    use num::Complex;

    #[test]
    fn test_wavelet_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 6; // Match output_dim to your layer's output
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: WaveletLayer = WaveletLayer::new();

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);

        let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        let _wavelet_output = wavelet_layer.forward(&layer_input);

        let mut previous_gradient = Gradient::new_default();
        previous_gradient.set_gradient_input_batch(previous_gradient_batch);
        let wavelet_gradient = wavelet_layer.backward(&previous_gradient);

        let (analytical_grad_batch, _analytical_grad) = (wavelet_gradient.get_gradient_input_batch(), wavelet_gradient.get_gradient_input());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            layer_input.set_input_batch(input.clone());
            let wavelet_output = wavelet_layer.forward(&layer_input);

            wavelet_output.get_output_batch()
        };

        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_sum_without_loss(&mut loss_fn, input_batch.clone(), epsilon);
        //let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad_batch);
        println!("\n numerical grad: {:?}", numerical_grad_batch);

        for b in 0..analytical_grad_batch.len() {
            for s in 0..analytical_grad_batch[b].len() {
                let analytical_row_sum: Complex<f64> = analytical_grad_batch[b][s].iter().sum();
                let numerical_row_sum: Complex<f64> = numerical_grad_batch[b][s].iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        let global_error = global_relative_error_l2(&numerical_grad_batch, &analytical_grad_batch);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_batch, &analytical_grad_batch, epsilon_test);
    }

    #[test]
    fn test_softmax_wavelet_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 6; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: WaveletLayer = WaveletLayer::new();
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim - 1, (output_dim - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        let wavelet_output = wavelet_layer.forward(&layer_input);
        let _softmax_batch_output = softmax_layer.forward(&wavelet_output.get_output_batch(), Some(padding_mask_batch.clone()));

        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let wavelet_gradient = wavelet_layer.backward(&softmax_gradient);

        let (analytical_grad_batch, analytical_grad) = (wavelet_gradient.get_gradient_input_batch(), wavelet_gradient.get_gradient_input());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let wavelet_output = wavelet_layer.forward(&layer_input);

            let softmax_batch_output = softmax_layer.forward(&wavelet_output.get_output_batch(), Some(padding_mask_batch.clone()));
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

            loss
        };

        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, input_batch.clone(), epsilon);
        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n numerical grad: {:?}", numerical_grad);

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        for b in 0..analytical_grad_batch.len() {
            for s in 0..analytical_grad_batch[b].len() {
                let analytical_row_sum: Complex<f64> = analytical_grad_batch[b][s].iter().sum();
                let numerical_row_sum: Complex<f64> = numerical_grad_batch[b][s].iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon_test);
        test_gradient_batch_error(&numerical_grad_batch, &analytical_grad_batch, epsilon_test);
    }

    #[test]
    fn test_softmax_wavelet_linear_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 6; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        let linear_output_dim = 7;
        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: WaveletLayer = WaveletLayer::new();
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, linear_output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim - 1, (output_dim - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        let wavelet_output = wavelet_layer.forward(&layer_input);
        layer_input.set_input_batch(wavelet_output.get_output_batch());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);

        let _softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), Some(padding_mask_batch.clone()));

        println!("softmax output batch: {:?}", _softmax_batch_output);
        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let linear_gradient = linear_layer.backward(&softmax_gradient);
        let wavelet_gradient = wavelet_layer.backward(&linear_gradient);

        let (analytical_grad_batch, analytical_grad) = (wavelet_gradient.get_gradient_input_batch(), wavelet_gradient.get_gradient_input());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let wavelet_output = wavelet_layer.forward(&layer_input);
            layer_input.set_input_batch(wavelet_output.get_output_batch());

            // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
            let linear_output = linear_layer.forward(&layer_input);

            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), Some(padding_mask_batch.clone()));
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

            loss
        };

        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, input_batch.clone(), epsilon);
        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n numerical grad: {:?}", numerical_grad);

        for b in 0..analytical_grad_batch.len() {
            for s in 0..analytical_grad_batch[b].len() {
                let analytical_row_sum: Complex<f64> = analytical_grad_batch[b][s].iter().sum();
                let numerical_row_sum: Complex<f64> = numerical_grad_batch[b][s].iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon_test);
        test_gradient_batch_error(&numerical_grad_batch, &analytical_grad_batch, epsilon_test);
    }

    #[test]
    fn test_derivative_cmor() {
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 5; // Match output_dim to your layer's output
        let epsilon = 1e-8;
        let epsilon_test = 1e-5;

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);

        let wavelet = CWTComplex {
            scales: vec![1.0],
            cw_type: ContinuousWaletetType::CMOR,
            sampling_period: 1.0,
            fc: 1.0,
            fb: 1.0,
            m: 1.0,
            frequencies: vec![],
        };

        let analytical_gradient_batch = cmor_derivative(&input_batch, &wavelet);
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> { input.iter().map(|batch| batch.iter().map(|row| row.iter().map(|value| cmorl(&value, &wavelet)).collect()).collect()).collect() };

        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\n analytical grad: {:?}", analytical_gradient_batch);
        println!("\n numerical grad: {:?}", numerical_grad_batch);

        let global_error = global_relative_error_l2(&numerical_grad_batch, &analytical_gradient_batch);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_batch, &analytical_gradient_batch, epsilon_test);
    }
}
