#[cfg(test)]
mod test_wavelet_discrete_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch, wavelet_discrete_layer::DiscreteWaveletLayer},
        utils::{
            derivative::{global_relative_error_2d_l2, numerical_gradient_input, numerical_gradient_input_batch, test_gradient_batch_error, test_gradient_error_2d},
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_wavelet_discrete_compression() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let feature_dim = 5;
        let seq_len = 8;

        // Create a simple LinearLayer with the given input and output dimensions
        let wavelet_discrete_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, seq_len, feature_dim);
        let input = &input_batch[0].clone();

        let (trend, _details, compression_dim) = wavelet_discrete_layer.compress_partial(input);
        println!("input: {:?}", input);
        println!("input dim: {} {}", input.len(), input[0].len());
        println!("trend dim: {} {}", trend.len(), trend[0].len());

        let decompresed = wavelet_discrete_layer.decompress_partial(&trend, 0, &compression_dim);

        println!("decompressed: {:?}", decompresed);
        println!("decomporessed dim: {} {}", decompresed.len(), decompresed[0].len());
    }

    #[test]
    fn test_softmax_discrete_wavelet_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1;
        let input_dim = 5;
        let output_dim = 18;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        // input includes target tokens + padding already !
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, input_dim, input_dim as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];
        // let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1, 1, 1, 1, 1, 1, 1, 1]; input_batch.len()];

        println!("input_batch: {:?}", input_batch);
        println!("target_token id batch: {:?}", target_token_id_batch);
        println!("padding mask batch: {:?}", padding_mask_batch);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_target_batch_ids(target_token_id_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let wavelet_output = wavelet_layer.forward(&layer_input);
        let wavelet_dwt = wavelet_output.get_output_batch();
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; wavelet_dwt[0].len()]; wavelet_dwt.len()];
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

        let seq_len = numerical_grad.len();
        let numerical_grad = numerical_grad[..seq_len.saturating_sub(input_dim)].to_vec();
        let seq_len = analytical_grad.len();
        let analytical_grad = analytical_grad[..seq_len.saturating_sub(input_dim)].to_vec();

        // //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n analytical grad dim : {:?} {}", analytical_grad.len(), analytical_grad[0].len());
        println!("\n numerical grad: {:?}", numerical_grad);
        println!("\n numerical grad dim: {:?} {}", numerical_grad.len(), numerical_grad[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        for b in 0..analytical_grad_batch.len() {
            for s in 0..analytical_grad_batch[b].len() {
                let analytical_row_sum: Complex<f64> = analytical_grad_batch[b][s].iter().sum();
                let numerical_row_sum: Complex<f64> = (numerical_grad_batch[b][s % numerical_grad_batch[0][0].len()]).iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon_test);
        test_gradient_batch_error(&numerical_grad_batch, &analytical_grad_batch, epsilon_test);
    }

    #[test]
    fn test_softmax_discrete_wavelet_linear_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 16; // Match the input dimension with your input batch
        let output_dim = 50; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        let linear_output_dim = 70;
        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, linear_output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, input_dim, input_dim as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        println!("target token id batch: {:?}", target_token_id_batch);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch);
        layer_input.set_target_batch_ids(target_token_id_batch.clone());

        let wavelet_output = wavelet_layer.forward(&layer_input);
        layer_input.set_input_batch(wavelet_output.get_output_batch());
        let wavelet_dwt = wavelet_output.get_output_batch();
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; wavelet_dwt[0].len()]; wavelet_dwt.len()];

        println!("wavelet_dwt batch: {:?} {} {}", wavelet_dwt.len(), wavelet_dwt[0].len(), wavelet_dwt[0][0].len());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);

        let _softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), Some(padding_mask_batch.clone()));

        // println!("softmax output batch: {:?}", _softmax_batch_output);
        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let linear_gradient: Gradient = linear_layer.backward(&softmax_gradient);
        let wavelet_gradient: Gradient = wavelet_layer.backward(&linear_gradient);

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

        let seq_len = numerical_grad.len();
        let numerical_grad = numerical_grad[..seq_len.saturating_sub(input_dim)].to_vec();
        let seq_len = numerical_grad.len();
        let analytical_grad = analytical_grad[..seq_len.saturating_sub(input_dim)].to_vec();

        //Check if gradient batch dimensions match expected shapes
        // println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n analytical grad dim : {:?} {}", analytical_grad.len(), analytical_grad[0].len());

        // println!("\n numerical grad: {:?}", numerical_grad);
        println!("\n numerical grad dim: {:?} {}", numerical_grad.len(), numerical_grad[0].len());

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
}
