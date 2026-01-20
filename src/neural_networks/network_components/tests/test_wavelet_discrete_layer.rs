#[cfg(test)]
mod test_wavelet_discrete_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
        network_layers::{feedforward_layer::FeedForwardLayer, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer, wavelet_discrete_layer::DiscreteWaveletLayer},
        network_types::{
            neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_sum_batch,
        },
        utils::{
            derivative::{global_relative_error_2d_l2, numerical_gradient_input, test_gradient_batch_error, test_gradient_error_2d},
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;
    use crate::neural_networks::utils::matrix::RowMajorMatrix;

    #[test]
    fn test_wavelet_discrete_compression() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 4; // Match the input dimension with your input batch
        let output_dim = 15; // Match output_dim to your layer's output

        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_padding_mask_batch(padding_mask_batch);
        layer_input.set_forward_only(false);
        layer_input.set_input_batch(input_batch.clone());

        let wavelet_output = wavelet_layer.forward(&layer_input);
        layer_input.set_input_batch(wavelet_output.get_output_batch());

        let wavelet_dwt = wavelet_output.get_output_batch();
        println!("wavelet_dwt batch: {:?} {} {}", wavelet_dwt.len(), wavelet_dwt[0].len(), wavelet_dwt[0][0].len());

        let wavelet_inverse_output = wavelet_layer.forward_inverse(&layer_input);
        layer_input.set_input_batch(wavelet_inverse_output.get_output_batch());

        let wavelet_inverse = wavelet_inverse_output.get_output_batch();
        println!("wavelet_dwt_inverse batch: {:?} {} {}", wavelet_inverse.len(), wavelet_inverse[0].len(), wavelet_inverse[0][0].len());

        println!("input batch: {:?}", input_batch);
        println!("wavelet_dwt_inverse batch: {:?} ", wavelet_inverse);

        test_gradient_batch_error(&input_batch, &wavelet_inverse, 1e-8);
    }

    #[test]
    fn test_softmax_discrete_wavelet_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1;
        let input_dim = 4;
        let output_dim = 80;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;
        let epsilon_test = 1e-1;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, input_dim);

        // Define a small input batch, [2][2][3]
        // input includes target tokens + padding already !
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, input_dim, input_dim as u32);
        // let target_token_id_batch: Vec<Vec<u32>> = vec![vec![1]; input_batch.len()];
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        // println!("input_batch: {:?}", input_batch);
        println!("target_token id batch: {:?}", target_token_id_batch);
        println!("padding mask batch: {:?}", padding_mask_batch);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_target_batch_ids(target_token_id_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let wavelet_output = wavelet_layer.forward(&layer_input);
        let wavelet_dwt = wavelet_output.get_output_batch();
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; wavelet_dwt[0].len()]; wavelet_dwt.len()];

        layer_input.set_input_batch(wavelet_output.get_output_batch());
        let _softmax_batch_output = softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let wavelet_gradient = wavelet_layer.backward(&softmax_gradient);

        let analytical_grad = wavelet_gradient.get_gradient_input();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let wavelet_output = wavelet_layer.forward(&layer_input);

            layer_input.set_input_batch(wavelet_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, input_batch.clone(), epsilon);

        // let seq_len = numerical_grad.len();
        // let numerical_grad = numerical_grad[..seq_len.saturating_sub(input_dim)].to_vec();
        // let seq_len = analytical_grad.len();
        // let analytical_grad = analytical_grad[..seq_len.saturating_sub(input_dim)].to_vec();

        // //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n analytical grad dim : {:?} {}", analytical_grad.len(), analytical_grad[0].len());
        println!("\n numerical grad: {:?}", numerical_grad);
        println!("\n numerical grad dim: {:?} {}", numerical_grad.len(), numerical_grad[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon_test);
    }

    #[test]
    fn test_softmax_discrete_wavelet_linear_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 60; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        let linear_output_dim = 70;
        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, linear_output_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, linear_output_dim);

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

        layer_input.set_input_batch(linear_output.get_output_batch());
        let _softmax_batch_output = softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        // println!("softmax output batch: {:?}", _softmax_batch_output);
        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let linear_gradient: Gradient = linear_layer.backward(&softmax_gradient);
        let wavelet_gradient: Gradient = wavelet_layer.backward(&linear_gradient);

        let analytical_grad = wavelet_gradient.get_gradient_input();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let wavelet_output = wavelet_layer.forward(&layer_input);
            layer_input.set_input_batch(wavelet_output.get_output_batch());

            // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
            let linear_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, input_batch.clone(), epsilon);

        // let seq_len = numerical_grad.len();
        // let numerical_grad = numerical_grad[..seq_len.saturating_sub(input_dim)].to_vec();
        // let seq_len = numerical_grad.len();
        // let analytical_grad = analytical_grad[..seq_len.saturating_sub(input_dim)].to_vec();

        //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n analytical grad dim : {:?} {}", analytical_grad.len(), analytical_grad[0].len());

        println!("\n numerical grad: {:?}", numerical_grad);
        println!("\n numerical grad dim: {:?} {}", numerical_grad.len(), numerical_grad[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        test_gradient_error_2d(&numerical_grad, &analytical_grad, epsilon_test);
    }

    #[test]
    fn test_softmax_discrete_wavelet_inverse_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 40; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;

        let linear_output_dim = 70;
        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(input_dim, linear_output_dim, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, linear_output_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, linear_output_dim);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, 4, 4 as u32);
        let mut padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        println!("target token id batch: {:?}", target_token_id_batch);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());
        layer_input.set_target_batch_ids(target_token_id_batch.clone());

        let wavelet_output = wavelet_layer.forward(&layer_input);
        layer_input.set_input_batch(wavelet_output.get_output_batch());
        layer_input.set_padding_mask_batch(wavelet_output.get_padding_mask_batch());
        padding_mask_batch = wavelet_output.get_padding_mask_batch();

        let ffn_output = ffn_layer.forward(&layer_input);
        layer_input.set_input_batch(ffn_output.get_output_batch());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);

        layer_input.set_input_batch(linear_output.get_output_batch());
        let _softmax_batch_output = softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        // println!("softmax output batch: {:?}", _softmax_batch_output);
        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let linear_gradient: Gradient = linear_layer.backward(&softmax_gradient);

        let linear_layer_gradient = linear_gradient.get_gradient_input_batch();
        println!(
            "linear_layer_gradient dim: {} {} {}",
            linear_layer_gradient.len(),
            linear_layer_gradient[0].len(),
            linear_layer_gradient[0][0].len()
        );

        let ffn_gradient: Gradient = ffn_layer.backward(&linear_gradient.get_gradient_input_batch());
        let wavelet_gradient: Gradient = wavelet_layer.backward(&ffn_gradient);

        let analytical_grad = wavelet_gradient.get_gradient_input();
        println!("analytical_grad dim: {} {}", analytical_grad.len(), analytical_grad[0].len());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let wavelet_output = wavelet_layer.forward(&layer_input);
            layer_input.set_input_batch(wavelet_output.get_output_batch());

            let ffn_output = ffn_layer.forward(&layer_input);
            layer_input.set_input_batch(ffn_output.get_output_batch());

            // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
            let linear_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, input_batch.clone(), epsilon);

        // //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n analytical grad dim : {:?} {}", analytical_grad.len(), analytical_grad[0].len());

        println!("\n numerical grad: {:?}", numerical_grad);
        println!("\n numerical grad dim: {:?} {}", numerical_grad.len(), numerical_grad[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        for b in 0..numerical_grad.len() {
            let analytical_row_sum: Complex<f64> = analytical_grad[b].iter().sum();
            let numerical_row_sum: Complex<f64> = numerical_grad[b].iter().sum();

            println!("analytical row sum: {:?}", analytical_row_sum);
            println!("numerical row sum: {:?}", numerical_row_sum);
        }

        test_gradient_error_2d(&numerical_grad, &analytical_grad, 1e-3);
    }

    #[test]
    fn test_softmax_discrete_wavelet_inverse_only_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 20; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;

        let linear_output_dim = 70;
        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, linear_output_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, linear_output_dim);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, input_dim, input_dim as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        println!("target token id batch: {:?}", target_token_id_batch);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());
        layer_input.set_target_batch_ids(target_token_id_batch.clone());

        let wavelet_output = wavelet_layer.forward(&layer_input);
        layer_input.set_input_batch(wavelet_output.get_output_batch());

        let wavelet_dwt: Vec<Vec<Vec<Complex<f64>>>> = wavelet_output.get_output_batch();
        println!("wavelet_dwt batch: {:?} {} {}", wavelet_dwt.len(), wavelet_dwt[0].len(), wavelet_dwt[0][0].len());

        let wavelet_inverse_output = wavelet_layer.forward_inverse(&layer_input);
        layer_input.set_input_batch(wavelet_inverse_output.get_output_batch());

        let wavelet_inverse = wavelet_inverse_output.get_output_batch();
        println!("wavelet_dwt_inverse batch: {:?} {} {}", wavelet_inverse.len(), wavelet_inverse[0].len(), wavelet_inverse[0][0].len());

        println!("input batch: {:?} {} {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());
        println!("wavelet_dwt_inverse batch: {:?} {} {} ", wavelet_inverse.len(), wavelet_inverse[0].len(), wavelet_inverse[0][0].len());

        println!("input batch: {:?}", input_batch);
        println!("wavelet_dwt_inverse batch: {:?} ", wavelet_inverse);

        test_gradient_batch_error(&input_batch, &wavelet_inverse, 1e-8);

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);
        
        layer_input.set_input_batch(linear_output.get_output_batch());
        let _softmax_batch_output = softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        // println!("softmax output batch: {:?}", _softmax_batch_output);
        let softmax_gradient: Gradient = softmax_layer.backward(&target_token_id_batch);
        let linear_gradient: Gradient = linear_layer.backward(&softmax_gradient);

        let linear_layer_gradient = linear_gradient.get_gradient_input_batch();
        println!(
            "linear_layer_gradient dim: {} {} {}",
            linear_layer_gradient.len(),
            linear_layer_gradient[0].len(),
            linear_layer_gradient[0][0].len()
        );

        let wavelet_gradient_inverse: Gradient = wavelet_layer.backward_inverse(&linear_gradient);

        let compressed_inversed_backward = wavelet_gradient_inverse.get_gradient_input_batch();
        println!(
            "compressed_inversed_backward dim: {} {} {}",
            compressed_inversed_backward.len(),
            compressed_inversed_backward[0].len(),
            compressed_inversed_backward[0][0].len()
        );

        let analytical_grad = wavelet_gradient_inverse.get_gradient_input();
        println!("analytical_grad dim: {} {}", analytical_grad.len(), analytical_grad[0].len());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let wavelet_output_inverse = wavelet_layer.forward_inverse(&layer_input);
            layer_input.set_input_batch(wavelet_output_inverse.get_output_batch());

            // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
            let linear_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, wavelet_dwt.clone(), epsilon);

        // //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n analytical grad dim : {:?} {}", analytical_grad.len(), analytical_grad[0].len());

        println!("\n numerical grad: {:?}", numerical_grad);
        println!("\n numerical grad dim: {:?} {}", numerical_grad.len(), numerical_grad[0].len());

        let global_error = global_relative_error_2d_l2(&numerical_grad, &analytical_grad);
        println!("\n\n global relative error input gradient: {:?}", &global_error);

        test_gradient_error_2d(&numerical_grad, &analytical_grad, 1e-3);
    }

    #[test]
    fn test_separate_input_target() {
        let batch_size = 1;
        let input_dim = 3;
        let output_dim = 10;

        // input dim = target dim = 3
        // Input = 5, Target = 3, no padding

        let wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, input_dim, input_dim as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let input = &input_batch[0];
        let target_ids = &target_token_id_batch[0];
        let padding_mask = &padding_mask_batch[0];

        let (input_s, target_s, padding_input_s) = wavelet_layer.separate_input_target(input, target_ids, padding_mask);

        // println!("\n input separated: {:?}", input_s);
        // println!("\n target separated: {:?}", target_s);
        // println!("\n padding mask separated: {:?}", padding_input_s);

        for i in 0..input_s.len() {
            assert_eq!(input_s[i], input[i]);
        }
        for i in 0..target_s.len() {
            assert_eq!(target_s[i], input[input_s.len() + i]);
        }

        assert_eq!(input_s.len(), input.len() - input_dim);
        assert_eq!(padding_input_s.len(), input.len() - input_dim);
        assert_eq!(target_s.len(), target_ids.len());

        // Padding mask
        // Padding length = 2
        let padding_mask: Vec<u32> = vec![1, 1, 1, 1, 1, 1, 1, 1, 0, 0];
        let (input_s, target_s, padding_input_s) = wavelet_layer.separate_input_target(input, target_ids, &padding_mask);

        println!("\n input original: {:?}", input);
        println!("\n padding mask original: {:?}", padding_mask_batch);

        println!("\n input separated: {:?}", input_s);
        println!("\n target separated: {:?}", target_s);
        println!("\n padding mask separated: {:?}", padding_input_s);

        assert_eq!(input_s.len(), input.len() - input_dim);
        assert_eq!(padding_input_s.len(), input.len() - input_dim);
        assert_eq!(target_s.len(), target_ids.len());

        for i in 0..input_s.len() - 2 {
            assert_eq!(input_s[i], input[i]);
        }

        let mut i_t = 0;
        for i in input_s.len()..input.len() {
            assert_eq!(target_s[i_t], input[i]);
            i_t += 1;
        }

        // Padding mask
        // Padding length = 2
        let padding_mask: Vec<u32> = vec![1, 1, 1, 1, 1, 0, 0, 1, 1, 1];
        let (input_s, target_s, padding_input_s) = wavelet_layer.separate_input_target(input, target_ids, &padding_mask);

        println!("\n input original: {:?}", input);
        println!("\n padding mask original: {:?}", padding_mask_batch);

        println!("\n input separated: {:?}", input_s);
        println!("\n target separated: {:?}", target_s);
        println!("\n padding mask separated: {:?}", padding_input_s);

        assert_eq!(input_s.len(), input.len() - input_dim);
        assert_eq!(padding_input_s.len(), input.len() - input_dim);
        assert_eq!(target_s.len(), target_ids.len());

        for i in 0..input_s.len() - 2 {
            assert_eq!(input_s[i], input[i]);
        }

        let mut i_t = 0;
        for i in input_s.len()..input.len() {
            assert_eq!(target_s[i_t], input[i]);
            i_t += 1;
        }
    }

    #[test]
    fn test_wavelet_discrete_rm_forward_backward_smoke() {
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();

        // 1 batch, 4 rows, 4 cols
        let input_rm = vec![RowMajorMatrix::from_data(
            4,
            4,
            vec![
                Complex::new(0.1, 0.0),
                Complex::new(0.2, 0.0),
                Complex::new(0.3, 0.0),
                Complex::new(0.4, 0.0),
                Complex::new(0.5, 0.0),
                Complex::new(0.6, 0.0),
                Complex::new(0.7, 0.0),
                Complex::new(0.8, 0.0),
                Complex::new(0.9, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.1, 0.0),
                Complex::new(1.2, 0.0),
                Complex::new(1.3, 0.0),
                Complex::new(1.4, 0.0),
                Complex::new(1.5, 0.0),
                Complex::new(1.6, 0.0),
            ],
        )];

        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; 4]];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_padding_mask_batch(padding_mask_batch);
        layer_input.set_forward_only(false);
        layer_input.set_input_batch_rm(input_rm);

        let out = wavelet_layer.forward(&layer_input);
        assert!(out.get_output_batch_rm_ref().is_some(), "Expected RM output from discrete wavelet forward");

        let out_rm = out.get_output_batch_rm();
        let prev_grad_rm: Vec<RowMajorMatrix<Complex<f64>>> = out_rm
            .iter()
            .map(|m| RowMajorMatrix::from_data(m.rows, m.cols, vec![Complex::new(1.0, 0.0); m.rows * m.cols]))
            .collect();

        let mut prev_gradient = Gradient::new_default();
        prev_gradient.set_gradient_input_batch_rm(prev_grad_rm);
        let g = wavelet_layer.backward(&prev_gradient);
        assert!(g.get_gradient_input_batch_rm_ref().is_some(), "Expected RM gradient from discrete wavelet backward");
    }
}
