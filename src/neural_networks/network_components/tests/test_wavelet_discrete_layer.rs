#[cfg(test)]
mod test_wavelet_discrete_layer {
    use crate::{
        neural_networks::{
            network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
            network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch, wavelet_discrete_layer::DiscreteWaveletLayer},
            utils::{
                derivative::{global_relative_error_2d_l2, numerical_gradient_dwt_1d, numerical_gradient_input, numerical_gradient_input_batch, numerical_gradient_input_batch_sum_without_loss, test_gradient_batch_error, test_gradient_error_2d},
                random_arrays::{generate_random_complex_2d, generate_random_complex_3d, generate_random_u32_batch},
            },
        },
        wavelet_transform::{
            dwt::{dwt_1d, get_ll_hh_1d, grad_dwt_1d_trend},
            dwt_types::DiscreteWaletetType,
            modes::WaveletMode,
        },
    };

    use assert_approx_eq::assert_approx_eq;
    use num::Complex;

    #[test]
    fn test_wavelet_discrete_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let feature_dim = 3;
        let seq_len = 8;
        let epsilon = 1e-8;
        let epsilon_test = 1e-6;
        let target_dim = 3;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_discrete_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, seq_len, feature_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, target_dim - 1, (seq_len - 1) as u32);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_target_batch_ids(target_token_id_batch.clone());

        println!("target token id batch: {:?}", target_token_id_batch);

        let _wavelet_output = wavelet_discrete_layer.forward(&layer_input);
        let dwt = _wavelet_output.get_output_batch();

        println!("input dim: {} {} {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());
        // println!("wavelet output: {:?}", dwt);
        println!("wavelet dim: {} {} {}", dwt.len(), dwt[0].len(), dwt[0][0].len());

        let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 0.0); dwt[0][0].len()]; dwt[0].len()]; dwt.len()];
        println!("previous gradient batch: {:?} {} {}", previous_gradient_batch.len(), previous_gradient_batch[0].len(), previous_gradient_batch[0][0].len());

        let mut previous_gradient = Gradient::new_default();
        previous_gradient.set_gradient_input_batch(previous_gradient_batch);
        let wavelet_gradient = wavelet_discrete_layer.backward(&previous_gradient);

        let (analytical_grad_batch, _analytical_grad) = (wavelet_gradient.get_gradient_input_batch(), wavelet_gradient.get_gradient_input());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            layer_input.set_input_batch(input.clone());
            let wavelet_output = wavelet_discrete_layer.forward(&layer_input);

            wavelet_output.get_output_batch()
        };

        let numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_sum_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad_batch);
        println!("\n analytical grad dim: {:?} {} {}", analytical_grad_batch.len(), analytical_grad_batch[0].len(), analytical_grad_batch[0][0].len());

        println!("\n numerical grad: {:?}", numerical_grad_batch);
        println!("\n numerical grad dim: {:?} {} {}", numerical_grad_batch.len(), numerical_grad_batch[0].len(), numerical_grad_batch[0][0].len());

        let emb_dim = analytical_grad_batch[0][0].len();
        for b in 0..analytical_grad_batch.len() {
            for s in 0..analytical_grad_batch[b].len() {
                let analytical_row_sum: Complex<f64> = (analytical_grad_batch[b][s][0..emb_dim]).iter().sum();
                let numerical_row_sum: Complex<f64> = (numerical_grad_batch[b][s][0..emb_dim]).iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        test_gradient_batch_error(&analytical_grad_batch, &numerical_grad_batch, epsilon_test);
    }

    #[test]
    fn test_1d_wavelet_discrete_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 1; // Match output_dim to your layer's output
        let epsilon = 1e-7;

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Complex<f64>>> = generate_random_complex_2d(output_dim, input_dim);
        let input_row_1 = &input_batch[0];
        let dwt_type = &DiscreteWaletetType::DB2;
        let dwt_mode = &WaveletMode::SYMMETRIC;

        let dwt_1 = dwt_1d(&input_row_1, dwt_type, dwt_mode);
        let dwt_trend = &get_ll_hh_1d(&dwt_1)[0];

        let prev_grad: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); dwt_trend.len()];

        let analytical_grad = grad_dwt_1d_trend(&prev_grad, dwt_type, dwt_mode);

        let loss_fn = |input: &Vec<Complex<f64>>| -> Complex<f64> {
            let dwt_1 = dwt_1d(&input.clone(), dwt_type, dwt_mode);
            get_ll_hh_1d(&dwt_1)[0].clone().iter().sum()
        };

        let numerical_grad: Vec<Complex<f64>> = numerical_gradient_dwt_1d(&loss_fn, &input_row_1, epsilon);

        println!("\n dwt_trend: {:?}", dwt_trend);
        println!("\n dwt_trend dim: {:?}", dwt_trend.len());

        //Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad: {:?}", analytical_grad);
        println!("\n analytical grad dim: {:?}", analytical_grad.len());
        println!("\n numerical grad: {:?}", numerical_grad);
        println!("\n numerical grad dim: {:?}", numerical_grad.len());

        let sum_analytical: Complex<f64> = analytical_grad.iter().sum();
        let sum_numerical: Complex<f64> = numerical_grad[0..dwt_trend.len()].iter().sum();

        println!("\n sum_analytical: {:?}", sum_analytical);
        println!("\n sum_numerical: {:?}", sum_numerical);

        assert_approx_eq!(sum_analytical.re, sum_numerical.re);
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
        let target_dim = 15;

        let linear_output_dim = 70;
        // Create a simple LinearLayer with the given input and output dimensions
        let mut wavelet_layer: DiscreteWaveletLayer = DiscreteWaveletLayer::new();
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, linear_output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, target_dim - 1, (output_dim + 1) as u32);

        println!("target token id batch: {:?}", target_token_id_batch);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
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
