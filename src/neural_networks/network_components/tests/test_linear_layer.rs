#[cfg(test)]
mod test_linear_layer {
    use std::time::Instant;

    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
        network_layers::{linear_layer::LinearLayer, multi_linear_layer::MultiLinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_sum_batch},
        utils::{
            derivative::{
                global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_bias, numerical_gradient_input_batch, numerical_gradient_weights, test_gradient_batch_error,
                test_gradient_error_1d, test_gradient_error_2d,
            },
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_softmax_linear_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 5; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, output_dim);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim - 1, (output_dim - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        //let target_token_id_batch = vec![vec![0]];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);

        println!("linear output: {:?}", linear_output.get_output_batch());

        layer_input.set_input_batch(linear_output.get_output_batch());
        softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        println!("softmax output: {:?}",  softmax_layer.cross_entropy_loss_batch.as_ref().unwrap());

        let gradient_softmax: Gradient = softmax_layer.backward_inner(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);
        let (grouped_linear_gradient, analytical_gradient_bias) = (gradient_linear.get_gradient_weights(), gradient_linear.get_gradient_bias());

        let linear_weights = linear_layer.weights.clone();

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();

            layer_input.set_input_batch(_input.clone());
            let linear_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-8;
        let numerical_grad_linear: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &linear_weights.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad weights: {:?}", grouped_linear_gradient);
        println!("\nnumerical grad weights: {:?}", numerical_grad_linear);

        // Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad weights dim: {:?}, {}", grouped_linear_gradient.len(), grouped_linear_gradient[0].len());
        println!("numerical grad weights dim: {:?}, {}", numerical_grad_linear.len(), numerical_grad_linear[0].len());

        for b in 0..grouped_linear_gradient.len() {
            let analytical_row_sum: Complex<f64> = grouped_linear_gradient[b].iter().sum();
            let numerical_row_sum: Complex<f64> = numerical_grad_linear[b].iter().sum();

            println!("analytical row sum: {:?}", analytical_row_sum);
            println!("numerical row sum: {:?}", numerical_row_sum);
        }

        let global_error = global_relative_error_2d_l2(&numerical_grad_linear, &grouped_linear_gradient);
        println!("\n\n global relative gradient error weights ffn: {:?}", &global_error);
        test_gradient_error_2d(&grouped_linear_gradient, &numerical_grad_linear, 1e-5);

        // TEST BIAS
        let linear_bias = linear_layer.bias.clone();
        linear_layer.weights = linear_weights;

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Complex<f64> {
            linear_layer.bias = bias.clone();

            layer_input.set_input_batch(_input.clone());
            let linear_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &linear_bias, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad bias: {:?}", analytical_gradient_bias);
        println!("\nnumerical grad bias: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&analytical_gradient_bias, &numerical_grad_linear_bias, 1e-5);
    }

    #[test]
    fn test_linear_softmax_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let input_dim = 15;
        let output_dim = 10;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, output_dim);

        // Define a small input batch, [2][6][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, input_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, 5, 5 as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let linear_output = linear_layer.forward(&layer_input);

        layer_input.set_input_batch(linear_output.get_output_batch());
        softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward_inner(&target_token_id_batch);

        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);

        let gradient_weights_batch: Vec<Vec<Complex<f64>>> = gradient_linear.get_gradient_weights();
        let gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = gradient_linear.get_gradient_input_batch();
        let weights: Vec<Vec<Complex<f64>>> = linear_layer.weights.clone();

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();

            layer_input.set_input_batch(_input.clone());
            let linear_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let num_gradient_weight_batch: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights, 1e-5);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical gradient_weights_batch: {:?}", gradient_weights_batch);
        println!("\n analytical gradient_weights: {:?} ", gradient_weights_batch);

        //println!("\n numerical grad: {:?}", num_gradient_weight_batch);
        println!("\n num_gradient_weight_batch {:?}", num_gradient_weight_batch);

        let global_error = global_relative_error_2d_l2(&num_gradient_weight_batch, &gradient_weights_batch);
        println!("global relative gradient error weights ffn: {:?}", &global_error);

        test_gradient_error_2d(&num_gradient_weight_batch, &gradient_weights_batch, 1e-5);

        // TEST GRADIENT OF THE INPUT BATCH
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let linear_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let num_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical gradient_weights_batch: {:?}", gradient_weights_batch);
        println!("\n analytical gradient_input_batch: {:?}", gradient_input_batch);
        println!(
            "\n anlytical gradient_input_batch dim: {} {} {}",
            gradient_input_batch.len(),
            gradient_input_batch[0].len(),
            gradient_input_batch[0][0].len()
        );

        //println!("\n numerical grad: {:?}", num_gradient_weight_batch);
        println!("\n numerical num_gradient_input_batch: {:?}", &num_gradient_input_batch);
        println!(
            "\n numerical num_gradient_input_batch dim: {} {} {}",
            num_gradient_input_batch.len(),
            num_gradient_input_batch[0].len(),
            num_gradient_input_batch[0][0].len()
        );

        for b in 0..num_gradient_input_batch.len() {
            for s in 0..gradient_input_batch[b].len() {
                let analytical_row_sum: Complex<f64> = gradient_input_batch[b][s].iter().sum();
                let numerical_row_sum: Complex<f64> = num_gradient_input_batch[b][s].iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        let global_error = global_relative_error_l2(&num_gradient_input_batch, &gradient_input_batch);
        println!("global relative gradient error gradient input batch: {:?}", &global_error);

        //test_gradient_batch_error(&num_gradient_input_batch, &gradient_input_batch, 1e-3);
    }

    #[test]
    fn test_multi_linear_softmax_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 3;
        let input_dim = 12;
        let output_dim = 20;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;

        // Create a simple MultiLinearLayer with the given input and output dimensions
        let mut multi_linear_layer: MultiLinearLayer = MultiLinearLayer::new(learning_rate, input_dim, output_dim, 5);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, output_dim);

        let (combined_weights, combined_bias) = multi_linear_layer.get_combined_weights();
        linear_layer.weights = combined_weights.clone();
        linear_layer.bias = combined_bias.clone();

        // Define a small input batch, [2][6][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, 12, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, 5, (output_dim - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let linear_output = multi_linear_layer.forward(&layer_input);

        layer_input.set_input_batch(linear_output.get_output_batch());
        softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward_inner(&target_token_id_batch);
        let gradient_linear: Gradient = multi_linear_layer.backward(&gradient_softmax);

        let anal_multilayer_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = gradient_linear.get_gradient_input_batch();

        println!("target_token_id_batch dim: {} {}", target_token_id_batch.len(), target_token_id_batch[0].len());

        // TEST GRADIENT OF THE INPUT BATCH
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let linear_output = multi_linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let num_multi_linear_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical gradient_weights_batch: {:?}", gradient_weights_batch);
        //println!("\n analytical gradient_input_batch: {:?}", anal_multilayer_gradient_input_batch);
        println!(
            "\n anlytical gradient_input_batch dim: {} {} {}",
            anal_multilayer_gradient_input_batch.len(),
            anal_multilayer_gradient_input_batch[0].len(),
            anal_multilayer_gradient_input_batch[0][0].len()
        );

        //println!("\n numerical grad: {:?}", num_gradient_weight_batch);
        //println!("\n numerical num_gradient_input_batch: {:?}", &num_multi_linear_gradient_input_batch);
        println!(
            "\n numerical num_gradient_input_batch dim: {} {} {}",
            num_multi_linear_gradient_input_batch.len(),
            num_multi_linear_gradient_input_batch[0].len(),
            num_multi_linear_gradient_input_batch[0][0].len()
        );

        let global_error = global_relative_error_l2(&num_multi_linear_gradient_input_batch, &anal_multilayer_gradient_input_batch);

        println!("global relative gradient error gradient input batch: {:?}", &global_error);

        test_gradient_batch_error(&num_multi_linear_gradient_input_batch, &anal_multilayer_gradient_input_batch, 1e-3);

        // TEST 2: Gradient of the normal linear layer with the same weights as the multi linear layer
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let linear_output = linear_layer.forward(&layer_input);

        layer_input.set_input_batch(linear_output.get_output_batch());
        softmax_layer.forward_inner(&layer_input, None, Some(target_token_id_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward_inner(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);

        let anal_linear_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = gradient_linear.get_gradient_input_batch();

        println!("target_token_id_batch dim: {} {}", target_token_id_batch.len(), target_token_id_batch[0].len());

        // TEST GRADIENT OF THE INPUT BATCH
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let linear_output = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward_inner(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let num_linear_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical gradient_weights_batch: {:?}", gradient_weights_batch);
        //println!("\n analytical gradient_input_batch: {:?}", anal_linear_gradient_input_batch);
        println!(
            "\n anlytical gradient_input_batch dim: {} {} {}",
            anal_linear_gradient_input_batch.len(),
            anal_linear_gradient_input_batch[0].len(),
            anal_linear_gradient_input_batch[0][0].len()
        );

        //println!("\n numerical grad: {:?}", num_gradient_weight_batch);
        //println!("\n numerical num_gradient_input_batch: {:?}", &num_linear_gradient_input_batch);
        println!(
            "\n numerical num_gradient_input_batch dim: {} {} {}",
            num_linear_gradient_input_batch.len(),
            num_linear_gradient_input_batch[0].len(),
            num_linear_gradient_input_batch[0][0].len()
        );

        let global_error = global_relative_error_l2(&num_linear_gradient_input_batch, &anal_linear_gradient_input_batch);

        println!("global relative gradient error gradient input batch: {:?}", &global_error);

        test_gradient_batch_error(&num_linear_gradient_input_batch, &anal_linear_gradient_input_batch, 1e-3);
        test_gradient_batch_error(&num_linear_gradient_input_batch, &num_multi_linear_gradient_input_batch, 1e-3);
    }

    #[test]
    fn test_compare_matrix_multiplication_speed() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let input_seq = 10;
        let input_dim = 20;
        let output_dim = 50;
        let learning_rate = 0.01;

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 0.0); input_dim]; input_seq]; batch_size];
        println!("input batch dim: {} {} {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());
        // Create a simple MultiLinearLayer with the given input and output dimensions
        let mut multi_linear_layer: MultiLinearLayer = MultiLinearLayer::new(learning_rate, input_dim, output_dim, 15);
        let (weights, bias) = multi_linear_layer.get_combined_weights();

        let mut linear_layer = LinearLayer::new(learning_rate, input_dim, output_dim, true);
        linear_layer.weights = weights.clone();
        linear_layer.bias = bias.clone();

        println!("multi linear layer weights dim: {} {} ", weights.len(), weights[0].len());

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        let start = Instant::now();
        let _multi_linear_output = multi_linear_layer.forward(&layer_input);
        println!("MultiLinearLayer forward pass took: {:?}", start.elapsed().as_secs_f64());

        let start = Instant::now();
        let _linear_output = linear_layer.forward(&layer_input);
        println!("LinearLayer forward pass took: {:?}", start.elapsed().as_secs_f64());

        let start = Instant::now();
        let _multi_linear_output = multi_linear_layer.forward(&layer_input);
        println!("MultiLinearLayer 2 forward pass took: {:?}", start.elapsed().as_secs_f64());

        let start = Instant::now();
        let _linear_output = linear_layer.forward(&layer_input);
        println!("LinearLayer 2 forward pass took: {:?}", start.elapsed().as_secs_f64());

        let start = Instant::now();
        let _multi_linear_output = multi_linear_layer.forward(&layer_input);
        println!("MultiLinearLayer 3 forward pass took: {:?}", start.elapsed().as_secs_f64());

        let start = Instant::now();
        let _linear_output = linear_layer.forward(&layer_input);
        println!("LinearLayer 3 forward pass took: {:?}", start.elapsed().as_secs_f64());
    }
}

