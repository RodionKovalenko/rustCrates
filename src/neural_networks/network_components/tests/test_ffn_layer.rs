#[cfg(test)]
pub mod test_ffn_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer::LayerEnum, layer_input_struct::LayerInput, layer_output_struct::LayerOutput, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{feedforward_layer::FeedForwardLayer, neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::{
            derivative::{global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_bias, numerical_gradient_input_batch, numerical_gradient_weights, test_gradient_error_1d, test_gradient_error_2d},
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_softmax_linear_with_loss_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 4;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 5; // Match the input dimension with your input batch
        let output_dim = 5; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(input_dim, output_dim, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][3][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim, 2);

        let mut layer_input = LayerInput::new_default();

        layer_input.set_input_batch(input_batch.clone());

        // Forward pass (initialize the input batch) [2][4][3]  * [3][4] => [2][2][4]
        let ffn_output = ffn_layer.forward(&layer_input);

        layer_input.set_input_batch(ffn_output.get_output_batch());
        let linear_output = linear_layer.forward(&layer_input);
        let _softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let analytical_grad_batch_softmax: Vec<Vec<Vec<Complex<f64>>>> = gradient_softmax.get_gradient_input_batch();

        let gradient_linear: Gradient = linear_layer.backward(&analytical_grad_batch_softmax);
        let (grouped_linear_gradient_weights, _analytical_gradient_bias_linear) = (gradient_linear.get_gradient_weights(), gradient_linear.get_gradient_bias());

        let weights_linear = linear_layer.weights.clone();
        let bias_linear = linear_layer.bias.clone();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();

            layer_input.set_input_batch(input.to_vec());
            let ffn_output = ffn_layer.forward(&layer_input);

            layer_input.set_input_batch(ffn_output.get_output_batch());
            let linear_output = linear_layer.forward(&layer_input);

            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-8;
        let numerical_grad_linear: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_linear, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad weights linear: {:?}", grouped_linear_gradient_weights);
        println!("\nnumerical grad weights linear: {:?}", numerical_grad_linear);

        test_gradient_error_2d(&grouped_linear_gradient_weights, &numerical_grad_linear, 1e-6);

        let global_error = global_relative_error_2d_l2(&grouped_linear_gradient_weights, &numerical_grad_linear);
        println!("global relative gradient error gradient_weights_batch: {:?}", &global_error);

        linear_layer.weights = weights_linear.clone();

        // TEST BIAS
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Complex<f64> {
            linear_layer.bias = bias.clone();

            layer_input.set_input_batch(input.to_vec());
            let ffn_output = ffn_layer.forward(&layer_input);

            layer_input.set_input_batch(ffn_output.get_output_batch());
            let linear_output = linear_layer.forward(&layer_input);

            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &bias_linear, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad bias linear: {:?}", _analytical_gradient_bias_linear);
        println!("\nnumerical grad bias linear: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&_analytical_gradient_bias_linear, &numerical_grad_linear_bias, 1e-6);
    }

    #[test]
    fn test_softmax_linear_ffn_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 2; // Update to match the input structure
        let input_dim = 4; // Match the input dimension with your input batch
        let output_dim = 5; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(input_dim, output_dim, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][3][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, 5, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, 5, (5 - 1) as u32);

        println!("input batch dim: {}, {}, {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let ffn_batch_output = ffn_layer.forward(&layer_input);

        layer_input.set_input_batch(ffn_batch_output.get_output_batch());
        let linear_output: LayerOutput = linear_layer.forward(&layer_input);
        let _softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax.get_gradient_input_batch());
        let gradient_ffn: Gradient = ffn_layer.backward(&gradient_linear.get_gradient_input_batch());

        let (grouped_ffn_gradient_weights, analytical_gradient_ffn_bias) = (gradient_ffn.get_gradient_weights(), gradient_ffn.get_gradient_bias());

        // TEST GRADIENT OF INPUT
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let ffn_batch_output = ffn_layer.forward(&layer_input);

            layer_input.set_input_batch(ffn_batch_output.get_output_batch());
            let linear_output: LayerOutput = linear_layer.forward(&layer_input);

            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_input_ffn: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);
        let analytical_grad_input_ffn: Vec<Vec<Vec<Complex<f64>>>> = gradient_ffn.get_gradient_input_batch();

        // Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad input dim: {:?}, {}", analytical_grad_input_ffn.len(), analytical_grad_input_ffn[0].len());
        println!("numerical grad input dim: {:?}, {}", numerical_grad_input_ffn.len(), numerical_grad_input_ffn[0].len());

        for b in 0..analytical_grad_input_ffn.len() {
            for s in 0..analytical_grad_input_ffn[b].len() {
                let analytical_row_sum: Complex<f64> = analytical_grad_input_ffn[b][s].iter().sum();
                let numerical_row_sum: Complex<f64> = numerical_grad_input_ffn[b][s].iter().sum();

                println!("analytical row sum: {:?}", analytical_row_sum);
                println!("numerical row sum: {:?}", numerical_row_sum);
            }
        }

        let global_error = global_relative_error_l2(&analytical_grad_input_ffn, &numerical_grad_input_ffn);
        println!("\n\n global relative gradient error input ffn: {:?}", &global_error);

        println!("\n\n  gradient input ffn numerical gradient: {:?}", &numerical_grad_input_ffn);
        println!("\n\n  gradient input ffn analytical gradient: {:?}", &analytical_grad_input_ffn);

        let weights_dense = match ffn_layer.layers.get(0) {
            Some(LayerEnum::Dense(dense_layer)) => dense_layer.weights.clone(),
            _ => vec![],
        };

        // println!("dense weights: {:?}", &weights_dense);

        let bias_dense = match ffn_layer.layers.get(0) {
            Some(LayerEnum::Dense(dense_layer)) => dense_layer.bias.clone(),
            _ => vec![],
        };

        // TEST GRADIENTS OF WEIGHTS
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            if let Some(LayerEnum::Dense(dense_layer)) = ffn_layer.layers.get_mut(0) {
                dense_layer.weights = weights.clone();
            } else {
                println!("Layer 2 does not exist!");
            }

            layer_input.set_input_batch(input.to_vec());
            let ffn_batch_output = ffn_layer.forward(&layer_input);

            layer_input.set_input_batch(ffn_batch_output.get_output_batch());
            let linear_output: LayerOutput = linear_layer.forward(&layer_input);

            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_weights_ffn: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_dense.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        // println!("\nanalytical grad weights: {:?}", grouped_ffn_gradient_weights);
        println!("\n analytical grad weights dim: {:?}, {}", grouped_ffn_gradient_weights.len(), grouped_ffn_gradient_weights[0].len());
        println!("\n numerical grad weights dim: {:?}, {}", numerical_grad_weights_ffn.len(), numerical_grad_weights_ffn[0].len());

        let global_error = global_relative_error_2d_l2(&grouped_ffn_gradient_weights, &numerical_grad_weights_ffn);
        println!("\n\n global relative gradient error weights ffn: {:?}", &global_error);

        test_gradient_error_2d(&grouped_ffn_gradient_weights, &numerical_grad_weights_ffn, 1e-4);

        if let Some(LayerEnum::Dense(dense_layer)) = ffn_layer.layers.get_mut(0) {
            dense_layer.weights = weights_dense.clone();
        } else {
            println!("Layer 2 does not exist!");
        }

        // TEST BIAS
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Complex<f64> {
            match ffn_layer.layers.get_mut(0) {
                Some(LayerEnum::Dense(dense_layer)) => dense_layer.bias = bias.clone(),
                _ => {}
            };

            layer_input.set_input_batch(input.to_vec());
            let ffn_batch_output = ffn_layer.forward(&layer_input);

            layer_input.set_input_batch(ffn_batch_output.get_output_batch());
            let linear_output: LayerOutput = linear_layer.forward(&layer_input);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &bias_dense, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad bias: {:?}", analytical_gradient_ffn_bias);
        println!("\nnumerical grad bias: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&analytical_gradient_ffn_bias, &numerical_grad_linear_bias, 1e-4);
    }
}
