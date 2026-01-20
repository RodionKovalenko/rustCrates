#[cfg(test)]
pub mod test_ffn_swiglu {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
        network_layers::{
            feedforward_layer::FeedForwardLayer,
            layer::{ActivationType, Layer, LayerEnum, LayerType},
            linear_layer::LinearLayer,
            softmax_output_layer::SoftmaxLayer,
        },
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
    fn test_array_splitting() {
        let column_dim = 64;
        let input_dim = 46;
        let hidden_dim = 2048;
        let batch_size = 1;
        let learning_rate = 0.01;
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, input_dim, column_dim);
        let mut dense_layer: Layer = Layer::new(column_dim, hidden_dim, &learning_rate, &ActivationType::SWiGLU, LayerType::DenseLayer);
        let mut layer_input = LayerInput::new_default();

        layer_input.set_input_batch(input_batch);
        let layer_output: LayerOutput = dense_layer.forward(&layer_input);
        let activated_output = layer_output.get_output_batch();
        assert!(activated_output[0][0].len() == hidden_dim / 2);

        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(column_dim, hidden_dim, learning_rate);
        let ffn_output: LayerOutput = ffn_layer.forward(&layer_input);
        let ffn_activated_output = ffn_output.get_output_batch();
        assert!(ffn_activated_output[0][0].len() == column_dim);
    }

    #[test]
    fn test_softmax_linear_ffn_swiglu_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 6; // Update to match the input structure
        let input_dim = 64; // Match the input dimension with your input batch
        let output_dim = 10; // Match output_dim to your layer's output
        let hidden_dim = 26;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon: f64 = 1e-5;
        let epsilon_test = 1e-2;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(input_dim, output_dim, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, hidden_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, hidden_dim);

        // Define a small input batch, [2][3][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, _seq_len, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, _seq_len - 1, _seq_len as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        println!("input batch dim: {}, {}, {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let ffn_batch_output = ffn_layer.forward(&layer_input);

        layer_input.set_input_batch(ffn_batch_output.get_output_batch());
        let output = ffn_batch_output.get_output_batch();
        println!("FFN output batch dim: {}, {}, {}", output.len(), output[0].len(), output[0][0].len());
        println!("linear layer process");
        let linear_output: LayerOutput = linear_layer.forward(&layer_input);

        println!(
            "linear output batch dim: {}, {}, {}",
            linear_output.get_output_batch().len(),
            linear_output.get_output_batch()[0].len(),
            linear_output.get_output_batch()[0][0].len()
        );

        layer_input.set_input_batch(linear_output.get_output_batch());
        let _softmax_batch_output = softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);
        let gradient_ffn: Gradient = ffn_layer.backward(&gradient_linear.get_gradient_input_batch());

        let ffn_layer_gradient: Gradient = match ffn_layer.layers.get(0) {
            Some(LayerEnum::Dense(dense_layer)) => dense_layer.gradient.clone().unwrap().clone(),
            _ => Gradient::new_default(),
        };

        let (grouped_ffn_gradient_weights, analytical_gradient_ffn_bias) = (ffn_layer_gradient.get_gradient_weights(), ffn_layer_gradient.get_gradient_bias());

        // TEST GRADIENT OF INPUT
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let ffn_batch_output = ffn_layer.forward(&layer_input);

            layer_input.set_input_batch(ffn_batch_output.get_output_batch());
            let linear_output: LayerOutput = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_input_ffn: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);
        let analytical_grad_input_ffn: Vec<Vec<Vec<Complex<f64>>>> = gradient_ffn.get_gradient_input_batch();

        println!("\n\n analytical grad input ffn: {:?}", &analytical_grad_input_ffn);
        println!("\n\n numerical grad input ffn: {:?}", &numerical_grad_input_ffn);

        // Check if gradient batch dimensions match expected shapes
        println!(
            "\n analytical grad input dim: {:?}, {} {}",
            analytical_grad_input_ffn.len(),
            analytical_grad_input_ffn[0].len(),
            analytical_grad_input_ffn[0][0].len()
        );
        println!(
            "numerical grad input dim: {:?}, {} {}",
            numerical_grad_input_ffn.len(),
            numerical_grad_input_ffn[0].len(),
            numerical_grad_input_ffn[0][0].len()
        );

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

        // println!("\n\n  gradient input ffn numerical gradient: {:?}", &numerical_grad_input_ffn);
        // println!("\n\n  gradient input ffn analytical gradient: {:?}", &analytical_grad_input_ffn);

        test_gradient_batch_error(&analytical_grad_input_ffn, &numerical_grad_input_ffn, epsilon_test);

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

            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_weights_ffn: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_dense.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        // println!("\nanalytical grad weights: {:?}", grouped_ffn_gradient_weights);
        println!("\n analytical grad weights dim: {:?}", grouped_ffn_gradient_weights);
        println!("\n numerical grad weights dim: {:?}", numerical_grad_weights_ffn);

        let global_error = global_relative_error_2d_l2(&grouped_ffn_gradient_weights, &numerical_grad_weights_ffn);
        println!("\n\n global relative gradient error weights ffn: {:?}", &global_error);

        test_gradient_error_2d(&grouped_ffn_gradient_weights, &numerical_grad_weights_ffn, epsilon_test);

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
            layer_input.set_input_batch(linear_output.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &bias_dense, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad bias: {:?}", analytical_gradient_ffn_bias);
        println!("\nnumerical grad bias: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&analytical_gradient_ffn_bias, &numerical_grad_linear_bias, epsilon_test);
    }
}
