#[cfg(test)]
mod test_linear_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::{
            derivative::{global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_bias, numerical_gradient_input_batch, numerical_gradient_weights, test_gradient_batch_error, test_gradient_error_1d, test_gradient_error_2d},
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
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim, 2);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        //let target_token_id_batch = vec![vec![0]];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_output = linear_layer.forward(&layer_input);
        let _softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);
        let (grouped_linear_gradient, analytical_gradient_bias) = (gradient_linear.get_gradient_weights(), gradient_linear.get_gradient_bias());

        let linear_weights = linear_layer.weights.clone();

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();

            let linear_output = linear_layer.forward(&layer_input);
            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

            loss
        };

        let epsilon = 1e-7;
        let numerical_grad_linear: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &linear_weights.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad weights: {:?}", grouped_linear_gradient);
        println!("\nnumerical grad weights: {:?}", numerical_grad_linear);

        test_gradient_error_2d(&grouped_linear_gradient, &numerical_grad_linear, 1e-5);

        // TEST BIAS
        let linear_bias = linear_layer.bias.clone();
        linear_layer.weights = linear_weights;

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Complex<f64> {
            linear_layer.bias = bias.clone();

            let linear_output = linear_layer.forward(&layer_input);
            let softmax_batch_output = softmax_layer.forward(&linear_output.get_output_batch(), None);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

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
        let input_dim = 20;
        let output_dim = 20;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][6][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim, 2);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        let linear_output = linear_layer.forward(&layer_input);
        let _softmax_batch_output: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&linear_output.get_output_batch(), None);

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);

        let gradient_weights_batch: Vec<Vec<Complex<f64>>> = gradient_linear.get_gradient_weights();
        let gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = gradient_linear.get_gradient_input_batch();
        let weights: Vec<Vec<Complex<f64>>> = linear_layer.weights.clone();

        println!("softmax output dim: {} {} {}", _softmax_batch_output.len(), _softmax_batch_output[0].len(), _softmax_batch_output[0][0].len());
        println!("target_token_id_batch dim: {} {}", target_token_id_batch.len(), target_token_id_batch[0].len());

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();

            let linear_output = linear_layer.forward(&layer_input);
            let softmax_batch_output: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&linear_output.get_output_batch(), None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

            loss
        };

        let num_gradient_weight_batch: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights, 1e-5);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical gradient_weights_batch: {:?}", gradient_weights_batch);
        println!("\n analytical gradient_weights_batch dim: {} {}", gradient_weights_batch.len(), gradient_weights_batch[0].len());

        //println!("\n numerical grad: {:?}", num_gradient_weight_batch);
        println!("\n num_gradient_weight_batch gradient dim: {} {}", num_gradient_weight_batch.len(), num_gradient_weight_batch[0].len());

        let global_error = global_relative_error_2d_l2(&num_gradient_weight_batch, &gradient_weights_batch);

        println!("global relative gradient error gradient_weights_batch: {:?}", &global_error);

        test_gradient_error_2d(&num_gradient_weight_batch, &gradient_weights_batch, 1e-5);

        // TEST GRADIENT OF THE INPUT BATCH
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let linear_output = linear_layer.forward(&layer_input);
            let softmax_batch_output: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&linear_output.get_output_batch(), None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch, &padding_mask_batch);

            loss
        };

        let num_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical gradient_weights_batch: {:?}", gradient_weights_batch);
        println!("\n analytical gradient_weights_batch dim: {} {}", gradient_input_batch.len(), gradient_input_batch[0].len());

        //println!("\n numerical grad: {:?}", num_gradient_weight_batch);
        println!("\n numerical num_gradient_input_batch gradient dim: {} {}", num_gradient_input_batch.len(), num_gradient_input_batch[0].len());

        let global_error = global_relative_error_l2(&num_gradient_input_batch, &gradient_input_batch);

        println!("global relative gradient error gradient input batch: {:?}", &global_error);

        test_gradient_batch_error(&num_gradient_input_batch, &gradient_input_batch, 1e-3);
    }
}
