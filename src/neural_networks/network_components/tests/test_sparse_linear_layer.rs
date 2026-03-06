#[cfg(test)]
mod test_sparse_linear_layer {

    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
        network_layers::{complex_to_linear_layer::ComplexToLinearLayer, softmax_output_layer::SoftmaxLayer, clustering_linear_layer::ClusteringLinearLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_sum_batch},
        utils::{
            derivative::{
                global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_bias_f32, numerical_gradient_input_batch, numerical_gradient_weights_f32, test_gradient_batch_error,
                test_gradient_error_1d, test_gradient_error_2d,
            },
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_sparse_linear_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let cols = 5; // Match the input dimension with your input batch
        let rows = 15; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        // Create a simple LinearLayer with the given input and output dimensions

        let embedding_dim = 16;
        let vocab_size = 32;
        let mut complex_to_linear_layer: ComplexToLinearLayer = ComplexToLinearLayer::new(cols, embedding_dim, learning_rate);
        let mut sparse_linear_layer: ClusteringLinearLayer = ClusteringLinearLayer::new(learning_rate, embedding_dim, vocab_size);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, cols);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, rows, cols);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, rows - 1, (rows - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        //let target_token_id_batch = vec![vec![0]];

        let mut layer_input = LayerInput::new_default();
        // Provide the metadata that SparseLinear + Softmax need for correct sparse CE behavior.
        // Without output indices, softmax backward can't map targets to sparse logits.
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());
        layer_input.set_target_batch_ids(target_token_id_batch.clone());
        // Avoid non-differentiability from top-k selection during finite-difference checks.
        layer_input.set_top_k_size(vocab_size);
        layer_input.set_input_batch(input_batch.clone());

        let complex_linear_output = complex_to_linear_layer.forward(&layer_input);
        layer_input.set_input_batch(complex_linear_output.get_output_batch());

        let linear_output = sparse_linear_layer.forward(&layer_input);
        layer_input.set_input_batch(linear_output.get_output_batch());
        layer_input.set_output_indices(linear_output.get_output_indices());

        softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_sparse_linear: Gradient = sparse_linear_layer.backward(&gradient_softmax);
        let _gradient_complex_linear: Gradient = complex_to_linear_layer.backward(&gradient_sparse_linear);

        let (grouped_linear_gradient, analytical_gradient_bias) = (gradient_sparse_linear.get_gradient_weights(), gradient_sparse_linear.get_gradient_bias());
        let gradient_input_batch = gradient_sparse_linear.get_gradient_input_batch();

        let linear_weights = sparse_linear_layer.weights.to_vec();

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<f32>>| -> Complex<f64> {
            *sparse_linear_layer.weights.write() = weights.clone();
            layer_input.set_input_batch(_input.clone());

            let complex_to_linear_output = complex_to_linear_layer.forward(&layer_input);
            layer_input.set_input_batch(complex_to_linear_output.get_output_batch());

            let linear_output = sparse_linear_layer.forward(&layer_input);
            layer_input.set_input_batch(linear_output.get_output_batch());
            layer_input.set_output_indices(linear_output.get_output_indices());

            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss: Complex<f64> = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let epsilon: f32 = 1e-4;
        let numerical_grad_linear: Vec<Vec<Complex<f64>>> = numerical_gradient_weights_f32(&mut loss_fn, input_batch.clone(), &linear_weights.clone(), epsilon);

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
        println!("\n\n global relative gradient error weights linear: {:?}", &global_error);
        test_gradient_error_2d(&grouped_linear_gradient, &numerical_grad_linear, 1e-3);

        // TEST BIAS
        let linear_bias = sparse_linear_layer.bias.clone();
        *sparse_linear_layer.weights.write() = linear_weights.clone();

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<f32>| -> Complex<f64> {
            sparse_linear_layer.bias = bias.clone();
            layer_input.set_input_batch(_input.clone());

            let complex_to_linear_output = complex_to_linear_layer.forward(&layer_input);
            layer_input.set_input_batch(complex_to_linear_output.get_output_batch());

            let linear_output = sparse_linear_layer.forward(&layer_input);
            layer_input.set_input_batch(linear_output.get_output_batch());
            layer_input.set_output_indices(linear_output.get_output_indices());

            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias_f32(&mut loss_fn, input_batch.clone(), &linear_bias, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\n analytical grad bias: {:?}", analytical_gradient_bias);
        println!("\n numerical grad bias: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&analytical_gradient_bias, &numerical_grad_linear_bias, 1e-3);

        // TEST GRADIENT OF THE INPUT BATCH
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());

            let linear_output = sparse_linear_layer.forward(&layer_input);
            layer_input.set_input_batch(linear_output.get_output_batch());
            layer_input.set_output_indices(linear_output.get_output_indices());

            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            loss
        };

        let epsilon: f64 = 1e-6;
        let num_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, complex_linear_output.get_output_batch().clone(), epsilon);

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

        test_gradient_batch_error(&num_gradient_input_batch, &gradient_input_batch, 1e-3);
    }
}
