#[cfg(test)]
mod test_norm_layer {
    use crate::neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, norm_layer::NormalNormLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::{
            derivative::{global_relative_error_l2, numerical_gradient_input_batch, test_gradient_batch_error},
            random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
        },
    };

    use num::Complex;

    #[test]
    fn test_norm_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 2; // Update to match the input structure
        let _input_dim = 3; // Match the input dimension with your input batch
        let _output_dim = 3;
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-8;

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(_batch_size, _output_dim, _input_dim);
        let input_batch_before: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(_batch_size, _output_dim, _input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(_batch_size, _output_dim, 2);

        let mut norm_layer = NormalNormLayer::new(input_batch[0][0].len(), 1e-8, learning_rate);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, OperationMode::TRAINING);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch_before(input_batch_before.clone());
        layer_input.set_input_batch(input_batch.clone());

        let norm_output = norm_layer.forward(&layer_input);
        let norm_output_batch: Vec<Vec<Vec<Complex<f64>>>> = norm_output.get_output_batch();
        let _softmax_batch_output = softmax_layer.forward(&norm_output_batch, None);

        let gradient: Gradient = softmax_layer.backward(&target_token_id_batch);

        println!("input batch :{:?}", &input_batch);
        println!("\ninput batch before :{:?}", &input_batch_before);

        norm_layer.previous_gradient_input_batch = Some(vec![vec![vec![Complex::new(1.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()]);

        let gradient_norm = norm_layer.backward(&gradient);
        let analytical_gradient_input_norm = gradient_norm.get_gradient_input_batch();

        //TEST 1: gradient of input batch before
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            //layer_input.set_input_batch(input.clone());
            layer_input.set_input_batch_before(input.clone());
            let norm_output = norm_layer.forward(&layer_input);
            let norm_output_batch = norm_output.get_output_batch();

            let softmax_batch_output = softmax_layer.forward(&norm_output_batch, None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_input_norm: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch_before.clone(), epsilon);

        println!("\nnumerical gradient norm: {:?}", &numerical_grad_input_norm);
        println!("\nanalytical gradient norm: {:?}", &analytical_gradient_input_norm);

        // for b in 0..analytical_gradient_input_norm.len() {
        //     for s in 0..analytical_gradient_input_norm[b].len() {
        //         let analytical_row_sum: Complex<f64> = analytical_gradient_input_norm[b][s].iter().sum();
        //         let numerical_row_sum: Complex<f64> = numerical_grad_input_norm[b][s].iter().sum();

        //         println!("analytical row sum: {:?}", analytical_row_sum);
        //         println!("numerical row sum: {:?}", numerical_row_sum);
        //     }
        // }

        let global_error = global_relative_error_l2(&numerical_grad_input_norm, &analytical_gradient_input_norm);

        println!("\n\nglobal relative gradient error: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_input_norm, &analytical_gradient_input_norm, 1e-5);

        //TEST 2: input batch itself
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            layer_input.set_input_batch_before(input_batch_before.clone());
            let norm_output = norm_layer.forward(&layer_input);
            let norm_output_batch = norm_output.get_output_batch();

            let softmax_batch_output = softmax_layer.forward(&norm_output_batch, None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_input_norm: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient norm: {:?}", &numerical_grad_input_norm);
        println!("\nanalytical gradient norm: {:?}", &analytical_gradient_input_norm);

        let global_error = global_relative_error_l2(&numerical_grad_input_norm, &analytical_gradient_input_norm);

        println!("\n\nglobal relative gradient error: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_input_norm, &analytical_gradient_input_norm, 1e-5);
    }
}
