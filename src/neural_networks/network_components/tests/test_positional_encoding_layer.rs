#[cfg(test)]
mod test_positional_encoding_layer {
    use crate::neural_networks::{
        network_components::{layer_input_struct::LayerInput, positional_encoding_layer::PositionalEncodingLayer},
        utils::{
            derivative::{global_relative_error_l2, numerical_gradient_input_batch_sum_without_loss, test_gradient_batch_error},
            random_arrays::generate_random_complex_3d,
        },
    };

    use num::Complex;

    #[test]
    fn test_positional_encoding_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 8; // Match the input dimension with your input batch
        let output_dim = 6; // Match output_dim to your layer's output
        let epsilon = 1e-8;
        let epsilon_test = 1e-3;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut positonal_encoding_layer: PositionalEncodingLayer = PositionalEncodingLayer::new(input_dim);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
        let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(1.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        let _positonal_encoding_output = positonal_encoding_layer.forward(&layer_input);
        let positonal_encoding_gradient = positonal_encoding_layer.backward(&previous_gradient_batch);

        let (analytical_grad_batch, _analytical_grad) = (positonal_encoding_gradient.get_gradient_input_batch(), positonal_encoding_gradient.get_gradient_input());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> { 
            layer_input.set_input_batch(input.clone());
            positonal_encoding_layer.forward(&layer_input)
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
}
