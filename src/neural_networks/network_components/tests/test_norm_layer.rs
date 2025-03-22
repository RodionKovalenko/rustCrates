#[cfg(test)]
mod test_rms_norm_layer {
    use crate::neural_networks::{
        network_components::add_and_norm_layer::NormalNormLayer,
        network_types::neural_network_generic::OperationMode,
        utils::derivative::{global_relative_error_l2, numerical_gradient_input_batch_without_loss, test_gradient_batch_error},
    };

    use num::Complex;

    #[test]
    fn test_norm_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-4;

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![
                Complex { re: 12.8783587878162713, im: 10.8745106130787896 },
                Complex { re: 32.634908282093512, im: -30.20236154707027212 },
                Complex { re: 20.6006278740248395, im: -21.1437859779264576 },
            ]],
            vec![vec![
                Complex { re: -20.34133995925522365, im: -10.21654721340666816 },
                Complex { re: 10.38135819845619007, im: -20.2937659597844681 },
                Complex { re: 50.06247390427697107, im: -30.01925084410848496 },
            ]],
        ];
        let input_batch_before: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![Complex { re: -0.1, im: 0.2 }, Complex { re: 0.3, im: 0.6 }, Complex { re: 1.0, im: 2.5 }]],
            vec![vec![Complex { re: -0.2, im: 4.0 }, Complex { re: 3.7, im: 4.8 }, Complex { re: 5.9, im: 5.0 }]],
        ];

        let mut norm_layer = NormalNormLayer::new(input_dim, epsilon, learning_rate);

        let rms_output_batch = norm_layer.forward(&input_batch, &input_batch_before);

        println!("input batch :{:?}", &input_batch);
        println!("\nrms output_batch: {:?}", &rms_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); rms_output_batch[0][0].len()]; rms_output_batch[0].len()]; rms_output_batch.len()];

        let gradient = norm_layer.backward(&previous_gradient);
        let analytical_gradient_norm = gradient.get_gradient_input_batch();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            let linear_batch_output = norm_layer.forward(input, &input_batch_before);

            linear_batch_output
        };

        //let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_jacobi_without_loss(&mut loss_fn, input_batch.clone(), epsilon);
        let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient norm: {:?}", &numerical_grad_rms);
        println!("\nanalytical gradient norm: {:?}", &analytical_gradient_norm);

        let global_error = global_relative_error_l2(&numerical_grad_rms, &analytical_gradient_norm);

        println!("global relative gradient error: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_rms, &analytical_gradient_norm, 1e-3);
    }
}
