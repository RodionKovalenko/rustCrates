#[cfg(test)]
mod test_rms_norm_layer {
    use crate::neural_networks::{
        network_components::{add_rms_norm_layer::RMSNormLayer, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
        network_types::neural_network_generic::OperationMode,
        utils::derivative::{global_relative_error_l2, numerical_gradient_input_batch_sum_without_loss, test_gradient_batch_error},
    };

    use crate::neural_networks::utils::matrix::RowMajorMatrix;

    use num::Complex;

    #[test]
    fn test_rms_norm_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-5;

        // Create a simple LinearLayer with the given input and output dimensions
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![
                Complex { re: 2.8783587878162713, im: 0.8745106130787896 },
                Complex { re: 2.634908282093512, im: -0.20236154707027212 },
                Complex { re: 0.6006278740248395, im: -1.1437859779264576 },
            ]],
            vec![vec![
                Complex { re: -0.34133995925522365, im: -0.21654721340666816 },
                Complex { re: 0.38135819845619007, im: -0.2937659597844681 },
                Complex { re: 0.06247390427697107, im: -0.01925084410848496 },
            ]],
        ];
        let input_batch_before: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![Complex { re: -0.1, im: 0.2 }, Complex { re: 0.3, im: 0.6 }, Complex { re: 1.0, im: 2.5 }]],
            vec![vec![Complex { re: -0.2, im: 4.0 }, Complex { re: 3.7, im: 4.8 }, Complex { re: 5.9, im: 5.0 }]],
        ];

        let mut rms_norm_layer = RMSNormLayer::new(input_dim, epsilon, learning_rate);

        let mut rms_input_layer = LayerInput::new_default();
        rms_input_layer.set_input_batch(input_batch.clone());
        rms_input_layer.set_input_batch_before(input_batch_before.clone());

        let rms_output = rms_norm_layer.forward(&rms_input_layer);
        let rms_output_batch = rms_output.get_output_batch();

        println!("input batch :{:?}", &input_batch);
        println!("\nrms output_batch: {:?}", &rms_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); rms_output_batch[0][0].len()]; rms_output_batch[0].len()]; rms_output_batch.len()];

        let gradient = rms_norm_layer.backward(&previous_gradient);
        let analytical_gradient_rms = gradient.get_gradient_input_batch();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            rms_input_layer.set_input_batch(input.clone());
            rms_input_layer.set_input_batch_before(input_batch_before.clone());
    
            let rms_output: LayerOutput = rms_norm_layer.forward(&rms_input_layer);

            rms_output.get_output_batch()
        };

        //let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_jacobi_without_loss(&mut loss_fn, input_batch.clone(), epsilon);
        let numerical_grad_rms: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_sum_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient rms: {:?}", &numerical_grad_rms);
        println!("\nanalytical gradient rms: {:?}", &analytical_gradient_rms);

        let global_error = global_relative_error_l2(&numerical_grad_rms, &analytical_gradient_rms);
        println!("global relative gradient error: {:?}", &global_error);

        test_gradient_batch_error(&numerical_grad_rms, &analytical_gradient_rms, epsilon);
    }

    #[test]
    fn test_rms_norm_forward_backward_rm_smoke() {
        let input_dim = 3;
        let learning_rate = 0.01;
        let epsilon = 1e-5;

        let input_batch_vec: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![Complex::new(1.0, 0.5), Complex::new(-2.0, 0.0), Complex::new(0.25, -0.1)]],
            vec![vec![Complex::new(0.1, -0.2), Complex::new(0.3, 0.6), Complex::new(-0.7, 0.0)]],
        ];
        let input_before_vec: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![Complex::new(0.1, 0.0), Complex::new(0.2, 0.0), Complex::new(0.3, 0.0)]],
            vec![vec![Complex::new(-0.1, 0.0), Complex::new(-0.2, 0.0), Complex::new(-0.3, 0.0)]],
        ];

        let input_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = input_batch_vec.iter().map(|m| RowMajorMatrix::from_rows(m)).collect();
        let input_before_rm: Vec<RowMajorMatrix<Complex<f64>>> = input_before_vec.iter().map(|m| RowMajorMatrix::from_rows(m)).collect();

        let mut rms_norm_layer = RMSNormLayer::new(input_dim, epsilon, learning_rate);
        let mut li = LayerInput::new_default();
        li.set_input_batch_rm(input_batch_rm);
        li.set_input_batch_before_rm(input_before_rm);

        let out = rms_norm_layer.forward(&li);
        assert!(out.get_output_batch_rm_ref().is_some(), "Expected RM output from RMSNorm forward");

        let out_rm = out.get_output_batch_rm();
        let prev_grad_rm: Vec<RowMajorMatrix<Complex<f64>>> = out_rm
            .iter()
            .map(|m| RowMajorMatrix::from_data(m.rows, m.cols, vec![Complex::new(1.0, 0.0); m.rows * m.cols]))
            .collect();

        let grad = rms_norm_layer.backward_rm(&prev_grad_rm);
        assert!(grad.get_gradient_input_batch_rm_ref().is_some(), "Expected RM gradient from RMSNorm backward_rm");
        assert!(
            grad.get_gradient_input_batch_ref().is_none(),
            "RM backward should not populate Vec gradient storage in this smoke test"
        );
    }
}
