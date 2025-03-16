#[cfg(test)]
pub mod test_ffn_layer {
    use crate::neural_networks::{
        network_components::{
            gradient_struct::Gradient,
            layer::{ActivationType, Layer, LayerEnum, LayerType},
            linear_layer::LinearLayer,
            softmax_output_layer::SoftmaxLayer,
        },
        network_types::{feedforward_layer::FeedForwardLayer, neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::derivative::{
            numerical_gradient_bias, numerical_gradient_bias_without_loss, numerical_gradient_input_batch_without_loss, numerical_gradient_weights, numerical_gradient_weights_multiple_layers_without_loss, numerical_gradient_weights_without_loss, test_gradient_batch_error, test_gradient_error_1d,
            test_gradient_error_2d,
        },
    };

    use num::Complex;

    #[test]
    fn test_ffn_dense_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 4; // Match the input dimension with your input batch
        let output_dim = 7; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon: f64 = 1e-3;

        let mut dense_layer: Layer = Layer::new(input_dim, output_dim, &learning_rate, &ActivationType::GELU, LayerType::DenseLayer);

        // Create a simple LinearLayer with the given input and output dimensions
        //let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0010, 0.20), Complex::new(0.0030, 0.50), Complex::new(0.60, 0.40)], vec![Complex::new(0.0010, 0.20), Complex::new(0.0030, 0.50), Complex::new(0.60, 0.40)]]];

        // Define a small input batch, [2][6][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(0.5, 1.0), Complex::new(0.8, 4.0), Complex::new(0.1, 0.0), Complex::new(0.3, 1.0)],
                vec![Complex::new(0.9, 0.4), Complex::new(-0.8, 5.0), Complex::new(0.1, 0.0), Complex::new(0.1, 2.0)],
                vec![Complex::new(1.2, 0.5), Complex::new(-0.3, 1.0), Complex::new(0.1, 0.0), Complex::new(0.5, 3.0)],
                vec![Complex::new(1.2, 0.6), Complex::new(-0.3, 2.0), Complex::new(0.1, 0.0), Complex::new(0.5, 4.0)],
                vec![Complex::new(1.2, 0.7), Complex::new(-0.3, 3.0), Complex::new(0.1, 0.0), Complex::new(0.5, 5.0)],
                vec![Complex::new(0.9, 0.8), Complex::new(-0.8, 4.0), Complex::new(0.1, 0.0), Complex::new(0.1, 6.0)],
            ],
            vec![
                vec![Complex::new(1.0, 0.1), Complex::new(2.0, 0.4), Complex::new(3.0, 3.0), Complex::new(0.4, 0.1)],
                vec![Complex::new(3.0, 0.2), Complex::new(3.0, 0.6), Complex::new(4.0, 2.5), Complex::new(0.5, 0.2)],
                vec![Complex::new(0.6, 0.3), Complex::new(0.8, 0.8), Complex::new(0.1, 6.3), Complex::new(0.6, 0.3)],
                vec![Complex::new(0.6, 0.4), Complex::new(0.8, 0.9), Complex::new(0.1, 6.4), Complex::new(0.6, 0.4)],
                vec![Complex::new(0.6, 0.5), Complex::new(0.8, 1.2), Complex::new(0.1, 6.6), Complex::new(0.6, 0.5)],
                vec![Complex::new(3.0, 0.6), Complex::new(3.0, 2.3), Complex::new(4.0, 1.7), Complex::new(0.5, 0.6)],
            ],
        ];

        let padding_mask_batch = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let dense_output_batch = dense_layer.forward(&input_batch, &padding_mask_batch);

        println!("\ninput batch :{:?}", &input_batch);
        println!("\ndense output_batch: {:?}", &dense_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); dense_output_batch[0][0].len()]; dense_output_batch[0].len()]; dense_output_batch.len()];

        let gradient = dense_layer.backward(&previous_gradient);
        let dense_analytical_gradient_weights_batch = gradient.get_gradient_weight_batch();
        let dense_analytical_input_gradient_batch = gradient.get_gradient_input_batch();
        let dense_analytical_gradient_bias_batch = gradient.get_gradient_bias();

        let weights = dense_layer.weights.clone();
        let bias = dense_layer.bias.clone();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            dense_layer.weights = weights.clone();
            let dense_batch_output = dense_layer.forward(input, &padding_mask_batch);

            dense_batch_output
        };

        let dense_numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_without_loss(&mut loss_fn, input_batch.clone(), &weights, epsilon);

        println!("\nnumerical weight gradient dense layer {:?}", dense_numerical_grad_batch);
        println!("\nanalytical weight gradient dense layer {:?}", dense_analytical_gradient_weights_batch);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&dense_numerical_grad_batch, &dense_analytical_gradient_weights_batch, epsilon);

        // Bias gradient check
        dense_layer.weights = weights.clone();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            dense_layer.bias = bias.clone();
            let dense_batch_output = dense_layer.forward(input, &padding_mask_batch);

            dense_batch_output
        };

        let dense_numerical_grad_bias_batch: Vec<Complex<f64>> = numerical_gradient_bias_without_loss(&mut loss_fn, input_batch.clone(), &bias.clone(), epsilon);

        println!("\nnumerical gradient dense layer {:?}", dense_numerical_grad_bias_batch);
        println!("\nanalytical gradient dense layer {:?}", dense_analytical_gradient_bias_batch);

        // For Gelu it can a little more deviation
        test_gradient_error_1d(&dense_numerical_grad_bias_batch, &dense_analytical_gradient_bias_batch, epsilon);

        // input gradient batch check
        dense_layer.bias = bias.clone();

        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Vec<Vec<Vec<Complex<f64>>>> { dense_layer.forward(input, &padding_mask_batch) };

        let numerical_grad_input_linear: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch_without_loss(&mut loss_fn, input_batch.clone(), epsilon);

        println!("\nnumerical gradient input batch linear: {:?}", &numerical_grad_input_linear);
        println!("\nanalytical gradient input batch linear: {:?}", &dense_analytical_input_gradient_batch);

        test_gradient_batch_error(&numerical_grad_input_linear, &dense_analytical_input_gradient_batch, epsilon);
    }

    #[test]
    fn test_ffn_network_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 6; // Match the input dimension with your input batch
        let output_dim = 8; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let _operation_mode = OperationMode::TRAINING;
        let epsilon: f64 = 1e-3;

        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(input_dim, output_dim, learning_rate);

        // Create a simple LinearLayer with the given input and output dimensions
        //let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.10, 0.20), Complex::new(0.30, 0.50), Complex::new(0.60, 0.40)]]];

        // Define a small input batch, [2][6][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(0.5, 1.0), Complex::new(0.8, 4.0), Complex::new(0.1, 0.0), Complex::new(0.3, 1.0)],
                vec![Complex::new(0.9, 0.4), Complex::new(-0.8, 5.0), Complex::new(0.1, 0.0), Complex::new(0.1, 2.0)],
                vec![Complex::new(1.2, 0.5), Complex::new(-0.3, 1.0), Complex::new(0.1, 0.0), Complex::new(0.5, 3.0)],
                vec![Complex::new(1.2, 0.6), Complex::new(-0.3, 2.0), Complex::new(0.1, 0.0), Complex::new(0.5, 4.0)],
                vec![Complex::new(1.2, 0.7), Complex::new(-0.3, 3.0), Complex::new(0.1, 0.0), Complex::new(0.5, 5.0)],
                vec![Complex::new(0.9, 0.8), Complex::new(-0.8, 4.0), Complex::new(0.1, 0.0), Complex::new(0.1, 6.0)],
            ],
            vec![
                vec![Complex::new(1.0, 0.1), Complex::new(2.0, 0.4), Complex::new(3.0, 3.0), Complex::new(0.4, 0.1)],
                vec![Complex::new(3.0, 0.2), Complex::new(3.0, 0.6), Complex::new(4.0, 2.5), Complex::new(0.5, 0.2)],
                vec![Complex::new(0.6, 0.3), Complex::new(0.8, 0.8), Complex::new(0.1, 6.3), Complex::new(0.6, 0.3)],
                vec![Complex::new(0.6, 0.4), Complex::new(0.8, 0.9), Complex::new(0.1, 6.4), Complex::new(0.6, 0.4)],
                vec![Complex::new(0.6, 0.5), Complex::new(0.8, 1.2), Complex::new(0.1, 6.6), Complex::new(0.6, 0.5)],
                vec![Complex::new(3.0, 0.6), Complex::new(3.0, 2.3), Complex::new(4.0, 1.7), Complex::new(0.5, 0.6)],
            ],
        ];

        let ffn_output_batch = ffn_layer.forward(&input_batch);

        println!("\ninput batch :{:?}", &input_batch);
        println!("\ndense output_batch: {:?}", &ffn_output_batch);

        let previous_gradient = vec![vec![vec![Complex::new(1.0, 0.0); ffn_output_batch[0][0].len()]; ffn_output_batch[0].len()]; ffn_output_batch.len()];

        let gradient = ffn_layer.backward(&previous_gradient);
        let dense_analytical_gradient_batch = gradient.get_gradient_weight_batch();

        // let weights_linear: Vec<Vec<Complex<f64>>> = match ffn_layer.layers.get(1) {
        //     Some(LayerEnum::Linear(linear_layer)) => linear_layer.weights.clone(),
        //     _ => vec![],
        // };

        let weights_dense: Vec<Vec<Complex<f64>>> = match ffn_layer.layers.get(0) {
            Some(LayerEnum::Dense(dense_layer)) => dense_layer.weights.clone(),
            _ => vec![],
        };

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Vec<Vec<Vec<Complex<f64>>>> {
            // if let Some(LayerEnum::Linear(linear_layer)) = ffn_layer.layers.get_mut(1) {
            //     linear_layer.weights = weights.clone();
            // } else {
            //     println!("Layer 2 does not exist!");
            // }
            if let Some(LayerEnum::Dense(dense_layer)) = ffn_layer.layers.get_mut(0) {
                dense_layer.weights = weights.clone();
            } else {
                println!("Layer 2 does not exist!");
            }

            let dense_batch_output = ffn_layer.forward(input);
            dense_batch_output
        };

        // println!("linear weights: {:?}", &weights_linear);
        let dense_numerical_grad_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_multiple_layers_without_loss(&mut loss_fn, input_batch.clone(), &weights_dense.clone(), ffn_output_batch.clone(), epsilon);

        println!("\nnumerical gradient ffn layer {:?}", dense_numerical_grad_batch);
        println!("\nanalytical gradient ffn layer {:?}", dense_analytical_gradient_batch);

        // For Gelu it can a little more deviation
        test_gradient_batch_error(&dense_numerical_grad_batch, &dense_analytical_gradient_batch, epsilon);
    }

    #[test]
    fn test_softmax_linear_with_loss_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let output_dim = 4; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(input_dim, output_dim, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.5, 0.0), Complex::new(0.8, 0.0), Complex::new(0.1, 0.0)]], vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]]];

        let target_token_id_batch = vec![vec![0], vec![1]];
        //let target_token_id_batch = vec![vec![0]];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let ffn_batch_output = ffn_layer.forward(&input_batch);
        let linear_batch_output = linear_layer.forward(&ffn_batch_output);
        let _softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let analytical_grad_batch_softmax: Vec<Vec<Vec<Complex<f64>>>> = gradient_softmax.get_gradient_input_batch();

        let gradient_linear: Gradient = linear_layer.backward(&analytical_grad_batch_softmax);
        let (_grouped_linear_gradient_weights, _analytical_gradient_bias_linear) = (gradient_linear.get_gradient_weights(), gradient_linear.get_gradient_bias());

        let weights_linear = linear_layer.weights.clone();
        let bias_linear = linear_layer.bias.clone();

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            linear_layer.weights = weights.clone();

            let ffn_batch_output = ffn_layer.forward(input);
            let linear_batch_output = linear_layer.forward(&ffn_batch_output);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-5;
        let numerical_grad_linear: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_linear, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad weights linear: {:?}", _grouped_linear_gradient_weights);
        println!("\nnumerical grad weights linear: {:?}", numerical_grad_linear);

        test_gradient_error_2d(&_grouped_linear_gradient_weights, &numerical_grad_linear, epsilon);
        linear_layer.weights = weights_linear.clone();

        // TEST BIAS
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>| -> Complex<f64> {
            linear_layer.bias = bias.clone();
            let ffn_batch_output = ffn_layer.forward(input);
            let linear_batch_output = linear_layer.forward(&ffn_batch_output);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &bias_linear, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad bias linear: {:?}", _analytical_gradient_bias_linear);
        println!("\nnumerical grad bias linear: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&_analytical_gradient_bias_linear, &numerical_grad_linear_bias, epsilon);
    }

    #[test]
    fn test_softmax_linear_ffn_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let _seq_len: usize = 1; // Update to match the input structure
        let input_dim = 4; // Match the input dimension with your input batch
        let output_dim = 8; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;
        let epsilon = 1e-4;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(input_dim, output_dim, learning_rate);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][1][3]
        //let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.5, 0.2), Complex::new(0.8, 0.3), Complex::new(0.1, 0.6)]], vec![vec![Complex::new(1.0, 0.9), Complex::new(2.0, 1.5), Complex::new(3.0, -0.4)]]];

        // Define a small input batch, [2][6][4]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(0.5, 1.0), Complex::new(0.8, 4.0), Complex::new(0.1, 0.0), Complex::new(0.3, 1.0)],
                vec![Complex::new(0.9, 0.4), Complex::new(-0.8, 5.0), Complex::new(0.1, 0.0), Complex::new(0.1, 2.0)],
                vec![Complex::new(1.2, 0.5), Complex::new(-0.3, 1.0), Complex::new(0.1, 0.0), Complex::new(0.5, 3.0)],
                vec![Complex::new(1.2, 0.6), Complex::new(-0.3, 2.0), Complex::new(0.1, 0.0), Complex::new(0.5, 4.0)],
                vec![Complex::new(1.2, 0.7), Complex::new(-0.3, 3.0), Complex::new(0.1, 0.0), Complex::new(0.5, 5.0)],
                vec![Complex::new(0.9, 0.8), Complex::new(-0.8, 4.0), Complex::new(0.1, 0.0), Complex::new(0.1, 6.0)],
            ],
            vec![
                vec![Complex::new(1.0, 0.1), Complex::new(2.0, 0.4), Complex::new(3.0, 3.0), Complex::new(0.4, 0.1)],
                vec![Complex::new(3.0, 0.2), Complex::new(3.0, 0.6), Complex::new(4.0, 2.5), Complex::new(0.5, 0.2)],
                vec![Complex::new(0.6, 0.3), Complex::new(0.8, 0.8), Complex::new(0.1, 6.3), Complex::new(0.6, 0.3)],
                vec![Complex::new(0.6, 0.4), Complex::new(0.8, 0.9), Complex::new(0.1, 6.4), Complex::new(0.6, 0.4)],
                vec![Complex::new(0.6, 0.5), Complex::new(0.8, 1.2), Complex::new(0.1, 6.6), Complex::new(0.6, 0.5)],
                vec![Complex::new(3.0, 0.6), Complex::new(3.0, 2.3), Complex::new(4.0, 1.7), Complex::new(0.5, 0.6)],
            ],
        ];

        println!("input batch dim: {}, {}, {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());

        let target_token_id_batch = vec![vec![0, 1, 1, 1, 1, 1], vec![1, 0, 1, 1, 1, 0]];
        //let target_token_id_batch = vec![vec![0]];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let ffn_batch_output = ffn_layer.forward(&input_batch);
        let linear_batch_output = linear_layer.forward(&ffn_batch_output);
        let _softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax.get_gradient_input_batch());
        let gradient_ffn: Gradient = ffn_layer.backward(&gradient_linear.get_gradient_input_batch());

        let (grouped_ffn_gradient_weights, analytical_gradient_ffn_bias) = (gradient_ffn.get_gradient_weights(), gradient_ffn.get_gradient_bias());

        let weights_dense = match ffn_layer.layers.get(0) {
            Some(LayerEnum::Dense(dense_layer)) => dense_layer.weights.clone(),
            _ => vec![],
        };

        println!("dense weights: {:?}", &weights_dense);

        let bias_dense = match ffn_layer.layers.get(0) {
            Some(LayerEnum::Dense(dense_layer)) => dense_layer.bias.clone(),
            _ => vec![],
        };

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            if let Some(LayerEnum::Dense(dense_layer)) = ffn_layer.layers.get_mut(0) {
                dense_layer.weights = weights.clone();
            } else {
                println!("Layer 2 does not exist!");
            }

            let ffn_batch_output = ffn_layer.forward(input);
            let linear_batch_output = linear_layer.forward(&ffn_batch_output);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_weights_ffn: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights_dense.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad weights: {:?}", grouped_ffn_gradient_weights);
        println!("\nnumerical grad weights: {:?}", numerical_grad_weights_ffn);

        test_gradient_error_2d(&grouped_ffn_gradient_weights, &numerical_grad_weights_ffn, 1e-2);

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

            let ffn_batch_output = ffn_layer.forward(input);
            let linear_batch_output = linear_layer.forward(&ffn_batch_output);
            let softmax_batch_output = softmax_layer.forward(&linear_batch_output, None);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let numerical_grad_linear_bias: Vec<Complex<f64>> = numerical_gradient_bias(&mut loss_fn, input_batch.clone(), &bias_dense, epsilon);

        // Check if gradient batch dimensions match expected shapes
        println!("\nanalytical grad bias: {:?}", analytical_gradient_ffn_bias);
        println!("\nnumerical grad bias: {:?}", numerical_grad_linear_bias);

        test_gradient_error_1d(&analytical_gradient_ffn_bias, &numerical_grad_linear_bias, 1e-2);
    }
}
