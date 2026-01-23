#[cfg(test)]
mod test_transformer {
    use num::Complex;

    use crate::{
        neural_networks::{
            network_components::{gradient_struct::Gradient, input::concat_batches, layer_input_struct::LayerInput},
            network_layers::{
                embedding_layer::EmbeddingLayer,
                feedforward_layer::FeedForwardLayer,
                layer::LayerEnum,
                linear_layer::LinearLayer,
                norm_layer::NormalNormLayer,
                positional_encoding_layer::PositionalEncodingLayer,
                softmax_output_layer::SoftmaxLayer,
                wavelet_complex_layer::ComplexWaveletLayer,
                wavelet_discrete_layer::DiscreteWaveletLayer,
                wavelet_network::DECOMPOSITION_LEVELS,
            },
            network_types::{
                neural_network_generic::{create, NeuralNetwork, OperationMode},
                transformer::{
                    masked_attention_head::MaskedAttentionHead,
                    self_attention_layer::SelfAttentionLayer,
                    transformer_network::{backward, cross_entropy_sum_batch, predict},
                },
            },
            utils::{
                derivative::{global_relative_error_2d_l2, numerical_gradient_input, numerical_gradient_weights, test_gradient_error_2d},
                random_arrays::{generate_random_complex_3d, generate_random_u32_batch},
                tokenizer::tokenize_batch,
            },
        },
    };

    #[test]
    #[ignore]
    fn test_simple_training() {
        let batch_size = 1;
        let seq_len = 5;
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        let feature_dim = 64;
        let col_dim = 128;

        let hidden_dim = 256;

        let epsilon = 1e-8;
        let num_attention_heads: usize = 4;
        // Create a simple LinearLayer with the given input and output dimensions

        let mut norm_layer = NormalNormLayer::new(feature_dim, epsilon, learning_rate);
        let mut positional_encoding_layer = PositionalEncodingLayer::new(feature_dim);

        let mut attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, feature_dim, feature_dim, learning_rate);
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(feature_dim, hidden_dim, learning_rate);

        let compressed_hidden = 16;
        let mut linear_layer_1 = LinearLayer::new(learning_rate, feature_dim, compressed_hidden, true);
        let mut linear_layer_2: LinearLayer = LinearLayer::new(learning_rate, compressed_hidden, col_dim, true);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode, col_dim);

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, seq_len, feature_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, seq_len - 1, (seq_len - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();

        // Forward pass

        let num_epochs = 20000;

        for epoch in 0..num_epochs {
            layer_input.set_input_batch(input_batch.clone());
            layer_input.set_padding_mask_batch(padding_mask_batch.clone());

            let norm_output = norm_layer.forward(&layer_input);
            layer_input.set_input_batch(norm_output.get_output_batch());

            let positional_encoding_output = positional_encoding_layer.forward(&layer_input);
            layer_input.set_input_batch(positional_encoding_output.clone());

            let attention_output = attention_layer.forward(&layer_input);
            layer_input.set_input_batch(attention_output.get_output_batch());

            let ffn_output = ffn_layer.forward(&layer_input);
            layer_input.set_input_batch(ffn_output.get_output_batch());

            let linear_output_1 = linear_layer_1.forward(&layer_input);
            layer_input.set_input_batch(linear_output_1.get_output_batch());

            let linear_output = linear_layer_2.forward(&layer_input);
            layer_input.set_input_batch(linear_output.get_output_batch());

            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_id_batch.clone()));

            // Backward pass
            let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
            let gradient_linear_2: Gradient = linear_layer_2.backward(&gradient_softmax);
            let gradient_linear_1: Gradient = linear_layer_1.backward(&gradient_linear_2);
            let ffn_gradient: Gradient = ffn_layer.backward(&gradient_linear_1.get_gradient_input_batch());
            let gradient_attention_layer: Gradient = attention_layer.backward(&ffn_gradient.get_gradient_input_batch());
            let positional_encoding_gradient = positional_encoding_layer.backward(&gradient_attention_layer.get_gradient_input_batch());
            norm_layer.backward(&positional_encoding_gradient);

            let cross_entropy_loss_batch = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_id_batch);

            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:?}", epoch, loss);
            }

            if loss.re < 0.001 {
                println!("Early stopping at epoch {}: Loss = {:?}", epoch, loss);
                break;
            }

            // softmax_layer.time_step = epoch + 1;
            // linear_layer_1.time_step = epoch + 1;
            // linear_layer_2.time_step = epoch + 1;
            // attention_layer.time_step = epoch + 1;
            // ffn_layer.time_step = epoch + 1;
            // norm_layer.time_step = epoch + 1;

            layer_input.set_time_step(epoch + 1);

            // update parameters
            softmax_layer.update_parameters();
            linear_layer_1.update_parameters();
            linear_layer_2.update_parameters();
            attention_layer.update_parameters();
            ffn_layer.update_parameters();
            norm_layer.update_parameters();
        }
    }

    #[test]
    #[ignore]
    fn test_transformer_backward() {
        let number_inputs: usize = 32;
        let number_outputs = 32;
        let number_of_hidden_layers: usize = 1;
        let number_of_hidden_neurons: usize = 32;
        let minibatch_size: usize = 50;
        let learning_rate: f64 = 0.01;
        let epsilon = 1e-9;

        let mut transformer_network: NeuralNetwork = create(number_inputs, number_outputs, number_of_hidden_layers, number_of_hidden_neurons, minibatch_size, learning_rate);

        //Add layers to the network
        let mut layers = transformer_network.layers;

        let embedding_dim_original: usize = 512;
        let base_2: i32 = 2;
        // embedding_dim_compressed  = 16
        let embedding_dim_compressed = (embedding_dim_original as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;
        let vocab_size: usize = 50254;

        let embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim_original);
        let positional_encoding_layer = PositionalEncodingLayer::new(embedding_layer.embedding_dim);

        let rows: usize = 16;
        let linear_layer = LinearLayer::new(learning_rate, rows, vocab_size, true);
        let softmax_layer = SoftmaxLayer::new(learning_rate, OperationMode::TRAINING, vocab_size);

        layers.push(LayerEnum::Embedding(Box::new(embedding_layer)));
        layers.push(LayerEnum::PositionalEncoding(Box::new(positional_encoding_layer)));

        let rows: usize = 16;
        let hidden_dim = 64;
        let ffn_layer: FeedForwardLayer = FeedForwardLayer::new(rows, hidden_dim, learning_rate);

        let num_self_attention_layer: usize = 1;
        for _i in 0..num_self_attention_layer {
            let num_attention_heads: usize = 4;
            let rows: usize = 16;
            let cols: usize = embedding_dim_compressed;

            let attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, rows, cols, learning_rate);
            layers.push(LayerEnum::SelfAttention(Box::new(attention_layer)));
        }

        layers.push(LayerEnum::FeedForward(Box::new(ffn_layer)));
        layers.push(LayerEnum::Linear(Box::new(linear_layer)));
        layers.push(LayerEnum::Softmax(Box::new(softmax_layer)));

        transformer_network.layers = layers;

        let input_str1: &str = "Hallo, wie geht es dir?";
        let input_batch_str: Vec<String> = vec![input_str1.to_string()];

        let target_str1: &str = "Mir geht es gut";
        let target_batch_str: Vec<String> = vec![target_str1.to_string()];

        let (_tokens, batch_ids) = tokenize_batch(&input_batch_str, false).unwrap();

        let mut layer_input = LayerInput::new_default();
        layer_input.set_batch_ids(batch_ids.clone());

        let network_output = predict(&mut transformer_network, &layer_input);
        let output_batch = network_output.get_output_batch();
        let (_tokens, target_ids) = tokenize_batch(&target_batch_str, true).unwrap();
        backward(&mut transformer_network, &target_ids, false);

        println!("\n\n output batch {:?}", &output_batch[0][0][0..100]);
        println!("\n output_batch dim {:?}, {}, {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());

        // Extract SelfAttention layer first
        let first_self_attention = transformer_network
            .layers
            .iter()
            .find_map(|layer| {
                if let LayerEnum::SelfAttention(attention_layer_box) = layer {
                    Some(attention_layer_box) // Dereference Box
                } else {
                    None
                }
            })
            .expect("No SelfAttention layer found");

        // Extract first attention head
        let first_attention_head: MaskedAttentionHead = first_self_attention.attention_heads.iter().next().expect("No attention heads found").clone();
        let analytical_gradient_weight_q_batch = first_attention_head.gradient.as_ref().unwrap().get_gradient_weights_q();

        // Clone weights_q before the next mutable borrow
        let weights_q = first_attention_head.weights_q.clone();

        //println!("\n\n weights_q before {:?}", weights_q);

        // // Weight Q ------------------------------------------------------------------------------------------- start
        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            let first_self_attention = transformer_network
                .layers
                .iter_mut()
                .find_map(|layer| {
                    if let LayerEnum::SelfAttention(attention_layer_box) = layer {
                        Some(attention_layer_box) // Dereference Box
                    } else {
                        None
                    }
                })
                .expect("No SelfAttention layer found");

            let first_attention_head = first_self_attention.attention_heads.iter_mut().next().expect("No attention head found in loss fn");
            first_attention_head.weights_q = weights.clone();

            let network_output = predict(&mut transformer_network, &layer_input);
            let loss = cross_entropy_sum_batch(&network_output.get_cross_entropy_loss_batch(), &batch_ids);

            loss
        };

        let numerical_grad_weight_q_batch: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, output_batch.clone(), &weights_q.clone(), epsilon);

        let global_error = global_relative_error_2d_l2(&analytical_gradient_weight_q_batch, &numerical_grad_weight_q_batch);
        println!("\n\n global relative gradient error: {:?}", &global_error);

        println!("\n\nnumerical gradient weight q attention layer {:?}", numerical_grad_weight_q_batch);
        println!("\n dim numerical gradient {:?}, {}", numerical_grad_weight_q_batch.len(), numerical_grad_weight_q_batch[0].len());

        println!("\n\nanalytical gradient weight q attention layer {:?}", analytical_gradient_weight_q_batch);
        println!(
            "\n dim nanalytical gradient {:?}, {}",
            analytical_gradient_weight_q_batch.len(),
            analytical_gradient_weight_q_batch[0].len()
        );

        test_gradient_error_2d(&numerical_grad_weight_q_batch, &analytical_gradient_weight_q_batch, 1e-4);
        // Weight Q ------------------------------------------------------------------------------------------- end
    }

    #[test]
    #[ignore]
    fn test_transformer_two_self_attention_layers_norm_backward() {
        let learning_rate: f64 = 0.01;
        let epsilon = 1e-7;

        let embedding_dim_original: usize = 512;
        let base_2: i32 = 2;
        let embedding_dim_compressed = (embedding_dim_original as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;
        let vocab_size: usize = 50;
        let rows: usize = 16;

        let mut embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim_original);
        let mut positional_encoding_layer = PositionalEncodingLayer::new(embedding_layer.embedding_dim);
        let mut discrete_wavelet_layer = DiscreteWaveletLayer::new();
        let mut complex_wavelet_layer = ComplexWaveletLayer::new();
        let mut linear_layer = LinearLayer::new(learning_rate, rows, vocab_size, true);
        let mut softmax_layer = SoftmaxLayer::new(learning_rate, OperationMode::TRAINING, vocab_size);

        let rows: usize = 16;
        let hidden_dim = 16;
        let mut ffn_layer_1: FeedForwardLayer = FeedForwardLayer::new(rows, hidden_dim, learning_rate);
        let mut ffn_layer_2: FeedForwardLayer = FeedForwardLayer::new(rows, hidden_dim, learning_rate);

        let batch_size = 1;
        let num_attention_heads: usize = 4;
        let rows: usize = 16;
        let cols: usize = embedding_dim_compressed;
        let input_len = 15;
        let target_len = 6;

        let mut attention_layer_1: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, rows, cols, learning_rate);
        let mut attention_layer_2: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, rows, cols, learning_rate);

        let input_token_ids = generate_random_u32_batch(batch_size, input_len, input_len as u32);
        let target_token_ids = generate_random_u32_batch(batch_size, target_len, target_len as u32);

        let concat_batch_ids = concat_batches(&input_token_ids, &target_token_ids);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_batch_ids(concat_batch_ids.clone());
        layer_input.set_calculate_gradient(true);
        layer_input.set_forward_only(false);
        layer_input.set_target_batch_ids(target_token_ids.clone());

        // forward
        let (embeddings, padding_mask_batch) = embedding_layer.forward(&layer_input);
        layer_input.set_input_batch(embeddings.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let discrete_wavelet_output = discrete_wavelet_layer.forward(&layer_input);
        layer_input.set_input_batch(discrete_wavelet_output.get_output_batch());

        let complex_wavelet_output = complex_wavelet_layer.forward(&layer_input);
        layer_input.set_input_batch(complex_wavelet_output.get_output_batch());

        let positional_encoding_output = positional_encoding_layer.forward(&layer_input);
        layer_input.set_input_batch(positional_encoding_output.clone());

        let output_attention_1 = attention_layer_1.forward(&layer_input);
        layer_input.set_input_batch(output_attention_1.get_output_batch());

        let output_ffn = ffn_layer_1.forward(&layer_input);
        layer_input.set_input_batch(output_ffn.get_output_batch());

        let output_attention_2 = attention_layer_2.forward(&layer_input);
        layer_input.set_input_batch(output_attention_2.get_output_batch());

        let output_ffn_2 = ffn_layer_2.forward(&layer_input);
        layer_input.set_input_batch(output_ffn_2.get_output_batch());

        let output_linear = linear_layer.forward(&layer_input);

        layer_input.set_input_batch(output_linear.get_output_batch());
        let _output_softmax = softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_ids.clone()));

        // backward
        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_ids);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax);
        let ffn_gradient_2 = ffn_layer_2.backward(&gradient_linear.get_gradient_input_batch());
        let gradient_attention_layer_2: Gradient = attention_layer_2.backward(&ffn_gradient_2.get_gradient_input_batch());
        let gradient_ffn: Gradient = ffn_layer_1.backward(&gradient_attention_layer_2.get_gradient_input_batch());
        let gradient_attention_layer_1: Gradient = attention_layer_1.backward(&gradient_ffn.get_gradient_input_batch());
        let pos_enc_gradient = positional_encoding_layer.backward(&gradient_attention_layer_1.get_gradient_input_batch());
        let _complex_layer_gradient = complex_wavelet_layer.backward(&pos_enc_gradient);
        let _discrete_layer_gradient = discrete_wavelet_layer.backward(&_complex_layer_gradient);

        let analytical_norm_gradient: Vec<Vec<Complex<f64>>> = _discrete_layer_gradient.get_gradient_input();

        println!("padding mask batch in test transformer: {:?}", &padding_mask_batch);
        println!("target tokens ids: {:?}", &target_token_ids);
        println!("final output dim: {} {} {}", _output_softmax.len(), _output_softmax[0].len(), _output_softmax[0][0].len());

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            layer_input.set_calculate_gradient(false);

            let discrete_wavelet_output = discrete_wavelet_layer.forward(&layer_input);
            layer_input.set_input_batch(discrete_wavelet_output.get_output_batch());

            let complex_wavelet_output = complex_wavelet_layer.forward(&layer_input);
            layer_input.set_input_batch(complex_wavelet_output.get_output_batch());

            let positional_encoding_output = positional_encoding_layer.forward(&layer_input);
            layer_input.set_input_batch(positional_encoding_output.clone());

            let output_attention_1 = attention_layer_1.forward(&layer_input);
            layer_input.set_input_batch(output_attention_1.get_output_batch());

            let output_ffn = ffn_layer_1.forward(&layer_input);
            layer_input.set_input_batch(output_ffn.get_output_batch());

            let output_attention_2 = attention_layer_2.forward(&layer_input);
            layer_input.set_input_batch(output_attention_2.get_output_batch());

            let output_ffn_2 = ffn_layer_2.forward(&layer_input);
            layer_input.set_input_batch(output_ffn_2.get_output_batch());

            let output_linear = linear_layer.forward(&layer_input);

            layer_input.set_input_batch(output_linear.get_output_batch());
            softmax_layer.forward(&layer_input, Some(padding_mask_batch.clone()), Some(target_token_ids.clone()));

            let cross_entropy_loss_batch: &Vec<Vec<Vec<Complex<f64>>>> = softmax_layer.cross_entropy_loss_batch.as_ref().unwrap();
            let loss = cross_entropy_sum_batch(&cross_entropy_loss_batch, &target_token_ids);

            loss
        };

        let numerical_grad_input_batch: Vec<Vec<Complex<f64>>> = numerical_gradient_input(&mut loss_fn, embeddings.clone(), epsilon);

        let seq_len = numerical_grad_input_batch.len() - 1;
        let numerical_grad_input_batch = numerical_grad_input_batch[..seq_len.saturating_sub(target_token_ids[0].len())].to_vec();
        let seq_len = analytical_norm_gradient.len() - 1;
        let analytical_norm_gradient = analytical_norm_gradient[..seq_len.saturating_sub(target_token_ids[0].len())].to_vec();

        let global_error = global_relative_error_2d_l2(&numerical_grad_input_batch, &analytical_norm_gradient);
        println!("\n\n global relative gradient error: {:?}", &global_error);

        println!("\n numerical gradient input batch norm {:?}", numerical_grad_input_batch);
        println!("\n dim numerical gradient {:?}, {}", numerical_grad_input_batch.len(), numerical_grad_input_batch[0].len());

        println!("\n analytical gradient input batch norm {:?}", analytical_norm_gradient);
        println!("\n dim analytical gradient {:?}, {}", analytical_norm_gradient.len(), analytical_norm_gradient[0].len());

        for (i, numerical_grad_input) in numerical_grad_input_batch.iter().enumerate() {
            let row_sum_numeric: Complex<f64> = numerical_grad_input.iter().sum();
            let row_sum_analytic: Complex<f64> = analytical_norm_gradient[i].iter().sum();

            println!("row_sum numerical: {:?}, {:?}", row_sum_numeric.re, row_sum_numeric.im);
            println!("row sum analytical: {:?}, {:?}", row_sum_analytic.re, row_sum_analytic.im);
        }

        test_gradient_error_2d(&numerical_grad_input_batch, &analytical_norm_gradient, 1e-3);
    }
}
