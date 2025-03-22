#[cfg(test)]
mod test_transformer {
    use num::Complex;

    use crate::neural_networks::{
        network_components::{embedding_layer::EmbeddingLayer, gradient_struct::Gradient, layer::LayerEnum, linear_layer::LinearLayer, positional_encoding_layer::PositionalEncodingLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{
            feedforward_layer::FeedForwardLayer,
            neural_network_generic::{create, NeuralNetwork, OperationMode},
            transformer::{
                masked_attention_head::MaskedAttentionHead,
                self_attention_layer::SelfAttentionLayer,
                transformer_network::{backward, cross_entropy_loss_batch, predict},
            },
            wavelet_network::DECOMPOSITION_LEVELS,
        },
        utils::{
            derivative::{global_relative_error_2d_l2, global_relative_error_l2, numerical_gradient_weights, numerical_gradient_weights_batch}, random_arrays::{generate_random_complex_3d, generate_random_u32_batch}, tokenizer::{tokenize, tokenize_batch}
        },
    };

    #[test]
    #[ignore]
    fn test_transformer_backward() {
        let number_inputs: usize = 32;
        let number_outputs = 32;
        let number_of_hidden_layers: usize = 1;
        let number_of_hidden_neurons: usize = 32;
        let minibatch_size: usize = 50;
        let learning_rate: f64 = 0.01;
        let epsilon = 1e-5;

        let mut transformer_network: NeuralNetwork = create(number_inputs, number_outputs, number_of_hidden_layers, number_of_hidden_neurons, minibatch_size, learning_rate);

        //Add layers to the network
        let mut layers = transformer_network.layers;

        let embedding_dim_original: usize = 512;
        let base_2: i32 = 2;
        // embedding_dim_compressed  = 16
        let embedding_dim_compressed = (embedding_dim_original as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;
        let vocab_size: usize = 50254;

        let embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim_original, false);
        let positional_encoding_layer = PositionalEncodingLayer::new(embedding_layer.embedding_dim);

        let rows: usize = 16;
        let linear_layer = LinearLayer::new(learning_rate, rows, vocab_size);
        let softmax_layer = SoftmaxLayer::new(learning_rate, OperationMode::TRAINING);

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

        let output_batch = predict(&mut transformer_network, &input_batch_str);
        let (_tokens, target_ids) = tokenize_batch(&target_batch_str).unwrap();
        backward(&mut transformer_network, &target_ids);

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

            //println!("\n\n weights_q in loss fn {:?}", weights);
            let softmax_batch_output = predict(&mut transformer_network, &input_batch_str);
            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_ids);

            loss
        };

        let numerical_grad_weight_q_batch: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, output_batch.clone(), &weights_q.clone(), epsilon);

        let global_error = global_relative_error_2d_l2(&analytical_gradient_weight_q_batch, &numerical_grad_weight_q_batch);
        println!("\n\n global relative gradient error: {:?}", &global_error);

        println!("\n\nnumerical gradient weight q attention layer {:?}", numerical_grad_weight_q_batch);
        println!("\n dim numerical gradient {:?}, {}", numerical_grad_weight_q_batch.len(), numerical_grad_weight_q_batch[0].len());

        println!("\n\nanalytical gradient weight q attention layer {:?}", analytical_gradient_weight_q_batch);
        println!("\n dim nanalytical gradient {:?}, {}", analytical_gradient_weight_q_batch.len(), analytical_gradient_weight_q_batch[0].len());

        // test_gradient_batch_error(&numerical_grad_weight_q_batch, &analytical_gradient_weights_q_batch, epsilon);
        // Weight Q ------------------------------------------------------------------------------------------- end
    }
    #[test]
    #[ignore]
    fn test_transformer_backward_separate() {
        let learning_rate: f64 = 0.01;
        let epsilon = 1e-3;
        let batch_size = 2;
        let output_dim = 16;
        let input_dim = 16;

        let embedding_dim_original: usize = 512;
        let base_2: i32 = 2;
        // embedding_dim_compressed  = 16
        let embedding_dim_compressed = (embedding_dim_original as i32 / base_2.pow(DECOMPOSITION_LEVELS)) as usize;
        let vocab_size: usize = 50254;

        let embedding_layer: EmbeddingLayer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim_original, false);
        //let positional_encoding_layer = PositionalEncodingLayer::new(embedding_layer.embedding_dim);

        let rows: usize = 16;
        let mut linear_layer = LinearLayer::new(learning_rate, rows, vocab_size);
        let mut softmax_layer = SoftmaxLayer::new(learning_rate, OperationMode::TRAINING);

        let rows: usize = 16;
        let hidden_dim = 16;
        let mut ffn_layer: FeedForwardLayer = FeedForwardLayer::new(rows, hidden_dim, learning_rate);

        let num_attention_heads: usize = 4;
        let rows: usize = 16;
        let cols: usize = embedding_dim_compressed;

        let mut attention_layer: SelfAttentionLayer = SelfAttentionLayer::new(num_attention_heads, rows, cols, learning_rate);

        // let input_str1: &str = "Hallo, wie geht es dir?";
        // let input_batch_str: Vec<String> = vec![input_str1.to_string()];

        // let target_str1: &str = "Mir geht es gut";
        // let target_batch_str: Vec<String> = vec![target_str1.to_string()];

        // let mut batch_token_ids: Vec<Vec<u32>> = vec![];
        // for input in &input_batch_str {
        //     let (_tokens, input_ids) = tokenize(input).unwrap();
        //     batch_token_ids.push(input_ids);
        // }

        // let (_tokens, target_token_ids) = tokenize_batch(&target_batch_str).unwrap();
      
          // Define a small input batch, [2][6][4]
          let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, output_dim, input_dim);
          let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, output_dim, 2);
          let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; output_dim]; batch_size];

        // forward
        let output_attention = attention_layer.forward(&input_batch, &padding_mask_batch);
        let output_ffn = ffn_layer.forward(&output_attention);
        let output_linear = linear_layer.forward(&output_ffn);
        let _output_softmax = softmax_layer.forward(&output_linear, Some(padding_mask_batch.clone()));

        // backward
        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_linear: Gradient = linear_layer.backward(&gradient_softmax.get_gradient_input_batch());
        let gradient_ffn: Gradient = ffn_layer.backward(&gradient_linear.get_gradient_input_batch());
        let _gradient_attention_layer: Gradient = attention_layer.backward(&gradient_ffn.get_gradient_input_batch());

        //let weights = attention_layer.attention_heads[0].weights_q.clone();
        //let weights = linear_layer.weights.clone();

        let weights: Vec<Vec<Complex<f64>>> = match ffn_layer.layers.get(0) {
            Some(LayerEnum::Dense(dense_layer)) => dense_layer.weights.clone(),
            _ => vec![],
        };

        //let gradient_batch_weight: Vec<Vec<Vec<Complex<f64>>>> = attention_layer.attention_heads[0].gradient.clone().unwrap().get_gradient_weights_q_batch();
        //let gradient_batch_weight: Vec<Vec<Vec<Complex<f64>>>> = linear_layer.gradient.clone().unwrap().get_gradient_weight_batch();
        let gradient_batch_weight: Vec<Vec<Complex<f64>>> = gradient_ffn.get_gradient_weights();

        println!("padding mask batch: {:?}", &padding_mask_batch);
        println!("target tokens ids: {:?}", &target_token_id_batch);

        println!("weights dim: {} {}", weights.len(), weights[0].len());

        // Define the loss function
        let mut loss_fn = |_input: &Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>| -> Complex<f64> {
            //attention_layer.attention_heads[0].weights_q = weights.clone();
            //linear_layer.weights = weights.clone();
            if let Some(LayerEnum::Dense(dense_layer)) = ffn_layer.layers.get_mut(0) {
                dense_layer.weights = weights.clone();
            } else {
                println!("Layer 2 does not exist!");
            }

            let output_attention = attention_layer.forward(&input_batch, &padding_mask_batch);
            let output_ffn = ffn_layer.forward(&output_attention);
            let output_linear = linear_layer.forward(&output_ffn);
            let output_softmax = softmax_layer.forward(&output_linear, Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&output_softmax, &target_token_id_batch);

            loss
        };

       // let numerical_grad_weight_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_weights_batch(&mut loss_fn, input_batch.clone(), &weight_q.clone(), epsilon);
        let numerical_grad_weight_batch: Vec<Vec<Complex<f64>>> = numerical_gradient_weights(&mut loss_fn, input_batch.clone(), &weights.clone(), epsilon);

        let global_error = global_relative_error_2d_l2(&numerical_grad_weight_batch, &gradient_batch_weight);
        println!("\n\n global relative gradient error: {:?}", &global_error);

        println!("\n\nnumerical gradient weight q attention layer {:?}", numerical_grad_weight_batch);
        println!("\n dim numerical gradient {:?}, {}", numerical_grad_weight_batch.len(), numerical_grad_weight_batch[0].len());

        println!("\n\nanalytical gradient weight q attention layer {:?}", gradient_batch_weight);
        println!("\n dim nanalytical gradient {:?}, {}", gradient_batch_weight.len(), gradient_batch_weight[0].len());

        // test_gradient_batch_error(&numerical_grad_weight_q_batch, &analytical_gradient_weights_q_batch, epsilon);
        // Weight Q ------------------------------------------------------------------------------------------- end
    }
}
