#[cfg(test)]
mod test_transformer {
    use num::Complex;

    use crate::neural_networks::{
        network_components::{embedding_layer::EmbeddingLayer, layer::LayerEnum, linear_layer::LinearLayer, positional_encoding_layer::PositionalEncodingLayer, softmax_output_layer::SoftmaxLayer},
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
        utils::{derivative::numerical_gradient_weights, tokenizer::tokenize_batch},
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
        let epsilon = 1e-3;

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

        let num_self_attention_layer: usize = 4;
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

        println!("\n\nnumerical gradient weight q attention layer {:?}", numerical_grad_weight_q_batch);
        println!("\n dim numerical gradient {:?}, {},", numerical_grad_weight_q_batch.len(), numerical_grad_weight_q_batch[0].len());

        println!("\n\nanalytical gradient weight q attention layer {:?}", analytical_gradient_weight_q_batch);
        println!("\n dim nanalytical gradient {:?}, {}", analytical_gradient_weight_q_batch.len(), analytical_gradient_weight_q_batch[0].len());

        // test_gradient_batch_error(&numerical_grad_weight_q_batch, &analytical_gradient_weights_q_batch, epsilon);
        // Weight Q ------------------------------------------------------------------------------------------- end
    }
}
