use num::Complex;

use crate::neural_networks::{
    network_components::{input::Dataset, layer::LayerEnum},
    network_types::neural_network_generic::NeuralNetwork,
    utils::{
        matrix::find_highest_index,
        tokenizer::{detokenize, tokenize},
    },
};

pub fn train(transformer_network: NeuralNetwork, dataset: Dataset<String, String>, num_epochs: usize) {
    for epoch in 0..num_epochs {
        for (input, target) in dataset.iter() {
            //     // Compute loss
            //     let predictions = predict(&transformer_network, &input);
            //     let (_tokens, target_ids) = tokenize(target).unwrap();
            //     let loss = cross_entropy_loss(&predictions, &target_ids);

            //     println!("Epoch: {:?}, Loss: {:?}", epoch, loss);

            // // Backward pass
            // let mut gradients = None;
            // for layer in transformer_network.layers.iter().rev() {
            //     gradients = Some(layer.backward(gradients.unwrap_or_else(|| predictions.clone())));
            // }

            // Update weights (implemented inside the layer's `backward` method or externally)
        }
    }
}

pub fn predict(transformer_network: &NeuralNetwork, input_batch: Vec<&str>) -> Vec<Vec<Vec<Complex<f64>>>> {
    // Forward pass
    let mut batch_ids: Vec<Vec<u32>> = vec![];
    for input in input_batch {
        let (_tokens, input_ids) = tokenize(input).unwrap();
        batch_ids.push(input_ids);
    }

    let mut output = None;

    for layer in transformer_network.layers.iter() {
        match layer {
            LayerEnum::Embedding(embedding_layer_box) => {
                let embedding_l = Some(embedding_layer_box).unwrap();
                let embeddings: Vec<Vec<Vec<Complex<f64>>>> = embedding_l.forward(&batch_ids);

                //println!("output embedding layer: {:?}, {:?}", &embeddings.len(), &embeddings[0].len());
                //println!("output embedding: {:?}", &embeddings);
                output = Some(embeddings);
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_output) = &output {
                    let positional_encoding_l = Some(positional_encoding_layer).unwrap();
                    println!("previous output embedding layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let positional_encodings = positional_encoding_l.forward(&previous_output);

                    println!("positional_encodings layer: {:?}, {:?}", &positional_encodings.len(), &positional_encodings[0].len());

                    output = Some(positional_encodings);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(previous_output) = &output {
                    println!("Previous output attention layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let output_attention = attention.forward(&previous_output);

                    println!("output attention layer: {:?}, {:?}", &output_attention.len(), &output_attention[0].len());
                    // Store the output for the next layer
                    output = Some(output_attention);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::FeedForward(dense) => {
                let dense_layer = Some(dense).unwrap();
                if let Some(previous_output) = &output {
                    println!("Previous output: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_dense = dense_layer.forward(&previous_output);

                    println!("Dense output: {:?}, {:?}", &output_dense.len(), &output_dense[0].len());

                    //println!("dense output: {:?}", &output_dense);
                    output = Some(output_dense);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                let linear_layer = Some(linear_layer).unwrap();
                if let Some(previous_output) = &output {
                    println!("Previous output in linear layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_linear = linear_layer.forward(&previous_output);

                    println!("Linear output: {:?}, {:?}", &output_linear.len(), &output_linear[0].len());

                    //println!("output_linear output: {:?}", &output_linear);
                    output = Some(output_linear);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                let softmax_layer = Some(softmax_layer).unwrap();
                if let Some(previous_output) = &output {
                    println!("Previous output in softmax layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let output_softmax: Vec<Vec<Vec<Complex<f64>>>> = softmax_layer.forward(&previous_output);
                    let high_token_index_batch: Vec<u32> = find_highest_index(&output_softmax).unwrap();

                    let mut predicted_token_batch: Vec<String> = vec![];

                    for high_token_index in high_token_index_batch.iter() {
                        let predicted_token: String = detokenize(&vec![*high_token_index]).unwrap();
                        predicted_token_batch.push(predicted_token);
                    }

                    println!("Softmax output: {:?}, {:?}", &output_softmax.len(), &output_softmax[0].len());
                    println!("softmax: {:?}", &high_token_index_batch);
                    println!("predicted token: {:?}", predicted_token_batch);

                    output = Some(output_softmax);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            _ => {}
        }
    }

    output.unwrap()
}

fn cross_entropy_loss(
    predictions: &Vec<Vec<Complex<f64>>>, // Complex-valued softmax output
    targets: &Vec<u32>,                   // Ground truth token indices
) -> f64 {
    let mut loss: f64 = 0.0;
    let batch_size = predictions.len();

    for i in 0..batch_size {
        let target_index = targets[i] as usize;
        let predicted_prob = predictions[i][target_index].norm(); // Magnitude of the complex number
        if predicted_prob > 0.0 {
            loss += -predicted_prob.ln(); // Negative log of the magnitude
        } else {
            panic!("Predicted probability is zero or negative, which is invalid!");
        }
    }

    loss / batch_size as f64 // Average loss over the minibatch
}
