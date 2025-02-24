use num::Complex;

use crate::neural_networks::{
    network_components::{
        gradient_struct::Gradient, input::{DataTrait, Dataset}, layer::LayerEnum
    },
    network_types::neural_network_generic::NeuralNetwork,
    utils::tokenizer::{tokenize, tokenize_batch},
};

pub fn train(transformer_network: &mut NeuralNetwork, dataset: Dataset<String, String>, num_epochs: usize) {
    for epoch in 0..num_epochs {
        for batch_dataset in dataset.split_into_batches(2) {
            let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

            let input_batch_extended = batch_dataset.extend_input_with_target(input_batch, target_batch);

            //println!("input batch extended: {:?}", &input_batch_extended);
            let predicted_softmax_batch = predict(transformer_network, &input_batch_extended);
            let (_tokens, target_ids) = tokenize_batch(target_batch).unwrap();
            let loss = cross_entropy_loss_batch(&predicted_softmax_batch, &target_ids);

            println!("Epoch: {:?}, Loss: {:?}", epoch, loss);

            backward(transformer_network, &target_ids);
        }
    }
}

pub fn predict(transformer_network: &mut NeuralNetwork, input_batch: &Vec<String>) -> Vec<Vec<Vec<Complex<f64>>>> {
    // Forward pass
    let mut batch_ids: Vec<Vec<u32>> = vec![];
    for input in input_batch {
        let (_tokens, input_ids) = tokenize(input).unwrap();
        batch_ids.push(input_ids);
    }

    //println!("tokens: {:?}", &batch_ids);

    println!("forward pass start ----------------------------------------------------------------------");

    let mut output = None;
    let mut padding_mask = None;

    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::Embedding(embedding_layer_box) => {
                let embedding_l = Some(embedding_layer_box).unwrap();
                let (embeddings, padding_m) = embedding_l.forward(&batch_ids);

                println!("Output embedding layer: {:?}, {:?}, {:?}", &embeddings.len(), &embeddings[0].len(), &embeddings[0][0].len());
                // println!("output embedding: {:?}", &embeddings[0]);
                output = Some(embeddings);
                padding_mask = Some(padding_m);
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_output) = &output {
                    let positional_encoding_l = Some(positional_encoding_layer).unwrap();
                    //println!("previous output embedding layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let positional_encodings = positional_encoding_l.forward(&previous_output);

                    println!("Output Positional_encodings layer: {:?}, {:?}, {:?}", &positional_encodings.len(), &positional_encodings[0].len(), &positional_encodings[0][0].len());

                    output = Some(positional_encodings);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let (Some(previous_output), Some(padding_m)) = (&output, &padding_mask) {
                    //println!("Previous output attention layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let output_attention = attention.forward(&previous_output, padding_m);

                     println!("Output attention layer: {:?}, {:?}, {:?}", &output_attention.len(), &output_attention[0].len(), &output_attention[0][0].len());
                    // Store the output for the next layer
                    output = Some(output_attention);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::FeedForward(dense) => {
                let dense_layer = Some(dense).unwrap();
                if let Some(previous_output) = &output {
                    // println!("Previous output: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_dense = dense_layer.forward(&previous_output);

                    println!("Output Dense FFN: {:?}, {:?},  {:?}", &output_dense.len(), &output_dense[0].len(), &output_dense[0][0].len());

                    //println!("dense output: {:?}", &output_dense);
                    output = Some(output_dense);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                let linear_layer = Some(linear_layer).unwrap();
                if let Some(previous_output) = &output {
                    // println!("Previous output in linear layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_linear = linear_layer.forward(&previous_output);

                    println!("Output Linear layer: {:?}, {:?}, {:?}", &output_linear.len(), &output_linear[0].len(), &output_linear[0][0].len());

                    //println!("output_linear output: {:?}", &output_linear);
                    output = Some(output_linear);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                let softmax_layer_clone = Some(softmax_layer).unwrap();
                if let Some(previous_output) = &output {
                    // println!("Previous output in softmax layer: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    let output_softmax: Vec<Vec<Vec<Complex<f64>>>> = softmax_layer_clone.forward(&previous_output);

                    println!("Output Softmax: {:?}, {:?}, {}", &output_softmax.len(), &output_softmax[0].len(), &output_softmax[0][0].len());

                    output = Some(output_softmax);
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            _ => {}
        }
    }

    println!("forward pass end ----------------------------------------------------------------------");

    output.unwrap()
}

pub fn backward(transformer_network: &mut NeuralNetwork, target_batch_ids: &Vec<Vec<u32>>) {
    // Backward pass

    let mut gradient: Option<Gradient> = None;

    for layer in transformer_network.layers.iter_mut().rev() {
        match layer {
            LayerEnum::Embedding(embedding_layer) => {
                if let Some(previous_gradient) = gradient {
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>>  = previous_gradient.get_gradient_input_batch();
                    
                    let gradient_batch: Gradient = embedding_layer.backward(&previous_gradient_batch);

                    embedding_layer.update_parameters(&target_batch_ids, transformer_network.learning_rate);

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Token Embedding Layer");
                }
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_gradient) = gradient {
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>>  = previous_gradient.get_gradient_input_batch();
                    
                    let gradient_batch: Gradient = positional_encoding_layer.backward(&previous_gradient_batch);

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Positional Encoding Layer");
                }
            }
            LayerEnum::SelfAttention(attention_layer) => {
                if let Some(previous_gradient) = gradient {
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>>  = previous_gradient.get_gradient_input_batch();
                    
                    let gradient_batch: Gradient = attention_layer.backward(&previous_gradient_batch);
                    // Update weights and biases
                    attention_layer.update_parameters();

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Self Attention Layer");
                }
            }
            LayerEnum::FeedForward(dense_layer) => {
                if let Some(previous_gradient) = gradient {
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>>  = previous_gradient.get_gradient_input_batch();
                    
                    let gradient_batch: Gradient = dense_layer.backward(&previous_gradient_batch);
                    // Update weights and biases
                    dense_layer.update_parameters();

                    let weight_gradient_batch = gradient_batch.get_gradient_weight_batch();
                    let _bias_gradient_batch = gradient_batch.get_gradient_bias_batch();

                    println!("weight gradients batch of ffn layer: {}, {}, {}", &weight_gradient_batch.len(), &weight_gradient_batch[0].len(), &weight_gradient_batch[0][0].len());
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Dense Layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>>  = previous_gradient.get_gradient_input_batch();
                    
                    let gradient_batch: Gradient = linear_layer.backward(&previous_gradient_batch);
                    // Update weights and biases
                    linear_layer.update_parameters();

                    let weight_gradient_batch = gradient_batch.get_gradient_weight_batch();
                    let _bias_gradient_batch = gradient_batch.get_gradient_bias_batch();

                    println!("weight gradients batch of linear layer: {}, {}, {}", &weight_gradient_batch.len(), &weight_gradient_batch[0].len(), &weight_gradient_batch[0][0].len());
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                let gradient_batch: Gradient = softmax_layer.backward(target_batch_ids);

                let input_gradient_batch = gradient_batch.get_gradient_input_batch();

                println!("gradients of softmax layer: {}, {}", &input_gradient_batch.len(), &input_gradient_batch[0].len());

                gradient = Some(gradient_batch);
            }
            _ => {}
        }
    }
}

pub fn cross_entropy_loss_batch(
    predicted_softmax_batch: &Vec<Vec<Vec<Complex<f64>>>>, // Complex-valued softmax output
    targets: &Vec<Vec<u32>>,
) -> Complex<f64> {
    // let high_token_index_batch: Vec<Vec<u32>> = find_highest_index_batch(&predicted_softmax_batch).unwrap();

    // let mut predicted_token_batch: Vec<String> = vec![];
    // for high_token_index in high_token_index_batch.iter() {
    //     let predicted_token: String = detokenize(&high_token_index).unwrap();
    //     predicted_token_batch.push(predicted_token);
    // }
    // println!("predicted token: {:?}", predicted_token_batch);

    let batch_len = predicted_softmax_batch.len() as f64;
    let mut total_loss: Complex<f64> = Complex::new(0.0, 0.0);

    // println!("softmax batch inside function cross entropy batch: {:?}", &predicted_softmax_batch);
    for (batch_ind, prediction) in predicted_softmax_batch.iter().enumerate() {
        total_loss += cross_entropy_loss(prediction, &targets[batch_ind]);
    }

    -total_loss / batch_len
}

fn cross_entropy_loss(predictions: &Vec<Vec<Complex<f64>>>, targets: &Vec<u32>) -> Complex<f64> {
    let mut loss: Complex<f64> = Complex::new(0.0, 0.0);
    let seq_len = predictions.len();
    let target_len: usize = targets.len();

    for i in 0..targets.len() {
        let target_index = targets[i] as usize;
        let target_moved_index = seq_len - target_len + i;

        //print!("target moved index: {}, ", &target_moved_index);
        let predicted_prob: Complex<f64> = predictions[target_moved_index][target_index];
        //println!("predicted prob: {:?}", &predicted_prob);
        if predicted_prob.norm() > 0.0 {
            loss += predicted_prob.ln(); // Negative log of the magnitude
        } else {
            panic!("Predicted probability is zero or negative, which is invalid! {:?}", &predicted_prob);
        }
    }

    loss / target_len as f64
}
