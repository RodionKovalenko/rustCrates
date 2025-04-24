use std::cmp::Ordering;

use num::Complex;

use crate::{
    database::sled_db::SLED_DB_TRANSFORMER_V1,
    neural_networks::{
        network_components::{
            gradient_struct::Gradient,
            input::{DataTrait, Dataset},
            layer::LayerEnum,
            layer_input_struct::LayerInput,
        },
        network_types::neural_network_generic::{save_to_sled, NeuralNetwork},
        utils::{
            matrix::check_nan_or_inf_3d,
            tokenizer::{detokenize, tokenize, tokenize_batch},
        },
    },
    utils::sampling_methods::{greedy_decoding, top_p_temperature_sampling},
};

pub fn train(transformer_network: &mut NeuralNetwork, dataset: Dataset<String, String>, num_epochs: usize) {
    'outer: for epoch in 0..num_epochs {
        for batch_dataset in dataset.split_into_batches(8) {
            let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

            let input_batch_extended = batch_dataset.extend_input_with_target(input_batch, target_batch);

            // println!("input batch extended: {:?}", &input_batch_extended);
            let predicted_softmax_batch: Vec<Vec<Vec<Complex<f64>>>> = predict(transformer_network, &input_batch_extended, epoch);
            let (_tokens, target_ids) = tokenize_batch(target_batch, true).unwrap();
            let loss = cross_entropy_loss_batch(&predicted_softmax_batch, &target_ids);

            // let (_tokens, input_ids) = tokenize_batch(input_batch).unwrap();
            // println!("input Ids: {:?}", &input_ids);
            // println!("target ids: {:?},", &target_ids);

            //println!("predicted softmax batch: {}, {}, {}", predicted_softmax_batch.len(), predicted_softmax_batch[0].len(), predicted_softmax_batch[0][0].len());

            if epoch % 10 == 0 {
                save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
            }

            if epoch % 5 == 0 || loss.norm() <= 0.01 {
                println!("Epoch: {:?}, Loss: {:?}", epoch, loss);

                let target_len = target_ids[0].len();
                let p = 0.9; // Top-p (Nucleus) threshold
                let temperature = 0.7; // Temperature for controlling randomness
                let predicted_softmax_targets: Vec<Vec<Vec<Complex<f64>>>> = predicted_softmax_batch
                    .iter()
                    .enumerate()
                    .filter_map(|(_batch_ind, input_seq)| {
                        let mut valid_seq_opt = None;

                        // Slide backwards to find a valid window of length `target_len`
                        for offset in (0..=input_seq.len() - target_len).rev() {
                            let window = &input_seq[offset..offset + target_len];
                            let max = window.iter().flat_map(|w| w.iter()).max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Less));

                            if let Some(max) = max {
                                if max.norm() > 0.0 {
                                    valid_seq_opt = Some(window.to_vec());
                                    break;
                                }
                            }
                        }

                        valid_seq_opt
                    })
                    .collect();

                //println!("predicted softmax targets at 0: {:?}", &predicted_softmax_targets[0][0][0..100]);

                let sampled_tokens = top_p_temperature_sampling(&predicted_softmax_targets, p, temperature);
                println!("Top-p + Temperature Sampling: {:?}", sampled_tokens);
                
                let predicted_token_batch: Vec<String> = sampled_tokens.iter().map(|token_indices| detokenize(token_indices).unwrap()).collect();
                println!("predicted token tempareture sampling: {:?}", predicted_token_batch);

                let sampled_greedy_tokens = greedy_decoding(&predicted_softmax_targets);
                let predicted_token_batch: Vec<String> = sampled_greedy_tokens.iter().map(|token_indices| detokenize(token_indices).unwrap()).collect();
                println!("predicted token highest index: {:?}", predicted_token_batch);

                if loss.norm() <= 0.01 {
                    println!("loss is smaller than 0.05. Break the training: {:?}", &loss);
                    save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
                    break 'outer;
                }
            }

            backward(transformer_network, &target_ids, true);
        }
    }
}

pub fn predict(transformer_network: &mut NeuralNetwork, input_batch: &Vec<String>, time_step: usize) -> Vec<Vec<Vec<Complex<f64>>>> {
    // Forward pass
    let mut batch_ids: Vec<Vec<u32>> = vec![];
    for input in input_batch {
        let (_tokens, input_ids) = tokenize(input).unwrap();
        // println!("input: {:?}", &input);
        // println!("input Ids: {:?}", &input_ids);
        batch_ids.push(input_ids);
    }

    //println!("tokens: {:?}", &batch_ids);
    // println!("forward pass start ----------------------------------------------------------------------");

    let mut output = None;
    let mut padding_mask = None;

    let mut layer_input = LayerInput::new_default();
    layer_input.set_time_step(time_step);

    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::Embedding(embedding_layer) => {
                layer_input.set_batch_ids(batch_ids.clone());
                let (mut embeddings, padding_m) = embedding_layer.forward(&layer_input);

                check_nan_or_inf_3d(&mut embeddings, "embedding output in forward");
                output = Some(embeddings);
                padding_mask = Some(padding_m);

                // println!("padding mask: {:?}", &padding_mask);
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_output) = &output {
                    let positional_encoding_l = Some(positional_encoding_layer).unwrap();

                    // println!("forward pos encoding");
                    let mut positional_encodings: Vec<Vec<Vec<Complex<f64>>>> = positional_encoding_l.forward(&previous_output);

                    check_nan_or_inf_3d(&mut positional_encodings, "positional encodings output in forward");
                    output = Some(positional_encodings);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::Norm(norm_layer) => {
                if let Some(previous_output) = &output {
                    layer_input.set_input_batch(previous_output.clone());

                    // println!("forward norm");
                    let norm_output = norm_layer.forward(&layer_input);

                    output = Some(norm_output.get_output_batch());
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let (Some(previous_output), Some(padding_m)) = (&output, &padding_mask) {
                    layer_input.set_input_batch(previous_output.clone());
                    layer_input.set_padding_mask_batch(padding_m.clone());

                    // println!("forward self-attention");
                    let output_attention = attention.forward(&layer_input);

                    check_nan_or_inf_3d(&mut output_attention.get_output_batch(), "output attention layer in forward");

                    output = Some(output_attention.get_output_batch());
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::FeedForward(dense) => {
                let dense_layer = Some(dense).unwrap();
                if let Some(previous_output) = &output {
                    //println!("Previous output in dense ffn layer: {:?}, {:?}, {}", &previous_output.len(), &previous_output[0].len(), &previous_output[0][0].len());

                    dense_layer.padding_mask_batch = padding_mask.clone();
                    // println!("forward ffn");

                    layer_input.set_input_batch(previous_output.to_vec());
                    let layer_output = dense_layer.forward(&layer_input);

                    output = Some(layer_output.get_output_batch());

                    check_nan_or_inf_3d(&mut layer_output.get_output_batch(), "output ffn dense");
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                let linear_layer = Some(linear_layer).unwrap();
                if let Some(previous_output) = &output {
                    // println!("forward linear");
                    layer_input.set_input_batch(previous_output.clone());
                    let output_linear = linear_layer.forward(&layer_input);

                    check_nan_or_inf_3d(&mut output_linear.get_output_batch(), "output linear layer");

                    output = Some(output_linear.get_output_batch());
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                let softmax_layer_clone = Some(softmax_layer).unwrap();
                if let Some(previous_output) = &output {
                    let mut output_softmax: Vec<Vec<Vec<Complex<f64>>>> = softmax_layer_clone.forward(&previous_output, padding_mask.clone());

                    // println!("forward softmax start");
                    //println!("Output Softmax: {:?}, {:?}, {}", &output_softmax.len(), &output_softmax[0].len(), &output_softmax[0][0].len());
                    check_nan_or_inf_3d(&mut output_softmax, "output softmax");
                    output = Some(output_softmax);
                    // println!("forward softmax end");
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            _ => {}
        }
    }

    //println!("forward pass end ----------------------------------------------------------------------");

    output.unwrap()
}

pub fn backward(transformer_network: &mut NeuralNetwork, target_batch_ids: &Vec<Vec<u32>>, update_params: bool) -> Option<Gradient> {
    // Backward pass

    let mut gradient: Option<Gradient> = None;
    let mut epoch = 1;

    for layer in transformer_network.layers.iter_mut().rev() {
        match layer {
            LayerEnum::Embedding(embedding_layer) => {
                if let Some(previous_gradient) = gradient {
                    //println!("backward embedding start");
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();
                    let gradient_batch: Gradient = embedding_layer.backward(&previous_gradient_batch);

                    if update_params {
                        embedding_layer.update_parameters(&target_batch_ids, transformer_network.learning_rate);
                    }
                    //println!("backward embedding end");

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Token Embedding Layer");
                }
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward positional encoding start");
                    let gradient_batch: Gradient = positional_encoding_layer.backward(&previous_gradient.get_gradient_input_batch());

                    //println!("backward positional encoding end");
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Positional Encoding Layer");
                }
            }
            LayerEnum::Norm(norm_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward norm start");
                    let gradient_batch: Gradient = norm_layer.backward(&previous_gradient.get_gradient_input_batch());
                    gradient = Some(gradient_batch);
                    //  println!("backward norm end");
                } else {
                    println!("No previous gradient for norm layer");
                }
            }
            LayerEnum::SelfAttention(attention_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward attention layer start");
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();

                    let gradient_batch: Gradient = attention_layer.backward(&previous_gradient_batch);
                    // Update weights and biases
                    if update_params {
                        attention_layer.update_parameters();
                    }
                    // println!("backward attention layer end");

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Self Attention Layer");
                }
            }
            LayerEnum::FeedForward(dense_layer) => {
                epoch = dense_layer.time_step;
                if let Some(previous_gradient) = gradient {
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();
                    //println!("backward dense start");
                    let gradient_batch: Gradient = dense_layer.backward(&previous_gradient_batch);
                    // Update weights and biases
                    if update_params {
                        dense_layer.update_parameters();
                    }

                    // println!("backward dense end");
                    let _weight_gradient_batch = gradient_batch.get_gradient_weight_batch();
                    let _bias_gradient_batch = gradient_batch.get_gradient_bias_batch();

                    //println!("weight gradients batch of ffn layer: {}, {}, {}", &weight_gradient_batch.len(), &weight_gradient_batch[0].len(), &weight_gradient_batch[0][0].len());
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Dense Layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();

                    //println!("backward linear start");
                    let gradient_batch: Gradient = linear_layer.backward(&previous_gradient_batch);
                    // Update weights and biases
                    if update_params {
                        linear_layer.update_parameters();
                    }
                    //println!("backward linear end");

                    let _weight_gradient_batch = gradient_batch.get_gradient_weight_batch();
                    let _bias_gradient_batch = gradient_batch.get_gradient_bias_batch();

                    //println!("weight gradients batch of linear layer: {}, {}, {}", &weight_gradient_batch.len(), &weight_gradient_batch[0].len(), &weight_gradient_batch[0][0].len());
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                //println!("backward softmax start");
                let gradient_batch: Gradient = softmax_layer.backward(target_batch_ids);
                let _input_gradient_batch = gradient_batch.get_gradient_input_batch();

                // println!("gradients of softmax layer: {}, {}, {}", &input_gradient_batch.len(), &input_gradient_batch[0].len(),  &input_gradient_batch[0][0].len());
                //println!("backward softmax end");
                gradient = Some(gradient_batch);
            }
            _ => {}
        }
    }

    transformer_network.update_step_lr_scheduler(epoch, 50, 0.9);

    gradient
}

pub fn cross_entropy_loss_batch(
    predicted_softmax_batch: &Vec<Vec<Vec<Complex<f64>>>>, // Complex-valued softmax output
    targets: &Vec<Vec<u32>>,
) -> Complex<f64> {
    let batch_len = predicted_softmax_batch.len() as f64;
    let mut total_loss: Complex<f64> = Complex::new(0.0, 0.0);

    // println!("softmax batch inside function cross entropy batch: {:?}", &predicted_softmax_batch);
    for (batch_ind, prediction) in predicted_softmax_batch.iter().enumerate() {
        total_loss += cross_entropy_loss(prediction, &targets[batch_ind]);
    }

    total_loss / batch_len
}

fn cross_entropy_loss(predictions: &Vec<Vec<Complex<f64>>>, target_tokens: &Vec<u32>) -> Complex<f64> {
    let mut loss: Complex<f64> = Complex::new(0.0, 0.0);
    let seq_len = predictions.len();
    let target_len = target_tokens.len();
    let seq_ind_start = seq_len - target_len;
    let mut count = 0.0;

    for (s, &target_idx) in target_tokens.iter().enumerate() {
        if target_idx == 1 {
            continue; // skip padding
        }

        let seq_ind = seq_ind_start + s;
        loss += -(predictions[seq_ind][target_idx as usize] + 1e-15).ln();
        count +=1.0;
    }

    loss / count
}
