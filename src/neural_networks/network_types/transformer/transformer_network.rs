use std::time::Instant;

use num::Complex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    database::sled_db::SLED_DB_TRANSFORMER_V1,
    neural_networks::{
        network_components::{
            gradient_struct::Gradient,
            input::{extend_input_with_bos, DataTrait, Dataset},
            layer::LayerEnum,
            layer_input_struct::LayerInput,
        },
        network_types::{
            neural_network_generic::{get_from_db, reset_previous_gradient, save_to_sled, NeuralNetwork, OperationMode},
            transformer::transformer_builder::create_transformer,
        },
        utils::{
            array_splitting::sliding_window_chunks_matrix,
            matrix::check_nan_or_inf_3d,
            tokenizer::{detokenize, tokenize_batch},
        },
    },
    utils::{data_converter::convert_c_to_f64_3d, sampling_methods::{get_target_predictions, greedy_decoding}},
};

pub const MAX_CONTEXT_WINDOW_SIZE: usize = 512;
pub const CONTEXT_OVERLAPPING: usize = 450;

pub fn train(transformer_network: &mut NeuralNetwork, dataset: Dataset<String, String>, num_epochs: usize) {
    let mut total_loss: Complex<f64>;
    let loss_threshold: f64 = 0.04;
    let now = Instant::now();
    let mut previous_last_losses: Vec<f64> = Vec::new();
    let mut total_loss_exp_ma = 0.0;
    let alpha = 0.2;

    let mut epoch_processed = 0;

    'outer: for epoch in 0..num_epochs {
        total_loss = Complex::new(0.0, 0.0);
        for batch_dataset in dataset.split_into_batches(5) {
            let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

            let seconds_elapsed = now.elapsed();
            let input_batch_extended = batch_dataset.extend_input_with_target(input_batch, target_batch);
            let target_batch_extended = batch_dataset.extend_target(target_batch);

            // println!("extended input batch: {:?}", &input_batch_extended);

            let (_tokens, mut batch_ids) = tokenize_batch(&input_batch_extended, false).unwrap();
            let (_tokens, mut target_ids) = tokenize_batch(&target_batch_extended, false).unwrap();

            let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap();

            if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
                let (input_batch_ids, target_batch_ids) = sliding_window_chunks_matrix(&batch_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
                batch_ids = input_batch_ids;
                target_ids = target_batch_ids;
            }

            let (predicted_softmax_batch, padding_mask_batch) = predict(transformer_network, &batch_ids, epoch, false);
            let loss = cross_entropy_loss_batch(&predicted_softmax_batch, &target_ids, &padding_mask_batch);
            total_loss += loss;

            if epoch > 0 && epoch % 10 == 0 || loss.norm() <= loss_threshold {
                println!("Epoch: {:?}, Loss: {:?}", epoch, loss);
                let predicted_softmax_targets: Vec<Vec<Vec<f64>>> = get_target_predictions(&predicted_softmax_batch, &target_ids, &padding_mask_batch);
                let sampled_tokens = greedy_decoding(&predicted_softmax_targets);

                let predicted_token_batch: Vec<String> = sampled_tokens.par_iter().map(|token_indices| detokenize(token_indices, false).unwrap()).collect();
                println!("Top-p tokens dim: {:?}", sampled_tokens[0].len() * sampled_tokens.len());
                println!("predicted tokens: {:?}", predicted_token_batch);

                let seconds_elapsed_end = now.elapsed();
                let duration = seconds_elapsed_end - seconds_elapsed;
                let seconds = duration.as_secs_f64();
                println!("time elapsed for forward pass in seconds: {:?}", seconds);
            }

            backward(transformer_network, &target_ids, true);

            if epoch > 0 && epoch % 10 == 0 {
                let seconds_elapsed_end = now.elapsed();
                let duration = seconds_elapsed_end - seconds_elapsed;
                let seconds = duration.as_secs_f64();
                println!("time elapsed for forward and backward pass in seconds: {:?}", seconds);
            }

            transformer_network.update_step_lr_scheduler(epoch, 100, 0.9);
        }

        if epoch % 10 == 0 {
            save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
        }

        if total_loss_exp_ma == 0.0 && epoch == 0 {
            total_loss_exp_ma = total_loss.re;
        }

        total_loss_exp_ma = alpha * total_loss.re + (1.0 - alpha) * total_loss_exp_ma;

        if epoch % 5 == 0 || total_loss.norm() <= loss_threshold {
            println!("Epoch: {:?}, TOTAL LOSS: {:?}", epoch, total_loss);
            println!("Epoch: {:?}, EXPONENTIAL MOVING AVARAGE LOSS: {:?}", epoch, total_loss_exp_ma);
        }

        if previous_last_losses.len() <= 4 {
            previous_last_losses.push(total_loss.re);
        }
        let len = previous_last_losses.len();
        previous_last_losses[epoch % len] = total_loss.re;

        if previous_last_losses.len() >= 4 {
            let end_ind = epoch % previous_last_losses.len();

            // Only continue if we have enough range to compute a start index safely
            if end_ind >= 4 {
                let start_ind = end_ind - 4;
                let mut loss_increasing_count = 0;

                for i in start_ind..end_ind - 1 {
                    if previous_last_losses[i] < previous_last_losses[i + 1] {
                        loss_increasing_count += 1;
                    }
                }

                if loss_increasing_count > 2 && epoch_processed != epoch {
                    println!("loss is increasing too much, reducing learning rate");
                    transformer_network.decay_learning_rate(0.5); // e.g., reduce LR by half
                    reset_previous_gradient(transformer_network);

                    epoch_processed = epoch;
                }
            }
        }

        if total_loss.norm() <= loss_threshold {
            println!("loss is smaller than {loss_threshold} Break the training: {:?}", &total_loss);
            save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
            break 'outer;
        }
    }
}

pub fn predict_token_by_token(transformer_network: &mut NeuralNetwork, input_batch: &Vec<String>) -> (Vec<Vec<Vec<f64>>>, Vec<String>) {
    let mut all_predictions: Vec<Vec<Vec<f64>>> = Vec::new();
    let time_step: usize = 0;
    let mut _padding_mask_batch: Vec<Vec<u32>> = Vec::new();

    let mut current_input_batch: Vec<String> = extend_input_with_bos(input_batch);
    let mut count_tokens_prediction = 0;
    let now = Instant::now();

    print!("Antwort: ");
    // transformer.learning_rate = learning_rate;
    let last_layer_index = transformer_network.layers.len() - 1;

    let layer = &mut transformer_network.layers[last_layer_index];
    match layer {
        LayerEnum::Softmax(_softmax_layer) => _softmax_layer.operation_mode = OperationMode::PRODUCTION,
        _ => {}
    }

    let (_tokens, mut batch_ids) = tokenize_batch(&current_input_batch, false).unwrap();

    // Continue predicting until EOS token is predicted
    loop {
        // println!("current input batch: {:?}", &current_input_batch);

        let _seconds_elapsed = now.elapsed();

        let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap();

        // println!("input tokens length in inference: {} {}", batch_ids.len(), batch_ids[0].len());

        if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
            batch_ids = batch_ids
                .into_iter()
                .map(|mut seq| {
                    if seq.len() > MAX_CONTEXT_WINDOW_SIZE {
                        seq.split_off(seq.len() - MAX_CONTEXT_WINDOW_SIZE) // keeps only the last CONTEXT_WINDOW_SIZE
                    } else {
                        seq
                    }
                })
                .collect();
        }

        let (current_predictions, _padding_mask_batch) = predict(transformer_network, &batch_ids, time_step, true);

        // let seconds_elapsed_end = now.elapsed();
        // let duration = seconds_elapsed_end - _seconds_elapsed;
        // let seconds = duration.as_secs_f64();
        // println!("time elapsed in seconds: {:?}", seconds);

        // If no more predictions are available, break out
        if current_predictions.is_empty() {
            break;
        }

        // Store the last predicted token's softmax probabilities
        all_predictions.push(current_predictions[current_predictions.len() - 1].clone());

        let predicted_softmax_targets: Vec<Vec<Vec<f64>>> = vec![current_predictions[current_predictions.len() -1].clone()];

        let sampled_tokens = greedy_decoding(&predicted_softmax_targets);
        batch_ids[0].push(sampled_tokens[0][0].clone());

        let predicted_token_batch: Vec<String> = sampled_tokens.par_iter().map(|token_indices| detokenize(token_indices, false).unwrap()).collect();
        // Check if the last predicted token is the EOS token
        let predicted_token = predicted_token_batch[predicted_token_batch.len() - 1].clone();

        // If EOS token is predicted, break the loop
        if predicted_token == "<eos>" {
            println!("\n\n <eos> predicted. Breaking ....");
            break;
        }

        print!("{}", predicted_token);
        // Add the predicted token to the current input batch
        current_input_batch[0] = format!("{}{}", current_input_batch[0], predicted_token);

        count_tokens_prediction += 1;

        if count_tokens_prediction > 20 {
            break;
        }
    }

    (all_predictions, current_input_batch)
}

pub fn predict(transformer_network: &mut NeuralNetwork, batch_ids: &Vec<Vec<u32>>, time_step: usize, forward_only: bool) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<u32>>) {
    // Forward pass

    // println!("forward pass start ----------------------------------------------------------------------");
    let mut output = None;
    let mut output_softmax = None;
    let mut padding_mask = None;

    let mut layer_input = LayerInput::new_default();
    layer_input.set_forward_only(forward_only);
    layer_input.set_time_step(time_step);

    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::Embedding(embedding_layer) => {
                layer_input.set_batch_ids(batch_ids.clone());
                let (embeddings, padding_m) = embedding_layer.forward(&layer_input);
                // println!("padding mask batch in embedding layer: {:?}", &padding_m);

                output = Some(embeddings);
                padding_mask = Some(padding_m);

                // println!("padding mask: {:?}", &padding_mask);
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_output) = &output {
                    let positional_encoding_l = Some(positional_encoding_layer).unwrap();

                    //println!("forward pos encoding");
                    let positional_encodings: Vec<Vec<Vec<Complex<f64>>>> = positional_encoding_l.forward(&previous_output);
                    output = Some(positional_encodings);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::Norm(norm_layer) => {
                if let Some(previous_output) = &output {
                    layer_input.set_input_batch(previous_output.clone());

                    //println!("forward norm");
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

                    //println!("forward self-attention start");
                    let output_attention = attention.forward(&layer_input);
                    output = Some(output_attention.get_output_batch());
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::FeedForward(dense) => {
                let dense_layer = Some(dense).unwrap();
                if let Some(previous_output) = &output {
                    dense_layer.padding_mask_batch = padding_mask.clone();

                    //println!("forward ffn start");
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
                    //println!("forward linear start");
                    layer_input.set_input_batch(previous_output.clone());
                    let output_linear = linear_layer.forward(&layer_input);

                    output = Some(output_linear.get_output_batch());
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                let softmax_layer_clone = Some(softmax_layer).unwrap();
                if let Some(previous_output) = &output {
                    //println!("forward softmax start");
                    if !forward_only {
                        let softmax_result: Vec<Vec<Vec<f64>>> = softmax_layer_clone.forward(&previous_output, padding_mask.clone());
                        output_softmax = Some(softmax_result);
                    } else {
                        output_softmax = Some(convert_c_to_f64_3d(previous_output));
                    }
                    // println!("forward softmax end");
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            _ => {}
        }
    }

    //println!("forward pass end ----------------------------------------------------------------------");
    (output_softmax.unwrap(), padding_mask.unwrap())
}

pub fn backward(transformer_network: &mut NeuralNetwork, target_batch_ids: &Vec<Vec<u32>>, update_params: bool) -> Option<Gradient> {
    // Backward pass

    let mut gradient: Option<Gradient> = None;

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
                    let gradient_batch: Gradient = norm_layer.backward(&previous_gradient);
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
                    //println!("backward linear start");
                    let gradient_batch: Gradient = linear_layer.backward(&previous_gradient);
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

    gradient
}

pub fn predict_by_text(input: &Vec<String>) -> Vec<String> {
    let mut transformer = match get_from_db(SLED_DB_TRANSFORMER_V1) {
        Ok(transformer) => {
            // Successfully loaded transformer from the database
            println!("Loaded transformer from the database!");
            transformer
        }
        Err(e) => {
            println!("error: {:?}", e);
            // Create a new transformer since the database didn't have one
            let transformer: NeuralNetwork = create_transformer(OperationMode::TRAINING);
            println!("Created a new transformer for training.");
            transformer
        }
    };

    let (_predicted_softmax_targets, all_predicted_tokens) = predict_token_by_token(&mut transformer, &input);

    all_predicted_tokens
}

pub fn cross_entropy_loss_batch(
    predicted_softmax_batch: &Vec<Vec<Vec<f64>>>, // Complex-valued softmax output
    targets: &Vec<Vec<u32>>,
    padding_mask: &Vec<Vec<u32>>,
) -> Complex<f64> {
    let batch_len = predicted_softmax_batch.len() as f64;
    let mut total_loss: Complex<f64> = Complex::new(0.0, 0.0);

    // println!("softmax batch inside function cross entropy batch: {:?}", &predicted_softmax_batch);
    for (batch_ind, prediction) in predicted_softmax_batch.iter().enumerate() {
        total_loss += cross_entropy_loss(prediction, &targets[batch_ind], &padding_mask[batch_ind]);
    }

    total_loss / batch_len
}

fn cross_entropy_loss(predictions: &Vec<Vec<f64>>, target_tokens: &Vec<u32>, padding_mask: &Vec<u32>) -> f64 {
    let mut loss: f64 = 0.0;
    let target_len = target_tokens.len();
    let mut count = 0.0;

    let mut sequence_len_unpadded: usize = 0;
    for &padding in padding_mask {
        if padding != 0 {
            sequence_len_unpadded += 1;
        }
    }

    let seq_ind_start = sequence_len_unpadded - target_len;

    for (s, &target_idx) in target_tokens.iter().enumerate() {
        if target_idx == 1 {
            continue; // skip padding
        }

        let seq_ind = seq_ind_start + s;

        let prob = predictions[seq_ind][target_idx as usize];
        let re_loss = -(prob + 1e-15).ln();
        loss += re_loss;
        count += 1.0;
    }

    loss / count
}
