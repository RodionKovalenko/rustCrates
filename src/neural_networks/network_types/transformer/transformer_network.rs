use std::time::Instant;

use num::Complex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    database::sled_db::SLED_DB_TRANSFORMER_V1,
    neural_networks::{
        network_components::{
            gradient_struct::Gradient,
            input::{concat_batches, extend_input_with_bos, DataTrait, Dataset},
            layer::LayerEnum,
            layer_input_struct::LayerInput,
            layer_output_struct::LayerOutput,
        },
        network_types::{
            neural_network_generic::{get_from_db, print_networt_structure, reset_previous_gradient, save_to_sled, NeuralNetwork, OperationMode},
            transformer::transformer_builder::create_transformer,
        },
        utils::{
            array_splitting::sliding_window_chunks_matrix,
            matrix::check_nan_or_inf_3d,
            tokenizer::{detokenize, tokenize_batch},
        },
    },
    utils::{
        data_converter::convert_c_to_f64_3d,
        sampling_methods::{get_target_predictions, greedy_decoding},
    },
};

pub const MAX_CONTEXT_WINDOW_SIZE: usize = 512;
pub const CONTEXT_OVERLAPPING: usize = 450;

pub fn train(transformer_network: &mut NeuralNetwork, dataset: Dataset<String, String>, num_epochs: usize, batch_size: usize) {
    let mut total_loss: Complex<f64>;
    let loss_threshold: f64 = 0.004;
    let now = Instant::now();
    let mut previous_last_losses: Vec<f64> = Vec::new();
    let mut total_loss_exp_ma = 0.0;
    let alpha = 0.2;
    let mut layer_input = LayerInput::new_default();
    let mut epoch_processed = 0;

    'outer: for epoch in 0..num_epochs {
        total_loss = Complex::new(0.0, 0.0);
        for batch_dataset in dataset.split_into_batches(batch_size) {
            let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

            let seconds_elapsed = now.elapsed();
            let input_batch_extended = extend_input_with_bos(input_batch);
            let target_batch_extended = batch_dataset.extend_target(target_batch);

            let (_tokens, input_ids) = tokenize_batch(&input_batch_extended, false).unwrap();
            let (_tokens, mut target_ids) = tokenize_batch(&target_batch_extended, false).unwrap();

            let mut batch_ids = concat_batches(&input_ids, &target_ids);

            // println!("input_ids tokens: {:?}", &input_ids);
            // println!("target tokens: {:?}", &target_ids);
            // println!("extended input tokens: {:?}", &batch_ids);
            // let original_text_extended: Vec<String> = batch_ids.par_iter().map(|token_indices| detokenize(token_indices, false).unwrap()).collect();
            // println!("original text: {:?}", &original_text_extended);

            let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap();

            if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
                let (input_batch_ids, target_batch_ids) = sliding_window_chunks_matrix(&batch_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
                batch_ids = input_batch_ids;
                target_ids = target_batch_ids;
            }

            layer_input.set_batch_ids(batch_ids);
            layer_input.set_time_step(epoch);
            layer_input.set_forward_only(false);
            layer_input.set_calculate_gradient(true);
            layer_input.set_target_batch_ids(target_ids.clone());

            let network_output = predict(transformer_network, &layer_input);
            let (predicted_softmax_batch, padding_mask_batch) = (network_output.get_output_batch_f64(), network_output.get_padding_mask_batch());

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
    let mut current_input_batch: Vec<String> = extend_input_with_bos(input_batch);
    let mut count_tokens_prediction = 0;

    let now = Instant::now();
    let mut layer_input = LayerInput::new_default();
    layer_input.set_calculate_gradient(false);
    layer_input.set_forward_only(true);

    print!("Antwort: ");

    let last_layer_index = transformer_network.layers.len() - 1;
    let _seconds_elapsed = now.elapsed();

    let layer = &mut transformer_network.layers[last_layer_index];
    match layer {
        LayerEnum::Softmax(_softmax_layer) => _softmax_layer.operation_mode = OperationMode::PRODUCTION,
        _ => {}
    }

    let (_tokens, mut batch_ids) = match tokenize_batch(&current_input_batch, false) {
        Ok(res) => res,
        Err(e) => {
            println!("Error tokenizing batch: {:?}", e);
            return (all_predictions, current_input_batch);
        }
    };

    let mut time_step = 0;

    // Continue predicting until EOS token is predicted
    loop {
        if batch_ids.is_empty() {
            println!("batch_ids is empty. Breaking.");
            break;
        }

        let max_seq_len = batch_ids.iter().map(|v| v.len()).max().unwrap_or(0);

        if max_seq_len == 0 {
            println!("All sequences in batch_ids are empty. Breaking.");
            break;
        }

        if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
            batch_ids = batch_ids
                .into_iter()
                .map(|mut seq| {
                    if seq.len() > MAX_CONTEXT_WINDOW_SIZE {
                        seq.split_off(seq.len() - MAX_CONTEXT_WINDOW_SIZE) // keep last tokens
                    } else {
                        seq
                    }
                })
                .collect();
        }

        layer_input.set_batch_ids(batch_ids.clone());

        if time_step > 0 {
            // let last_tokens: Vec<Vec<u32>> = batch_ids.iter().map(|seq| vec![*seq.last().unwrap()]).collect();
            let last_n = 1;

            let last_tokens_batch: Vec<Vec<u32>> = batch_ids
                .iter()
                .map(|seq| {
                    let len = seq.len();
                    if len >= last_n {
                        seq[len - last_n..].to_vec() // take last n tokens
                    } else {
                        seq.to_vec() // fallback: return the whole sequence
                    }
                })
                .collect();

            // println!("batch ids {:?}", batch_ids);
            // println!("last token id {:?}", last_tokens_batch);
            layer_input.set_batch_ids(last_tokens_batch);
        }

        // println!("batch ids {:?}", batch_ids);

        layer_input.set_time_step(time_step);

        let network_output = predict(transformer_network, &layer_input);
        let current_predictions = network_output.get_output_batch_f64();

        if current_predictions.is_empty() || current_predictions[0].is_empty() {
            println!("Empty predictions. Breaking.");
            break;
        }

        // Store the last predicted token's softmax probabilities
        all_predictions.push(current_predictions[current_predictions.len() - 1].clone());

        let last_pred = current_predictions[0].last().unwrap();

        let predicted_softmax_targets: Vec<Vec<Vec<f64>>> = vec![vec![last_pred.clone()]];

        let sampled_tokens = greedy_decoding(&predicted_softmax_targets);

        if batch_ids.is_empty() || sampled_tokens.is_empty() || sampled_tokens[0].is_empty() {
            println!("Error: batch_ids or sampled_tokens is empty. Breaking.");
            break;
        }

        batch_ids[0].push(sampled_tokens[0][0].clone());

        let predicted_token_batch: Vec<String> = sampled_tokens
            .par_iter()
            .map(|token_indices| {
                detokenize(token_indices, false).unwrap_or_else(|e| {
                    println!("Error detokenizing: {:?}", e);
                    "<detokenize_error>".to_string()
                })
            })
            .collect();

        let predicted_token = predicted_token_batch.last().unwrap_or(&"<none>".to_string()).clone();

        if predicted_token == "<eos>" {
            println!("\n\n <eos> predicted. Breaking ....");
            break;
        }

        print!("{}", predicted_token);
        current_input_batch[0] = format!("{}{}", current_input_batch[0], predicted_token);

        count_tokens_prediction += 1;
        if count_tokens_prediction > 20 {
            println!("\nMax token prediction limit reached. Breaking.");
            break;
        }

        if time_step == 0 {
            time_step = batch_ids[0].len() - 1;
        } else {
            time_step += 1;
        }
    }

    let seconds_elapsed_end = now.elapsed();
    let duration = seconds_elapsed_end - _seconds_elapsed;
    let seconds = duration.as_secs_f64();
    println!("\ntime elapsed in seconds: {:?}", seconds);

    (all_predictions, current_input_batch)
}

pub fn predict(transformer_network: &mut NeuralNetwork, layer_input: &LayerInput) -> LayerOutput {
    // Forward pass

    // println!("forward pass start ----------------------------------------------------------------------");
    let mut output = None;
    let mut output_softmax = None;
    let mut padding_mask = None;

    let batch_ids = layer_input.get_batch_ids();
    let forward_only = layer_input.get_forward_only();

    let mut layer_input = layer_input.clone();

    if forward_only {
        layer_input.set_calculate_gradient(false);
    }

    let now = Instant::now();
    let _start = now.elapsed();

    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::Embedding(embedding_layer) => {
                layer_input.set_batch_ids(batch_ids.clone());

                //let seconds_elapsed = now.elapsed();
                let (embeddings, padding_m) = embedding_layer.forward(&layer_input);

                output = Some(embeddings);
                padding_mask = Some(padding_m.clone());
                layer_input.set_padding_mask_batch(padding_m);

                //println!("time elapsed in seconds in embedding: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                // println!("padding mask: {:?}", &padding_mask);
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_output) = &output {
                    let positional_encoding_l = Some(positional_encoding_layer).unwrap();

                    //println!("forward pos encoding");
                    //let seconds_elapsed = now.elapsed();
                    layer_input.set_input_batch(previous_output.clone());
                    let positional_encodings: Vec<Vec<Vec<Complex<f64>>>> = positional_encoding_l.forward(&layer_input);

                    //println!("time elapsed in seconds in positional encoding: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                    output = Some(positional_encodings);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::Norm(norm_layer) => {
                if let Some(previous_output) = &output {
                    layer_input.set_input_batch(previous_output.clone());

                    //println!("forward norm");
                    //let seconds_elapsed = now.elapsed();
                    let norm_output = norm_layer.forward(&layer_input);

                    //println!("time elapsed in seconds in norm layer: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
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
                    //let seconds_elapsed = now.elapsed();
                    let output_attention = attention.forward(&layer_input);

                    //println!("time elapsed in seconds in self attention layer: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                    output = Some(output_attention.get_output_batch());
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::FeedForward(dense_layer) => {
                if let Some(previous_output) = &output {
                    dense_layer.padding_mask_batch = padding_mask.clone();

                    //println!("forward ffn start");
                    layer_input.set_input_batch(previous_output.to_vec());

                    //let seconds_elapsed = now.elapsed();
                    let layer_output = dense_layer.forward(&layer_input);

                    //println!("time elapsed in seconds in ffn layer: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                    output = Some(layer_output.get_output_batch());

                    check_nan_or_inf_3d(&mut layer_output.get_output_batch(), "output ffn dense");
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                if let Some(previous_output) = &output {
                    //println!("forward linear start");
                    layer_input.set_input_batch(previous_output.clone());

                    //let seconds_elapsed = now.elapsed();
                    let output_linear = linear_layer.forward(&layer_input);

                    //println!("time elapsed in seconds in linear layer: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                    output = Some(output_linear.get_output_batch());
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                if let Some(previous_output) = &output {
                    //println!("forward linear start");
                    layer_input.set_input_batch(previous_output.clone());

                    //let seconds_elapsed = now.elapsed();
                    let output_linear = multi_linear_layer.forward(&layer_input);

                    //println!("time elapsed in seconds in MultiLinear layer: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                    output = Some(output_linear.get_output_batch());
                } else {
                    println!("No previous output for Multilinear layer");
                }
            }
            LayerEnum::Wavelet(wavelet_layer) => {
                if let Some(previous_output) = &output {
                    //println!("forward wavelet layer start");
                    layer_input.set_input_batch(previous_output.clone());

                    //let seconds_elapsed = now.elapsed();
                    let output_cwt = wavelet_layer.forward(&layer_input);

                    //println!("time elapsed in seconds in wavelet layer: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                    output = Some(output_cwt.get_output_batch());
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                if let Some(previous_output) = &output {
                    //println!("forward discrete wavelet layer start");
                    layer_input.set_input_batch(previous_output.clone());

                    //let seconds_elapsed = now.elapsed();
                    let output_dwt = wavelet_layer.forward(&layer_input);

                    padding_mask = Some(output_dwt.get_padding_mask_batch());
                    layer_input.set_padding_mask_batch(output_dwt.get_padding_mask_batch());

                    //println!("time elapsed in seconds in wavelet layer: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                    output = Some(output_dwt.get_output_batch());
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                if let Some(previous_output) = &output {
                    //println!("forward softmax start");

                    //let seconds_elapsed = now.elapsed();
                    if !forward_only {
                        let softmax_result: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&previous_output, padding_mask.clone());
                        output_softmax = Some(softmax_result);
                    } else {
                        output_softmax = Some(convert_c_to_f64_3d(previous_output));
                    }

                    //println!("time elapsed in seconds in softmax layer: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                    // println!("forward softmax end");
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            _ => {}
        }
    }

    let mut layer_output = LayerOutput::new_default();
    layer_output.set_output_batch_f64(output_softmax.unwrap());
    layer_output.set_padding_mask_batch(padding_mask.unwrap());

    if !forward_only {
        println!("time elapsed in forward pass in predict: {:?}", (now.elapsed() - _start).as_secs_f64());
    }

    //println!("forward pass end ----------------------------------------------------------------------");
    layer_output
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

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Dense Layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    //println!("backward linear start");
                    let gradient_batch: Gradient = linear_layer.backward(&previous_gradient);

                    if update_params {
                        linear_layer.update_parameters();
                    }
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    //println!("backward linear start");
                    let gradient_batch: Gradient = multi_linear_layer.backward(&previous_gradient);

                    if update_params {
                        multi_linear_layer.update_parameters();
                    }
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::Wavelet(wavelet_layer) => {
                if let Some(previous_gradient) = gradient {
                    //println!("backward linear start");
                    let gradient_batch: Gradient = wavelet_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                if let Some(previous_gradient) = gradient {
                    //println!("backward linear start");
                    let gradient_batch: Gradient = wavelet_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                //println!("backward softmax start");
                let gradient_batch: Gradient = softmax_layer.backward(target_batch_ids);
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

    print_networt_structure(&mut transformer);
    let (_predicted_softmax_targets, all_predicted_tokens) = predict_token_by_token(&mut transformer, &input);

    println!("prediction is: {:?}", all_predicted_tokens);

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

    let mut _sequence_len_unpadded: usize = 0;
    for padding in padding_mask.iter() {
        if *padding != 0 {
            _sequence_len_unpadded += 1;
        }
    }

    // let seq_ind_start = _sequence_len_unpadded - target_len - 1;
    // let end_ind = _sequence_len_unpadded - 1;
    let seq_ind_start = predictions.len() - target_len - 1;
    let end_ind = predictions.len() - 1;

    for (s, &target_idx) in target_tokens.iter().enumerate() {
        if target_idx == 1 {
            continue; // skip padding
        }

        let seq_ind = seq_ind_start + s;

        if seq_ind >= end_ind {
            break;
        }

        let prob = predictions[seq_ind][target_idx as usize];
        // for softmax
        let re_loss = -(prob + 1e-15).ln();
        //for log softmax
        //let re_loss = -prob;
        loss += re_loss;
        count += 1.0;
    }

    loss / count
}
