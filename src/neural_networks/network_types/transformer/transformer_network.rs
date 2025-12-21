use colored::*;
use num::Complex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::time::Instant;

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
            neural_network_generic::{get_from_db, print_networt_structure, save_to_sled, NeuralNetwork, OperationMode},
            transformer::{
                transformer_builder::create_transformer,
                transformer_updater::{update_transformer, VERBOSE},
            },
        },
        utils::{
            array_splitting::sliding_window_chunks_matrix,
            tokenizer::{detokenize, tokenize_batch},
        },
    },
    utils::{data_converter::convert_c_to_f64_3d, sampling_methods::greedy_decoding},
};

pub const MAX_CONTEXT_WINDOW_SIZE: usize = 50280;
pub const CONTEXT_OVERLAPPING: usize = 16;
pub const EMA_SCALER: f64 = 1.1;

pub fn train(transformer_network: &mut NeuralNetwork, dataset: Dataset<String, String>, num_epochs: usize, batch_size: usize) {
    let mut total_loss: Complex<f64>;
    let loss_threshold: f64 = 0.004;
    let now = Instant::now();
    let mut previous_last_losses: Vec<f64> = Vec::new();
    let mut total_loss_exp_ma = 0.0;
    let alpha = 0.2;
    let mut layer_input = LayerInput::new_default();
    let mut epoch_processed = 0;
    let mut timestep = 0;

    'outer: for epoch in 0..num_epochs {
        total_loss = Complex::new(0.0, 0.0);
        for (record_ind, batch_dataset) in dataset.split_into_batches(batch_size).iter().enumerate() {
            let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

            let seconds_elapsed = now.elapsed();
            let input_batch_extended = extend_input_with_bos(input_batch);
            let target_batch_extended = batch_dataset.extend_target(target_batch);

            let (_tokens, input_ids) = tokenize_batch(&input_batch_extended, false).unwrap();
            let (_tokens, target_ids) = tokenize_batch(&target_batch_extended, false).unwrap();

            let batch_ids: Vec<Vec<u32>> = concat_batches(&input_ids, &target_ids);
            // shift one position to the right in the array
            let mut target_ids: Vec<Vec<u32>> = batch_ids
                .iter()
                .map(|seq| {
                    if seq.is_empty() {
                        return vec![];
                    }

                    let mut shifted = Vec::with_capacity(seq.len());
                    shifted.extend_from_slice(&seq[1..seq.len()]);
                    shifted
                })
                .collect();

            let mut batch_ids: Vec<Vec<u32>> = batch_ids
                .iter()
                .map(|seq| {
                    if seq.is_empty() {
                        return vec![];
                    }

                    let mut shifted = Vec::with_capacity(seq.len());
                    shifted.extend_from_slice(&seq[..seq.len() - 1]);
                    shifted
                })
                .collect();

            // print!("\n batch ids: {:?}\n", &batch_ids);
            // print!("\n target ids: {:?}\n", &target_ids);

            let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap();

            if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
                let (input_batch_ids, target_batch_ids) = sliding_window_chunks_matrix(&batch_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
                batch_ids = input_batch_ids;
                target_ids = target_batch_ids;
            }

            layer_input.set_batch_ids(batch_ids.clone());
            layer_input.set_time_step(timestep + 1);
            layer_input.set_batch_size(batch_size);
            layer_input.set_forward_only(false);
            layer_input.set_calculate_gradient(true);
            layer_input.set_target_batch_ids(target_ids.clone());
            layer_input.set_record_index(record_ind);

            transformer_network.minibatch_size = batch_size;
            transformer_network.time_step = timestep + 1;

            if record_ind % batch_size == 0 {
                timestep += 1;
            }

            let network_output = predict(transformer_network, &layer_input);
            let (_predicted_softmax_batch, _padding_mask_batch) = (network_output.get_output_batch_f64(), network_output.get_padding_mask_batch());

            let loss: Complex<f64> = cross_entropy_sum_batch(&network_output.get_cross_entropy_loss_batch(), &target_ids);
            total_loss += loss;

            if epoch > 0 && epoch == (num_epochs - 1) || loss.norm() <= loss_threshold || epoch % 50 == 0 {
                println!("Epoch: {:?}, Loss: {:?}", epoch, loss);
                // let predicted_softmax_targets: Vec<Vec<Vec<f64>>> = get_target_predictions(&predicted_softmax_batch, &target_ids, &padding_mask_batch);
                // let sampled_tokens = greedy_decoding(&predicted_softmax_targets);

                // let predicted_token_batch: Vec<String> = sampled_tokens.par_iter().map(|token_indices| detokenize(token_indices, false).unwrap()).collect();
                // println!("Top-p tokens dim: {:?}", sampled_tokens[0].len() * sampled_tokens.len());
                // println!("predicted tokens: {:?}", predicted_token_batch);

                let seconds_elapsed_end = now.elapsed();
                let duration = seconds_elapsed_end - seconds_elapsed;
                let seconds = duration.as_secs_f64();
                println!("time elapsed for forward pass in seconds: {:?}", seconds);
            }

            backward(transformer_network, &target_ids, &layer_input, true);

            if epoch > 0 && epoch % 10 == 0 && record_ind == 0 || VERBOSE {
                let seconds_elapsed_end = now.elapsed();
                let duration = seconds_elapsed_end - seconds_elapsed;
                let seconds = duration.as_secs_f64();
                println!("batch size: {}", batch_ids.len());
                println!("TOTAL time elapsed for FORWARD AND BACKWARD pass in seconds: {}", seconds.to_string().green().bold());
            }

            transformer_network.update_step_lr_scheduler(epoch, 500, 0.9);
        }

        if epoch % 10 == 0 {
            save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
        }

        if total_loss_exp_ma == 0.0 && epoch == 0 {
            total_loss_exp_ma = total_loss.re;
        }

        total_loss_exp_ma = alpha * total_loss.re + (1.0 - alpha) * total_loss_exp_ma;

        if epoch % 5 == 0 || total_loss.norm() <= loss_threshold {
            println!("Epoch: {}, TOTAL LOSS: {}", epoch.to_string().blue().bold(), total_loss.re.to_string().red().bold());
            // println!("Epoch: {:?}, EXPONENTIAL MOVING AVARAGE LOSS: {:?}", epoch, total_loss_exp_ma);
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

                if loss_increasing_count > 5 && epoch_processed != epoch {
                    println!("loss is increasing too much, reducing learning rate");
                    transformer_network.decay_learning_rate(0.5); // e.g., reduce LR by half
                                                                  // reset_previous_gradient(transformer_network);

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
    layer_input.set_calculate_k_v_cache(true);

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

        if max_seq_len >= MAX_CONTEXT_WINDOW_SIZE {
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

        if time_step > 0 && layer_input.get_forward_only() {
            // let last_tokens: Vec<Vec<u32>> = batch_ids.iter().map(|seq| vec![*seq.last().unwrap()]).collect();
            let last_n = 4; // window_size * 2
            // let last_n = 1; // window_size * 2

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

            layer_input.set_batch_ids(last_tokens_batch);
        }

        layer_input.set_time_step(time_step);
        // layer_input.set_padding_mask_batch(vec![vec![1; batch_ids[0].len()]; batch_ids.len()]); // assuming all tokens are valid

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
        if count_tokens_prediction > 50 {
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
    let mut output: Option<Vec<Vec<Vec<Complex<f64>>>>> = None;
    let mut output_softmax = None;
    let mut padding_mask = None;

    let batch_ids = layer_input.get_batch_ids();
    let forward_only = layer_input.get_forward_only();

    let mut layer_input = layer_input.clone();
    let target_batch_ids_option = Some(layer_input.get_target_batch_ids());

    if forward_only {
        layer_input.set_calculate_gradient(false);
    }

    let now = Instant::now();

    let mut layer_output = LayerOutput::new_default();

    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(adaptive_avg_pooling_layer) => {
                if let Some(previous_output) = &output {
                    layer_input.set_input_batch(previous_output.clone());

                    //println!("forward norm");
                    //let start = Instant::now();
                    let layer_output: LayerOutput = adaptive_avg_pooling_layer.forward(&layer_input);
                    //println!("time elapsed in seconds in norm layer: {:?}",  start.elapsed().as_secs_f64());
                    output = Some(layer_output.get_output_batch());
                } else {
                    println!("No previous output for Attention layer");
                }
            }
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

                    if VERBOSE {
                        println!("forward pos encoding");
                    }
                    let start = Instant::now();
                    layer_input.set_input_batch(previous_output.clone());
                    let positional_encodings: Vec<Vec<Vec<Complex<f64>>>> = positional_encoding_l.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in positional encoding: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(positional_encodings);
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::Norm(norm_layer) => {
                if let Some(previous_output) = &output {
                    layer_input.set_input_batch(previous_output.clone());
                    layer_input.set_input_batch_before(previous_output.clone());

                    //println!("forward norm");
                    //let start = Instant::now();
                    let norm_output = norm_layer.forward(&layer_input);

                    //println!("time elapsed in seconds in norm layer: {:?}",  start.elapsed().as_secs_f64());
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

                    if VERBOSE {
                        println!("forward self-attention start");
                    }
                    let start = Instant::now();
                    let output_attention = attention.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in self attention layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(output_attention.get_output_batch());
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SparseSelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let (Some(previous_output), Some(padding_m)) = (&output, &padding_mask) {
                    layer_input.set_input_batch(previous_output.clone());
                    layer_input.set_padding_mask_batch(padding_m.clone());

                    if VERBOSE {
                        println!("forward sparse self-attention start");
                    }
                    let start = Instant::now();
                    let output_attention = attention.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in sparse self attention layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(output_attention.get_output_batch());
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttentionApproximation(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let (Some(previous_output), Some(padding_m)) = (&output, &padding_mask) {
                    layer_input.set_input_batch(previous_output.clone());
                    layer_input.set_padding_mask_batch(padding_m.clone());

                    if VERBOSE {
                        println!("forward self-attention approximation start");
                    }
                    let start = Instant::now();
                    let output_attention = attention.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in self attention approximation layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(output_attention.get_output_batch());
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::FeedForward(dense_layer) => {
                if let Some(previous_output) = &output {
                    dense_layer.padding_mask_batch = padding_mask.clone();

                    if VERBOSE {
                        println!("forward feed-forward network start");
                    }
                    layer_input.set_input_batch(previous_output.to_vec());

                    let start = Instant::now();
                    let layer_output = dense_layer.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in ffn layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(layer_output.get_output_batch());

                    //check_nan_or_inf_3d(&mut layer_output.get_output_batch(), "output ffn dense");
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                if let Some(previous_output) = &output {
                    if VERBOSE {
                        println!("forward linear start");
                    }
                    layer_input.set_input_batch(previous_output.clone());

                    let start = Instant::now();
                    let output_linear = linear_layer.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in linear layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(output_linear.get_output_batch());
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                if let Some(previous_output) = &output {
                    if VERBOSE {
                        println!("forward multilinear start");
                    }
                    layer_input.set_input_batch(previous_output.clone());

                    let start = Instant::now();
                    let output_linear = multi_linear_layer.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in multilayer layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(output_linear.get_output_batch());
                } else {
                    println!("No previous output for Multilinear layer");
                }
            }
            LayerEnum::Wavelet(wavelet_layer) => {
                if let Some(previous_output) = &output {
                    //println!("forward wavelet layer start");
                    layer_input.set_input_batch(previous_output.clone());

                    let start = Instant::now();
                    let output_cwt = wavelet_layer.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in wavelet layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(output_cwt.get_output_batch());
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                if let Some(previous_output) = &output {
                    //println!("forward discrete wavelet layer start");
                    layer_input.set_input_batch(previous_output.clone());

                    let start = Instant::now();
                    let output_dwt = wavelet_layer.forward(&layer_input);

                    padding_mask = Some(output_dwt.get_padding_mask_batch());
                    layer_input.set_padding_mask_batch(output_dwt.get_padding_mask_batch());

                    if VERBOSE {
                        println!("time elapsed in seconds in discrete wavelet layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = Some(output_dwt.get_output_batch());
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                if let Some(previous_output) = &output {
                    //println!("forward softmax start");
                    let start = Instant::now();

                    layer_input.set_input_batch(previous_output.clone());
                    if !forward_only {
                        let softmax_result: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&layer_input, padding_mask.clone(), target_batch_ids_option.clone());
                        output_softmax = Some(softmax_result);
                        layer_output.set_cross_entropy_loss_batch(softmax_layer.cross_entropy_loss_batch.clone().unwrap());
                    } else {
                        output_softmax = Some(convert_c_to_f64_3d(&previous_output));
                    }

                    if VERBOSE {
                        println!("time elapsed in seconds in softmax layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    // println!("forward softmax end");
                } else {
                    println!("No previous output for Dense layer");
                }
            }
            _ => {
                println!("Layer type not supported for backward pass");
            }
        }
    }

    layer_output.set_output_batch_f64(output_softmax.unwrap());
    layer_output.set_padding_mask_batch(padding_mask.unwrap());

    if VERBOSE {
        let whole_duration_forward = now.elapsed();
        println!("TOTAL time elapsed in seconds in forward pass: {}", whole_duration_forward.as_secs_f64().to_string().green().bold());
    }
    //println!("forward pass end ----------------------------------------------------------------------");
    layer_output
}

pub fn backward(transformer_network: &mut NeuralNetwork, target_batch_ids: &Vec<Vec<u32>>, layer_input: &LayerInput, update_params: bool) -> Option<Gradient> {
    // Backward pass

    let mut gradient: Option<Gradient> = None;
    let batch_size = transformer_network.get_minibatch_size();
    let record_ind = layer_input.get_record_index();
    let batch_ids = layer_input.get_batch_ids();
    let update_gradients: bool = (record_ind * batch_ids.len()) % batch_size == 0 && update_params;

    let start_time = Instant::now();

    for layer in transformer_network.layers.iter_mut().rev() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(adaptive_pooling) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward adaptive avg pool start");
                    let gradient_batch: Gradient = adaptive_pooling.backward(&previous_gradient);
                    gradient = Some(gradient_batch);
                    // println!("backward adaptive avg pool end");
                } else {
                    println!("No previous gradient for norm layer");
                }
            }
            LayerEnum::Embedding(embedding_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward embedding start");
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();

                    let start = Instant::now();
                    let gradient_batch: Gradient = embedding_layer.backward(&previous_gradient_batch);
                    if VERBOSE {
                        println!("time elapsed in seconds in embedding layer backward: {}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Token Embedding Layer");
                }
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward positional encoding start");
                    let start = Instant::now();
                    let gradient_batch: Gradient = positional_encoding_layer.backward(&previous_gradient.get_gradient_input_batch());

                    // println!("backward positional encoding end");
                    if VERBOSE {
                        println!("time elapsed in seconds in positional encoding layer backward: {}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                } else {
                    // println!("No previous gradient in Positional Encoding Layer");
                }
            }
            LayerEnum::Norm(norm_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward norm start");

                    let start = Instant::now();
                    let gradient_batch: Gradient = norm_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in norm layer backward: {}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient for norm layer");
                }
            }
            LayerEnum::SelfAttention(attention_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward attention layer start");
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();

                    let start = Instant::now();
                    let gradient_batch: Gradient = attention_layer.backward(&previous_gradient_batch);
                    if VERBOSE {
                        println!("time elapsed in seconds in self attention layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Self Attention Layer");
                }
            }
            LayerEnum::SparseSelfAttention(attention_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward attention layer start");
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();

                    let start = Instant::now();
                    let gradient_batch: Gradient = attention_layer.backward(&previous_gradient_batch);
                    if VERBOSE {
                        println!("time elapsed in seconds in sparse self attention layer backward: {:?}", start.elapsed().as_secs_f64());
                    }

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Self Attention Layer");
                }
            }
            LayerEnum::SelfAttentionApproximation(attention_layer) => {
                if let Some(previous_gradient) = gradient {
                    // println!("backward attention layer start");
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();

                    let start = Instant::now();
                    let gradient_batch: Gradient = attention_layer.backward(&previous_gradient_batch);
                    if VERBOSE {
                        println!("time elapsed in seconds in self attention approximation layer backward: {:?}", start.elapsed().as_secs_f64());
                    }

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Self Attention Layer");
                }
            }
            LayerEnum::FeedForward(dense_layer) => {
                if let Some(previous_gradient) = gradient {
                    let previous_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();
                    let start = Instant::now();
                    let gradient_batch: Gradient = dense_layer.backward(&previous_gradient_batch);
                    // println!("backward dense end");
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in ffn layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in Dense Layer");
                }
            }
            LayerEnum::Linear(linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = linear_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);
                    if VERBOSE {
                        println!("time elapsed in seconds in linear layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = multi_linear_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in multi linear layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::Wavelet(wavelet_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = wavelet_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in wavelet layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = wavelet_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in discrete wavelet layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                softmax_layer.batch_size = batch_size;
                // println!("backward softmax start");
                // let gradient_batch: Gradient = softmax_layer.backward(target_batch_ids);
                // gradient = Some(gradient_batch);

                gradient = Some(softmax_layer.gradient.as_ref().unwrap().clone());
                // println!("backward softmax end");
            }
            _ => {
                println!("Layer type not supported for backward pass");
            }
        }
    }

    if update_gradients {
        let start = Instant::now();
        update_transformer(transformer_network, &target_batch_ids);
        // reset_previous_gradient(transformer_network);
        if VERBOSE {
            println!("time elapsed in seconds in update transformer: {}", start.elapsed().as_secs_f64().to_string().green().bold());
        }
    }

    if VERBOSE {
        println!("TOTAL time elapsed in seconds in BACKWARD pass: {}", start_time.elapsed().as_secs_f64().to_string().green().bold());
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

pub fn cross_entropy_sum_batch(cross_entropy_loss_batch: &Vec<Vec<Vec<Complex<f64>>>>, _targets: &Vec<Vec<u32>>) -> Complex<f64> {
    let mut total_loss = Complex::new(0.0, 0.0);

    for batch in cross_entropy_loss_batch {
        for seq in batch {
            for token_loss in seq {
                if token_loss.is_nan() || token_loss.is_infinite() {
                    continue;
                }
                total_loss += *token_loss;
            }
        }
    }

    total_loss
}

// pub fn cross_entropy_loss_batch(
//     predicted_softmax_batch: &Vec<Vec<Vec<f64>>>, // Complex-valued softmax output
//     targets: &Vec<Vec<u32>>,
//     padding_mask: &Vec<Vec<u32>>,
//     batch_size: usize,
// ) -> Complex<f64> {
//     let mut total_loss: Complex<f64> = Complex::new(0.0, 0.0);

//     // println!("softmax batch inside function cross entropy batch: {:?}", &predicted_softmax_batch);
//     for (batch_ind, prediction) in predicted_softmax_batch.iter().enumerate() {
//         total_loss += cross_entropy_loss(prediction, &targets[batch_ind], &padding_mask[batch_ind]);
//     }

//     total_loss / batch_size as f64
// }

// fn cross_entropy_loss(predictions: &Vec<Vec<f64>>, target_tokens: &Vec<u32>, padding_mask: &Vec<u32>) -> f64 {
//     let mut loss: f64 = 0.0;
//     // let target_len = target_tokens.len();
//     let mut count = 0.0;

//     let mut _sequence_len_unpadded: usize = 0;
//     for padding in padding_mask.iter() {
//         if *padding != 0 {
//             _sequence_len_unpadded += 1;
//         }
//     }

//     let mut target_len_unpadded = 0.0;
//     for (_t, &target_class) in target_tokens.iter().enumerate() {
//         if target_class != 1 {
//             target_len_unpadded += 1.0;
//         }
//     }

//     let seq_ind_start = _sequence_len_unpadded - target_len_unpadded as usize;
//     let end_ind = _sequence_len_unpadded;
//     // let seq_ind_start = predictions.len() - target_len;
//     // let end_ind = predictions.len();

//     for (s, &target_idx) in target_tokens.iter().enumerate() {
//         if target_idx == 1 {
//             continue; // skip padding
//         }

//         let seq_ind = seq_ind_start + s;

//         if seq_ind >= end_ind {
//             break;
//         }

//         let prob = predictions[seq_ind][target_idx as usize];
//         // for softmax
//         let re_loss = -(prob + 1e-15).ln();
//         //for log softmax
//         //let re_loss = -prob;
//         loss += re_loss;
//         count += 1.0;
//     }

//     loss / count
// }
