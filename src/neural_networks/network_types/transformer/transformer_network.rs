use crate::neural_networks::network_layers::layer::LayerEnum;
use crate::neural_networks::utils::dtype::{r, C, Real, ZERO};
use crate::neural_networks::utils::matrix::RowMajorMatrix;
use colored::*;
use std::time::Instant;

use crate::{
    database::sled_db::SLED_DB_TRANSFORMER_V1,
    neural_networks::{
        network_components::{
            gradient_struct::Gradient,
            input::{concat_batches, extend_input_with_bos, DataTrait, Dataset},
            layer_input_struct::LayerInput,
            layer_output_struct::LayerOutput,
        },
        network_types::{
            neural_network_generic::{get_from_db, print_networt_structure, save_to_sled, NeuralNetwork, OperationMode},
            transformer::{
                transformer_builder::{create_transformer, SPARSE_WINDOW_SIZE},
                transformer_updater::{update_k_mean_clusters, update_transformer, VERBOSE},
            },
        },
        utils::{
            array_splitting::sliding_window_chunks_matrix,
            tokenizer::{detokenize, tokenize_batch},
        },
    },
};

pub const MAX_CONTEXT_WINDOW_SIZE: usize = 50280;
pub const CONTEXT_OVERLAPPING: usize = 16;
pub const EMA_SCALER: f64 = 1.1;
pub const TOP_K_SIZE: usize = 100;

pub fn train(transformer_network: &mut NeuralNetwork, mut dataset: Dataset<String, String>, num_epochs: usize, batch_size: usize) {
    // Setup data splits: 90% train, 10% validation (test set remains separate)
    dataset.setup_splits(None);

    let mut total_loss: C;
    let loss_threshold: Real = r(0.01);
    let now = Instant::now();
    let mut previous_last_losses: Vec<Real> = Vec::new();
    let mut total_loss_exp_ma: Real = ZERO;
    let alpha: Real = r(0.2);
    let mut layer_input = LayerInput::new_default();
    let mut epoch_processed = 0;
    let mut timestep = 1;
    let mut total_valid_target_tokens_epoch: usize;

    // Early stopping parameters
    let mut best_val_loss = Real::INFINITY;
    let mut best_epoch = 0;
    let patience = 100; // Stop if no improvement for 100 epochs
    let mut epochs_without_improvement = 0;

    'outer: for epoch in 0..num_epochs {
        total_loss = C::new(ZERO, ZERO);
        total_valid_target_tokens_epoch = 0;
        for (batch_ind, batch_dataset) in dataset.split_into_batches(batch_size).iter().enumerate() {
            let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

            let seconds_elapsed = now.elapsed();
            let input_batch_extended = extend_input_with_bos(input_batch);
            let target_batch_extended = batch_dataset.extend_target(target_batch);

            let (_tokens, input_ids) = tokenize_batch(&input_batch_extended, false).unwrap();
            let (_tokens, target_ids) = tokenize_batch(&target_batch_extended, false).unwrap();

            let valid_target_tokens_batch: usize = target_ids.iter().map(|seq| seq.iter().filter(|&&id| id != 1).count()).sum();
            total_valid_target_tokens_epoch += valid_target_tokens_batch;

            let batch_ids: Vec<Vec<u32>> = concat_batches(&input_ids, &target_ids);

            //NOTE: no-op for now.
            // Kept intentionally for future changes to target shifting logic
            let mut target_ids: Vec<Vec<u32>> = target_ids
                .iter()
                .map(|seq| {
                    if seq.is_empty() {
                        return vec![];
                    }

                    let mut shifted = Vec::with_capacity(seq.len());
                    shifted.extend_from_slice(&seq[0..seq.len()]);
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

            if VERBOSE {
                print!("\n batch ids: {:?}\n", &batch_ids);
                print!("\n target ids: {:?}\n", &target_ids);
            }

            let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap();
            let actual_batch_size = batch_ids.len();

            if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
                let (input_batch_ids, target_batch_ids) = sliding_window_chunks_matrix(&batch_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
                batch_ids = input_batch_ids;
                target_ids = target_batch_ids;
            }

            layer_input.set_batch_ids(batch_ids.clone());
            layer_input.set_rm_strict(true);
            layer_input.set_time_step(timestep);
            layer_input.set_batch_size(actual_batch_size);
            layer_input.set_forward_only(false);
            layer_input.set_calculate_gradient(true);
            layer_input.set_target_batch_ids(target_ids.clone());
            layer_input.set_top_k_size(TOP_K_SIZE);

            transformer_network.minibatch_size = actual_batch_size;
            transformer_network.time_step = timestep;

            let network_output = predict(transformer_network, &layer_input);
            let _padding_mask_batch = network_output.get_padding_mask_batch();

            // If there are no valid targets in this batch, skip loss/backward so we don't
            // early-stop on a meaningless 0.0 loss.
            if valid_target_tokens_batch == 0 {
                if epoch == 0 && batch_ind == 0 {
                    println!("WARNING: batch has 0 valid target tokens (all empty/padding). Skipping backward.");
                }
                timestep += 1;
                continue;
            }

            let ce_loss_batch = network_output.get_cross_entropy_loss_batch();
            if ce_loss_batch.is_empty() {
                panic!(
                    "Cross-entropy loss batch is empty in TRAINING. This usually means the Softmax layer ran in PRODUCTION mode or loss generation was skipped. valid_target_tokens_batch={}",
                    valid_target_tokens_batch
                );
            }

            let loss: C = cross_entropy_sum_batch(&ce_loss_batch, &target_ids);
            total_loss += loss;

            if epoch > 0 && epoch == (num_epochs - 1) || loss.norm() <= loss_threshold || epoch % 50 == 0 {
                println!("Epoch: {:?}, Loss: {:?}", epoch, loss);
                let seconds_elapsed_end = now.elapsed();
                let duration = seconds_elapsed_end - seconds_elapsed;
                let seconds = duration.as_secs_f64();
                println!("time elapsed for forward pass in seconds: {:?}", seconds);
            }

            backward(transformer_network, &target_ids, true);

            // time step is incremented after each batch update
            timestep += 1;

            if epoch > 0 && epoch % 10 == 0 && batch_ind == 0 || VERBOSE {
                let seconds_elapsed_end = now.elapsed();
                let duration = seconds_elapsed_end - seconds_elapsed;
                let seconds = duration.as_secs_f64();
                println!("batch size: {}", batch_ids.len());
                println!("total number of tokens: {}", batch_ids.iter().map(|seq| seq.len()).sum::<usize>());
                println!("TOTAL time elapsed for FORWARD AND BACKWARD pass in seconds: {}", seconds.to_string().green().bold());
            }

            transformer_network.update_step_lr_scheduler(epoch, 500, 0.9);
        }

        if epoch % 10 == 0 {
            save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
        }

        update_k_mean_clusters(transformer_network, epoch);

        if total_loss_exp_ma == ZERO && epoch == 0 {
            total_loss_exp_ma = total_loss.re;
        }

        total_loss_exp_ma = alpha * total_loss.re + (r(1.0) - alpha) * total_loss_exp_ma;

        if epoch % 5 == 0 || total_loss.norm() <= loss_threshold {
            println!("Epoch: {}, TRAINING LOSS: {}", epoch.to_string().blue().bold(), total_loss.re.to_string().red().bold());
        }

        if dataset.total_validation_records_size > 120 {
            // ========== VALIDATION PHASE (NO GRADIENT UPDATES) ==========
            let val_loss = evaluate_validation(transformer_network, &dataset, batch_size, &mut layer_input);

            if epoch % 5 == 0 {
                println!("Epoch: {}, VALIDATION LOSS: {}", epoch.to_string().blue().bold(), val_loss.to_string().yellow().bold());
            }

            // Early stopping check with improvement threshold
            let improvement_threshold: Real = r(1e-4); // Consider it an improvement if loss decreases by at least this amount

            if val_loss < best_val_loss - improvement_threshold {
                let improvement = best_val_loss - val_loss;
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_without_improvement = 0; // Reset patience counter
                                                // Save best model
                println!(
                    "✨ New best validation loss: {} at epoch {} (improved by {})",
                    best_val_loss.to_string().green().bold(),
                    epoch,
                    improvement.to_string().green()
                );
                save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
            } else {
                epochs_without_improvement += 1;

                if epoch % 5 == 0 || epochs_without_improvement >= patience - 2 {
                    println!("No improvement for {} epochs (best: {} at epoch {})", epochs_without_improvement, best_val_loss, best_epoch);
                }

                if epochs_without_improvement >= patience {
                    println!("⛔ Early stopping triggered! No improvement for {} epochs.", patience);
                    println!("Best validation loss: {} at epoch {}", best_val_loss, best_epoch);
                    save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
                    break 'outer;
                }
            }
        }
        // ============================================================

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

                if loss_increasing_count > 3 && epoch_processed != epoch {
                    println!("loss is increasing too much, reducing learning rate");
                    transformer_network.decay_learning_rate(0.5); // e.g., reduce LR by half
                                                                  // reset_previous_gradient(transformer_network);

                    epoch_processed = epoch;
                }
            }
        }

        if total_valid_target_tokens_epoch == 0 {
            println!(
                "WARNING: epoch {} had 0 valid target tokens across all batches; skipping early-stop on loss threshold.",
                epoch
            );
        } else if total_loss.norm() <= loss_threshold {
            println!("loss is smaller than {loss_threshold} Break the training: {:?}", &total_loss);
            save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
            break 'outer;
        }
    }
}

pub fn predict_token_by_token(transformer_network: &mut NeuralNetwork, input_batch: &Vec<String>) -> Vec<String> {
    let mut current_input_batch: Vec<String> = extend_input_with_bos(input_batch);
    let mut count_tokens_prediction = 0;

    let now = Instant::now();
    let mut layer_input = LayerInput::new_default();
    layer_input.set_calculate_gradient(false);
    layer_input.set_forward_only(true);
    layer_input.set_calculate_k_v_cache(true);
    layer_input.set_top_k_size(TOP_K_SIZE); // Set k for sparse linear layer

    print!("Antwort: ");

    let _seconds_elapsed = now.elapsed();

    let (_tokens, mut batch_ids) = match tokenize_batch(&current_input_batch, false) {
        Ok(res) => res,
        Err(e) => {
            println!("Error tokenizing batch: {:?}", e);
            return current_input_batch;
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
            // window_size * 2
            let last_n = SPARSE_WINDOW_SIZE * 2;
            // let last_n = 1;

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
        let current_predictions = network_output.get_output_batch_real();
        let output_indices = network_output.get_output_indices();

        // let start = Instant::now();

        if current_predictions.is_empty() || current_predictions[0].is_empty() {
            println!("Empty predictions. Breaking.");
            break;
        }

        // Get reference to last prediction (avoid clone)
        let last_pred = current_predictions[0].last().unwrap();

        // Greedy decoding: find argmax in sparse array, then map to actual token ID
        let predicted_token_id = if !output_indices.is_empty() && !output_indices[0].is_empty() {
            let last_indices = output_indices[0].last().unwrap();
            let sparse_idx = last_pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            last_indices[sparse_idx] as u32
        } else {
            // Fallback for non-sparse output
            last_pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
        };

        batch_ids[0].push(predicted_token_id);

        // let duration = start.elapsed().as_secs_f64();
        // println!("\ntime elapsed for token prediction in seconds: {}", duration.to_string().red().bold());

        // Detokenize single token directly (no parallel overhead)
        let predicted_token = match detokenize(&vec![predicted_token_id], false) {
            Ok(token) => token,
            Err(e) => {
                println!("Error detokenizing: {:?}", e);
                "<detokenize_error>".to_string()
            }
        };

        if predicted_token == "<eos>" {
            println!("\n\n <eos> predicted. Breaking ....");
            break;
        }

        print!("{}", predicted_token);
        // Use push_str instead of format! for efficiency
        current_input_batch[0].push_str(&predicted_token);

        count_tokens_prediction += 1;
        if count_tokens_prediction > 50 {
            println!("\nMax token prediction limit reached. Breaking.");
            break;
        }

        time_step = if time_step == 0 { batch_ids[0].len() - 1 } else { time_step + 1 };
    }

    let seconds_elapsed_end = now.elapsed();
    let duration = seconds_elapsed_end - _seconds_elapsed;
    let seconds = duration.as_secs_f64();
    println!("\ntime elapsed in seconds: {:?}", seconds);

    current_input_batch
}

pub fn predict(transformer_network: &mut NeuralNetwork, layer_input: &LayerInput) -> LayerOutput {
    // Forward pass

    // println!("forward pass start ----------------------------------------------------------------------");
    let mut output: Option<Vec<Vec<Vec<C>>>> = None;
    let mut output_softmax: Option<Vec<Vec<Vec<Real>>>> = None;
    let mut padding_mask = None;

    let batch_ids = layer_input.get_batch_ids();
    let forward_only = layer_input.get_forward_only();

    let mut layer_input = layer_input.clone();
    let target_batch_ids_option = Some(layer_input.get_target_batch_ids());
    let mut linear_output_indices = vec![];

    // Compute total valid tokens across batch for proper gradient normalization
    if !forward_only {
        let target_batch_ids = layer_input.get_target_batch_ids();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

        let total_valid_tokens: usize = if !target_batch_ids.is_empty() && !padding_mask_batch.is_empty() {
            target_batch_ids
                .iter()
                .zip(padding_mask_batch.iter())
                .map(|(targets, mask)| {
                    let target_len = targets.len();
                    // Calculate offset from the VALID sequence length, not total padded length
                    let valid_seq_len = mask.iter().filter(|&&m| m != 0).count();
                    let offset = valid_seq_len.saturating_sub(target_len);

                    // Count only positions where both:
                    // 1. Target token is not padding (id != 1)
                    // 2. Corresponding mask position is non-zero
                    targets
                        .iter()
                        .enumerate()
                        .filter(|(i, &target_id)| {
                            target_id != 1 && // not padding token
                            mask[offset + i] != 0 // mask is valid at this position
                        })
                        .count()
                })
                .sum()
        } else if !target_batch_ids.is_empty() {
            // Fallback: count non-padding tokens in targets
            target_batch_ids.iter().map(|targets| targets.iter().filter(|&&id| id != 1).count()).sum()
        } else {
            batch_ids.iter().map(|b| b.len()).sum()
        };

        layer_input.set_total_valid_tokens(total_valid_tokens);
    }

    if forward_only {
        layer_input.set_calculate_gradient(false);
    }

    let now = Instant::now();

    let mut layer_output = LayerOutput::new_default();

    // Optional contiguous row-major activations; used to avoid conversions between consecutive Linear layers.
    let mut output_rm: Option<Vec<RowMajorMatrix<C>>> = None;

    let layers_len = transformer_network.layers.len();
    for layer_idx in 0..layers_len {
        let next_supports_rm = matches!(
            transformer_network.layers.get(layer_idx + 1),
            Some(
                LayerEnum::Linear(_)
                    | LayerEnum::SparseLinear(_)
                    | LayerEnum::SparseLinearRm(_)
                    | LayerEnum::MultiLinear(_)
                    | LayerEnum::Norm(_)
                    | LayerEnum::NormRm(_)
                    | LayerEnum::RMSNorm(_)
                    | LayerEnum::PositionalEncoding(_)
                    | LayerEnum::PositionalEncodingRm(_)
                    | LayerEnum::FeedForward(_)
                    | LayerEnum::FeedForwardRm(_)
                    | LayerEnum::DenseRm(_)
                    | LayerEnum::SelfAttention(_)
                    | LayerEnum::SparseSelfAttention(_)
                    | LayerEnum::SparseSelfAttentionRm(_)
                    | LayerEnum::SelfAttentionApproximationRm(_)
                    | LayerEnum::ComplexToLinear(_)
                    | LayerEnum::Softmax(_)
                    | LayerEnum::SoftmaxRm(_)
            )
        );

        let layer = transformer_network.layers.get_mut(layer_idx).expect("layer index");
        match layer {
            LayerEnum::AdaptiveAvgPool1d(adaptive_avg_pooling_layer) => {
                let previous_output = match output.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Attention layer");
                        continue;
                    }
                };

                layer_input.set_input_batch(previous_output);
                let mut layer_output: LayerOutput = adaptive_avg_pooling_layer.forward(&layer_input);
                output = layer_output.take_output_batch();
            }
            LayerEnum::Embedding(embedding_layer) => {
                layer_input.set_batch_ids(batch_ids.clone());

                // Prefer RM embeddings when the downstream path supports RM.
                if next_supports_rm {
                    let (embeddings, padding_m) = embedding_layer.forward(&layer_input);
                    let embeddings_rm = embeddings.iter().map(|m| RowMajorMatrix::from_rows(m)).collect();
                    output_rm = Some(embeddings_rm);
                    output = None;
                    padding_mask = Some(padding_m.clone());
                    layer_input.set_padding_mask_batch(padding_m);
                } else {
                    let (embeddings, padding_m) = embedding_layer.forward(&layer_input);
                    output = Some(embeddings);
                    output_rm = None;
                    padding_mask = Some(padding_m.clone());
                    layer_input.set_padding_mask_batch(padding_m);
                }

                //println!("time elapsed in seconds in embedding: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
                // println!("padding mask: {:?}", &padding_mask);
            }
            LayerEnum::EmbeddingRm(embedding_layer) => {
                layer_input.set_batch_ids(batch_ids.clone());

                let (embeddings_rm, padding_m) = embedding_layer.forward(&layer_input);
                padding_mask = Some(padding_m.clone());
                layer_input.set_padding_mask_batch(padding_m);

                if next_supports_rm {
                    output_rm = Some(embeddings_rm);
                    output = None;
                } else {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: EmbeddingRm would need RM->Vec conversion for downstream Vec layer");
                    }
                    let embeddings = embeddings_rm.iter().map(|m| m.to_rows()).collect();
                    output = Some(embeddings);
                    output_rm = None;
                }
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if VERBOSE {
                    println!("forward pos encoding (vec)");
                }
                let start = Instant::now();

                if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: PositionalEncoding(Vec) would consume RM output");
                    }
                    let vec_out: Vec<Vec<Vec<C>>> = output_rm.take().unwrap().iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                }

                let previous_output = match output.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Attention layer");
                        continue;
                    }
                };

                layer_input.set_input_batch(previous_output);
                let positional_encodings: Vec<Vec<Vec<C>>> = positional_encoding_layer.forward(&layer_input);
                output = Some(positional_encodings);
                output_rm = None;

                if VERBOSE {
                    println!("time elapsed in seconds in positional encoding: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::PositionalEncodingRm(positional_encoding_layer) => {
                if VERBOSE {
                    println!("forward pos encoding (rm)");
                }
                let start = Instant::now();

                if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_rm.take().unwrap());
                } else {
                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Attention layer");
                            continue;
                        }
                    };
                    layer_input.set_input_batch(previous_output);
                }

                let enc_rm = positional_encoding_layer.forward(&layer_input);
                output_rm = Some(enc_rm);

                if next_supports_rm {
                    output = None;
                } else {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: PositionalEncodingRm produced RM but next layer requires Vec");
                    }
                    let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                    output_rm = None;
                }

                if VERBOSE {
                    println!("time elapsed in seconds in positional encoding: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::Norm(norm_layer) => {
                if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: Norm(Vec) would consume RM output");
                    }
                    let vec_out: Vec<Vec<Vec<C>>> = output_rm.take().unwrap().iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                }

                let previous_output = match output.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Attention layer");
                        continue;
                    }
                };

                layer_input.set_input_batch_before(previous_output.clone());
                layer_input.set_input_batch(previous_output);

                let mut norm_output = norm_layer.forward(&layer_input);
                output = norm_output.take_output_batch();
                output_rm = None;
            }
            LayerEnum::NormRm(norm_layer) => {
                if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_rm.take().unwrap());
                } else {
                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Attention layer");
                            continue;
                        }
                    };
                    layer_input.set_input_batch(previous_output);
                }

                let mut norm_output = norm_layer.forward(&layer_input);
                output_rm = norm_output.take_output_batch_rm();

                if next_supports_rm {
                    output = None;
                } else {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: NormRm produced RM but next layer requires Vec");
                    }
                    let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                    output_rm = None;
                }
            }
            LayerEnum::SelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(padding_m) = &padding_mask {
                    if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                        layer_input.clear_input_batch();
                        layer_input.set_input_batch_rm(output_rm.take().unwrap());
                        layer_input.set_padding_mask_batch(padding_m.clone());

                        if VERBOSE {
                            println!("forward self-attention start (rm)");
                        }
                        let start = Instant::now();
                        let mut output_attention = attention.forward(&layer_input);
                        output_rm = output_attention.take_output_batch_rm();

                        if VERBOSE {
                            println!("time elapsed in seconds in self attention layer (rm): {:?}", start.elapsed().as_secs_f64());
                        }

                        if next_supports_rm {
                            output = None;
                        } else {
                            if layer_input.get_rm_strict() {
                                panic!("RM strict mode violation: SelfAttention produced RM but next layer requires Vec");
                            }
                            let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                            output = Some(vec_out);
                            output_rm = None;
                        }
                    } else {
                        let previous_output = match output.take() {
                            Some(v) => v,
                            None => {
                                println!("No previous output for Attention layer");
                                continue;
                            }
                        };

                        layer_input.set_input_batch(previous_output);
                        layer_input.set_padding_mask_batch(padding_m.clone());

                        if VERBOSE {
                            println!("forward self-attention start");
                        }
                        let start = Instant::now();
                        let mut output_attention = attention.forward(&layer_input);

                        if VERBOSE {
                            println!("time elapsed in seconds in self attention layer: {:?}", start.elapsed().as_secs_f64());
                        }
                        output = output_attention.take_output_batch();
                        output_rm = None;
                    }
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SparseSelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(padding_m) = &padding_mask {
                    // Vec-only layer: convert RM -> Vec when needed.
                    if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: SparseSelfAttention(Vec) would consume RM output; use SparseSelfAttentionRm");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = output_rm.take().unwrap().iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                    }

                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Attention layer");
                            continue;
                        }
                    };

                    layer_input.set_input_batch(previous_output);
                    layer_input.set_padding_mask_batch(padding_m.clone());

                    if VERBOSE {
                        println!("forward sparse self-attention start (vec)");
                    }
                    let start = Instant::now();
                    let mut output_attention = attention.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in sparse self attention layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = output_attention.take_output_batch();
                    output_rm = None;
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SparseSelfAttentionRm(attention) => {
                if let Some(padding_m) = &padding_mask {
                    if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                        layer_input.clear_input_batch();
                        layer_input.set_input_batch_rm(output_rm.take().unwrap());
                        layer_input.set_padding_mask_batch(padding_m.clone());
                    } else {
                        let previous_output = match output.take() {
                            Some(v) => v,
                            None => {
                                println!("No previous output for Attention layer");
                                continue;
                            }
                        };
                        let rm_out: Vec<RowMajorMatrix<C>> = previous_output.iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
                        layer_input.clear_input_batch();
                        layer_input.set_input_batch_rm(rm_out);
                        layer_input.set_padding_mask_batch(padding_m.clone());
                    }

                    if VERBOSE {
                        println!("forward sparse self-attention start (rm)");
                    }
                    let start = Instant::now();
                    let mut output_attention = attention.forward(&layer_input);
                    output_rm = output_attention.take_output_batch_rm();

                    if VERBOSE {
                        println!("time elapsed in seconds in sparse self attention layer (rm): {:?}", start.elapsed().as_secs_f64());
                    }

                    if next_supports_rm {
                        output = None;
                    } else {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: SparseSelfAttentionRm produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    }
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttentionApproximation(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(padding_m) = &padding_mask {
                    let previous_output: Vec<Vec<Vec<C>>> = if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: SelfAttentionApproximation (Vec) cannot consume RM; use SelfAttentionApproximationRm");
                        }
                        output_rm
                            .take()
                            .unwrap()
                            .iter()
                            .map(|m| m.to_rows())
                            .collect()
                    } else {
                        match output.take() {
                            Some(v) => v,
                            None => {
                                println!("No previous output for Attention layer");
                                continue;
                            }
                        }
                    };

                    layer_input.set_input_batch(previous_output);
                    layer_input.set_padding_mask_batch(padding_m.clone());
                    layer_input.set_input_batch_rm(vec![]);

                    if VERBOSE {
                        println!("forward self-attention approximation start");
                    }
                    let start = Instant::now();
                    let mut output_attention = attention.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in self attention approximation layer: {:?}", start.elapsed().as_secs_f64());
                    }

                    output = output_attention.take_output_batch();
                    output_rm = None;
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttentionApproximationRm(attention) => {
                if let Some(padding_m) = &padding_mask {
                    if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                        layer_input.clear_input_batch();
                        layer_input.set_input_batch_rm(output_rm.take().unwrap());
                        layer_input.set_padding_mask_batch(padding_m.clone());
                    } else {
                        let previous_output = match output.take() {
                            Some(v) => v,
                            None => {
                                println!("No previous output for Attention layer");
                                continue;
                            }
                        };
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: SelfAttentionApproximationRm requires RM input");
                        }
                        let previous_output_rm: Vec<RowMajorMatrix<C>> = previous_output
                            .iter()
                            .map(|rows: &Vec<Vec<C>>| RowMajorMatrix::from_rows(rows.as_slice()))
                            .collect();
                        layer_input.clear_input_batch();
                        layer_input.set_input_batch_rm(previous_output_rm);
                        layer_input.set_padding_mask_batch(padding_m.clone());
                    }

                    if VERBOSE {
                        println!("forward self-attention approximation start (rm)");
                    }
                    let start = Instant::now();
                    let mut output_attention = attention.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in self attention approximation layer (rm): {:?}", start.elapsed().as_secs_f64());
                    }

                    output_rm = output_attention.take_output_batch_rm();
                    if next_supports_rm {
                        output = None;
                    } else {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: SelfAttentionApproximationRm produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    }
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::FeedForward(dense_layer) => {
                dense_layer.padding_mask_batch = padding_mask.clone();

                if VERBOSE {
                    println!("forward feed-forward network start");
                }

                let start = Instant::now();

                if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_rm.take().unwrap());

                    let mut layer_output = dense_layer.forward(&layer_input);
                    output_rm = layer_output.take_output_batch_rm();

                    if next_supports_rm {
                        output = None;
                    } else {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: FeedForward produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    }
                } else {
                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Dense layer");
                            continue;
                        }
                    };
                    layer_input.set_input_batch(previous_output);
                    let mut layer_output = dense_layer.forward(&layer_input);
                    output = layer_output.take_output_batch();
                    output_rm = None;
                }

                if VERBOSE {
                    println!("time elapsed in seconds in ffn layer: {:?}", start.elapsed().as_secs_f64());
                }

                //check_nan_or_inf_3d(&mut layer_output.get_output_batch(), "output ffn dense");
            }
            LayerEnum::FeedForwardRm(ffn_layer) => {
                ffn_layer.padding_mask_batch = padding_mask.clone();

                if VERBOSE {
                    println!("forward feed-forward RM network start");
                }

                let start = Instant::now();

                if output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_rm.take().unwrap());

                    let mut layer_output = ffn_layer.forward(&layer_input);
                    output_rm = layer_output.take_output_batch_rm();

                    if next_supports_rm {
                        output = None;
                    } else {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: FeedForwardRm produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    }
                } else {
                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for FeedForwardRm layer");
                            continue;
                        }
                    };

                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: FeedForwardRm requires RM input but received Vec");
                    }
                    let previous_output_rm: Vec<RowMajorMatrix<C>> = previous_output.iter().map(|m| RowMajorMatrix::from_rows(m)).collect();
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(previous_output_rm);
                    let mut layer_output = ffn_layer.forward(&layer_input);
                    output_rm = layer_output.take_output_batch_rm();
                    output = None;
                }

                if VERBOSE {
                    println!("time elapsed in seconds in ffn_rm layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::Linear(linear_layer) => {
                if VERBOSE {
                    println!("forward linear start");
                }

                // Run RM-only mode whenever we have a contiguous RM activation available from the previous layer
                // and we deliberately withheld the legacy Vec output (output == None).
                let use_rm_only = linear_layer.is_complex && output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty());

                if use_rm_only {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_rm.take().unwrap());
                } else {
                    // Ensure we have a Vec batch for layers that still operate on Vec<Vec<...>>.
                    if output.is_none() {
                        if let Some(rm) = &output_rm {
                            if layer_input.get_rm_strict() {
                                panic!("RM strict mode violation: Linear requires Vec input but RM is present");
                            }
                            let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                            output = Some(vec_out);
                        }
                    }

                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Dense layer");
                            continue;
                        }
                    };
                    layer_input.set_input_batch(previous_output);
                    // Clear any stale RM input to avoid accidentally generating legacy outputs.
                    layer_input.set_input_batch_rm(vec![]);
                }

                let start = Instant::now();
                let mut output_linear = linear_layer.forward(&layer_input);

                linear_output_indices = output_linear.get_output_indices();

                if VERBOSE {
                    println!("time elapsed in seconds in linear layer: {:?}", start.elapsed().as_secs_f64());
                }

                output_rm = output_linear.take_output_batch_rm();
                if next_supports_rm {
                    // Keep everything in RM form; avoid Vec conversion on the hot path.
                    output = None;
                } else {
                    // Next layer needs Vec form; convert once at the boundary.
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: Linear produced RM but next layer requires Vec");
                    }
                    let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                    output_rm = None;
                }
            }
            LayerEnum::SparseLinear(sparse_linear_layer) => {
                if VERBOSE {
                    println!("forward sparse linear start");
                }

                if output.is_none() {
                    if let Some(rm) = &output_rm {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: SparseLinear(Vec) requires Vec input but RM is present");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    }
                }

                let previous_output = match output.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for SparseLinear(Vec) layer");
                        continue;
                    }
                };
                layer_input.set_input_batch(previous_output);
                layer_input.set_input_batch_rm(vec![]);

                let start = Instant::now();
                let mut output_linear = sparse_linear_layer.forward(&layer_input);

                linear_output_indices = output_linear.get_output_indices();

                if VERBOSE {
                    println!("time elapsed in seconds in sparse linear layer: {:?}", start.elapsed().as_secs_f64());
                }

                output = output_linear.take_output_batch();
                output_rm = None;
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                if VERBOSE {
                    println!("forward sparse linear_rm start");
                }

                if output_rm.is_none() {
                    if let Some(vec_out) = &output {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: SparseLinearRm requires RM input but Vec is present");
                        }
                        output_rm = Some(vec_out.iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect());
                        output = None;
                    }
                }

                let rm_in = match output_rm.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous RM output for SparseLinearRm layer");
                        continue;
                    }
                };

                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(rm_in);

                let start = Instant::now();
                let mut output_linear = sparse_linear_layer.forward(&layer_input);
                linear_output_indices = output_linear.get_output_indices();

                if VERBOSE {
                    println!("time elapsed in seconds in sparse linear_rm layer: {:?}", start.elapsed().as_secs_f64());
                }

                output_rm = output_linear.take_output_batch_rm();
                if next_supports_rm {
                    output = None;
                } else {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: SparseLinearRm produced RM but next layer requires Vec");
                    }
                    let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                    output_rm = None;
                }
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                if VERBOSE {
                    println!("forward multilinear start");
                }

                // Prefer RM-only path when available.
                let use_rm_only = output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty());
                if use_rm_only {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_rm.take().unwrap());
                } else {
                    if output.is_none() {
                        if let Some(rm) = &output_rm {
                            if layer_input.get_rm_strict() {
                                panic!("RM strict mode violation: MultiLinear requires Vec input but RM is present");
                            }
                            let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                            output = Some(vec_out);
                        }
                    }

                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Multilinear layer");
                            continue;
                        }
                    };
                    layer_input.set_input_batch(previous_output);
                    layer_input.set_input_batch_rm(vec![]);
                }

                let start = Instant::now();
                let mut output_linear = multi_linear_layer.forward(&layer_input);

                if VERBOSE {
                    println!("time elapsed in seconds in multilayer layer: {:?}", start.elapsed().as_secs_f64());
                }

                output_rm = output_linear.take_output_batch_rm();
                if next_supports_rm {
                    output = None;
                } else {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: MultiLinear produced RM but next layer requires Vec");
                    }
                    let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                    output_rm = None;
                }
            }
            LayerEnum::Wavelet(wavelet_layer) => {
                let use_rm_only = output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty());
                if use_rm_only {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: Wavelet (Vec) received RM-only input");
                    }
                    let rm = output_rm.take().unwrap();
                    let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                    output_rm = None;
                }

                if output.is_none() {
                    println!("No previous output for Wavelet layer");
                    continue;
                }

                let previous_output = output.take().unwrap();
                layer_input.set_input_batch(previous_output);
                layer_input.set_input_batch_rm(vec![]);

                let start = Instant::now();
                let mut output_cwt = wavelet_layer.forward(&layer_input);

                if VERBOSE {
                    println!("time elapsed in seconds in wavelet layer: {:?}", start.elapsed().as_secs_f64());
                }

                output_rm = output_cwt.take_output_batch_rm();
                if next_supports_rm {
                    output = None;
                } else {
                    if let Some(rm) = &output_rm {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: Wavelet produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    } else {
                        output = output_cwt.take_output_batch();
                        output_rm = None;
                    }
                }
            }
            LayerEnum::WaveletRm(wavelet_layer) => {
                if output_rm.as_ref().is_none_or(|rm| rm.is_empty()) {
                    if output.is_some() {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: WaveletRm requires RM input but Vec is present");
                        }
                        let vec_batch = output.take().unwrap();
                        let rm_out: Vec<RowMajorMatrix<C>> = vec_batch.iter().map(|m| RowMajorMatrix::from_rows(m)).collect();
                        output_rm = Some(rm_out);
                    }
                }

                let use_rm_only = output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty());
                if !use_rm_only {
                    println!("No previous output for WaveletRm layer");
                    continue;
                }

                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(output_rm.take().unwrap());

                let start = Instant::now();
                let mut output_cwt = wavelet_layer.forward(&layer_input);

                if VERBOSE {
                    println!("time elapsed in seconds in wavelet rm layer: {:?}", start.elapsed().as_secs_f64());
                }

                output_rm = output_cwt.take_output_batch_rm();
                if next_supports_rm {
                    output = None;
                } else {
                    if let Some(rm) = &output_rm {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: WaveletRm produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    } else {
                        output = output_cwt.take_output_batch();
                        output_rm = None;
                    }
                }
            }
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                let use_rm_only = output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty());
                if use_rm_only {
                    if layer_input.get_rm_strict() {
                        panic!("RM strict mode violation: DiscreteWavelet (Vec) received RM-only input");
                    }
                    let rm = output_rm.take().unwrap();
                    let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                    output = Some(vec_out);
                    output_rm = None;
                }

                if output.is_none() {
                    println!("No previous output for DiscreteWavelet layer");
                    continue;
                }

                let previous_output = output.take().unwrap();
                layer_input.set_input_batch(previous_output);
                layer_input.set_input_batch_rm(vec![]);

                let start = Instant::now();
                let mut output_dwt = wavelet_layer.forward(&layer_input);

                let new_padding_mask = output_dwt.get_padding_mask_batch();
                padding_mask = Some(new_padding_mask.clone());
                layer_input.set_padding_mask_batch(new_padding_mask);

                if VERBOSE {
                    println!("time elapsed in seconds in discrete wavelet layer: {:?}", start.elapsed().as_secs_f64());
                }

                output_rm = output_dwt.take_output_batch_rm();
                if next_supports_rm {
                    output = None;
                } else {
                    if let Some(rm) = &output_rm {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: DiscreteWavelet produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    } else {
                        output = output_dwt.take_output_batch();
                        output_rm = None;
                    }
                }
            }
            LayerEnum::DiscreteWaveletRm(wavelet_layer) => {
                if output_rm.as_ref().is_none_or(|rm| rm.is_empty()) {
                    if output.is_some() {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: DiscreteWaveletRm requires RM input but Vec is present");
                        }
                        let vec_batch = output.take().unwrap();
                        let rm_out: Vec<RowMajorMatrix<C>> = vec_batch.iter().map(|m| RowMajorMatrix::from_rows(m)).collect();
                        output_rm = Some(rm_out);
                    }
                }

                let use_rm_only = output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty());
                if !use_rm_only {
                    println!("No previous output for DiscreteWaveletRm layer");
                    continue;
                }

                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(output_rm.take().unwrap());

                let start = Instant::now();
                let mut output_dwt = wavelet_layer.forward(&layer_input);

                let new_padding_mask = output_dwt.get_padding_mask_batch();
                padding_mask = Some(new_padding_mask.clone());
                layer_input.set_padding_mask_batch(new_padding_mask);

                if VERBOSE {
                    println!("time elapsed in seconds in discrete wavelet rm layer: {:?}", start.elapsed().as_secs_f64());
                }

                output_rm = output_dwt.take_output_batch_rm();
                if next_supports_rm {
                    output = None;
                } else {
                    if let Some(rm) = &output_rm {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: DiscreteWaveletRm produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    } else {
                        output = output_dwt.take_output_batch();
                        output_rm = None;
                    }
                }
            }
            LayerEnum::ComplexToLinear(ctl_layer) => {
                // Prefer RM-only path when available.
                if output.is_none() && output_rm.as_ref().is_some_and(|rm| !rm.is_empty()) {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_rm.take().unwrap());

                    let start = Instant::now();
                    let mut output_ctl = ctl_layer.forward(&layer_input);
                    if VERBOSE {
                        println!("time elapsed in seconds in complex to linear layer: {:?}", start.elapsed().as_secs_f64());
                    }

                    output_rm = output_ctl.take_output_batch_rm();
                    if next_supports_rm {
                        output = None;
                    } else {
                        if layer_input.get_rm_strict() {
                            panic!("RM strict mode violation: ComplexToLinear produced RM but next layer requires Vec");
                        }
                        let vec_out: Vec<Vec<Vec<C>>> = output_rm.as_ref().unwrap().iter().map(|m| m.to_rows()).collect();
                        output = Some(vec_out);
                        output_rm = None;
                    }
                } else {
                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Complex to Linear layer");
                            continue;
                        }
                    };
                    layer_input.set_input_batch(previous_output);
                    layer_input.set_input_batch_rm(vec![]);

                    let start = Instant::now();
                    let mut output_ctl = ctl_layer.forward(&layer_input);
                    if VERBOSE {
                        println!("time elapsed in seconds in complex to linear layer: {:?}", start.elapsed().as_secs_f64());
                    }
                    output = output_ctl.take_output_batch();
                    output_rm = None;
                }
            }
            LayerEnum::Softmax(softmax_layer) => {
                //println!("forward softmax start");
                let start = Instant::now();

                if forward_only {
                    // In inference, we often only need the raw logits (or last row).
                    // If we have RM activations, avoid converting to nested Vec<Complex>.
                    if output.is_none() {
                        if let Some(rm) = &output_rm {
                            let out_real: Vec<Vec<Vec<Real>>> = rm
                                .iter()
                                .map(|m| {
                                    (0..m.rows)
                                        .map(|r| {
                                            let row = m.row_range(r);
                                            m.data[row].iter().map(|c| c.re).collect::<Vec<Real>>()
                                        })
                                        .collect::<Vec<Vec<Real>>>()
                                })
                                .collect();
                            output_softmax = Some(out_real);
                        }
                    } else {
                        let input_ref = layer_input.get_input_batch_ref().expect("softmax input batch missing");
                        output_softmax = Some(
                            input_ref
                                .iter()
                                .map(|m| m.iter().map(|row| row.iter().map(|z| z.re).collect()).collect())
                                .collect(),
                        );
                    }
                } else {
                    // Training always requires CE loss + gradients; ensure Softmax is in TRAINING
                    // even if a prior inference call switched it to PRODUCTION.
                    softmax_layer.operation_mode = OperationMode::TRAINING;

                    // Vec-only softmax: ensure logits are in Vec form.
                    if output.is_none() {
                        if let Some(rm) = &output_rm {
                            if layer_input.get_rm_strict() {
                                panic!("RM strict mode violation: Softmax(Vec) training requires Vec input but RM logits are present");
                            }
                            let vec_out: Vec<Vec<Vec<C>>> = rm.iter().map(|m| m.to_rows()).collect();
                            output = Some(vec_out);
                            output_rm = None;
                        }
                    }

                    let previous_output = match output.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Softmax(Vec) layer");
                            continue;
                        }
                    };

                    layer_input.set_input_batch(previous_output);
                    layer_input.set_input_batch_rm(vec![]);

                    layer_input.set_output_indices(linear_output_indices.clone());
                    let softmax_result_real: Vec<Vec<Vec<Real>>> = softmax_layer.forward(&layer_input, padding_mask.clone(), target_batch_ids_option.clone());
                    output_softmax = Some(softmax_result_real);
                    layer_output.set_cross_entropy_loss_batch(softmax_layer.cross_entropy_loss_batch.clone().unwrap());
                }

                if VERBOSE {
                    println!("time elapsed in seconds in softmax layer: {:?}", start.elapsed().as_secs_f64());
                }
                // println!("forward softmax end");
            }
            LayerEnum::SoftmaxRm(softmax_layer) => {
                let start = Instant::now();

                if forward_only {
                    if let Some(rm) = &output_rm {
                        let out_real: Vec<Vec<Vec<Real>>> = rm
                            .iter()
                            .map(|m| {
                                (0..m.rows)
                                    .map(|r| {
                                        let row = m.row_range(r);
                                        m.data[row].iter().map(|c| c.re).collect::<Vec<Real>>()
                                    })
                                    .collect::<Vec<Vec<Real>>>()
                            })
                            .collect();
                        output_softmax = Some(out_real);
                    } else if let Some(vec_logits) = layer_input.get_input_batch_ref() {
                        output_softmax = Some(
                            vec_logits
                                .iter()
                                .map(|m| m.iter().map(|row| row.iter().map(|z| z.re).collect()).collect())
                                .collect(),
                        );
                    }
                } else {
                    softmax_layer.operation_mode = OperationMode::TRAINING;

                    // RM-only softmax: ensure logits are in RM form.
                    if output_rm.is_none() {
                        if let Some(vec_logits) = output.take() {
                            if layer_input.get_rm_strict() {
                                panic!("RM strict mode violation: SoftmaxRm requires RM logits but Vec logits are present");
                            }
                            output_rm = Some(vec_logits.iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect());
                        }
                    }

                    let logits_rm = match output_rm.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous RM output for SoftmaxRm layer");
                            continue;
                        }
                    };

                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(logits_rm);

                    layer_input.set_output_indices(linear_output_indices.clone());
                    let softmax_result_real: Vec<Vec<Vec<Real>>> = softmax_layer.forward(&layer_input, padding_mask.clone(), target_batch_ids_option.clone());
                    output_softmax = Some(softmax_result_real);
                    layer_output.set_cross_entropy_loss_batch(softmax_layer.cross_entropy_loss_batch.clone().unwrap());
                }

                if VERBOSE {
                    println!("time elapsed in seconds in softmax_rm layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            _ => {
                panic!("Layer type not supported for backward pass");
            }
        }
    }

    layer_output.set_output_batch_real(output_softmax.unwrap());
    layer_output.set_padding_mask_batch(padding_mask.unwrap());
    layer_output.set_output_indices(linear_output_indices);

    if VERBOSE {
        let whole_duration_forward = now.elapsed();
        println!("TOTAL time elapsed in seconds in forward pass: {}", whole_duration_forward.as_secs_f64().to_string().green().bold());
    }
    //println!("forward pass end ----------------------------------------------------------------------");
    layer_output
}

pub fn backward(transformer_network: &mut NeuralNetwork, target_batch_ids: &Vec<Vec<u32>>, update_params: bool) -> Option<Gradient> {
    // Backward pass

    let mut gradient: Option<Gradient> = None;
    let batch_size = transformer_network.get_minibatch_size();
    let update_gradients: bool = update_params;

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
                    let start = Instant::now();
                    let grad_vec: Vec<Vec<Vec<C>>> = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref().filter(|g| !g.is_empty()) {
                        gr_rm.iter().map(|m| m.to_rows()).collect()
                    } else {
                        previous_gradient.get_gradient_input_batch()
                    };
                    let gradient_batch: Gradient = embedding_layer.backward(&grad_vec);
                    if VERBOSE {
                        println!("time elapsed in seconds in embedding layer backward: {}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Token Embedding Layer");
                }
            }
            LayerEnum::EmbeddingRm(embedding_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gr_rm = previous_gradient
                        .get_gradient_input_batch_rm_ref()
                        .filter(|g| !g.is_empty())
                        .expect("EmbeddingRm expects RM gradients");
                    let gradient_batch: Gradient = embedding_layer.backward(gr_rm);
                    if VERBOSE {
                        println!("time elapsed in seconds in embedding_rm layer backward: {}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Token Embedding Layer");
                }
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();

                    let grad_vec: Vec<Vec<Vec<C>>> = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        gr_rm.iter().map(|m| m.to_rows()).collect()
                    } else {
                        previous_gradient.get_gradient_input_batch()
                    };

                    let gradient_batch: Gradient = positional_encoding_layer.backward(&grad_vec);

                    if VERBOSE {
                        println!("time elapsed in seconds in positional encoding layer backward: {}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                }
            }
            LayerEnum::PositionalEncodingRm(positional_encoding_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gr_rm = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        gr_rm.to_vec()
                    } else {
                        previous_gradient.get_gradient_input_batch_rm()
                    };

                    let gradient_batch: Gradient = positional_encoding_layer.backward(&gr_rm);

                    if VERBOSE {
                        println!("time elapsed in seconds in positional encoding layer backward: {}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                }
            }
            LayerEnum::Norm(norm_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();

                    let gradient_batch: Gradient = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        let mut pg = previous_gradient.clone();
                        pg.set_gradient_input_batch(gr_rm.iter().map(|m| m.to_rows()).collect());
                        pg.set_gradient_input_batch_rm(vec![]);
                        norm_layer.backward(&pg)
                    } else {
                        norm_layer.backward(&previous_gradient)
                    };
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in norm layer backward: {}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient for norm layer");
                }
            }
            LayerEnum::NormRm(norm_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = if previous_gradient.get_gradient_input_batch_rm_ref().is_some() {
                        norm_layer.backward(&previous_gradient)
                    } else {
                        let mut pg = previous_gradient.clone();
                        let gr_rm = pg.get_gradient_input_batch_rm();
                        pg.set_gradient_input_batch_rm(gr_rm);
                        norm_layer.backward(&pg)
                    };
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
                    let start = Instant::now();
                    let gradient_batch: Gradient = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        attention_layer.backward_rm(gr_rm)
                    } else {
                        attention_layer.backward(&previous_gradient.get_gradient_input_batch())
                    };

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
                    let start = Instant::now();
                    let gradient_batch: Gradient = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        let previous_gradient_batch: Vec<Vec<Vec<C>>> = gr_rm.iter().map(|m| m.to_rows()).collect();
                        attention_layer.backward(&previous_gradient_batch)
                    } else {
                        attention_layer.backward(&previous_gradient.get_gradient_input_batch())
                    };
                    if VERBOSE {
                        println!("time elapsed in seconds in sparse self attention layer backward: {:?}", start.elapsed().as_secs_f64());
                    }

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Self Attention Layer");
                }
            }
            LayerEnum::SparseSelfAttentionRm(attention_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();

                    let gradient_batch: Gradient = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        attention_layer.backward_rm(gr_rm)
                    } else {
                        let previous_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient
                            .get_gradient_input_batch()
                            .iter()
                            .map(|rows| RowMajorMatrix::from_rows(rows))
                            .collect();
                        attention_layer.backward_rm(&previous_gradient_batch_rm)
                    };

                    if VERBOSE {
                        println!("time elapsed in seconds in sparse self attention layer backward (rm): {:?}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in SparseSelfAttentionLayerRm");
                }
            }
            LayerEnum::SelfAttentionApproximation(attention_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        let previous_gradient_batch: Vec<Vec<Vec<C>>> = gr_rm.iter().map(|m| m.to_rows()).collect();
                        attention_layer.backward(&previous_gradient_batch)
                    } else {
                        attention_layer.backward(&previous_gradient.get_gradient_input_batch())
                    };
                    if VERBOSE {
                        println!("time elapsed in seconds in self attention approximation layer backward: {:?}", start.elapsed().as_secs_f64());
                    }

                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Self Attention Layer");
                }
            }
            LayerEnum::SelfAttentionApproximationRm(attention_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();

                    let gradient_batch: Gradient = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        attention_layer.backward_rm(gr_rm)
                    } else {
                        let previous_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient
                            .get_gradient_input_batch()
                            .iter()
                            .map(|rows| RowMajorMatrix::from_rows(rows))
                            .collect();
                        attention_layer.backward_rm(&previous_gradient_batch_rm)
                    };

                    if VERBOSE {
                        println!("time elapsed in seconds in self attention approximation layer backward (rm): {:?}", start.elapsed().as_secs_f64());
                    }
                    gradient = Some(gradient_batch);
                } else {
                    println!("No previous gradient in Self Attention Layer");
                }
            }
            LayerEnum::FeedForward(dense_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                        // FeedForwardLayer is Vec-only now; convert RM -> Vec.
                        let previous_gradient_batch: Vec<Vec<Vec<C>>> = gr_rm.iter().map(|m| m.to_rows()).collect();
                        dense_layer.backward(&previous_gradient_batch)
                    } else {
                        let previous_gradient_batch: Vec<Vec<Vec<C>>> = previous_gradient.get_gradient_input_batch();
                        dense_layer.backward(&previous_gradient_batch)
                    };
                    // println!("backward dense end");
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in ffn layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in Dense Layer");
                }
            }
            LayerEnum::FeedForwardRm(ffn_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();

                    let gradient_batch: Gradient = if previous_gradient.get_gradient_input_batch_rm_ref().is_some() {
                        ffn_layer.backward(&previous_gradient)
                    } else {
                        let previous_gradient_batch: Vec<RowMajorMatrix<C>> = previous_gradient
                            .get_gradient_input_batch()
                            .iter()
                            .map(|rows| RowMajorMatrix::from_rows(rows))
                            .collect();
                        let mut g = Gradient::new_default();
                        g.set_gradient_input_batch_rm(previous_gradient_batch);
                        g.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());
                        ffn_layer.backward(&g)
                    };

                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in ffn_rm layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in FeedForwardRm Layer");
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
            LayerEnum::SparseLinear(sparse_linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = sparse_linear_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in sparse linear layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in Linear Layer");
                }
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = sparse_linear_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in sparse linear_rm layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in SparseLinearRm Layer");
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
            LayerEnum::WaveletRm(wavelet_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();

                    let gradient_batch: Gradient = if previous_gradient.get_gradient_input_batch_rm_ref().is_some() {
                        wavelet_layer.backward(&previous_gradient)
                    } else {
                        let previous_gradient_batch: Vec<RowMajorMatrix<C>> = previous_gradient
                            .get_gradient_input_batch()
                            .iter()
                            .map(|rows| RowMajorMatrix::from_rows(rows))
                            .collect();
                        let mut g = Gradient::new_default();
                        g.set_gradient_input_batch_rm(previous_gradient_batch);
                        g.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());
                        wavelet_layer.backward(&g)
                    };

                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in wavelet rm layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in WaveletRm Layer");
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
            LayerEnum::DiscreteWaveletRm(wavelet_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();

                    let gradient_batch: Gradient = if previous_gradient.get_gradient_input_batch_rm_ref().is_some() {
                        wavelet_layer.backward(&previous_gradient)
                    } else {
                        let previous_gradient_batch: Vec<RowMajorMatrix<C>> = previous_gradient
                            .get_gradient_input_batch()
                            .iter()
                            .map(|rows| RowMajorMatrix::from_rows(rows))
                            .collect();
                        let mut g = Gradient::new_default();
                        g.set_gradient_input_batch_rm(previous_gradient_batch);
                        g.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());
                        wavelet_layer.backward(&g)
                    };

                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in discrete wavelet rm layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in DiscreteWaveletRm Layer");
                }
            }
            LayerEnum::ComplexToLinear(complex_to_linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = complex_to_linear_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in complex to linear layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous gradient in Complex to Linear Layer");
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
            LayerEnum::SoftmaxRm(softmax_layer) => {
                softmax_layer.batch_size = batch_size;
                gradient = Some(softmax_layer.gradient.as_ref().unwrap().clone());
            }
            _ => {
                panic!("Layer type not supported for backward pass");
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
    let all_predicted_tokens = predict_token_by_token(&mut transformer, &input);

    println!("prediction is: {:?}", all_predicted_tokens);

    all_predicted_tokens
}

pub fn cross_entropy_sum_batch(cross_entropy_loss_batch: &Vec<Vec<Vec<C>>>, _targets: &Vec<Vec<u32>>) -> C {
    let mut total_loss = C::new(ZERO, ZERO);

    for batch in cross_entropy_loss_batch {
        for seq in batch {
            for token_loss in seq {
                if token_loss.is_nan() || token_loss.is_infinite() {
                    panic!("Invalid token loss value encountered in cross-entropy loss computation.");
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

/// Clear all caches in attention layers (needed when changing batch size)
fn clear_network_caches(transformer_network: &mut NeuralNetwork) {
    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::SparseSelfAttention(attention_layer) => {
                for head in attention_layer.attention_heads.iter_mut() {
                    head.clear_cache();
                }
            }
            LayerEnum::SelfAttention(attention_layer) => {
                for head in attention_layer.attention_heads.iter_mut() {
                    head.clear_cache();
                }
            }
            _ => {}
        }
    }
}

/// Evaluate model on validation set (NO gradient updates, NO shuffling)
fn evaluate_validation(transformer_network: &mut NeuralNetwork, dataset: &Dataset<String, String>, batch_size: usize, layer_input: &mut LayerInput) -> Real {
    // Clear all caches before validation (important for correct batch size handling)
    clear_network_caches(transformer_network);

    let mut total_val_loss: Real = ZERO;
    let mut num_batches = 0;

    // Get validation batches (NOT shuffled)
    let val_batches = dataset.get_validation_batches(batch_size);

    if val_batches.is_empty() {
        println!("⚠️  Warning: No validation data available! Validation loss will be unreliable.");
        println!(
            "   Total dataset size: {}, Training size: {}, Validation size: {}",
            dataset.input.len(),
            dataset.total_training_records_size,
            dataset.total_validation_records_size
        );
        return Real::INFINITY;
    }

    for batch_dataset in val_batches.iter() {
        let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

        let input_batch_extended = extend_input_with_bos(input_batch);
        let target_batch_extended = batch_dataset.extend_target(target_batch);

        let (_tokens, input_ids) = tokenize_batch(&input_batch_extended, false).unwrap();
        let (_tokens, target_ids) = tokenize_batch(&target_batch_extended, false).unwrap();

        let batch_ids: Vec<Vec<u32>> = concat_batches(&input_ids, &target_ids);

        let mut target_ids: Vec<Vec<u32>> = target_ids
            .iter()
            .map(|seq| {
                if seq.is_empty() {
                    return vec![];
                }
                let mut shifted = Vec::with_capacity(seq.len());
                shifted.extend_from_slice(&seq[0..seq.len()]);
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

        let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap_or(0);

        if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
            let (input_batch_ids, target_batch_ids) = sliding_window_chunks_matrix(&batch_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
            batch_ids = input_batch_ids;
            target_ids = target_batch_ids;
        }

        let actual_batch_size = batch_ids.len();

        layer_input.set_batch_ids(batch_ids.clone());
        layer_input.set_batch_size(actual_batch_size);
        layer_input.set_forward_only(false); // MUST be false to compute loss!
        layer_input.set_calculate_gradient(false); // NO GRADIENT COMPUTATION
        layer_input.set_target_batch_ids(target_ids.clone());
        layer_input.set_top_k_size(TOP_K_SIZE);

        transformer_network.minibatch_size = actual_batch_size; // Update network's batch size too

        // Forward pass only
        let network_output = predict(transformer_network, &layer_input);
        let loss: C = cross_entropy_sum_batch(&network_output.get_cross_entropy_loss_batch(), &target_ids);

        total_val_loss += loss.re;
        num_batches += 1;
    }

    if num_batches == 0 {
        println!("⚠️  Warning: No validation batches processed!");
        return Real::INFINITY;
    }

    let avg_val_loss = total_val_loss / r(num_batches as f64);

    // Debug: print validation statistics on first epoch
    if transformer_network.time_step < 100 {
        println!("📊 Validation: {} batches, total loss: {:.4}, avg loss: {:.4}", num_batches, total_val_loss, avg_val_loss);
    }

    avg_val_loss
}

/// Evaluate model on TEST set - call this ONLY ONCE after training is complete
pub fn evaluate_test(transformer_network: &mut NeuralNetwork, dataset: &Dataset<String, String>, batch_size: usize) -> Real {
    // Clear all caches before test evaluation (important for correct batch size handling)
    clear_network_caches(transformer_network);

    let mut total_test_loss: Real = ZERO;
    let mut num_batches = 0;
    let mut layer_input = LayerInput::new_default();

    // Get test batches (NOT shuffled)
    let test_batches = dataset.get_test_batches(batch_size);

    if test_batches.is_empty() {
        println!("Warning: No test data available");
        return Real::INFINITY;
    }

    println!("\n{}", "=".repeat(60).bright_cyan());
    println!("{}", "FINAL TEST SET EVALUATION".bright_cyan().bold());
    println!("{}", "=".repeat(60).bright_cyan());

    for batch_dataset in test_batches.iter() {
        let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

        let input_batch_extended = extend_input_with_bos(input_batch);
        let target_batch_extended = batch_dataset.extend_target(target_batch);

        let (_tokens, input_ids) = tokenize_batch(&input_batch_extended, false).unwrap();
        let (_tokens, target_ids) = tokenize_batch(&target_batch_extended, false).unwrap();

        let batch_ids: Vec<Vec<u32>> = concat_batches(&input_ids, &target_ids);

        let mut target_ids: Vec<Vec<u32>> = target_ids
            .iter()
            .map(|seq| {
                if seq.is_empty() {
                    return vec![];
                }
                let mut shifted = Vec::with_capacity(seq.len());
                shifted.extend_from_slice(&seq[0..seq.len()]);
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

        let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap_or(0);

        if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
            let (input_batch_ids, target_batch_ids) = sliding_window_chunks_matrix(&batch_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
            batch_ids = input_batch_ids;
            target_ids = target_batch_ids;
        }

        let actual_batch_size = batch_ids.len();

        layer_input.set_batch_ids(batch_ids.clone());
        layer_input.set_batch_size(actual_batch_size);
        layer_input.set_forward_only(false); // MUST be false to compute loss!
        layer_input.set_calculate_gradient(false); // NO GRADIENT COMPUTATION
        layer_input.set_target_batch_ids(target_ids.clone());
        layer_input.set_top_k_size(TOP_K_SIZE);

        transformer_network.minibatch_size = actual_batch_size; // Update network's batch size too

        // Forward pass only
        let network_output = predict(transformer_network, &layer_input);
        let loss: C = cross_entropy_sum_batch(&network_output.get_cross_entropy_loss_batch(), &target_ids);

        total_test_loss += loss.re;
        num_batches += 1;
    }

    let avg_test_loss = total_test_loss / r(num_batches as f64);

    println!("\n{}", "FINAL TEST LOSS:".bright_cyan().bold());
    println!("{}", avg_test_loss.to_string().bright_green().bold());
    println!("{}\n", "=".repeat(60).bright_cyan());

    avg_test_loss
}
