use crate::neural_networks::network_layers::layer::LayerEnum;
use crate::neural_networks::utils::dtype::{r, Real, C, ZERO};
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
        utils::tokenizer::{detokenize, tokenize_batch},
    },
};

pub const MAX_CONTEXT_WINDOW_SIZE: usize = 50280;
pub const CONTEXT_OVERLAPPING: usize = 16;
pub const EMA_SCALER: f64 = 1.1;
pub const TOP_K_SIZE: usize = 500;

#[inline]
fn complex_batch_to_real_batch(data: &[Vec<Vec<C>>]) -> Vec<Vec<Vec<Real>>> {
    data.iter().map(|seq| seq.iter().map(|row| row.iter().map(|z| z.re).collect()).collect()).collect()
}

fn shift_targets_for_next_token_prediction(target_ids: &[Vec<u32>]) -> Vec<Vec<u32>> {
    target_ids.to_vec()
}

fn shift_inputs_for_next_token_prediction(batch_ids: &[Vec<u32>]) -> Vec<Vec<u32>> {
    batch_ids
        .iter()
        .map(|seq| {
            if seq.is_empty() {
                Vec::new()
            } else {
                seq[..seq.len() - 1].to_vec()
            }
        })
        .collect()
}

fn sliding_window_start_positions(len: usize, window_size: usize, stride: usize) -> Vec<usize> {
    if len <= window_size {
        return vec![0];
    }

    let step = stride.max(1);
    let mut starts = Vec::new();
    let mut start = 0;

    while start + window_size <= len {
        starts.push(start);
        start += step;
    }

    let last_start = len - window_size;
    if starts.last().copied() != Some(last_start) {
        starts.push(last_start);
    }

    starts
}

fn split_shifted_batch_with_targets(
    batch_ids: &[Vec<u32>],
    target_ids: &[Vec<u32>],
    window_size: usize,
    stride: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut input_chunks = Vec::new();
    let mut target_chunks = Vec::new();

    for (seq, targets) in batch_ids.iter().zip(target_ids.iter()) {
        if seq.is_empty() {
            input_chunks.push(Vec::new());
            target_chunks.push(Vec::new());
            continue;
        }

        let target_start_in_seq = seq.len().saturating_sub(targets.len());

        for start in sliding_window_start_positions(seq.len(), window_size, stride) {
            let end = usize::min(start + window_size, seq.len());
            input_chunks.push(seq[start..end].to_vec());

            let supervised_start = target_start_in_seq.max(start);
            let target_start = supervised_start.saturating_sub(target_start_in_seq);
            let target_end = end.saturating_sub(target_start_in_seq).min(targets.len());

            if target_start < target_end {
                target_chunks.push(targets[target_start..target_end].to_vec());
            } else {
                target_chunks.push(Vec::new());
            }
        }
    }

    (input_chunks, target_chunks)
}

pub fn train(transformer_network: &mut NeuralNetwork, mut dataset: Dataset<String, String>, num_epochs: usize, batch_size: usize) {
    // Setup data splits: 90% train, 10% validation (test set remains separate)
    dataset.setup_splits(None);

    let mut total_loss: C;
    let loss_threshold: Real = r(0.001);
    let now = Instant::now();
    let mut total_loss_exp_ma: Real = ZERO;
    let alpha: Real = r(0.2);
    let mut layer_input = LayerInput::new_default();
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

            let batch_ids: Vec<Vec<u32>> = concat_batches(&input_ids, &target_ids);

            // Causal next-token training over the target span:
            // the last prompt token predicts `<sep>`, then `<sep>` predicts the first answer token, etc.
            let mut target_ids: Vec<Vec<u32>> = shift_targets_for_next_token_prediction(&target_ids);
            let mut batch_ids: Vec<Vec<u32>> = shift_inputs_for_next_token_prediction(&batch_ids);

            // Count valid target tokens AFTER shifting.
            let valid_target_tokens_batch: usize = target_ids.iter().map(|seq| seq.iter().filter(|&&id| id != 1).count()).sum();
            total_valid_target_tokens_epoch += valid_target_tokens_batch;

            if VERBOSE {
                print!("\n batch ids: {:?}\n", &batch_ids);
                print!("\n target ids: {:?}\n", &target_ids);
            }

            let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap();
            let actual_batch_size = batch_ids.len();

            if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
                let (input_batch_ids, target_batch_ids) =
                    split_shifted_batch_with_targets(&batch_ids, &target_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
                batch_ids = input_batch_ids;
                target_ids = target_batch_ids;
            }

            layer_input.set_batch_ids(batch_ids.clone());
            // layer_input.set_rm_strict(true);
            layer_input.set_time_step(timestep);
            layer_input.set_batch_size(actual_batch_size);
            layer_input.set_forward_only(false);
            layer_input.set_calculate_gradient(true);
            layer_input.set_target_batch_ids(target_ids.clone());
            layer_input.set_top_k_size(TOP_K_SIZE);
            layer_input.set_total_valid_tokens(valid_target_tokens_batch.max(1));

            transformer_network.minibatch_size = actual_batch_size;
            transformer_network.time_step = timestep;

            let network_output = predict(transformer_network, &layer_input);

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
                let batch_loss_per_token = loss.re / r(valid_target_tokens_batch.max(1) as f64);
                println!(
                    "Epoch: {:?}, Batch loss/token: {:?}, Batch loss(sum): {:?}",
                    epoch, batch_loss_per_token, loss
                );
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

            // NOTE: learning-rate scheduling is owned entirely by the warmup+cosine
            // schedule inside the AdamW step (see `get_current_learning_rate`). Do not
            // add competing schedulers here — stacking several LR mutators was a source
            // of the erratic effective LR and loss oscillation.
        }

        if epoch % 10 == 0 {
            save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
        }

        update_k_mean_clusters(transformer_network, epoch);

        let train_loss = if total_valid_target_tokens_epoch > 0 {
            total_loss.re / r(total_valid_target_tokens_epoch as f64)
        } else {
            total_loss.re
        };

        if total_loss_exp_ma == ZERO && epoch == 0 {
            total_loss_exp_ma = train_loss;
        }

        total_loss_exp_ma = alpha * train_loss + (r(1.0) - alpha) * total_loss_exp_ma;

        if epoch % 5 == 0 || train_loss <= loss_threshold {
            println!("Epoch: {}, TRAINING LOSS: {}", epoch.to_string().blue().bold(), train_loss.to_string().red().bold());
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
                    println!("â›” Early stopping triggered! No improvement for {} epochs.", patience);
                    println!("Best validation loss: {} at epoch {}", best_val_loss, best_epoch);
                    save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
                    break 'outer;
                }
            }
        }
        // ============================================================

        if total_valid_target_tokens_epoch == 0 {
            println!("WARNING: epoch {} had 0 valid target tokens across all batches; skipping early-stop on loss threshold.", epoch);
        } else if train_loss <= loss_threshold {
            println!("loss is smaller than {loss_threshold} Break the training: {:?}", &train_loss);
            save_to_sled(SLED_DB_TRANSFORMER_V1, &transformer_network);
            break 'outer;
        }
    }
}

pub fn predict_token_by_token(transformer_network: &mut NeuralNetwork, input_batch: &Vec<String>) -> Vec<String> {
    let mut current_input_batch: Vec<String> = extend_input_with_bos(input_batch);

    clear_network_caches(transformer_network);

    // Option A inference: ensure `<sep>` is present at the end of the prompt.
    // Training conditions the model to start generating AFTER `<sep>`.
    for s in &mut current_input_batch {
        if !s.trim_end().ends_with("<sep>") {
            s.push_str(" <sep>");
        }
    }
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
            let last_n = if SPARSE_WINDOW_SIZE != 0 { SPARSE_WINDOW_SIZE * 2 } else { 1 };

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
    let mut output_batch: Option<Vec<Vec<Vec<C>>>> = None;
    let mut output_batch_real: Option<Vec<Vec<Vec<Real>>>> = Some(Vec::new());
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
    let mut output_batch_rm: Option<Vec<RowMajorMatrix<C>>> = None;

    let layers_len = transformer_network.layers.len();
    for layer_idx in 0..layers_len {
        // Only keep RM activations when the next layer can consume RM directly.
        // Vec-only layers (e.g. Norm(Vec), PositionalEncoding(Vec), SparseLinear(Vec)) require RM->Vec conversion,
        // which is forbidden in rm_strict mode.

        let layer = transformer_network.layers.get_mut(layer_idx).expect("layer index");
        match layer {
            LayerEnum::AdaptiveAvgPool1d(adaptive_avg_pooling_layer) => {
                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Attention layer");
                        continue;
                    }
                };

                layer_input.set_input_batch(previous_output);
                let mut layer_output: LayerOutput = adaptive_avg_pooling_layer.forward(&layer_input);
                output_batch = layer_output.take_output_batch();
            }
            LayerEnum::Embedding(embedding_layer) => {
                layer_input.set_batch_ids(batch_ids.clone());

                let (embeddings, padding_m) = embedding_layer.forward_inner(&layer_input);
                output_batch = Some(embeddings);

                padding_mask = Some(padding_m.clone());
                layer_input.set_padding_mask_batch(padding_m);
            }
            LayerEnum::EmbeddingRm(embedding_layer) => {
                layer_input.set_batch_ids(batch_ids.clone());

                let (embeddings_rm, padding_m) = embedding_layer.forward_inner(&layer_input);
                padding_mask = Some(padding_m.clone());
                layer_input.set_padding_mask_batch(padding_m);

                output_batch_rm = Some(embeddings_rm);
            }
            LayerEnum::PositionalEncoding(positional_encoding_layer) => {
                if VERBOSE {
                    println!("forward pos encoding (vec)");
                }
                let start = Instant::now();

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Attention layer");
                        continue;
                    }
                };

                layer_input.set_input_batch(previous_output);
                let positional_encodings: Vec<Vec<Vec<C>>> = positional_encoding_layer.forward(&layer_input).get_output_batch();
                output_batch = Some(positional_encodings);

                if VERBOSE {
                    println!("time elapsed in seconds in positional encoding: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::PositionalEncodingRm(positional_encoding_layer) => {
                if VERBOSE {
                    println!("forward pos encoding (rm)");
                }
                let start = Instant::now();

                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(output_batch_rm.take().unwrap());

                let enc_rm = positional_encoding_layer.forward_inner(&layer_input);
                output_batch_rm = Some(enc_rm);

                if VERBOSE {
                    println!("time elapsed in seconds in positional encoding rm: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::Norm(norm_layer) => {
                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Attention layer");
                        continue;
                    }
                };

                layer_input.set_input_batch(previous_output);

                let mut norm_output = norm_layer.forward(&layer_input);
                output_batch = norm_output.take_output_batch();
            }
            LayerEnum::NormRm(norm_layer) => {
                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(output_batch_rm.take().unwrap());

                let mut norm_output = norm_layer.forward(&layer_input);
                output_batch_rm = norm_output.take_output_batch_rm();
            }
            LayerEnum::SelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(padding_m) = &padding_mask {
                    let previous_output = match output_batch.take() {
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
                    output_batch = output_attention.take_output_batch();
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SparseSelfAttention(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(padding_m) = &padding_mask {
                    let previous_output = match output_batch.take() {
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
                    output_batch = output_attention.take_output_batch();
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SparseSelfAttentionRm(attention) => {
                if let Some(padding_m) = &padding_mask {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_batch_rm.take().unwrap());
                    layer_input.set_padding_mask_batch(padding_m.clone());

                    if VERBOSE {
                        println!("forward sparse self-attention start (rm)");
                    }

                    let start = Instant::now();
                    let mut output_attention = attention.forward(&layer_input);
                    output_batch_rm = output_attention.take_output_batch_rm();

                    if VERBOSE {
                        println!("time elapsed in seconds in sparse self attention layer (rm): {:?}", start.elapsed().as_secs_f64());
                    }
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttentionApproximation(attention) => {
                // Ensure there's an output from the previous layer before forwarding
                if let Some(padding_m) = &padding_mask {
                    let previous_output: Vec<Vec<Vec<C>>> = match output_batch.take() {
                        Some(v) => v,
                        None => {
                            println!("No previous output for Attention layer");
                            continue;
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

                    output_batch = output_attention.take_output_batch();
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::SelfAttentionApproximationRm(attention) => {
                if let Some(padding_m) = &padding_mask {
                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(output_batch_rm.take().unwrap());
                    layer_input.set_padding_mask_batch(padding_m.clone());

                    if VERBOSE {
                        println!("forward self-attention approximation start (rm)");
                    }
                    let start = Instant::now();
                    let mut output_attention = attention.forward(&layer_input);

                    if VERBOSE {
                        println!("time elapsed in seconds in self attention approximation layer (rm): {:?}", start.elapsed().as_secs_f64());
                    }

                    output_batch_rm = output_attention.take_output_batch_rm();
                } else {
                    println!("No previous output for Attention layer");
                }
            }
            LayerEnum::FeedForward(dense_layer) => {
                if VERBOSE {
                    println!("forward feed-forward network start");
                }

                let start = Instant::now();

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Dense layer");
                        continue;
                    }
                };
                layer_input.set_input_batch(previous_output);
                layer_input.set_padding_mask_batch(padding_mask.clone().unwrap());

                let mut layer_output = dense_layer.forward(&layer_input);
                output_batch = layer_output.take_output_batch();

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

                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(output_batch_rm.take().unwrap());

                let mut layer_output = ffn_layer.forward(&layer_input);
                output_batch_rm = layer_output.take_output_batch_rm();

                if VERBOSE {
                    println!("time elapsed in seconds in ffn_rm layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::Linear(linear_layer) => {
                if VERBOSE {
                    println!("forward linear start");
                }

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Dense layer");
                        continue;
                    }
                };
                layer_input.set_input_batch(previous_output);

                let start = Instant::now();
                let mut output_linear = linear_layer.forward(&layer_input);

                linear_output_indices = output_linear.get_output_indices();
                output_batch = output_linear.take_output_batch();

                if VERBOSE {
                    println!("time elapsed in seconds in linear layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::ClusteringLinear(clustering_linear_layer) => {
                if VERBOSE {
                    println!("forward clustering linear start");
                }

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for ClusteringLinear(Vec) layer");
                        continue;
                    }
                };
                layer_input.set_input_batch(previous_output);

                let start = Instant::now();
                let mut output_linear = clustering_linear_layer.forward(&layer_input);

                linear_output_indices = output_linear.get_output_indices();
                output_batch = output_linear.take_output_batch();

                if VERBOSE {
                    println!("time elapsed in seconds in clustering linear layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::AdaptiveLinear(adaptive_linear_layer) => {
                if VERBOSE {
                    println!("forward adaptive linear start");
                }

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for AdaptiveLinear layer");
                        continue;
                    }
                };
                layer_input.set_input_batch(previous_output);

                let start = Instant::now();
                let mut output_linear = adaptive_linear_layer.forward(&layer_input);
                linear_output_indices = output_linear.get_output_indices();
                if !forward_only {
                    let adaptive_loss_batch = output_linear.get_cross_entropy_loss_batch();
                    if !adaptive_loss_batch.is_empty() {
                        layer_output.set_cross_entropy_loss_batch(adaptive_loss_batch);
                    }
                } else {
                    let adaptive_logits = output_linear.get_output_batch();
                    output_batch_real = Some(complex_batch_to_real_batch(&adaptive_logits));
                }
                output_batch = output_linear.take_output_batch();

                if VERBOSE {
                    println!("time elapsed in seconds in adaptive linear layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::Eml(eml_linear_layer) => {
                if VERBOSE {
                    println!("forward eml linear start");
                }

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Eml layer");
                        continue;
                    }
                };
                layer_input.set_input_batch(previous_output.clone());

                let start = Instant::now();
                let mut output_linear = eml_linear_layer.forward(&layer_input);
                if !forward_only {
                    let eml_loss_batch = output_linear.get_cross_entropy_loss_batch();
                    if !eml_loss_batch.is_empty() {
                        layer_output.set_cross_entropy_loss_batch(eml_loss_batch);
                    }
                    // Terminal teacher-forced head on the training path: no dense logits, so
                    // pass the hidden states through unchanged for any downstream consumer.
                    output_batch = Some(previous_output);
                } else {
                    // Inference: the head emits sparse top-k token scores + indices, exactly
                    // like AdaptiveLinear, so the greedy decoder can argmax and map to ids.
                    linear_output_indices = output_linear.get_output_indices();
                    let eml_logits = output_linear.get_output_batch();
                    output_batch_real = Some(complex_batch_to_real_batch(&eml_logits));
                    output_batch = output_linear.take_output_batch();
                }

                if VERBOSE {
                    println!("time elapsed in seconds in eml linear layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                if VERBOSE {
                    println!("forward sparse linear_rm start");
                }

                let rm_in = match output_batch_rm.take() {
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
                output_batch_rm = output_linear.take_output_batch_rm();

                if VERBOSE {
                    println!("time elapsed in seconds in sparse linear_rm layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                if VERBOSE {
                    println!("forward multilinear start");
                }

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        panic!("No previous output for Multilinear layer");
                    }
                };
                layer_input.set_input_batch(previous_output);

                let start = Instant::now();
                let mut output_linear = multi_linear_layer.forward(&layer_input);
                output_batch = output_linear.take_output_batch();

                if VERBOSE {
                    println!("time elapsed in seconds in multilayer layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::Wavelet(wavelet_layer) => {
                if output_batch.is_none() {
                    panic!("No previous output for Wavelet layer");
                }

                let previous_output = output_batch.take().unwrap();
                layer_input.set_input_batch(previous_output);

                let start = Instant::now();
                let mut output_cwt = wavelet_layer.forward(&layer_input);
                output_batch = output_cwt.take_output_batch();

                if VERBOSE {
                    println!("time elapsed in seconds in wavelet layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::WaveletRm(wavelet_layer) => {
                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(output_batch_rm.take().unwrap());

                let start = Instant::now();
                let mut output_cwt = wavelet_layer.forward(&layer_input);
                output_batch_rm = output_cwt.take_output_batch_rm();

                if VERBOSE {
                    println!("time elapsed in seconds in wavelet rm layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                if output_batch.is_none() {
                    panic!("No previous output for DiscreteWavelet layer");
                }

                let previous_output = output_batch.take().unwrap();
                layer_input.set_input_batch(previous_output);

                let start = Instant::now();
                let mut output_dwt = wavelet_layer.forward(&layer_input);

                let new_padding_mask = output_dwt.get_padding_mask_batch();
                padding_mask = Some(new_padding_mask.clone());
                layer_input.set_padding_mask_batch(new_padding_mask);
                output_batch = output_dwt.take_output_batch();

                if VERBOSE {
                    println!("time elapsed in seconds in discrete wavelet layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::DiscreteWaveletRm(wavelet_layer) => {
                if output_batch_rm.is_none() {
                    panic!("No previous output for DiscreteWaveletRm layer");
                }

                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(output_batch_rm.take().unwrap());

                let start = Instant::now();
                let mut output_dwt = wavelet_layer.forward(&layer_input);

                let new_padding_mask = output_dwt.get_padding_mask_batch();
                padding_mask = Some(new_padding_mask.clone());
                layer_input.set_padding_mask_batch(new_padding_mask);
                output_batch_rm = output_dwt.take_output_batch_rm();

                if VERBOSE {
                    println!("time elapsed in seconds in discrete wavelet rm layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::ComplexToLinear(ctl_layer) => {
                if output_batch.is_none() {
                    panic!("No previous output for ComplexToLinear layer");
                }

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Complex to Linear layer");
                        continue;
                    }
                };
                layer_input.set_input_batch(previous_output);

                let start = Instant::now();
                let mut output_ctl = ctl_layer.forward(&layer_input);
                output_batch = output_ctl.take_output_batch();

                if VERBOSE {
                    println!("time elapsed in seconds in complex to linear layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::ComplexToLinearRm(ctl_layer) => {
                if output_batch_rm.is_none() {
                    panic!("No previous output for ComplexToLinearRm layer");
                }

                // Prefer RM-only path when available.
                layer_input.clear_input_batch();
                layer_input.set_input_batch_rm(output_batch_rm.take().unwrap());

                let start = Instant::now();
                let mut output_ctl = ctl_layer.forward(&layer_input);
                if VERBOSE {
                    println!("time elapsed in seconds in complex to linear layer: {:?}", start.elapsed().as_secs_f64());
                }

                output_batch_rm = output_ctl.take_output_batch_rm();
            }
            LayerEnum::Softmax(softmax_layer) => {
                let start = Instant::now();

                if !forward_only && !layer_output.get_cross_entropy_loss_batch().is_empty() {
                    if VERBOSE {
                        println!("skipping standalone softmax because adaptive output layer already produced CE loss");
                    }
                    continue;
                }

                let previous_output = match output_batch.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous output for Softmax(Vec) layer");
                        continue;
                    }
                };

                if !forward_only {
                    // Training always requires CE loss + gradients; ensure Softmax is in TRAINING
                    // even if a prior inference call switched it to PRODUCTION.
                    softmax_layer.operation_mode = OperationMode::TRAINING;

                    layer_input.set_input_batch(previous_output);
                    layer_input.set_output_indices(linear_output_indices.clone());

                    let _softmax_result: Vec<Vec<Vec<C>>> = softmax_layer.forward_inner(&layer_input, padding_mask.clone(), target_batch_ids_option.clone());
                    layer_output.set_cross_entropy_loss_batch(softmax_layer.cross_entropy_loss_batch.clone().unwrap());
                } else {
                    output_batch_real = Some(complex_batch_to_real_batch(&previous_output));
                }

                if VERBOSE {
                    println!("time elapsed in seconds in softmax layer: {:?}", start.elapsed().as_secs_f64());
                }
            }
            LayerEnum::SoftmaxRm(softmax_layer) => {
                let start = Instant::now();

                if !forward_only && !layer_output.get_cross_entropy_loss_batch().is_empty() {
                    if VERBOSE {
                        println!("skipping standalone softmax_rm because adaptive output layer already produced CE loss");
                    }
                    continue;
                }

                let logits_rm = match output_batch_rm.take() {
                    Some(v) => v,
                    None => {
                        println!("No previous RM output for SoftmaxRm layer");
                        continue;
                    }
                };

                if !forward_only {
                    softmax_layer.operation_mode = OperationMode::TRAINING;

                    layer_input.clear_input_batch();
                    layer_input.set_input_batch_rm(logits_rm);

                    layer_input.set_output_indices(linear_output_indices.clone());
                    let _softmax_result: Vec<Vec<Vec<C>>> = softmax_layer.forward_inner(&layer_input, padding_mask.clone(), target_batch_ids_option.clone());
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

    layer_output.set_output_batch_real(output_batch_real.unwrap());
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
                    let gradient_batch: Gradient = embedding_layer.backward_inner(&grad_vec);
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
                    let gr_rm = previous_gradient.get_gradient_input_batch_rm_ref().filter(|g| !g.is_empty()).expect("EmbeddingRm expects RM gradients");
                    let gradient_batch: Gradient = embedding_layer.backward_inner(gr_rm);
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

                    let gradient_batch: Gradient = positional_encoding_layer.backward(&previous_gradient);

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

                    let gradient_batch: Gradient = positional_encoding_layer.backward_inner(&gr_rm);

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
                    let gradient_batch: Gradient = attention_layer.backward(&previous_gradient);

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
                    let gradient_batch: Gradient = attention_layer.backward(&previous_gradient);
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
                        let previous_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient.get_gradient_input_batch().iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
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
                        let previous_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient.get_gradient_input_batch().iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
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
                    let gradient_batch: Gradient = dense_layer.backward(&previous_gradient);
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

                    let gradient_batch: Gradient = ffn_layer.backward(&previous_gradient);
                    
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
            LayerEnum::ClusteringLinear(sparse_linear_layer) => {
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
            LayerEnum::AdaptiveLinear(adaptive_linear_layer) => {
                if let Some(previous_gradient) = gradient {
                    let start = Instant::now();
                    let gradient_batch: Gradient = adaptive_linear_layer.backward(&previous_gradient);
                    gradient = Some(gradient_batch);

                    if VERBOSE {
                        println!("time elapsed in seconds in adaptive linear layer backward: {:?}", start.elapsed().as_secs_f64());
                    }
                } else if let Some(stored_gradient) = adaptive_linear_layer.gradient.as_ref() {
                    gradient = Some(stored_gradient.clone());
                } else {
                    println!("No previous gradient in Adaptive Linear Layer");
                }
            }
            LayerEnum::Eml(eml_linear_layer) => {
                // Terminal head: the gradient was computed during the training forward.
                if let Some(stored_gradient) = eml_linear_layer.gradient.as_ref() {
                    gradient = Some(stored_gradient.clone());
                } else {
                    println!("No stored gradient in Eml Linear Layer");
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
                        let previous_gradient_batch: Vec<RowMajorMatrix<C>> = previous_gradient.get_gradient_input_batch().iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
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
                        let previous_gradient_batch: Vec<RowMajorMatrix<C>> = previous_gradient.get_gradient_input_batch().iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
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
    let mut total_valid_target_tokens: usize = 0;

    // Get validation batches (NOT shuffled)
    let val_batches = dataset.get_validation_batches(batch_size);

    if val_batches.is_empty() {
        println!("âš ï¸  Warning: No validation data available! Validation loss will be unreliable.");
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

        let mut target_ids: Vec<Vec<u32>> = shift_targets_for_next_token_prediction(&target_ids);
        let mut batch_ids: Vec<Vec<u32>> = shift_inputs_for_next_token_prediction(&batch_ids);

        let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap_or(0);

        if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
            let (input_batch_ids, target_batch_ids) =
                split_shifted_batch_with_targets(&batch_ids, &target_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
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
        let valid_target_tokens_batch: usize = target_ids.iter().map(|seq| seq.iter().filter(|&&id| id != 1).count()).sum();
        layer_input.set_total_valid_tokens(valid_target_tokens_batch.max(1));

        transformer_network.minibatch_size = actual_batch_size; // Update network's batch size too

        // Forward pass only
        let network_output = predict(transformer_network, &layer_input);
        let loss: C = cross_entropy_sum_batch(&network_output.get_cross_entropy_loss_batch(), &target_ids);

        total_val_loss += loss.re;
        total_valid_target_tokens += valid_target_tokens_batch;
    }

    if total_valid_target_tokens == 0 {
        println!("âš ï¸  Warning: No validation batches processed!");
        return Real::INFINITY;
    }

    let avg_val_loss = total_val_loss / r(total_valid_target_tokens as f64);

    // Debug: print validation statistics on first epoch
    if transformer_network.time_step < 100 {
        println!(
            "ðŸ“Š Validation: {} valid tokens, total loss: {:.4}, avg loss/token: {:.4}",
            total_valid_target_tokens, total_val_loss, avg_val_loss
        );
    }

    avg_val_loss
}

/// Evaluate model on TEST set - call this ONLY ONCE after training is complete
pub fn evaluate_test(transformer_network: &mut NeuralNetwork, dataset: &Dataset<String, String>, batch_size: usize) -> Real {
    // Clear all caches before test evaluation (important for correct batch size handling)
    clear_network_caches(transformer_network);

    let mut total_test_loss: Real = ZERO;
    let mut total_valid_target_tokens: usize = 0;
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

        let mut target_ids: Vec<Vec<u32>> = shift_targets_for_next_token_prediction(&target_ids);
        let mut batch_ids: Vec<Vec<u32>> = shift_inputs_for_next_token_prediction(&batch_ids);

        let max_seq_len: usize = batch_ids.iter().map(|v| v.len()).max().unwrap_or(0);

        if max_seq_len > MAX_CONTEXT_WINDOW_SIZE {
            let (input_batch_ids, target_batch_ids) =
                split_shifted_batch_with_targets(&batch_ids, &target_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);
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
        let valid_target_tokens_batch: usize = target_ids.iter().map(|seq| seq.iter().filter(|&&id| id != 1).count()).sum();
        layer_input.set_total_valid_tokens(valid_target_tokens_batch.max(1));

        transformer_network.minibatch_size = actual_batch_size; // Update network's batch size too

        // Forward pass only
        let network_output = predict(transformer_network, &layer_input);
        let loss: C = cross_entropy_sum_batch(&network_output.get_cross_entropy_loss_batch(), &target_ids);

        total_test_loss += loss.re;
        total_valid_target_tokens += valid_target_tokens_batch;
    }

    if total_valid_target_tokens == 0 {
        println!("Warning: No valid target tokens in test set");
        return Real::INFINITY;
    }

    let avg_test_loss = total_test_loss / r(total_valid_target_tokens as f64);

    println!("\n{}", "FINAL TEST LOSS:".bright_cyan().bold());
    println!("{}", avg_test_loss.to_string().bright_green().bold());
    println!("{}\n", "=".repeat(60).bright_cyan());

    avg_test_loss
}

#[cfg(test)]
mod tests {
    use super::split_shifted_batch_with_targets;

    #[test]
    fn sliding_window_keeps_targets_aligned_to_chunk_suffix() {
        let batch_ids = vec![vec![10, 11, 12, 13, 14, 20, 21, 22, 23]];
        let target_ids = vec![vec![19, 20, 21, 22, 23]];

        let (input_chunks, target_chunks) = split_shifted_batch_with_targets(&batch_ids, &target_ids, 5, 3);

        assert_eq!(
            input_chunks,
            vec![
                vec![10, 11, 12, 13, 14],
                vec![13, 14, 20, 21, 22],
                vec![14, 20, 21, 22, 23],
            ]
        );
        assert_eq!(target_chunks, vec![vec![19], vec![19, 20, 21, 22], vec![19, 20, 21, 22, 23]]);
    }
}
