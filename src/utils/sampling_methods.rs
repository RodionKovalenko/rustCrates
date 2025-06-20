use rand::prelude::*;
use rand_distr::weighted::WeightedIndex;
use std::cmp::Ordering;

pub fn greedy_decoding(predictions: &Vec<Vec<Vec<f64>>>) -> Vec<Vec<u32>> {
    predictions
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    let max_index = seq.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0);
                    max_index as u32
                })
                .collect()
        })
        .collect()
}

pub fn temperature_sampling(predictions: &Vec<Vec<Vec<f64>>>, temperature: f64) -> Vec<Vec<u32>> {
    predictions
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    let probs: Vec<f64> = seq.iter().map(|&val| (val / temperature).exp()).collect();
                    let sum_probs: f64 = probs.iter().sum();
                    let normalized_probs: Vec<f64> = probs.iter().map(|&prob| prob / sum_probs).collect();
                    sample_from_distribution(&normalized_probs)
                })
                .collect()
        })
        .collect()
}

fn _top_k_sampling(predicted_softmax_batch: &Vec<Vec<Vec<f64>>>, k: usize, temperature: f64) -> Vec<Vec<u32>> {
    predicted_softmax_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    let scaled_probs: Vec<f64> = seq.iter().map(|&val| (val / temperature).exp()).collect();

                    let mut top_k_indices: Vec<usize> = (0..scaled_probs.len()).collect();
                    top_k_indices.sort_by(|&i, &j| scaled_probs[j].partial_cmp(&scaled_probs[i]).unwrap());

                    let top_k_indices = top_k_indices.into_iter().take(k).collect::<Vec<_>>();
                    let selected_token = top_k_indices[rand::rng().random_range(0..k)];

                    selected_token as u32
                })
                .collect()
        })
        .collect()
}

fn _top_p_sampling(predicted_softmax_batch: &Vec<Vec<Vec<f64>>>, p: f64, temperature: f64) -> Vec<Vec<u32>> {
    predicted_softmax_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    let scaled_probs: Vec<f64> = seq.iter().map(|&val| (val / temperature).exp()).collect();

                    let mut token_probs: Vec<(usize, f64)> = scaled_probs.iter().enumerate().map(|(idx, &prob)| (idx, prob)).collect();
                    token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    let mut cumulative_prob = 0.0;
                    let mut selected_tokens = Vec::new();
                    for (idx, prob) in token_probs.iter() {
                        cumulative_prob += *prob;
                        selected_tokens.push(*idx);
                        if cumulative_prob >= p {
                            break;
                        }
                    }

                    let selected_token = selected_tokens[rand::rng().random_range(0..selected_tokens.len())];
                    selected_token as u32
                })
                .collect()
        })
        .collect()
}

fn sample_from_distribution(probs: &Vec<f64>) -> u32 {
    let mut rng = rand::rng();
    let random_value: f64 = rng.random();

    let mut cumulative_sum = 0.0;
    for (idx, &prob) in probs.iter().enumerate() {
        cumulative_sum += prob;
        if random_value < cumulative_sum {
            return idx as u32;
        }
    }

    probs.len() as u32 - 1
}

pub fn get_target_predictions(predicted_softmax_batch: &Vec<Vec<Vec<f64>>>, target_ids: &Vec<Vec<u32>>, padding_mask_batch: &Vec<Vec<u32>>) -> Vec<Vec<Vec<f64>>> {
    predicted_softmax_batch
        .iter()
        .zip(target_ids.clone())
        .enumerate()
        .filter_map(|(batch_ind, (input_seq, target_seq))| {
            let target_len = target_seq.len();
            let mut valid_seq_opt = None;
            let padding_mask = &padding_mask_batch[batch_ind];

            let mut _sequence_len_unpadded: usize = 0;
            for padding in padding_mask.iter() {
                if *padding != 0 {
                    _sequence_len_unpadded += 1;
                }
            }

            // let ind_end = _sequence_len_unpadded - target_len;
            let ind_end = input_seq.len() - target_len;

            // Slide backwards to find a valid window of length `target_len`
            for offset in (0..=ind_end).rev() {
                let window = &input_seq[offset..offset + target_len];
                let max = window.iter().flat_map(|w| w.iter()).max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Less));

                if let Some(max_num) = max {
                    if max_num.is_finite() {
                        valid_seq_opt = Some(window.to_vec());
                        break;
                    }
                }
            }

            valid_seq_opt
        })
        .collect()
}

pub fn top_p_sampling_from_softmax(probs_batch: &Vec<Vec<Vec<f64>>>, p: f64) -> Vec<Vec<u32>> {
    let mut rng = rand::rng(); // ✅ This is correct and compatible

    probs_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|probs| {
                    // Sort probabilities in descending order with their indices
                    let mut token_probs: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
                    token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    // Select top-p tokens
                    let mut cumulative = 0.0;
                    let mut top_p_tokens = Vec::new();
                    for (idx, prob) in token_probs {
                        cumulative += prob;
                        top_p_tokens.push((idx, prob));
                        if cumulative >= p {
                            break;
                        }
                    }

                    // Normalize the selected top-p probabilities
                    let prob_sum: f64 = top_p_tokens.iter().map(|&(_, p)| p).sum();
                    let normalized = top_p_tokens.iter().map(|&(idx, prob)| (idx as u32, prob / prob_sum)).collect::<Vec<_>>();

                    // Sample using weighted distribution
                    let weights = normalized.iter().map(|&(_, p)| p);
                    let dist = WeightedIndex::new(weights).unwrap();
                    let sampled_idx = dist.sample(&mut rng);
                    normalized[sampled_idx].0
                })
                .collect()
        })
        .collect()
}
