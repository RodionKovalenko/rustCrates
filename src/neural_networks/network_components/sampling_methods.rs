use num::Complex;
use rand::seq::IndexedRandom;
use rand::Rng;

pub fn greedy_decoding(predictions: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<u32>> {
    predictions
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    // Choose the most probable token (maximum real part of complex value)
                    let max_index = seq.iter().enumerate().max_by(|(_, a), (_, b)| a.re.partial_cmp(&b.re).unwrap()).map(|(i, _)| i).unwrap_or(0);
                    max_index as u32
                })
                .collect()
        })
        .collect()
}

pub fn temperature_sampling(predictions: &Vec<Vec<Vec<Complex<f64>>>>, temperature: f64) -> Vec<Vec<u32>> {
    predictions
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    let probs: Vec<f64> = seq.iter().map(|&complex| (complex.re / temperature).exp()).collect();
                    let sum_probs: f64 = probs.iter().sum();
                    let normalized_probs: Vec<f64> = probs.iter().map(|&prob| prob / sum_probs).collect();
                    sample_from_distribution(&normalized_probs)
                })
                .collect()
        })
        .collect()
}

fn _top_k_sampling(predicted_softmax_batch: &Vec<Vec<Vec<Complex<f64>>>>, k: usize, temperature: f64) -> Vec<Vec<u32>> {
    predicted_softmax_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    // Apply temperature scaling
                    let scaled_probs: Vec<f64> = seq.iter().map(|&complex| (complex.re / temperature).exp()).collect();

                    // Get the top-k token indices based on scaled probabilities
                    let mut top_k_indices: Vec<usize> = (0..scaled_probs.len()).collect();
                    top_k_indices.sort_by(|&i, &j| scaled_probs[j].partial_cmp(&scaled_probs[i]).unwrap());

                    // Select one token from the top-k tokens using a random sample
                    let top_k_indices = top_k_indices.into_iter().take(k).collect::<Vec<_>>();
                    // Choose one token from the top-k tokens using a random sample
                    let selected_token = top_k_indices[rand::rng().random_range(0..k)];

                    selected_token as u32 // Convert index to token ID
                })
                .collect()
        })
        .collect()
}

fn _top_p_sampling(predicted_softmax_batch: &Vec<Vec<Vec<Complex<f64>>>>, p: f64, temperature: f64) -> Vec<Vec<u32>> {
    predicted_softmax_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    // Apply temperature scaling
                    let scaled_probs: Vec<f64> = seq.iter().map(|&complex| (complex.re / temperature).exp()).collect();

                    // Sort tokens by probability and compute cumulative probability
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

                    // Choose one token from the selected set based on probabilities
                    let selected_token = selected_tokens[rand::rng().random_range(0..selected_tokens.len())];
                    selected_token as u32
                })
                .collect()
        })
        .collect()
}

fn sample_from_distribution(probs: &Vec<f64>) -> u32 {
    let mut rng = rand::rng();
    let random_value: f64 = rng.random::<f64>(); // Random value between 0 and 1

    let mut cumulative_sum = 0.0;
    for (idx, &prob) in probs.iter().enumerate() {
        cumulative_sum += prob;
        if random_value < cumulative_sum {
            return idx as u32;
        }
    }

    // Fallback (should never happen if probabilities sum to 1)
    probs.len() as u32 - 1
}

pub fn top_p_temperature_sampling(predicted_softmax_batch: &Vec<Vec<Vec<Complex<f64>>>>, p: f64, temperature: f64) -> Vec<Vec<u32>> {
    predicted_softmax_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    // Apply temperature scaling (avoiding overflow)
                    let scaled_probs: Vec<f64> = seq.iter().map(|&complex| ((complex.re / temperature).max(-10.0).min(10.0)).exp()).collect();

                    // Sort probabilities in descending order
                    let mut token_probs: Vec<(usize, f64)> = scaled_probs.iter().enumerate().map(|(idx, &prob)| (idx, prob)).collect();

                    token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Descending order

                    // Select top-p tokens
                    let mut cumulative_prob = 0.0;
                    let mut selected_tokens = Vec::new();
                    for &(idx, prob) in token_probs.iter() {
                        cumulative_prob += prob;
                        selected_tokens.push(idx);
                        if cumulative_prob >= p {
                            break;
                        }
                    }

                    // Use `rng` to randomly pick one from selected tokens
                    let mut rng = rand::rng();
                    let selected_token = selected_tokens.choose(&mut rng).unwrap(); // Randomly choose an index from the list

                    *selected_token as u32
                })
                .collect()
        })
        .collect()
}
