use rand::seq::IndexedRandom;
use rand::Rng;

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

pub fn top_p_temperature_sampling(predicted_softmax_batch: &Vec<Vec<Vec<f64>>>, p: f64, temperature: f64) -> Vec<Vec<u32>> {
    predicted_softmax_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|seq| {
                    let scaled_probs: Vec<f64> = seq.iter().map(|&val| ((val / temperature).max(-10.0).min(10.0)).exp()).collect();

                    let mut token_probs: Vec<(usize, f64)> = scaled_probs.iter().enumerate().map(|(idx, &prob)| (idx, prob)).collect();
                    token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    let mut cumulative_prob = 0.0;
                    let mut selected_tokens = Vec::new();
                    for &(idx, prob) in token_probs.iter() {
                        cumulative_prob += prob;
                        selected_tokens.push(idx);
                        if cumulative_prob >= p {
                            break;
                        }
                    }

                    let mut rng = rand::rng();
                    let selected_token = selected_tokens.choose(&mut rng).unwrap();
                    *selected_token as u32
                })
                .collect()
        })
        .collect()
}
