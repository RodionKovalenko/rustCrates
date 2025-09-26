use num::Complex;

use crate::neural_networks::network_components::adaptive_pooling::adaptive_avg_pool1d_layer::{CompressionMetadata, CompressionType};

// ContentAwarePooling struct and implementation
pub struct ContentAwarePoolingLayer {
    pub target_length: usize,
}

impl ContentAwarePoolingLayer {
    pub fn new(target_length: usize) -> Self {
        Self { target_length }
    }

    pub fn forward(&self, input: &Vec<Vec<Vec<Complex<f64>>>>) -> (Vec<Vec<Vec<Complex<f64>>>>, CompressionMetadata) {
        self.compress(input)
    }

    pub fn compress(&self, input: &Vec<Vec<Vec<Complex<f64>>>>) -> (Vec<Vec<Vec<Complex<f64>>>>, CompressionMetadata) {
        if input.is_empty() {
            return (
                Vec::new(),
                CompressionMetadata {
                    original_length: 0,
                    compressed_length: 0,
                    window_mappings: Vec::new(),
                    compression_type: CompressionType::ContentAware,
                },
            );
        }
        let seq_len = input[0].len();
        if seq_len <= self.target_length {
            return (
                input.clone(),
                CompressionMetadata {
                    original_length: seq_len,
                    compressed_length: seq_len,
                    window_mappings: (0..seq_len).map(|i| vec![i]).collect(),
                    compression_type: CompressionType::ContentAware,
                },
            );
        }

        // Calculate window mappings based on importance scores (first batch)
        let mut window_mappings = Vec::new();
        if !input.is_empty() {
            let importance_scores = self.calculate_importance_scores(&input[0]);
            window_mappings = self.create_importance_based_windows_distribution(&importance_scores, seq_len);
        }

        let mut output = Vec::with_capacity(input.len());
        for seq in input.iter() {
            let compressed_seq = self.compress_with_importance_and_windows(seq, &window_mappings);
            output.push(compressed_seq);
        }

        (
            output,
            CompressionMetadata {
                original_length: seq_len,
                compressed_length: self.target_length,
                window_mappings,
                compression_type: CompressionType::ContentAware,
            },
        )
    }

    pub fn decompress(&self, compressed: &Vec<Vec<Vec<Complex<f64>>>>, metadata: &CompressionMetadata) -> Vec<Vec<Vec<Complex<f64>>>> {
        if compressed.is_empty() || metadata.original_length == 0 {
            return Vec::new();
        }
        let batch_size = compressed.len();
        let hidden_dim = if compressed[0].len() > 0 { compressed[0][0].len() } else { 0 };
        let mut decompressed = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let batch_seq = &compressed[batch_idx];
            decompressed.push(self.decompress_sequence_content_aware(batch_seq, metadata.original_length, &metadata.window_mappings, hidden_dim));
        }
        decompressed
    }

    fn compress_with_importance_and_windows(&self, sequence: &Vec<Vec<Complex<f64>>>, window_mappings: &[Vec<usize>]) -> Vec<Vec<Complex<f64>>> {
        let seq_len = sequence.len();
        let hidden_dim = if seq_len > 0 { sequence[0].len() } else { 0 };
        let importance_scores = self.calculate_importance_scores(sequence);
        let mut compressed = Vec::with_capacity(self.target_length);

        for window in window_mappings.iter() {
            let mut pooled_token = vec![Complex::new(0.0, 0.0); hidden_dim];
            let mut total_importance = 0.0;
            for &pos in window.iter() {
                if pos < seq_len && pos < importance_scores.len() {
                    let importance = importance_scores[pos];
                    total_importance += importance;
                    for dim in 0..hidden_dim {
                        pooled_token[dim] += sequence[pos][dim] * importance;
                    }
                }
            }
            if total_importance > 0.0 {
                for dim in 0..hidden_dim {
                    pooled_token[dim] /= total_importance;
                }
            }
            compressed.push(pooled_token);
        }
        compressed
    }

    fn decompress_sequence_content_aware(&self, compressed: &Vec<Vec<Complex<f64>>>, original_length: usize, window_mappings: &[Vec<usize>], hidden_dim: usize) -> Vec<Vec<Complex<f64>>> {
        let mut decompressed = vec![vec![Complex::new(0.0, 0.0); hidden_dim]; original_length];
        for (comp_idx, window) in window_mappings.iter().enumerate() {
            if comp_idx >= compressed.len() {
                break;
            }
            let token = &compressed[comp_idx];
            for &pos in window.iter() {
                if pos < original_length {
                    decompressed[pos] = token.clone();
                }
            }
        }
        decompressed
    }

    fn calculate_importance_scores(&self, sequence: &Vec<Vec<Complex<f64>>>) -> Vec<f64> {
        sequence
            .iter()
            .map(|token| {
                let total_mag: f64 = token.iter().map(|c| c.norm()).sum();
                total_mag / token.len() as f64
            })
            .collect()
    }

    fn create_importance_based_windows_distribution(&self, importance_scores: &[f64], seq_len: usize) -> Vec<Vec<usize>> {
        let mut windows = Vec::new();
        if seq_len == 0 || importance_scores.is_empty() {
            return windows;
        }
        let total_importance: f64 = importance_scores.iter().sum();
        if total_importance == 0.0 {
            let base_len = seq_len / self.target_length;
            let rem = seq_len % self.target_length;
            let mut pos = 0;
            for i in 0..self.target_length {
                let sz = if i < rem { base_len + 1 } else { base_len };
                let window: Vec<usize> = (pos..pos + sz).collect();
                pos += sz;
                windows.push(window);
            }
            return windows;
        }
        let target_imp = total_importance / self.target_length as f64;
        let mut current_win = Vec::new();
        let mut current_imp = 0.0;
        let mut created = 0;
        for (pos, &imp) in importance_scores.iter().enumerate() {
            current_win.push(pos);
            current_imp += imp;
            if current_imp >= target_imp || pos == seq_len - 1 || created == self.target_length - 1 {
                windows.push(current_win.clone());
                current_win.clear();
                current_imp = 0.0;
                created += 1;
                if created >= self.target_length {
                    break;
                }
            }
        }
        if !current_win.is_empty() && created < self.target_length {
            windows.push(current_win);
        }
        while windows.len() < self.target_length && !windows.is_empty() {
            let (largest_idx, _) = windows.iter().enumerate().max_by_key(|(_, w)| w.len()).unwrap();
            let largest = windows.remove(largest_idx);
            let mid = largest.len() / 2;
            if mid > 0 {
                let (first, second) = largest.split_at(mid);
                windows.insert(largest_idx, first.to_vec());
                windows.insert(largest_idx + 1, second.to_vec());
            } else {
                windows.insert(largest_idx, largest);
                break;
            }
        }
        while windows.len() > self.target_length && windows.len() > 1 {
            let mut min_sum = f64::INFINITY;
            let mut min_idx = 0;
            for i in 0..windows.len() - 1 {
                let s: f64 = windows[i].iter().chain(windows[i + 1].iter()).map(|&pos| importance_scores[pos]).sum();
                if s < min_sum {
                    min_sum = s;
                    min_idx = i;
                }
            }
            let mut merged = windows.remove(min_idx);
            merged.extend(windows.remove(min_idx));
            windows.insert(min_idx, merged);
        }
        windows
    }
}
