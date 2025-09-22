use num_complex::Complex;

// Compression metadata for decompression
#[derive(Clone, Debug)]
pub struct CompressionMetadata {
    pub original_length: usize,
    pub compressed_length: usize,
    pub window_mappings: Vec<Vec<usize>>,
    pub compression_type: CompressionType,
}

#[derive(Clone, Debug)]
pub enum CompressionType {
    Adaptive,
    Strided { stride: usize, kernel_size: usize },
    ContentAware,
    MultiStage,
}

// AdaptiveAvgPool1d struct and implementation
pub struct AdaptiveAvgPool1d {
    pub output_size: usize,
}

impl AdaptiveAvgPool1d {
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }

    // Forward: compress input sequences
    pub fn forward(&self, input: &[Vec<Vec<Complex<f64>>>]) -> (Vec<Vec<Vec<Complex<f64>>>>, CompressionMetadata) {
        self.compress(input)
    }

    // Calculate gradient w.r.t. input given grad w.r.t. output
    pub fn backward(&self, grad_output: &[Vec<Vec<Complex<f64>>>], metadata: &CompressionMetadata) -> Vec<Vec<Vec<Complex<f64>>>> {
        let batch_size = grad_output.len();
        let original_length = metadata.original_length;
        let hidden_dim = if batch_size > 0 && !grad_output[0].is_empty() { grad_output[0][0].len() } else { 0 };
        let mut grad_input = vec![vec![vec![Complex::new(0.0, 0.0); hidden_dim]; original_length]; batch_size];
        for batch_idx in 0..batch_size {
            for (win_idx, window) in metadata.window_mappings.iter().enumerate() {
                let window_size = window.len() as f64;
                for &pos in window {
                    if pos < original_length && win_idx < grad_output[batch_idx].len() {
                        for dim in 0..hidden_dim {
                            grad_input[batch_idx][pos][dim] += grad_output[batch_idx][win_idx][dim] / window_size;
                        }
                    }
                }
            }
        }
        grad_input
    }

    // Compression implementation
    pub fn compress(&self, input: &[Vec<Vec<Complex<f64>>>]) -> (Vec<Vec<Vec<Complex<f64>>>>, CompressionMetadata) {
        if input.is_empty() {
            return (
                Vec::new(),
                CompressionMetadata {
                    original_length: 0,
                    compressed_length: 0,
                    window_mappings: Vec::new(),
                    compression_type: CompressionType::Adaptive,
                },
            );
        }

        let seq_len = input[0].len();
        for (i, seq) in input.iter().enumerate() {
            if seq.len() != seq_len {
                panic!("Batch {} sequence length {} does not match expected {}", i, seq.len(), seq_len);
            }
        }
        let hidden_dim = if seq_len > 0 && !input[0].is_empty() { input[0][0].len() } else { 0 };

        if seq_len <= self.output_size {
            let clone_input = input.to_vec();
            let window_mappings = (0..seq_len).map(|i| vec![i]).collect();
            return (
                clone_input,
                CompressionMetadata {
                    original_length: seq_len,
                    compressed_length: seq_len,
                    window_mappings,
                    compression_type: CompressionType::Adaptive,
                },
            );
        }

        let windows = self.calculate_pooling_windows(seq_len);
        let mut output = Vec::with_capacity(input.len());

        for seq in input.iter() {
            let compressed_seq = Self::compress_sequence_with_windows(seq, &windows, hidden_dim);
            output.push(compressed_seq);
        }

        (
            output,
            CompressionMetadata {
                original_length: seq_len,
                compressed_length: self.output_size,
                window_mappings: windows,
                compression_type: CompressionType::Adaptive,
            },
        )
    }

    // Decompression implementation
    pub fn decompress(&self, compressed: &[Vec<Vec<Complex<f64>>>], metadata: &CompressionMetadata) -> Vec<Vec<Vec<Complex<f64>>>> {
        if compressed.is_empty() || metadata.original_length == 0 {
            return Vec::new();
        }
        for (i, seq) in compressed.iter().enumerate() {
            if seq.len() != metadata.compressed_length {
                panic!("Batch {} compressed length {} != metadata {}", i, seq.len(), metadata.compressed_length);
            }
        }
        let hidden_dim = if !compressed[0].is_empty() { compressed[0][0].len() } else { 0 };
        let mut decompressed = Vec::with_capacity(compressed.len());

        for batch_seq in compressed.iter() {
            decompressed.push(Self::decompress_sequence(batch_seq, metadata.original_length, &metadata.window_mappings, hidden_dim));
        }
        decompressed
    }

    // Helper compress function
    fn compress_sequence_with_windows(sequence: &Vec<Vec<Complex<f64>>>, windows: &[Vec<usize>], hidden_dim: usize) -> Vec<Vec<Complex<f64>>> {
        let mut compressed = Vec::with_capacity(windows.len());
        for window in windows.iter() {
            let mut pooled_token = vec![Complex::new(0.0, 0.0); hidden_dim];
            let window_size = window.len();
            if window_size > 0 {
                for &pos in window.iter() {
                    if pos < sequence.len() {
                        let token = &sequence[pos];
                        let use_dim = usize::min(token.len(), hidden_dim);
                        for dim in 0..use_dim {
                            pooled_token[dim] += token[dim];
                        }
                    }
                }
                let divisor = window_size as f64;
                for dim in 0..hidden_dim {
                    pooled_token[dim] /= divisor;
                }
            }
            compressed.push(pooled_token);
        }
        compressed
    }

    // Helper decompress function
    fn decompress_sequence(compressed: &Vec<Vec<Complex<f64>>>, original_length: usize, window_mappings: &[Vec<usize>], hidden_dim: usize) -> Vec<Vec<Complex<f64>>> {
        let mut decompressed = vec![vec![Complex::new(0.0, 0.0); hidden_dim]; original_length];
        for (comp_idx, window) in window_mappings.iter().enumerate() {
            if comp_idx >= compressed.len() {
                break;
            }
            let token = &compressed[comp_idx];
            for &pos in window.iter() {
                if pos < original_length {
                    if token.len() == hidden_dim {
                        decompressed[pos] = token.clone();
                    } else {
                        let mut tmp = vec![Complex::new(0.0, 0.0); hidden_dim];
                        let n = usize::min(token.len(), hidden_dim);
                        for i in 0..n {
                            tmp[i] = token[i];
                        }
                        decompressed[pos] = tmp;
                    }
                }
            }
        }
        decompressed
    }

    // Calculates pooling windows for adaptive average pooling
    fn calculate_pooling_windows(&self, seq_len: usize) -> Vec<Vec<usize>> {
        let mut windows = Vec::with_capacity(self.output_size);
        let base_size = seq_len / self.output_size;
        let remainder = seq_len % self.output_size;
        let mut current_pos = 0;

        for i in 0..self.output_size {
            let window_size = if i < remainder { base_size + 1 } else { base_size };
            let mut window = Vec::with_capacity(window_size);
            for j in 0..window_size {
                if current_pos + j < seq_len {
                    window.push(current_pos + j);
                }
            }
            windows.push(window);
            current_pos += window_size;
        }
        windows
    }

    pub fn update()
    {
        
    }
}

// Dynamic Sequence Compressor (uses AdaptiveAvgPool1d internally)
pub struct DynamicSequenceCompressor {
    target_min: usize,
    target_max: usize,
}

impl DynamicSequenceCompressor {
    pub fn new(target_min: usize, target_max: usize) -> Self {
        Self { target_min, target_max }
    }

    pub fn forward(&self, input: &Vec<Vec<Vec<Complex<f64>>>>) -> (Vec<Vec<Vec<Complex<f64>>>>, CompressionMetadata) {
        self.compress(input)
    }

    pub fn decompress(&self, compressed: &Vec<Vec<Vec<Complex<f64>>>>, metadata: &CompressionMetadata) -> Vec<Vec<Vec<Complex<f64>>>> {
        let pool = AdaptiveAvgPool1d::new(metadata.compressed_length);
        pool.decompress(compressed, metadata)
    }

    pub fn compress(&self, input: &Vec<Vec<Vec<Complex<f64>>>>) -> (Vec<Vec<Vec<Complex<f64>>>>, CompressionMetadata) {
        if input.is_empty() {
            return (
                Vec::new(),
                CompressionMetadata {
                    original_length: 0,
                    compressed_length: 0,
                    window_mappings: Vec::new(),
                    compression_type: CompressionType::Adaptive,
                },
            );
        }

        let seq_len = input[0].len();

        if seq_len <= self.target_max {
            return (
                input.clone(),
                CompressionMetadata {
                    original_length: seq_len,
                    compressed_length: seq_len,
                    window_mappings: (0..seq_len).map(|i| vec![i]).collect(),
                    compression_type: CompressionType::Adaptive,
                },
            );
        }

        let target_size = self.calculate_target_size(seq_len);
        let pool = AdaptiveAvgPool1d::new(target_size);
        pool.compress(input)
    }

    fn calculate_target_size(&self, input_len: usize) -> usize {
        let target_middle = (self.target_min + self.target_max) / 2;
        if input_len / target_middle > 0 {
            let ideal_ratio = input_len / target_middle;
            let target_size = input_len / ideal_ratio;
            target_size.max(self.target_min).min(self.target_max)
        } else {
            target_middle
        }
    }
}

// StridedPooling implementation with forward/backward/backward_gradient
pub struct StridedPooling {
    pub stride: usize,
    pub kernel_size: usize,
}

impl StridedPooling {
    pub fn new(stride: usize, kernel_size: Option<usize>) -> Self {
        Self { stride, kernel_size: kernel_size.unwrap_or(stride) }
    }

    pub fn forward(&self, input: &Vec<Vec<Vec<Complex<f64>>>>) -> (Vec<Vec<Vec<Complex<f64>>>>, CompressionMetadata) {
        if input.is_empty() {
            return (
                Vec::new(),
                CompressionMetadata {
                    original_length: 0,
                    compressed_length: 0,
                    window_mappings: Vec::new(),
                    compression_type: CompressionType::Strided { stride: self.stride, kernel_size: self.kernel_size },
                },
            );
        }

        let batch_size = input.len();
        let seq_len = input[0].len();
        let hidden_dim = if seq_len > 0 { input[0][0].len() } else { 0 };
        let output_len = if seq_len >= self.kernel_size { (seq_len - self.kernel_size) / self.stride + 1 } else { 0 };

        let mut window_mappings = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let start_pos = i * self.stride;
            let end_pos = (start_pos + self.kernel_size).min(seq_len);
            window_mappings.push((start_pos..end_pos).collect());
        }

        let mut output = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let batch_input = &input[batch_idx];
            let pooled = self.apply_strided_pooling(batch_input, output_len, hidden_dim);
            output.push(pooled);
        }

        (
            output,
            CompressionMetadata {
                original_length: seq_len,
                compressed_length: output_len,
                window_mappings,
                compression_type: CompressionType::Strided { stride: self.stride, kernel_size: self.kernel_size },
            },
        )
    }

    pub fn decompress(&self, compressed: &Vec<Vec<Vec<Complex<f64>>>>, metadata: &CompressionMetadata) -> Vec<Vec<Vec<Complex<f64>>>> {
        if compressed.is_empty() || metadata.original_length == 0 {
            return Vec::new();
        }
        let batch_size = compressed.len();
        let hidden_dim = if !compressed[0].is_empty() { compressed[0][0].len() } else { 0 };
        let mut decompressed = Vec::with_capacity(batch_size);
        for batch_seq in compressed.iter() {
            decompressed.push(self.decompress_sequence_strided(batch_seq, metadata.original_length, &metadata.window_mappings, hidden_dim));
        }
        decompressed
    }

    pub fn backward(&self, grad_output: &[Vec<Vec<Complex<f64>>>], metadata: &CompressionMetadata) -> Vec<Vec<Vec<Complex<f64>>>> {
        // Similar to AdaptiveAvgPool1d.backward_gradient but adjusted for overlapping windows
        if grad_output.is_empty() || metadata.original_length == 0 {
            return Vec::new();
        }
        let batch_size = grad_output.len();
        let original_length = metadata.original_length;
        let hidden_dim = if batch_size > 0 && !grad_output[0].is_empty() { grad_output[0][0].len() } else { 0 };
        let mut grad_input = vec![vec![vec![Complex::new(0.0, 0.0); hidden_dim]; original_length]; batch_size];

        for batch_idx in 0..batch_size {
            for (comp_idx, window) in metadata.window_mappings.iter().enumerate() {
                if comp_idx >= grad_output[batch_idx].len() {
                    continue;
                }
                let window_len = window.len() as f64;
                for &pos in window {
                    if pos < original_length {
                        for dim in 0..hidden_dim {
                            grad_input[batch_idx][pos][dim] += grad_output[batch_idx][comp_idx][dim] / window_len;
                        }
                    }
                }
            }
        }
        grad_input
    }

    fn apply_strided_pooling(&self, sequence: &Vec<Vec<Complex<f64>>>, output_len: usize, hidden_dim: usize) -> Vec<Vec<Complex<f64>>> {
        let mut output = Vec::with_capacity(output_len);
        for i in 0..output_len {
            let start_pos = i * self.stride;
            let end_pos = (start_pos + self.kernel_size).min(sequence.len());
            let mut pooled_token = vec![Complex::new(0.0, 0.0); hidden_dim];
            let window_size = end_pos - start_pos;

            for pos in start_pos..end_pos {
                for dim in 0..hidden_dim {
                    pooled_token[dim] += sequence[pos][dim];
                }
            }
            if window_size > 0 {
                let window_size_f64 = window_size as f64;
                for dim in 0..hidden_dim {
                    pooled_token[dim] /= window_size_f64;
                }
            }
            output.push(pooled_token);
        }
        output
    }

    fn decompress_sequence_strided(&self, compressed: &Vec<Vec<Complex<f64>>>, original_length: usize, window_mappings: &[Vec<usize>], hidden_dim: usize) -> Vec<Vec<Complex<f64>>> {
        let mut decompressed = vec![vec![Complex::new(0.0, 0.0); hidden_dim]; original_length];
        let mut position_counts = vec![0usize; original_length];
        for (comp_idx, window) in window_mappings.iter().enumerate() {
            if comp_idx >= compressed.len() {
                continue;
            }
            let compressed_token = &compressed[comp_idx];
            for &pos in window {
                if pos < original_length {
                    for dim in 0..hidden_dim {
                        decompressed[pos][dim] += compressed_token[dim];
                    }
                    position_counts[pos] += 1;
                }
            }
        }
        for pos in 0..original_length {
            if position_counts[pos] > 1 {
                let count_f64 = position_counts[pos] as f64;
                for dim in 0..hidden_dim {
                    decompressed[pos][dim] /= count_f64;
                }
            }
        }
        decompressed
    }
}

// ContentAwarePooling struct and implementation
pub struct ContentAwarePooling {
    pub target_length: usize,
}

impl ContentAwarePooling {
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

// Interpolation-based decompression methods
pub struct InterpolationDecompressor;

impl InterpolationDecompressor {
    /// Linear interpolation decompression
    pub fn linear_interpolate(compressed: &Vec<Vec<Vec<Complex<f64>>>>, target_length: usize) -> Vec<Vec<Vec<Complex<f64>>>> {
        if compressed.is_empty() || target_length == 0 {
            return Vec::new();
        }

        let batch_size = compressed.len();
        let compressed_length = compressed[0].len();
        let hidden_dim = if compressed_length > 0 { compressed[0][0].len() } else { 0 };

        let mut output = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let batch_compressed = &compressed[batch_idx];
            let interpolated = Self::interpolate_sequence(batch_compressed, target_length, hidden_dim);
            output.push(interpolated);
        }

        output
    }

    fn interpolate_sequence(compressed: &Vec<Vec<Complex<f64>>>, target_length: usize, hidden_dim: usize) -> Vec<Vec<Complex<f64>>> {
        if compressed.is_empty() || target_length == 0 {
            return Vec::new();
        }

        let compressed_length = compressed.len();
        if compressed_length == target_length {
            return compressed.clone();
        }

        let mut interpolated = Vec::with_capacity(target_length);
        let scale = (compressed_length - 1) as f64 / (target_length - 1) as f64;

        for i in 0..target_length {
            let float_idx = i as f64 * scale;
            let left_idx = float_idx.floor() as usize;
            let right_idx = (left_idx + 1).min(compressed_length - 1);
            let weight = float_idx - left_idx as f64;

            let mut interpolated_token = vec![Complex::new(0.0, 0.0); hidden_dim];

            if left_idx == right_idx {
                // No interpolation needed
                interpolated_token = compressed[left_idx].clone();
            } else {
                // Linear interpolation between two tokens
                for dim in 0..hidden_dim {
                    let left_val = compressed[left_idx][dim];
                    let right_val = compressed[right_idx][dim];
                    interpolated_token[dim] = left_val * (1.0 - weight) + right_val * weight;
                }
            }

            interpolated.push(interpolated_token);
        }

        interpolated
    }
}

// Helper function to calculate reconstruction error (Mean Squared Error)
pub fn calculate_reconstruction_error(original: &Vec<Vec<Vec<Complex<f64>>>>, reconstructed: &Vec<Vec<Vec<Complex<f64>>>>) -> f64 {
    if original.len() != reconstructed.len() || original.is_empty() {
        return f64::INFINITY;
    }

    let mut total_error = 0.0;
    let mut total_elements = 0;

    for (batch_orig, batch_recon) in original.iter().zip(reconstructed.iter()) {
        if batch_orig.len() != batch_recon.len() {
            return f64::INFINITY;
        }

        for (seq_orig, seq_recon) in batch_orig.iter().zip(batch_recon.iter()) {
            if seq_orig.len() != seq_recon.len() {
                return f64::INFINITY;
            }

            for (orig_val, recon_val) in seq_orig.iter().zip(seq_recon.iter()) {
                let diff = orig_val - recon_val;
                total_error += diff.norm_sqr();
                total_elements += 1;
            }
        }
    }

    if total_elements > 0 {
        total_error / total_elements as f64
    } else {
        0.0
    }
}

// Test different decompression methods
pub fn test_decompression_methods(original: &Vec<Vec<Vec<Complex<f64>>>>, compressed: &Vec<Vec<Vec<Complex<f64>>>>, metadata: &CompressionMetadata) {
    let adaptive_pool = AdaptiveAvgPool1d::new(metadata.compressed_length);

    // Method 1: Window-based decompression
    let decompressed_windows = adaptive_pool.decompress(compressed, metadata);
    let window_error = calculate_reconstruction_error(original, &decompressed_windows);

    // Method 2: Linear interpolation
    let decompressed_interpolation = InterpolationDecompressor::linear_interpolate(compressed, metadata.original_length);
    let interpolation_error = calculate_reconstruction_error(original, &decompressed_interpolation);

    println!("Window-based decompression error: {:.6}", window_error);
    println!("Interpolation decompression error: {:.6}", interpolation_error);

    if window_error < interpolation_error {
        println!("→ Window-based decompression is better for this case");
    } else {
        println!("→ Linear interpolation is better for this case");
    }
}

// Helper function to create random complex input (reused from previous version)
pub fn create_random_input(batch_size: usize, seq_len: usize, hidden_dim: usize) -> Vec<Vec<Vec<Complex<f64>>>> {
    use std::f64::consts::PI;

    let mut input = Vec::with_capacity(batch_size);

    for batch in 0..batch_size {
        let mut sequence = Vec::with_capacity(seq_len);

        for seq in 0..seq_len {
            let mut features = Vec::with_capacity(hidden_dim);

            for dim in 0..hidden_dim {
                // Create some realistic complex values
                let phase = 2.0 * PI * (seq as f64) / (seq_len as f64) + (dim as f64) * 0.1;
                let magnitude = 1.0 / (1.0 + (seq as f64) * 0.001); // Decay over sequence

                let real = magnitude * phase.cos() + (batch as f64) * 0.01;
                let imag = magnitude * phase.sin() + (dim as f64) * 0.001;

                features.push(Complex::new(real, imag));
            }

            sequence.push(features);
        }

        input.push(sequence);
    }

    input
}

#[cfg(test)]
mod adaptive_pooling_complex_tests {
    use super::*;

    #[test]
    fn test_compression_decompression_roundtrip() {
        let input = create_random_input(1, 100, 16);
        let pool = AdaptiveAvgPool1d::new(25);

        let (compressed, metadata) = pool.compress(&input);
        let decompressed = pool.decompress(&compressed, &metadata);

        assert_eq!(input.len(), decompressed.len());
        assert_eq!(input[0].len(), decompressed[0].len());
        assert_eq!(input[0][0].len(), decompressed[0][0].len());

        // Error should be finite (not perfect reconstruction, but reasonable)
        let error = calculate_reconstruction_error(&input, &decompressed);
        println!("Reconstruction error: {:?}", error);
        assert!(error.is_finite() && error >= 0.0);
    }

    #[test]
    fn test_metadata_consistency() {
        let input = create_random_input(2, 200, 32);
        let compressor = DynamicSequenceCompressor::new(40, 50);

        let (compressed, metadata) = compressor.compress(&input);

        assert_eq!(metadata.original_length, 200);
        assert!(metadata.compressed_length >= 40 && metadata.compressed_length <= 50);
        assert_eq!(compressed[0].len(), metadata.compressed_length);
    }

    #[test]
    fn test_strided_pooling_decompression() {
        let input = create_random_input(1, 50, 8);
        let pool = StridedPooling::new(5, Some(5));

        let (compressed, metadata) = pool.forward(&input);
        let decompressed = pool.decompress(&compressed, &metadata);

        assert_eq!(decompressed[0].len(), 50);

        // Test that overlapping regions are handled correctly
        let error = calculate_reconstruction_error(&input, &decompressed);
        println!("Strided Pooling Reconstruction error: {:?}", error);
        assert!(error.is_finite());
    }

    #[test]
    fn test_interpolation_decompression() {
        let compressed = create_random_input(1, 10, 4);
        let interpolated = InterpolationDecompressor::linear_interpolate(&compressed, 25);

        assert_eq!(interpolated[0].len(), 25);
        assert_eq!(interpolated[0][0].len(), 4);
    }
}
