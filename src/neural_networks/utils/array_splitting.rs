pub fn sliding_window_chunks(tokens: &[u32], window_size: usize, stride: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut input_chunks: Vec<Vec<u32>> = Vec::new();
    let mut target_chunks: Vec<Vec<u32>> = Vec::new();

    let mut start = 0;
    let len = tokens.len();

    while start < len.saturating_sub(1) {
        let end = usize::min(start + window_size, len - 1);
        let input = tokens[start..end].to_vec();

        let target_end = usize::min(end + 1, len);
        let target = tokens[start + 1..target_end].to_vec();

        input_chunks.push(input);
        target_chunks.push(target);

        if start + stride >= len {
            break;
        }

        start += stride;
    }

    (input_chunks, target_chunks)
}

pub fn sliding_window_chunks_matrix(tokens_batch: &Vec<Vec<u32>>, window_size: usize, stride: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut inputs: Vec<Vec<u32>> = Vec::new();
    let mut targets: Vec<Vec<u32>> = Vec::new();

    for tokens in tokens_batch {
        let (input_chunks, target_chunks) = sliding_window_chunks(tokens, window_size, stride);

        for (ind, input_chunk) in input_chunks.iter().enumerate() {
            inputs.push(input_chunk.to_vec());
            targets.push(target_chunks[ind].to_vec());
        }
    }

    (inputs, targets)
}
