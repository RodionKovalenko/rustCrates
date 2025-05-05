// 454 batch len
// 50 windows zie
// 15 stride
// num_chunks = floor((batch_len - window_size) / stride) + 1
// num_chunks = 26

// (454 - 40)/30 +1 = 14,8 = 14
pub fn sliding_window_chunks(tokens: &[u32], window_size: usize, stride: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut input_chunks = Vec::new();
    let mut target_chunks = Vec::new();

    let mut start = 0;
    let len = tokens.len();

    while start + window_size <= len {
        let input = tokens[start..start + window_size].to_vec();
        let target = tokens[start + 1..usize::min(start + window_size + 1, len)].to_vec();

        input_chunks.push(input);
        target_chunks.push(target);

        start += stride;

        // include the last tokens
        if start + window_size > len && start < len {
            let input = tokens[len - window_size - 1..len - 1].to_vec();
            let target = tokens[len - window_size..len].to_vec();

            input_chunks.push(input);
            target_chunks.push(target);
        }
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
