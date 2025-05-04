pub fn sliding_window_chunks(tokens: &[u32], window_size: usize, stride: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    // let mut chunks: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    let mut start: usize = 0;

    let mut input_chunks: Vec<Vec<u32>> = vec![];
    let mut target_chunk: Vec<Vec<u32>> = vec![];

    while start + window_size <= tokens.len() {
        let end: usize = start + window_size;
        let input: Vec<u32> = tokens[start..end].to_vec();
        let target: Vec<u32> = tokens[start + 1..end + 1].to_vec(); // Teacher forcing shift

        input_chunks.push(input);
        target_chunk.push(target);

        start += stride;
    }

    (input_chunks, target_chunk)
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
