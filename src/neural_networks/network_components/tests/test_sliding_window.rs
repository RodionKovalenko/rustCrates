#[cfg(test)]
pub mod test_sliding_window {
    use crate::neural_networks::utils::array_splitting::sliding_window_chunks;

    #[test]
    fn test_sliding_window_for_training() {
        // Example usage:
        let tokens: Vec<u32> = (0..450).collect(); // Dummy token IDs (0..449)
        let window_size = 40;
        let stride = 15; // Overlap = 15 tokens
        let (input_chunks, target_chunks) = sliding_window_chunks(&tokens, window_size, stride);

        println!("Number of chunks: {}", input_chunks.len());
        for (i, input_chunk) in input_chunks.iter().enumerate() {
            println!("Chunk {}: Input={:?}, Target={:?}", i, &input_chunk, &target_chunks[i]);
            // Print first 5 tokens
        }
    }
}
