#[cfg(test)]
pub mod test_sliding_window {
    use crate::neural_networks::{
        network_components::input::{DataTrait, Dataset},
        network_types::transformer::transformer_network::{CONTEXT_OVERLAPPING, MAX_CONTEXT_WINDOW_SIZE},
        utils::{
            array_splitting::sliding_window_chunks_matrix,
            tokenizer::{detokenize, tokenize_batch},
        },
    };

    #[test]
    fn test_sliding_window_for_training() {
        let input: &str = "Context: Die Verteidigung der Panthers gab nur 308 Punkte ab und belegte den sechsten Platz in der Liga, während sie die NFL mit 24 Interceptions in dieser Kategorie anführte und sich mit vier Pro Bowl-Selektionen rühmen konnte. Pro Bowl Defensive Tackle Kawann Short führte das Team mit 11 Sacks an, erzwang zudem drei Fumbles und erzielte zwei Fumble Recoverys. Mario Addison, ebenfalls Lineman, addierte 6½ Sacks hinzu. Die Panthers-Line präsentierte auch den erfahrenen Defensive End Jared Allen, einen 5-fachen Pro-Bowler, der mit 136 Sacks der aktive Anführer in der NFL-Kategorie Karriere-Sacks war, sowie den Defensive End Kony Ealy, der 5 Sacks in nur 9 Starts erzielte. Nach ihnen wurden zwei der drei Linebacker der Panthers ausgewählt, um im Pro Bowl zu spielen: Thomas Davis und Luke Kuechly. Davis erzielte 5½ Sacks, vier erzwungene Fumbles und vier Interceptions, während Kuechly das Team bei den Tackles anführte (118), zwei Fumbles erzwang und vier Pässe abfing. Carolinas Secondarys bestanden aus dem Pro Bowl-Safety Kurt Coleman, der das Team mit einem Karrierehoch von sieben Interceptions anführte und gleichzeitig 88 Tackles erzielen konnte, und Pro Bowl-Cornerback Josh Norman, der sich während der Saison zur Shutdown Corner entwickelte und vier Interceptions erzielte, von denen zwei zu Touchdowns für sein Team wurden. \n <sep> Question: Wie viele Punkte gab die Verteidigung der Panthers ab?";
        let target = "308";
        let dataset = Dataset::new(vec![input.to_string()], vec![target.to_string()]);
        let batch_dataset = &dataset.split_into_batches(1)[0];
        let (input_batch, target_batch) = (batch_dataset.get_input(), batch_dataset.get_target());

        let input_batch_extended = batch_dataset.extend_input_with_target(input_batch, target_batch);
        let target_batch_extended = batch_dataset.extend_target(target_batch);

        let (_tokens, batch_ids) = tokenize_batch(&input_batch_extended, false).unwrap();
        let (_tokens, _target_ids) = tokenize_batch(&target_batch_extended, false).unwrap();

        println!("extended input batch: {:?}", input_batch_extended);
        println!("batch ids len: {}", batch_ids[0].len());

        let (input_chunks, target_chunks) = sliding_window_chunks_matrix(&batch_ids, MAX_CONTEXT_WINDOW_SIZE, CONTEXT_OVERLAPPING);

        println!("Number of chunks: {}", input_chunks.len());
        for (i, input_chunk) in input_chunks.iter().enumerate() {
            println!("\n\n======================================================================");
            println!("Chunk {}: Input={:?}, \nTarget={:?}", i, input_chunk, &target_chunks[i]);

            println!("input tokens len: {:?}", input_chunk.len());
            println!("target tokens len: {:?}", target_chunks[i].len());

            let input_tokens: String = detokenize(input_chunk, false).unwrap();
            let target_tokens: String = detokenize(&target_chunks[i], false).unwrap();

            println!("input tokens chunk: {:?}", input_tokens);
            println!("target tokens chunk: {:?}", target_tokens);
            println!("======================================================================");
            // Print first 5 tokens
        }
    }
}
