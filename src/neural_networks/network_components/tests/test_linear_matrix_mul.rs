#[cfg(test)]
mod test_linear_matrix_mul {
    use std::time::Instant;

    use num::Complex;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

    use crate::neural_networks::{
        network_components::{layer_input_struct::LayerInput, linear_layer::LinearLayer},
        utils::{derivative::test_gradient_batch_error, random_arrays::generate_random_complex_3d},
    };

    #[test]
    fn test_linear_matrix_multiplication() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 2;
        let seq_len = 512;
        let dim = 8;
        let learning_rate = 0.001;
        let vocab_size = 50280;
        let compression = 1;
        let output_dim = (vocab_size as f64 / compression as f64) as usize;

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = generate_random_complex_3d(batch_size, seq_len, dim);
        let mut linear_layer = LinearLayer::new(learning_rate, dim, vocab_size);

        let mut lin_layers: Vec<LinearLayer> = (0..compression).map(|_| LinearLayer::new(learning_rate, dim, output_dim)).collect();

        test_forward_pass(&input_batch, &mut linear_layer, &mut lin_layers);

        let mut layer_input: LayerInput = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Test 2:
        let now = Instant::now();
        let mut combined_output_weigths: Vec<Vec<Complex<f64>>> = vec![Vec::with_capacity(vocab_size); dim];

        for lin_layer in lin_layers.iter_mut() {
            let weights = &lin_layer.weights;

            for i in 0..dim {
                combined_output_weigths[i].extend_from_slice(&weights[i].clone());
            }
        }

        println!("combined_output_weigths in cocatenate: {} {}", combined_output_weigths.len(), combined_output_weigths[0].len());

        linear_layer.weights = combined_output_weigths;

        let seconds_elapsed = now.elapsed();
        let lin_output = linear_layer.forward(&layer_input);
        let output_batch = lin_output.get_output_batch();
        println!("lin output size: {} {} {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());
        println!("time elapsed in seconds in forward linear: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());

        // Test 4
        let seconds_elapsed = now.elapsed();
        layer_input.set_input_batch(input_batch.clone());

        let lin_layer_output_chunks = lin_layers
            .par_iter_mut()
            .map(|lin_layer| {
                let lin_output = lin_layer.forward(&layer_input);
                lin_output.get_output_batch()
            })
            .collect::<Vec<_>>();

        println!("Compressed output chunks: {}", lin_layer_output_chunks.len());

        let mut concatenated_output: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); 0]; seq_len]; batch_size];

        // Parallelize over (b, i) pairs
        concatenated_output.par_iter_mut().enumerate().for_each(|(b, batch_row)| {
            for i in 0..seq_len {
                let mut combined_output = Vec::new();
                for lin_layer_output_chunk in &lin_layer_output_chunks {
                    combined_output.extend_from_slice(&lin_layer_output_chunk[b][i]);
                }
                batch_row[i] = combined_output;
            }
        });

        println!("time elapsed in seconds in chunks: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());

        test_gradient_batch_error(&concatenated_output, &output_batch, 1e-5);

        test_forward_pass(&input_batch, &mut linear_layer, &mut lin_layers);
        test_forward_pass(&input_batch, &mut linear_layer, &mut lin_layers);
        test_forward_pass(&input_batch, &mut linear_layer, &mut lin_layers);
        test_forward_pass(&input_batch, &mut linear_layer, &mut lin_layers);
        test_forward_pass(&input_batch, &mut linear_layer, &mut lin_layers);
        test_forward_pass(&input_batch, &mut linear_layer, &mut lin_layers);
    }
    fn test_forward_pass(input_batch: &Vec<Vec<Vec<Complex<f64>>>>, linear_layer: &mut LinearLayer, lin_layers: &mut Vec<LinearLayer>) {
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        // repeat the test:
        let now = Instant::now();
        let seconds_elapsed = now.elapsed();
        let mut layer_input: LayerInput = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());
        // multiply linear_layer with input
        let lin_output = linear_layer.forward(&layer_input);
        let output_batch = lin_output.get_output_batch();
        println!("lin output size: {} {} {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());

        println!("time elapsed in seconds in forward linear: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());

        // test with chunks
        let seconds_elapsed = now.elapsed();
        layer_input.set_input_batch(input_batch.clone());

        let lin_layer_output_chunks = lin_layers
            .par_iter_mut()
            .map(|lin_layer| {
                let lin_output = lin_layer.forward(&layer_input);
                lin_output.get_output_batch()
            })
            .collect::<Vec<_>>();

        println!("Compressed output chunks: {}", lin_layer_output_chunks.len());

        let mut concatenated_output: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); 0]; seq_len]; batch_size];

        // Parallelize over (b, i) pairs
        concatenated_output.par_iter_mut().enumerate().for_each(|(b, batch_row)| {
            for i in 0..seq_len {
                let mut combined_output = Vec::new();
                for lin_layer_output_chunk in &lin_layer_output_chunks {
                    combined_output.extend_from_slice(&lin_layer_output_chunk[b][i]);
                }
                batch_row[i] = combined_output;
            }
        });

        // concatenate them together:
        println!("concatenated lin output: {} {} {}", concatenated_output.len(), concatenated_output[0].len(), concatenated_output[0][0].len());
        println!("time elapsed in seconds for chunking: {:?}", (now.elapsed() - seconds_elapsed).as_secs_f64());
    }
}
