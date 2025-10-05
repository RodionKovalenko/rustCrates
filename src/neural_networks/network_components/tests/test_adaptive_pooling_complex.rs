#[cfg(test)]
mod test_adaptive_pooling_complex {
    use num::Complex;

    use crate::neural_networks::{
        network_components::{
            adaptive_pooling::{
                adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer,
                adaptive_pooling_complex::{calculate_reconstruction_error, create_random_input, test_decompression_methods},
                content_aware_pooling_layer::ContentAwarePoolingLayer,
                dynamic_sequence_compressor_layer::DynamicSequenceCompressorLayer,
                interpalation_decompressor_layer::InterpolationDecompressorLayer,
                strided_pooling_layer::StridedPoolingLayer,
            },
            gradient_struct::Gradient,
            layer_input_struct::LayerInput,
            layer_output_struct::LayerOutput,
            linear_layer::LinearLayer,
            softmax_output_layer::SoftmaxLayer,
        },
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::{
            derivative::{global_relative_error_l2, numerical_gradient_input_batch, test_gradient_batch_error},
            random_arrays::generate_random_u32_batch,
        },
    };
    #[test]
    fn test_adaptive_avg_pooling_complex() {
        // Create your complex input data
        let input: Vec<Vec<Vec<Complex<f64>>>> = create_random_input(2, 1024, 64);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input.clone());

        // Adaptive pooling to exactly 45 tokens
        let mut pool: AdaptiveAvgPool1dLayer = AdaptiveAvgPool1dLayer::new(45);
        let pooling_output = pool.forward(&layer_input);
        let compressed = pooling_output.get_output_batch();

        println!("Original length: {} {} {}", input.len(), input[0].len(), input[0][0].len());
        println!("Compressed length dim: {} {} {}", compressed.len(), compressed[0].len(), compressed[0][0].len());

        assert_eq!(compressed[0].len(), 45);
        println!("Adaptive pooling test passed.");

        test_decompression_methods(&input, &compressed, &pool);
        //TEST 2
        // Create your complex input data
        let input: Vec<Vec<Vec<Complex<f64>>>> = create_random_input(2, 10, 4);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input.clone());

        // Adaptive pooling to exactly 10 tokens
        let mut pool = AdaptiveAvgPool1dLayer::new(5);
        let pooling_output = pool.forward(&layer_input);
        let compressed = pooling_output.get_output_batch();
        let decompressed = pool.decompress(&compressed);

        println!("Original length: {}, Compressed length: {}", input[0].len(), compressed[0].len());
        println!("Original length dim: {} {} {}", input.len(), input[0].len(), input[0][0].len());
        println!("Compressed length dim: {} {} {}", compressed.len(), compressed[0].len(), compressed[0][0].len());

        println!("Input data: {:?}", input);
        println!("Decompressed: {:?}", decompressed);

        test_decompression_methods(&input, &compressed, &pool);
    }

    #[test]
    fn test_benchmarks() {
        // TEST 3:
        // Create example input: batch_size=2, seq_len=1024, hidden_dim=64
        let batch_size = 2;
        let seq_len = 1024;
        let hidden_dim = 64;

        println!("Creating input tensor: {}x{}x{}", batch_size, seq_len, hidden_dim);
        let input = create_random_input(batch_size, seq_len, hidden_dim);
        println!("Input shape: [{}, {}, {}]", input.len(), input[0].len(), input[0][0].len());

        // Test 1: Adaptive pooling with compression and decompression
        println!("\n=== Test 1: Adaptive Pooling Compression/Decompression ===");
        let mut adaptive_pool = AdaptiveAvgPool1dLayer::new(45);

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input.clone());

        // Compress
        let pool_output = adaptive_pool.forward(&layer_input);
        let (compressed, metadata) = (pool_output.get_output_batch(), pool_output.get_pooling_metadata().unwrap());
        println!("Compressed shape: [{}, {}, {}]", compressed.len(), compressed[0].len(), compressed[0][0].len());
        println!("Compression metadata: {:?}", metadata.compression_type);

        // Decompress
        let decompressed = adaptive_pool.decompress(&compressed);
        println!("Decompressed shape: [{}, {}, {}]", decompressed.len(), decompressed[0].len(), decompressed[0][0].len());

        // Calculate reconstruction error
        let reconstruction_error = calculate_reconstruction_error(&input, &decompressed);
        println!("Reconstruction error (MSE): {:.6}", reconstruction_error);

        // Test 2: Dynamic compression with different input sizes
        println!("\n=== Test 2: Dynamic Compression/Decompression ===");
        let mut dynamic_compressor = DynamicSequenceCompressorLayer::new(40, 50);

        let pooling_output = dynamic_compressor.compress(&input);
        let (compressed2, _metadata2) = (pooling_output.get_output_batch(), pooling_output.get_pooling_metadata().unwrap());
        println!("Dynamic compressed shape: [{}, {}, {}]", compressed2.len(), compressed2[0].len(), compressed2[0][0].len());

        let decompressed2 = dynamic_compressor.decompress(&compressed2);
        println!("Dynamic decompressed shape: [{}, {}, {}]", decompressed2.len(), decompressed2[0].len(), decompressed2[0][0].len());

        // Test 3: Strided pooling compression/decompression
        println!("\n=== Test 3: Strided Pooling Compression/Decompression ===");
        let strided_pool = StridedPoolingLayer::new(21, Some(21));

        let (compressed3, metadata3) = strided_pool.forward(&input);
        println!("Strided compressed shape: [{}, {}, {}]", compressed3.len(), compressed3[0].len(), compressed3[0][0].len());

        let decompressed3 = strided_pool.decompress(&compressed3, &metadata3);
        println!("Strided decompressed shape: [{}, {}, {}]", decompressed3.len(), decompressed3[0].len(), decompressed3[0][0].len());

        let strided_error = calculate_reconstruction_error(&input, &decompressed3);
        println!("Strided reconstruction error: {:.6}", strided_error);

        // Test 4: Content-aware compression/decompression
        println!("\n=== Test 4: Content-Aware Compression/Decompression ===");
        let content_aware = ContentAwarePoolingLayer::new(45);

        let (compressed4, metadata4) = content_aware.compress(&input);
        println!("Content-aware compressed shape: [{}, {}, {}]", compressed4.len(), compressed4[0].len(), compressed4[0][0].len());

        let decompressed4 = content_aware.decompress(&compressed4, &metadata4);
        println!("Content-aware decompressed shape: [{}, {}, {}]", decompressed4.len(), decompressed4[0].len(), decompressed4[0][0].len());

        // Test 5: Linear interpolation decompression
        println!("\n=== Test 5: Linear Interpolation Decompression ===");
        let interpolated = InterpolationDecompressorLayer::linear_interpolate(&compressed, seq_len);
        println!("Interpolated shape: [{}, {}, {}]", interpolated.len(), interpolated[0].len(), interpolated[0][0].len());

        let interpolation_error = calculate_reconstruction_error(&input, &interpolated);
        println!("Interpolation reconstruction error: {:.6}", interpolation_error);

        // Test 6: Compare different decompression methods
        println!("\n=== Test 6: Decompression Method Comparison ===");
        test_decompression_methods(&input, &compressed, &adaptive_pool);

        // Test 7: Variable input lengths
        println!("\n=== Test 7: Variable Length Compression/Decompression ===");
        let test_lengths = vec![128, 256, 512, 1024, 2048];

        let mut layer_input = LayerInput::new_default();

        for &length in &test_lengths {
            let test_input = create_random_input(1, length, 32);
            layer_input.set_input_batch(test_input.clone());
            let pooling_output = adaptive_pool.forward(&layer_input);
            let (compressed_test, _metadata_test) = (pooling_output.get_output_batch(), pooling_output.get_pooling_metadata().unwrap());
            let decompressed_test = adaptive_pool.decompress(&compressed_test);

            let error = calculate_reconstruction_error(&test_input, &decompressed_test);
            println!("Length {}: {} -> {} -> {} (error: {:.6})", length, test_input[0].len(), compressed_test[0].len(), decompressed_test[0].len(), error);
        }
    }

    #[test]
    fn test_adaptive_avg_pooling_gradient() {
        // Create your complex input data
        let learning_rate = 0.001;
        let epsilon = 1e-6;
        let batch_size = 2;
        let sequent_len = 55;
        let input_dim = 64;
        let compressed_dim = 16;

        let operation_mode = OperationMode::TRAINING;
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = create_random_input(2, sequent_len, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, compressed_dim, compressed_dim as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; compressed_dim]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Adaptive pooling to exactly 45 tokens
        let mut pool: AdaptiveAvgPool1dLayer = AdaptiveAvgPool1dLayer::new(compressed_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        let pooling_output = pool.forward(&layer_input);
        let (compressed, metadata) = (pooling_output.get_output_batch(), pooling_output.get_pooling_metadata().unwrap());

        println!("compressed dim: {} {} {}", compressed.len(), compressed[0].len(), compressed[0][0].len());
        let _softmax_batch_output: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&compressed, Some(padding_mask_batch.clone()));

        let mut gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        gradient_softmax.set_pooling_metadata(metadata);
        let anal_linear_gradient_input_batch = pool.backward(&gradient_softmax).get_gradient_input_batch();

        // TEST GRADIENT OF THE INPUT BATCH
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());
            let pooling_output = pool.forward(&layer_input);
            let (compressed, _metadata) = (pooling_output.get_output_batch(), pooling_output.get_pooling_metadata().unwrap());
            let _softmax_batch_output: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&compressed, Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&_softmax_batch_output, &target_token_id_batch, &padding_mask_batch, batch_size);

            loss
        };

        let num_linear_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical gradient_weights_batch: {:?}", gradient_weights_batch);
        //println!("\n analytical gradient_input_batch: {:?}", anal_linear_gradient_input_batch);
        println!("\n anlytical gradient_input_batch dim: {} {} {}", anal_linear_gradient_input_batch.len(), anal_linear_gradient_input_batch[0].len(), anal_linear_gradient_input_batch[0][0].len());

        //println!("\n numerical grad: {:?}", num_gradient_weight_batch);
        //println!("\n numerical num_gradient_input_batch: {:?}", &num_linear_gradient_input_batch);
        println!("\n numerical num_gradient_input_batch dim: {} {} {}", num_linear_gradient_input_batch.len(), num_linear_gradient_input_batch[0].len(), num_linear_gradient_input_batch[0][0].len());

        let global_error = global_relative_error_l2(&num_linear_gradient_input_batch, &anal_linear_gradient_input_batch);

        println!("global relative gradient error gradient input batch: {:?}", &global_error);

        test_gradient_batch_error(&num_linear_gradient_input_batch, &anal_linear_gradient_input_batch, 1e-3);
    }

    #[test]
    fn test_adaptive_avg_pooling_two_layers_gradient() {
        // Create your complex input data
        let epsilon = 1e-6;
        let batch_size = 2;
        let output_dim = 64;
        let sequent_len = 36;
        let input_dim = 64;
        let compression_dim = 15;
        let learning_rate = 0.001;
        let linear_hidden_dim = 80;

        let operation_mode = OperationMode::TRAINING;
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = create_random_input(2, sequent_len, input_dim);
        let target_token_id_batch: Vec<Vec<u32>> = generate_random_u32_batch(batch_size, 5, (output_dim - 1) as u32);
        let padding_mask_batch: Vec<Vec<u32>> = vec![vec![1; input_batch[0].len()]; input_batch.len()];

        let mut layer_input = LayerInput::new_default();
        layer_input.set_input_batch(input_batch.clone());

        // Adaptive pooling to exactly 45 tokens
        let mut pool: AdaptiveAvgPool1dLayer = AdaptiveAvgPool1dLayer::new(compression_dim);
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, linear_hidden_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Input e.g. 36x64
        // 1. Pooling Compression -> dim 12x64
        // 2. Linear layer -> dim 12x64
        // 3. Pooling Decompression -> dim 36x64
        // 4. Softmax -> dim 36x64

        let pooling_output = pool.forward(&layer_input);
        layer_input.set_input_batch(pooling_output.get_output_batch());
        let linear_output: LayerOutput = linear_layer.forward(&layer_input);
        let compressed: Vec<Vec<Vec<Complex<f64>>>> = pooling_output.get_output_batch();
        let pooling_output_decompression: Vec<Vec<Vec<Complex<f64>>>> = pool.decompress(&linear_output.get_output_batch());
        let _softmax_batch_output: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&pooling_output_decompression, Some(padding_mask_batch.clone()));

        println!("original input dim: {} {} {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());
        println!("compressed dim: {} {} {}", compressed.len(), compressed[0].len(), compressed[0][0].len());

        let gradient_softmax: Gradient = softmax_layer.backward(&target_token_id_batch);
        let gradient_decompressed = pool.backward(&gradient_softmax);
        let linear_gradient = linear_layer.backward(&gradient_decompressed);
        let anal_pool_gradient_input_batch = pool.backward(&linear_gradient).get_gradient_input_batch();

        // TEST GRADIENT OF THE INPUT BATCH
        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>| -> Complex<f64> {
            layer_input.set_input_batch(input.clone());

            let pooling_output = pool.forward(&layer_input);
            layer_input.set_input_batch(pooling_output.get_output_batch());

            let linear_output: LayerOutput = linear_layer.forward(&layer_input);
            let pooling_output_decompression: Vec<Vec<Vec<Complex<f64>>>> = pool.decompress(&linear_output.get_output_batch());
            let _softmax_batch_output: Vec<Vec<Vec<f64>>> = softmax_layer.forward(&pooling_output_decompression, Some(padding_mask_batch.clone()));

            let loss = cross_entropy_loss_batch(&_softmax_batch_output, &target_token_id_batch, &padding_mask_batch, batch_size);

            loss
        };

        let num_linear_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient_input_batch(&mut loss_fn, input_batch.clone(), epsilon);

        // Check if gradient batch dimensions match expected shapes
        //println!("\n analytical gradient_weights_batch: {:?}", gradient_weights_batch);
        //println!("\n analytical gradient_input_batch: {:?}", anal_linear_gradient_input_batch);
        println!("\n anlytical gradient_input_batch dim: {} {} {}", anal_pool_gradient_input_batch.len(), anal_pool_gradient_input_batch[0].len(), anal_pool_gradient_input_batch[0][0].len());

        //println!("\n numerical grad: {:?}", num_gradient_weight_batch);
        //println!("\n numerical num_gradient_input_batch: {:?}", &num_linear_gradient_input_batch);
        println!("\n numerical num_gradient_input_batch dim: {} {} {}", num_linear_gradient_input_batch.len(), num_linear_gradient_input_batch[0].len(), num_linear_gradient_input_batch[0][0].len());

        let global_error = global_relative_error_l2(&num_linear_gradient_input_batch, &anal_pool_gradient_input_batch);

        println!("global relative gradient error gradient input batch: {:?}", &global_error);

        test_gradient_batch_error(&num_linear_gradient_input_batch, &anal_pool_gradient_input_batch, 1e-3);
    }
}
