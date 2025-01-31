#[cfg(test)]
mod tests {
    use std::f64::consts::E;

    use crate::neural_networks::{
        network_components::{linear_layer::LinearLayer, softmax_output_layer::SoftmaxLayer},
        network_types::{neural_network_generic::OperationMode, transformer::transformer_network::cross_entropy_loss_batch},
        utils::{activation::softmax_complex, derivative::numerical_gradient},
    };

    use num::Complex;

    #[test]
    fn test_linear_layer_backward() {
        // Define some small batch size and input dimensions for simplicity
        let _batch_size = 2;
        let seq_len: usize = 1; // Update to match the input structure
        let input_dim = 3; // Match the input dimension with your input batch
        let output_dim = 4; // Match output_dim to your layer's output
        let learning_rate = 0.01;
        let operation_mode = OperationMode::TRAINING;

        // Create a simple LinearLayer with the given input and output dimensions
        let mut linear_layer: LinearLayer = LinearLayer::new(learning_rate, input_dim, output_dim);
        let mut softmax_layer: SoftmaxLayer = SoftmaxLayer::new(learning_rate, operation_mode);

        // Define a small input batch, [2][2][3]
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![vec![Complex::new(0.5, 0.0), Complex::new(0.8, 0.0), Complex::new(0.1, 0.0)]],
            //vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]],
        ];


        let target_token_id_batch = vec![vec![1]];

        // Forward pass (initialize the input batch) [2][2][3]  * [3][4] => [2][2][4]
        let linear_batch_output = linear_layer.forward(&input_batch);

        println!("linear batch output: {:?}", &linear_batch_output);
        let softmax_batch_output = softmax_layer.forward(&linear_batch_output);
        let analytical_grad = softmax_layer.backward_batch(&target_token_id_batch);

        // Define the loss function
        let mut loss_fn = |input: &Vec<Vec<Vec<Complex<f64>>>>, linear_layer: &mut LinearLayer| -> Complex<f64> {
            let softmax_batch_output = softmax_layer.forward(&input);

            //println!("softmax batch output numerical loss {:?}", &softmax_batch_output);
            let loss = cross_entropy_loss_batch(&softmax_batch_output, &target_token_id_batch);

            loss
        };

        let epsilon = 1e-07;
        let numerical_grad: Vec<Vec<Vec<Complex<f64>>>> = numerical_gradient(&mut loss_fn, linear_batch_output.clone(), epsilon, &mut linear_layer);

        // Check if gradient batch dimensions match expected shapes
        println!("analytical grad: {:?}", analytical_grad);
        println!("numerical grad: {:?}", numerical_grad);
    }

    fn softmax(logits: &[f64]) -> Vec<f64> {
        let sum_exp: f64 = logits.iter().map(|&x| E.powf(x)).sum();
        logits.iter().map(|&x| E.powf(x) / sum_exp).collect()
    }

    // Analytical gradient of softmax with cross-entropy loss
    fn analytical_gradient(logits: &[f64], labels: &[f64]) -> Vec<f64> {
        let softmax_probs = softmax(logits);
        softmax_probs.iter().zip(labels.iter()).map(|(p, y)| p - y).collect()
    }

    fn cross_entropy_loss(predictions: &[f64], labels: &[f64]) -> f64 {
        -labels.iter().zip(predictions.iter()).map(|(y, p)| y * p.ln()).sum::<f64>()
    }

    fn numerical_gradient_function<F>(logits: &[f64], labels: &[f64], epsilon: f64, loss_fn: F) -> Vec<f64>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let mut gradients = Vec::with_capacity(logits.len());

        for i in 0..logits.len() {
            // Perturb the logit
            let mut logits_plus = logits.to_vec();
            logits_plus[i] += epsilon;

            let mut logits_minus = logits.to_vec();
            logits_minus[i] -= epsilon;

            // Compute loss for perturbed logits
            let loss_plus = loss_fn(&softmax(&logits_plus), labels);
            let loss_minus = loss_fn(&softmax(&logits_minus), labels);

            // Approximate gradient using finite difference
            let gradient = (loss_plus - loss_minus) / (2.0 * epsilon);
            gradients.push(gradient);
        }

        gradients
    }

    // Computes gradients for a batch of sequences (3D tensor)
    fn compute_gradients_batch(logits_batch: &[Vec<Vec<f64>>], labels_batch: &[Vec<Vec<f64>>], epsilon: f64) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>) {
        let mut analytical_grads = Vec::new();
        let mut numerical_grads = Vec::new();

        for (logits_seq, labels_seq) in logits_batch.iter().zip(labels_batch.iter()) {
            let mut seq_analytical_grads = Vec::new();
            let mut seq_numerical_grads = Vec::new();

            for (logits, labels) in logits_seq.iter().zip(labels_seq.iter()) {
                let analytical = analytical_gradient(logits, labels);
                let numerical = numerical_gradient_function(logits, labels, epsilon, cross_entropy_loss);

                seq_analytical_grads.push(analytical);
                seq_numerical_grads.push(numerical);
            }

            analytical_grads.push(seq_analytical_grads);
            numerical_grads.push(seq_numerical_grads);
        }

        (analytical_grads, numerical_grads)
    }


    pub fn backward_batch(softmax_output_batch:  &Vec<Vec<Vec<Complex<f64>>>>, target_token_ids: &Vec<Vec<u32>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let batch_size = softmax_output_batch.len();
        let seq_len = softmax_output_batch[0].len();
        let vocab_dim = softmax_output_batch[0][0].len();

        // Initialize gradient_batch with zeros
        let mut gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); vocab_dim]; seq_len]; batch_size];

        // Iterate over the batch of softmax outputs and target token IDs
        for (batch_index, (softmax_output, target_tokens)) in softmax_output_batch.iter().zip(target_token_ids.iter()).enumerate() {
            let seq_len = softmax_output.len();
            let target_len = target_tokens.len();

            if target_len > seq_len {
                panic!("Target length exceeds sequence length!");
            }

            let seq_ind_start = seq_len - target_len;

            for (sample_index, softmax_sample) in softmax_output[seq_ind_start..seq_len].iter().enumerate() {
                for (column_index, softmax_prob) in softmax_sample.iter().enumerate() {
                    let target = if target_tokens[sample_index] == column_index as u32 {
                        Complex::new(1.0, 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };

                    if target.norm() == 1.0 {
                        println!("target is 1: {}", column_index);
                    }

                    // Compute gradient
                    let gradient = softmax_prob - target;

                    // Store in batch-indexed gradient storage
                    gradient_batch[batch_index][sample_index][column_index] = gradient;
                }
            }
        }

        // Return the final gradient_batch
        gradient_batch
    }

    #[test]
    fn test_gradient() {
        let logits = vec![0.5, 0.8, 0.1];
        let labels = vec![0.0, 1.0, 0.0];
        let epsilon = 1e-8;

        let cross_entropy_loss_calculated = cross_entropy_loss(&softmax(&logits), &labels);

        let logits_batch = vec![vec![vec![Complex::new(0.5, 0.0), Complex::new(0.8, 0.0), Complex::new(0.1, 0.0)]]];
        let softmax_batch: Vec<Vec<Vec<Complex<f64>>>> =  logits_batch
        .iter() // Parallel iterator for the input batch
        .map(|input| softmax_complex(input)) // Apply `softmax_complex` to each input
        .collect();
    
        let labels_batch = vec![vec![1]];
        let cross_entropy_loss_complex = cross_entropy_loss_batch(&softmax_batch, &labels_batch);

        println!("calculated entropy loss f64: {:?}", cross_entropy_loss_calculated);
        println!("calculated entropy loss Complex f64: {:?}", cross_entropy_loss_complex);

        let analytical_grad_batch = backward_batch(&softmax_batch, &labels_batch);

        println!("analytical gradient batch Complex f64: {:?}", analytical_grad_batch);

        let analytical_grad = analytical_gradient(&logits, &labels);
        let numerical_grad = numerical_gradient_function(&logits, &labels, epsilon, cross_entropy_loss);

        println!("Analytical Gradient:");
        for (i, grad) in analytical_grad.iter().enumerate() {
            println!("dL/dz_{} = {}", i, grad);
        }

        println!("\nNumerical Gradient:");
        for (i, grad) in numerical_grad.iter().enumerate() {
            println!("dL/dz_{} = {}", i, grad);
        }
    }
}
