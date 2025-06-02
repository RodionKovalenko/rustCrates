#[cfg(test)]
mod test_attention_head_approximation {
    use num::Complex;

    use crate::neural_networks::{
        network_types::transformer::masked_attention_head::scale_attention_scores,
        utils::{
            activation::softmax_complex_real,
            matrix::{multiply_complex, multiply_f64_complex, transpose},
            matrix_approximation::softmax_re_qk_approx_stable,
            random_arrays::generate_random_complex_3d,
        },
    };

    #[test]
    fn test_matrix_multiplication_approximation() {
        // Define some small batch size and input dimensions for simplicity
        let batch_size = 1;
        let seq_len = 5;
        let dim = 5;

        // Simulate query and key matrices
        let query_batch = generate_random_complex_3d(batch_size, seq_len, dim);
        let key_batch = generate_random_complex_3d(batch_size, seq_len, dim);

        let r: usize = 64;

        for (q, k) in query_batch.iter().zip(&key_batch) {
            let v = &q.clone();

            let approx = softmax_re_qk_approx_stable(q, k, v, r);

            let attn_scores = multiply_complex(q, &transpose(&k));
            let scaled_scores = scale_attention_scores(&attn_scores, k[0].len() as f64);
            let softmax_exact = softmax_complex_real(&scaled_scores);
            let attention_weight = multiply_f64_complex(&softmax_exact, v);

            println!("Exact: {:?}", attention_weight);
            println!("Approx: {:?}", approx);

            let error = frobenius_error(&attention_weight, &approx);
            println!("Relative Frobenius error: {:.4}", error);

            assert!(error.norm() < 0.3, "Approximation error too high: {}", error);
        }
    }

    pub fn frobenius_error(a: &Vec<Vec<Complex<f64>>>, b: &Vec<Vec<Complex<f64>>>) -> Complex<f64> {
        let mut sum_sq = Complex::new(0.0, 0.0);
        let mut denom = Complex::new(0.0, 0.0);
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                let diff = a[i][j] - b[i][j];
                sum_sq += diff * diff;
                denom += a[i][j] * a[i][j];
            }
        }
        (sum_sq / denom).sqrt()
    }
}
