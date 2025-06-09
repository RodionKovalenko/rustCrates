#[cfg(test)]
mod test_attention_head_approximation {
    use num::Complex;

    use crate::neural_networks::{
        network_components::{performer::Performer, performer_complex::PerformerComplex},
        network_types::transformer::masked_attention_head::{scale_attention_scores, scale_attention_scores_f64},
        utils::{
            activation::{softmax_complex_real, softmax_matrix_f64},
            matrix::{multiply, multiply_complex, multiply_f64_complex, transpose},
            random_arrays::{generate_random_complex_2d, generate_random_f64_2d},
        },
    };

    #[test]
    fn test_complex_performer_approximation() {
        let d = 16; // input dimension
        let m = d * d; // number of random features
        let temp = (d as f64).sqrt();
        let rows = 15;
        let num_runs = 100;

        let mut errors = Vec::new();

        for _i in 0..num_runs {
            let approx = PerformerComplex::new(d, m, temp);

            // Example inputs (batch_size=2, d=4)
            let q = generate_random_complex_2d(rows, d);
            let k = generate_random_complex_2d(rows, d);
            let v = generate_random_complex_2d(rows, rows);

            // println!("\n q: {:?} \n v: {:?}, \n k: {:?}", &q, &v, &k);

            let result = approx.approximate_attention(&q, &k, &v);

            let attention_scores = multiply_complex(&q, &transpose(&k));
            let attention_scores_scaled = scale_attention_scores(&attention_scores, k[0].len() as f64);
            let attention_scores_activated = softmax_complex_real(&attention_scores_scaled);
            let attention_weights = multiply_f64_complex(&attention_scores_activated, &v);

            // if _i % 100 == 0 {
            //     println!("\n Approximated attention output:{:?}", &result);
            //     println!("\n Exact attention output:{:?}", &attention_weights);
            // }

            let frob_err = frobenius_error_complex(&attention_weights, &result);
            // println!("frobenius_error_f64: {:.4}", frob_err);
            errors.push(frob_err);
        }

        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let stddev = (errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
        println!("Mean Frobenius error: {mean_error}, Stddev: {stddev}");
    }

    fn frobenius_error_complex(a: &[Vec<Complex<f64>>], b: &[Vec<Complex<f64>>]) -> f64 {
        let mut num_sq = 0.0;
        let mut denom_sq = 0.0;
        for (row_a, row_b) in a.iter().zip(b.iter()) {
            for (val_a, val_b) in row_a.iter().zip(row_b.iter()) {
                let diff = *val_a - *val_b;
                num_sq += diff.norm_sqr();
                denom_sq += val_a.norm_sqr();
            }
        }

        (num_sq.sqrt()) / (denom_sq.sqrt() + 1e-12) // epsilon to avoid divide-by-zero
    }

    pub fn frobenius_error_f64(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> f64 {
        let mut num_sq = 0.0;
        let mut denom_sq = 0.0;

        for i in 0..a.len() {
            for j in 0..a[0].len() {
                let diff = a[i][j] - b[i][j];
                num_sq += diff * diff;
                denom_sq += a[i][j] * a[i][j];
            }
        }

        (num_sq.sqrt()) / (denom_sq.sqrt() + 1e-12) // epsilon to avoid divide-by-zero
    }

    pub fn mean_squared_error(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        let mut error = 0.0;
        for (v1, v2) in a.iter().zip(b.iter()) {
            for (&x1, &x2) in v1.iter().zip(v2.iter()) {
                error += (x1 - x2).powi(2);
            }
        }
        error / (a.len() as f64)
    }

    #[test]
    fn test_performer_approximation() {
        let d = 16; // input dimension
        let m = 4 * d; // number of random features
        let temp = (d as f64).sqrt();
        let rows = 10;
        let num_runs = 100;

        let mut errors = Vec::new();

        for _i in 0..num_runs {
            let approx = Performer::new(d, m, temp);

            // Example inputs (batch_size=2, d=4)
            let q = generate_random_f64_2d(rows, d);
            let k = generate_random_f64_2d(rows, d);
            let v = generate_random_f64_2d(rows, rows);

            // println!("\n q: {:?} \n v: {:?}, \n k: {:?}", &q, &v, &k);

            let result = approx.approximate_attention(&q, &k, &v);

            let attention_scores = multiply(&q, &transpose(&k));
            let attention_scores_scaled = scale_attention_scores_f64(&attention_scores, k[0].len() as f64);
            let attention_scores_activated = softmax_matrix_f64(&attention_scores_scaled);
            let attention_weights = multiply(&attention_scores_activated, &v);

            // println!("\n Approximated attention output:{:?}", &result);
            // println!("\n Exact attention output:{:?}", &attention_weights);

            let _error = mean_squared_error(&attention_weights, &result);
            // println!("mean_squared_error: {:.4}", error);

            let frob_err = frobenius_error_f64(&attention_weights, &result);
            // println!("frobenius_error_f64: {:.4}", frob_err);
            errors.push(frob_err);
        }

        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let stddev = (errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
        println!("Mean Frobenius error: {mean_error}, Stddev: {stddev}");
    }
}
