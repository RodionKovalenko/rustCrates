#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::neural_networks::utils::dtype::{c, c_to_f64, r, C};

    use crate::neural_networks::utils::{
        low_rank_approx::{low_rank_approx, reconstruction_error},
        matrix::{multiply, multiply_complex, transpose},
        random_arrays::generate_random_complex_2d,
    };

    #[cfg(feature = "dtype-f64")]
    use crate::neural_networks::utils::{
        matrix::multiply_complex_fear,
        random_arrays::generate_random_complex_3d,
    };

    fn relative_l2_error(a: &[Vec<C>], b: &[Vec<C>]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (row_a, row_b) in a.iter().zip(b.iter()) {
            for (za, zb) in row_a.iter().zip(row_b.iter()) {
                let za64 = c_to_f64(*za);
                let zb64 = c_to_f64(*zb);
                let dr = za64.re - zb64.re;
                let di = za64.im - zb64.im;
                num += dr * dr + di * di;
                den += za64.re * za64.re + za64.im * za64.im;
            }
        }
        (num.sqrt()) / (den.sqrt() + 1e-12)
    }

    #[test]
    fn test_multiply_arrays() {
        // 2x3 * 2x3 =
        let m1: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let m2: Vec<Vec<f64>> = vec![vec![5.0, 6.0, 7.0], vec![7.0, 8.0, 9.0]];

        let product = multiply(&m1, &m2);

        assert_eq!(product, [[38.0, 50.0], [92.0, 122.0]]);

        let m1: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let m2: Vec<Vec<i32>> = vec![vec![5, 6, 7], vec![7, 8, 9]];

        let product = multiply(&m1, &m2);

        assert_eq!(product, [[38.0, 50.0], [92.0, 122.0]]);

        let m1: Vec<Vec<i32>> = vec![vec![1, 4], vec![2, 5], vec![3, 6]];
        let m2: Vec<Vec<i32>> = vec![vec![5, 6, 7], vec![7, 8, 9]];

        let product = multiply(&m1, &m2);

        assert_eq!(product, [[33.0, 38.0, 43.0], [45.0, 52.0, 59.0], [57.0, 66.0, 75.0]]);

        let m1: Vec<Vec<f32>> = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        let m2: Vec<Vec<i32>> = vec![vec![5, 6, 7], vec![7, 8, 9]];

        let product = multiply(&m1, &m2);

        assert_eq!(product, [[33.0, 38.0, 43.0], [45.0, 52.0, 59.0], [57.0, 66.0, 75.0]]);

        // 4x3 * 4x2 =>   3x4 * 4x2 => 3x2
        let m1: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![4, 5, 6], vec![4, 5, 6]];
        let m2: Vec<Vec<i32>> = vec![vec![5, 6], vec![7, 8], vec![9, 5], vec![9, 5]];

        let product = multiply(&transpose(&m1), &m2);

        assert_eq!(product, [[105.0, 78.0], [135.0, 102.0], [165.0, 126.0]]);

        // 4x3 * 4x2 =>  3x4 * 4x2 => 3x2
        let m1: Vec<Vec<C>> = vec![
            vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)],
            vec![c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)],
            vec![c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)],
            vec![c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)],
        ];
        let m2: Vec<Vec<C>> = vec![
            vec![c(5.0, 0.0), c(6.0, 0.0)],
            vec![c(7.0, 0.0), c(8.0, 0.0)],
            vec![c(9.0, 0.0), c(5.0, 0.0)],
            vec![c(9.0, 0.0), c(5.0, 0.0)],
        ];

        let product: Vec<Vec<C>> = multiply_complex(&transpose(&m1), &m2);

        assert_eq!(
            product,
            [
                [c(105.0, 0.0), c(78.0, 0.0)],
                [c(135.0, 0.0), c(102.0, 0.0)],
                [c(165.0, 0.0), c(126.0, 0.0)]
            ]
        );

        // 3x2 * 2x6 =>  3x6
        let m1: Vec<Vec<C>> = vec![vec![c(1.0, 0.0), c(2.0, 0.0)], vec![c(3.0, 0.0), c(4.0, 0.0)], vec![c(5.0, 0.0), c(6.0, 0.0)]];
        let m2: Vec<Vec<C>> = vec![
            vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)],
            vec![c(7.0, 0.0), c(8.0, 0.0), c(9.0, 0.0), c(10.0, 0.0), c(11.0, 0.0), c(12.0, 0.0)],
        ];

        let product: Vec<Vec<C>> = multiply_complex(&m1, &m2);

        println!("product : {:?}", product);

        assert_eq!(
            product,
            [
                [
                    c(15.0, 0.0),
                    c(18.0, 0.0),
                    c(21.0, 0.0),
                    c(24.0, 0.0),
                    c(27.0, 0.0),
                    c(30.0, 0.0),
                ],
                [
                    c(31.0, 0.0),
                    c(38.0, 0.0),
                    c(45.0, 0.0),
                    c(52.0, 0.0),
                    c(59.0, 0.0),
                    c(66.0, 0.0),
                ],
                [
                    c(47.0, 0.0),
                    c(58.0, 0.0),
                    c(69.0, 0.0),
                    c(80.0, 0.0),
                    c(91.0, 0.0),
                    c(102.0, 0.0),
                ]
            ]
        );
    }

    #[test]
    #[cfg(feature = "dtype-f64")]
    fn test_multiply_complex_arrays() {
        let batch_size = 1;
        let seq_len = 5;
        let dim = 5;

        let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
        let weights = generate_random_complex_3d(batch_size, dim, seq_len);

        for (x, w) in input_batch.iter().zip(&weights) {
            let matrix_multip_cuda = multiply_complex(x, w);
            let matrix_multip_fear = multiply_complex_fear(x, w);

            println!("\n matrix cuda: {:?}", matrix_multip_cuda);
            println!("\n matrix fear: {:?}", matrix_multip_fear);

            crate::neural_networks::utils::derivative::test_gradient_error_2d(
                &matrix_multip_cuda,
                &matrix_multip_fear,
                1e-8,
            );
        }
    }

    #[test]
    fn test_low_rank_approx() {
        // Low-rank approximation
        let m: Vec<Vec<C>> = generate_random_complex_2d(10, 5);
        let rank = 6;
        let (u, v) = low_rank_approx(&m, rank, 200, r(1e-6));

        println!("\n Rank: {}", rank);
        println!("\n Original matrix: {:?}", m);
        println!("\n U matrix: {:?}", u);
        println!("\n V matrix: {:?}", v);

        // Verify dimensions
        assert_eq!(u.len(), m.len());
        assert_eq!(u[0].len(), rank);
        // reconstruct
        let reconstructed = multiply_complex(&u, &v);
        assert_eq!(reconstructed.len(), m.len());
        assert_eq!(reconstructed[0].len(), m[0].len());

        println!("\n Reconstructed matrix: {:?}", reconstructed);
        let error = reconstruction_error(&m, &u, &v);
        println!("\n Reconstruction error (Frobenius norm): {}", error);
        assert!(error < r(1e-5));
    }
    #[test]
    fn test_input_multiplication_with_low_rank_approx() {
        // Example input matrix (replace with your data)
        let start = Instant::now();
        let input: Vec<Vec<C>> = generate_random_complex_2d(50, 10);
        let m: Vec<Vec<C>> = generate_random_complex_2d(10, 100);
        println!("\n Generating random arrays took: {:?}", start.elapsed().as_secs_f64());

        // direkt multiplication
        let start = Instant::now();
        let matrix_result = multiply_complex(&input, &m);
        println!("\n Direct multiplication took: {:?}", start.elapsed().as_secs_f64());

        // Low-rank approximation
        let start = Instant::now();
        let rank = 16;
        let (u, v) = low_rank_approx(&m, rank, 400, r(1e-6));

        println!("\n Rank: {}", rank);

        // reconstruct
        let result_1 = multiply_complex(&input, &u);
        let result_2 = multiply_complex(&result_1, &v);
        println!("\nLow-rank approximation took: {:?}", start.elapsed().as_secs_f64());

        let rel_err = relative_l2_error(&matrix_result, &result_2);
        let tol = if cfg!(feature = "dtype-f64") { 1e-5 } else { 1e-2 };
        println!("\nRelative L2 error: {}", rel_err);
        assert!(rel_err < tol);
    }
}
