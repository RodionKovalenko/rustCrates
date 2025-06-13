#[cfg(test)]
mod tests {
    use num::Complex;

    use crate::neural_networks::utils::{
        derivative::test_gradient_error_2d,
        matrix::{multiply, multiply_complex, transpose},
        random_arrays::generate_random_complex_3d,
    };

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
        let m1: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)],
            vec![Complex::new(4.0, 0.0), Complex::new(5.0, 0.0), Complex::new(6.0, 0.0)],
            vec![Complex::new(4.0, 0.0), Complex::new(5.0, 0.0), Complex::new(6.0, 0.0)],
            vec![Complex::new(4.0, 0.0), Complex::new(5.0, 0.0), Complex::new(6.0, 0.0)],
        ];
        let m2: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(5.0, 0.0), Complex::new(6.0, 0.0)],
            vec![Complex::new(7.0, 0.0), Complex::new(8.0, 0.0)],
            vec![Complex::new(9.0, 0.0), Complex::new(5.0, 0.0)],
            vec![Complex::new(9.0, 0.0), Complex::new(5.0, 0.0)],
        ];

        let product: Vec<Vec<Complex<f64>>> = multiply_complex(&transpose(&m1), &m2);

        assert_eq!(
            product,
            [
                [Complex { re: 105.0, im: 0.0 }, Complex { re: 78.0, im: 0.0 }],
                [Complex { re: 135.0, im: 0.0 }, Complex { re: 102.0, im: 0.0 }],
                [Complex { re: 165.0, im: 0.0 }, Complex { re: 126.0, im: 0.0 }]
            ]
        );

        // 3x2 * 2x6 =>  3x6
        let m1: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)], vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)], vec![Complex::new(5.0, 0.0), Complex::new(6.0, 0.0)]];
        let m2: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0), Complex::new(4.0, 0.0), Complex::new(5.0, 0.0), Complex::new(6.0, 0.0)],
            vec![Complex::new(7.0, 0.0), Complex::new(8.0, 0.0), Complex::new(9.0, 0.0), Complex::new(10.0, 0.0), Complex::new(11.0, 0.0), Complex::new(12.0, 0.0)],
        ];

        let product: Vec<Vec<Complex<f64>>> = multiply_complex(&m1, &m2);

        println!("product : {:?}", product);

        assert_eq!(
            product,
            [
                [
                    Complex { re: 15.0, im: 0.0 },
                    Complex { re: 18.0, im: 0.0 },
                    Complex { re: 21.0, im: 0.0 },
                    Complex { re: 24.0, im: 0.0 },
                    Complex { re: 27.0, im: 0.0 },
                    Complex { re: 30.0, im: 0.0 },
                ],
                [
                    Complex { re: 31.0, im: 0.0 },
                    Complex { re: 38.0, im: 0.0 },
                    Complex { re: 45.0, im: 0.0 },
                    Complex { re: 52.0, im: 0.0 },
                    Complex { re: 59.0, im: 0.0 },
                    Complex { re: 66.0, im: 0.0 },
                ],
                [
                    Complex { re: 47.0, im: 0.0 },
                    Complex { re: 58.0, im: 0.0 },
                    Complex { re: 69.0, im: 0.0 },
                    Complex { re: 80.0, im: 0.0 },
                    Complex { re: 91.0, im: 0.0 },
                    Complex { re: 102.0, im: 0.0 },
                ]
            ]
        );
    }

    #[test]
    fn test_multiply_complex_arrays() {
        let batch_size = 1;
        let seq_len = 5;
        let dim = 5;

        let input_batch = generate_random_complex_3d(batch_size, seq_len, dim);
        let weights = generate_random_complex_3d(batch_size, dim, seq_len);

        for (x, w) in input_batch.iter().zip(&weights) {
            let matrix_multip_cuda = multiply_complex(x, w);
            let matrix_multip_fear = multiply_complex_fear(x, w);

            println!("\n matrix cude: {:?}", matrix_multip_cuda);
            println!("\n matrix fear: {:?}", matrix_multip_fear);

            test_gradient_error_2d(&matrix_multip_cuda, &matrix_multip_fear, 1e-8);
        }
    }
}
