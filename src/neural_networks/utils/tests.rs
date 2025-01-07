#[cfg(test)]
mod tests {
    use num::Complex;

    use crate::neural_networks::utils::matrix::{multiply, multiply_complex};

    #[test]
    fn test_multiply_arrays() {

        // 2x3 * 2x3 = 
        let m1: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let m2: Vec<Vec<f64>> = vec![vec![5.0, 6.0, 7.0], vec![7.0, 8.0, 9.0]];

        let product = multiply(&m1, &m2);

        assert_eq!(
            product,
            [[38.0, 50.0], [92.0, 122.0]]
        );

        let m1: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let m2: Vec<Vec<i32>> = vec![vec![5, 6, 7], vec![7, 8, 9]];

        let product = multiply(&m1, &m2);

        assert_eq!(
            product,
            [[38.0, 50.0], [92.0, 122.0]]
        );

        let m1: Vec<Vec<i32>> = vec![vec![1, 4], vec![2, 5], vec![3, 6]];
        let m2: Vec<Vec<i32>> = vec![vec![5, 6, 7], vec![7, 8, 9]];

        let product = multiply(&m1, &m2);

        assert_eq!(
            product,
            [[33.0, 38.0, 43.0], [45.0, 52.0, 59.0], [57.0, 66.0, 75.0]]
        );

        let m1: Vec<Vec<f32>> = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        let m2: Vec<Vec<i32>> = vec![vec![5, 6, 7], vec![7, 8, 9]];

        let product = multiply(&m1, &m2);

        assert_eq!(
            product,
            [[33.0, 38.0, 43.0], [45.0, 52.0, 59.0], [57.0, 66.0, 75.0]]
        );

        // 4x3 * 4x2 =>   3x4 * 4x2 => 3x2
        let m1: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![4, 5, 6], vec![4, 5, 6]];
        let m2: Vec<Vec<i32>> = vec![vec![5, 6], vec![7, 8], vec![9, 5], vec![9, 5]];

        let product = multiply(&m1, &m2);

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

        let product: Vec<Vec<Complex<f64>>> = multiply_complex(&m1, &m2);

        assert_eq!(
            product,
            [
                [
                    Complex { re: 105.0, im: 0.0 },
                    Complex { re: 78.0, im: 0.0 }
                ],
                [
                    Complex { re: 135.0, im: 0.0 },
                    Complex { re: 102.0, im: 0.0 }
                ],
                [
                    Complex { re: 165.0, im: 0.0 },
                    Complex { re: 126.0, im: 0.0 }
                ]
            ]
        );
    }
}
