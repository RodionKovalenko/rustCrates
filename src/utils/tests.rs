#[cfg(test)]
mod tests {
    use crate::utils::array::convolve;
    use crate::utils::convolution_modes::ConvolutionMode;
    use crate::utils::linalg::{gaussian_elimination_inverse, generate_random_matrix, get_determinant, is_identity_matrix, multiply_matrices};
    use std::time::Instant;

    #[test]
    fn test_convolve_mode_full() {
        let kernel: Vec<f64> = vec![0.0, 1.0, 0.5];
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];

        let result = convolve(&kernel, &data, &ConvolutionMode::FULL);
        assert_eq!(result, vec![0.0, 1.0, 2.5, 4.0, 1.5]);

        let kernel: Vec<f64> = vec![1.0, 0.5];
        let result = convolve(&kernel, &data, &ConvolutionMode::FULL);
        assert_eq!(result, vec![1.0, 2.5, 4.0, 1.5]);
        let result = convolve(&data, &kernel, &ConvolutionMode::FULL);
        assert_eq!(result, vec![1.0, 2.5, 4.0, 1.5]);
    }
    #[test]
    fn test_convolve_mode_same() {
        let kernel: Vec<f64> = vec![0.0, 1.0, 0.5];
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];

        let result = convolve(&data, &kernel, &ConvolutionMode::SAME);
        assert_eq!(result, vec![1.0, 2.5, 4.0]);

        let result = convolve(&kernel, &data, &ConvolutionMode::SAME);
        assert_eq!(result, vec![1.0, 2.5, 4.0]);

        let kernel: Vec<f64> = vec![1.0, 0.5];
        let result = convolve(&kernel, &data, &ConvolutionMode::SAME);
        assert_eq!(result, vec![1.0, 2.5, 4.0]);
        let result = convolve(&data, &kernel, &ConvolutionMode::SAME);
        assert_eq!(result, vec![1.0, 2.5, 4.0]);
    }
    #[test]
    fn test_convolve_mode_valid() {
        let kernel: Vec<f64> = vec![0.0, 1.0, 0.5];
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];

        let result = convolve(&data, &kernel, &ConvolutionMode::VALID);
        assert_eq!(result, vec![2.5]);

        let result = convolve(&kernel, &data, &ConvolutionMode::VALID);
        assert_eq!(result, vec![2.5]);

        let kernel: Vec<f64> = vec![1.0, 0.5];
        let result = convolve(&kernel, &data, &ConvolutionMode::VALID);
        assert_eq!(result, vec![2.5, 4.0]);
        let result = convolve(&data, &kernel, &ConvolutionMode::VALID);
        assert_eq!(result, vec![2.5, 4.0]);
    }
    #[test]
    fn test_determinant() {
        let matrix: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 1.0, 4.0],
            vec![1.0, 6.0, 3.0],
        ];
        let determinant = get_determinant(&matrix);

        assert_eq!(determinant, 32.0);
    
        println!("determinant 3x3 : {}", determinant);
    
        let matrix: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 1.0, 4.0, 5.0],
            vec![1.0, 6.0, 3.0, 6.0],
            vec![2.0, 4.0, 7.0, 1.0],
        ];
        let determinant = get_determinant(&matrix);
        let rounded_determinant = (determinant * 1e10).round() / 1e10;
    
        println!("determinant 4x4: {}", rounded_determinant);

        assert_eq!(rounded_determinant, -254.0);
    }

    #[test]
    fn test_inverse_matrix_lib() {
        let start = Instant::now(); // Record the start time
    
        let matrix = generate_random_matrix(5);
        match gaussian_elimination_inverse(&matrix) {
            Some(inverse) => {
                println!("Inverse Matrix:");
                for row in &inverse {
                    println!("{:?}", row);
                }
    
                // Test if the product of the original matrix and its inverse is close to identity
                let product = multiply_matrices(&matrix, &inverse);
                println!("Matrix * Inverse:");
                for row in &product {
                    println!("{:?}", row);
                }
    
                // Check if the result is close to an identity matrix
                let tolerance = 1e-9;
                let is_identity = is_identity_matrix(&product, tolerance);
    
                if is_identity {
                    println!("The matrix is successfully inverted.");
                }
                
                // Ensure the matrix is an identity matrix
                assert_eq!(is_identity, true);
            }
            None => println!("Matrix is singular and cannot be inverted."),
        }
    
        let duration = start.elapsed(); // Get the elapsed time
        println!("Time elapsed in your_method: {:?}", duration);
    }
    
}