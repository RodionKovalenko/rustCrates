#[cfg(test)]
mod tests {
    use crate::utils::array::convolve;
    use crate::utils::convolution_modes::ConvolutionMode;
    use crate::utils::linalg::get_determinant;

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
}