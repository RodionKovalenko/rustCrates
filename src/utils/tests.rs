#[cfg(test)]
mod tests {
    use crate::utils::array::convolve;
    use crate::utils::convolution_modes::ConvolutionMode;

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
}