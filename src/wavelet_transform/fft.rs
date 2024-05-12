extern crate num_traits;

use std::f64::consts::PI;
use num::Complex;

pub fn fft_real1_d(input: &Vec<f64>) -> Vec<Complex<f64>> {
    let complex_input: Vec<Complex<f64>> = input.iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft_1d(&complex_input)
}

pub fn fft_real2_d(input: &Vec<Vec<f64>>) -> Vec<Vec<Complex<f64>>> {
    let complex_input: Vec<Vec<Complex<f64>>> = input.iter()
        .map(|x| x.iter().map(|&y| Complex::new(y, 0.0)).collect())
        .collect();
    fft_2d(complex_input)
}

pub fn fft_1d(signal: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = signal.len();
    if n <= 1 {
        return signal.to_vec();
    }

    let mut result = vec![Complex::default(); n];

    for k in 0..n {
        let mut sum = Complex::default();
        for j in 0..n {
            let exp_factor = Complex::from_polar(1.0, -2.0 * PI * j as f64 * k as f64 / n as f64);
            sum += signal[j] * exp_factor;
        }
        result[k] = sum;
    }

    result
}

pub fn fft_2d(input: Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let rows = input.len();
    let cols = input[0].len();

    // Apply FFT to each row
    let row_fft: Vec<Vec<Complex<f64>>> = input.into_iter()
        .map(|row| fft_1d(&row))
        .collect();

    // Transpose to apply FFT to columns
    let mut transposed: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); rows]; cols];
    for (i, row) in row_fft.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            transposed[j][i] = val;
        }
    }

    // Apply FFT to each column in the transposed matrix
    let col_fft = transposed.into_iter()
        .map(|col| fft_1d(&col))
        .collect::<Vec<_>>();

    // Transpose back
    let mut final_output: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
    for (i, row) in col_fft.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            final_output[j][i] = val;
        }
    }

    final_output
}
