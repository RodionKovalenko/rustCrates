use num::Complex;

pub fn normalize(input: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let len = input.len() as f64;

    let mean: Complex<f64> = input.iter().sum::<Complex<f64>>() / len;

    let variance: Complex<f64> = input
        .iter()
        .map(|x| {
            let diff = *x - mean;
            diff * diff
        })
        .sum::<Complex<f64>>()
        / len;

    let stddev = (variance + 0.00000000001).sqrt();

    input.iter().map(|x| (*x - mean) / stddev).collect()
}

pub fn normalize_matrix(input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    input.iter().map(|x| normalize(x)).collect()
}

pub fn normalize_f64(input: &Vec<f64>) -> Vec<f64> {
    let len = input.len() as f64;

    let mean: f64 = input.iter().sum::<f64>() / len;

    let variance: f64 = input
        .iter()
        .map(|x| {
            let diff = *x - mean;
            diff * diff
        })
        .sum::<f64>()
        / len;

    let stddev = (variance + 0.00000000001).sqrt();

    input.iter().map(|x| (*x - mean) / stddev).collect()
}

pub fn normalize_matrix_f64(input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    input.iter().map(|x| normalize_f64(x)).collect()
}