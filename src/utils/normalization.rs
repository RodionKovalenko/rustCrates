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