use num_traits::abs;
use crate::utils::convolution_modes::ConvolutionMode;

pub fn arange(start: &f64, end: &f64, step: &f64) -> Vec<f64> {
    let size = ((end - start) / step) as usize;
    let mut result: Vec<f64> = vec![0.0; size];
    let mut tmp = start.clone();

    for i in 0..result.len() {
        result[i] = tmp.clone();

        tmp += step.clone();
    }

    result
}

pub fn linspace(start: &f64, end: &f64, n_samples: &i32, include_end: bool) -> Vec<f64> {
    let span = (end - start) as f64;
    let n = n_samples.clone() as f64;
    let mut step: f64 = span / n.clone();

    let mut result: Vec<f64> = vec![0.0; n_samples.clone() as usize];
    let last_i = result.len().clone() - 1;
    let mut tmp: f64 = start.clone() as f64;

    if include_end {
        step = span.clone() / (n.clone() - 1.0);
    }

    for i in 0..result.len() {
        result[i] = tmp.clone();
        tmp += step.clone();
    }

    if include_end {
        result[last_i] = end.clone() as f64;
    }

    result
}

pub fn convolve(kernel: &Vec<f64>, data: &Vec<f64>, conv_mode: &ConvolutionMode) -> Vec<f64> {
    let mut filter_padded: Vec<f64> = kernel.clone();
    let data_rev: Vec<f64> = data.to_vec().iter().rev().cloned().collect();
    let n = data.len() + kernel.len() - 1;
    let padding = n - kernel.len();

    for _i in 0..padding {
        filter_padded.insert(0, 0.0);
        filter_padded.push(0.0);
    }

    let mut result: Vec<f64> = vec![0.0; n];
    let mut sum: f64;
    let mut d;

    for j in 0..result.len() {
        sum = 0.0;

        for i in 0..kernel.len() {
            d = (i.clone() + j.clone());
            if i >= data.len() {
                break;
            }
            sum += filter_padded[d.clone() % filter_padded.len()].clone() * &data_rev[i.clone()];
        }
        result[j] = sum;
    }

    result
}
