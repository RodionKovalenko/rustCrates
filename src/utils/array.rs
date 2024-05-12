use std::cmp::{max, min};
use crate::utils::convolution_modes::ConvolutionMode;

/*
    Aranges a vector of f64 values from start to end with a step value
    e.g. arange(0.0, 10.0, 1.0) -> [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
 */
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

/*
    Creates a vector of f64 values from start to end with n_samples
    e.g. linspace(0.0, 10.0, 5, false) -> [0.0, 2.5, 5.0, 7.5, 10.0]
 */
pub fn linspace(start: &f64, end: &f64, n_samples: &i32, include_end: bool) -> Vec<f64> {
    let span = end - start;
    let n = n_samples.clone() as f64;
    let mut step: f64 = span / n.clone();

    let mut result: Vec<f64> = vec![0.0; n_samples.clone() as usize];
    let last_i = result.len().clone() - 1;
    let mut tmp: f64 = start.clone();

    if include_end {
        step = span.clone() / (n.clone() - 1.0);
    }

    for i in 0..result.len() {
        result[i] = tmp.clone();
        tmp += step.clone();
    }

    if include_end {
        result[last_i] = end.clone();
    }

    result
}

/*
     Convolves a kernel with a data vector
     three modes are possible: full, same, valid
 */
pub fn convolve(data: &Vec<f64>, kernel: &Vec<f64>, conv_mode: &ConvolutionMode) -> Vec<f64> {
    let mut data_clone = data.clone();
    let mut kernel_clone = kernel.clone();

    if kernel.len() < data.len() {
        data_clone = kernel.clone();
        kernel_clone = data.clone();
    }

    let mut filter_padded: Vec<f64> = kernel_clone.clone();
    let data_rev: Vec<f64> = data_clone.to_vec().iter().rev().cloned().collect();
    let n: usize;
    let padding: usize;

    match conv_mode {
        ConvolutionMode::FULL => {
            n = data.len() + kernel.len() - 1;
            padding = n - kernel_clone.len();

            for _i in 0..padding {
                filter_padded.insert(0, 0.0);
                filter_padded.push(0.0);
            }
        }
        ConvolutionMode::SAME => {
            n = max(data.len(), kernel.len());
            padding = n - kernel_clone.len() + 1;

            for _i in 0..padding {
                filter_padded.insert(0, 0.0);
                filter_padded.push(0.0);
            }
        }
        ConvolutionMode::VALID => {
            n = max(data.len(), kernel.len()) - min(data.len(), kernel.len()) + 1;
        }
    }

    let mut result: Vec<f64> = vec![0.0; n];
    let mut sum: f64;
    let mut d;

    for j in 0..result.len() {
        sum = 0.0;

        for i in 0..kernel_clone.len() {
            d = i.clone() + j.clone();
            if i >= data_clone.len() {
                break;
            }
            sum += filter_padded[d.clone() % filter_padded.len()].clone() * &data_rev[i.clone()];
        }
        result[j] = sum;
    }

    result
}

/*
    Convolves a kernel with a 2-d data vector using the convolve function
 */
pub fn convolve_2d(kernel: &Vec<f64>, data: &Vec<Vec<f64>>, conv_mode: &ConvolutionMode) -> Vec<Vec<f64>> {
    let mut convolved: Vec<Vec<f64>> = Vec::new();

    for (_i, row) in data.iter().enumerate() {
        convolved.push(convolve(kernel, row, conv_mode));
    }

    convolved
}

pub fn convolve_3d(kernel: &Vec<f64>, data: &Vec<Vec<Vec<f64>>>, conv_mode: &ConvolutionMode) -> Vec<Vec<Vec<f64>>> {
    let mut convolved: Vec<Vec<Vec<f64>>> = Vec::new();

    for (_i, row) in data.iter().enumerate() {
        convolved.push(convolve_2d(kernel, row, conv_mode));
    }

    convolved
}

/*
    Integrates a vector using the rectangle integration method.
 */
pub fn integrate(x: &Vec<f64>, y: &Vec<f64>, scale: &f64) -> Vec<f64> {
    let step = x[1].clone() - x[0].clone();
    let mut integral: Vec<f64> = Vec::new();
    let mut integral_psi_scaled: Vec<f64> = Vec::new();
    let mut sum: f64;

    let start = 0.0;
    let range = x[x.len() - 1].clone() - (x[0].clone());
    let end = (scale * range) + 1.0;
    let arange = arange(&start, &end, &1.0);

    for i in 0..y.len() {
        sum = y[i].clone() * step.clone();

        if (i.clone() as i32) > 0 {
            sum += integral[i.clone() - 1].clone();
        }
        integral.push(sum);
    }

    for v in arange.iter() {
        let index = (v / (scale.clone() * step.clone())) as usize;

        integral_psi_scaled.push(integral[index].clone());
    }

    integral_psi_scaled.to_vec().iter().rev().cloned().collect()
}

/*
    returns the array of differences multiplied by the square root of the scale
 */
pub fn get_coef(a: &Vec<f64>, scale: &f64) -> Vec<f64> {
    let mut diff: Vec<f64> = Vec::new();
    let scaled_sqrt = -1.0 * scale.clone().sqrt();
    let mut v;

    for i in 0..(a.len() - 1) {
        v = &scaled_sqrt * (&a[i + 1] - &a[i.clone()]);
        diff.push(v);
    }
    diff
}

/*
    returns the 2-d array of differences multiplied by the square root of the scale
 */
pub fn get_coef_2d(a: &Vec<Vec<f64>>, scale: &f64) -> Vec<Vec<f64>> {
    let mut diff: Vec<Vec<f64>> = Vec::new();
    let mut vec: Vec<f64>;

    for i in 0..(a.len()) {
        vec = get_coef(&a[i], scale);
        diff.push(vec);
    }

    diff
}

pub fn get_coef_3d(a: &Vec<Vec<Vec<f64>>>, scale: &f64) -> Vec<Vec<Vec<f64>>> {
    let mut diff: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut vec: Vec<Vec<f64>>;

    for i in 0..(a.len()) {
        vec = get_coef_2d(&a[i], scale);
        diff.push(vec);
    }

    diff
}