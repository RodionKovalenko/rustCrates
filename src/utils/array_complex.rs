/*
    Creates a vector of f64 values from start to end with n_samples
    e.g. linspace(0.0, 10.0, 5, false) -> [0.0, 2.5, 5.0, 7.5, 10.0]
 */
use std::cmp::{max, min};
use num_complex::Complex;
use crate::utils::array::arange;
use crate::utils::convolution_modes::ConvolutionMode;

pub fn linspace_complex(start: &f64, end: &f64, n_samples: &i32, include_end: bool) -> Vec<Complex<f64>> {
    let span: f64 = end - start;
    let n: f64 = n_samples.clone() as f64;
    let mut step: f64 = span / n.clone();

    let mut result: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_samples.clone() as usize];
    let last_i = result.len().clone() - 1;
    let mut tmp: Complex<f64> = Complex::new(start.clone(), 0.0);

    if include_end {
        step = span.clone() / (n.clone() - 1.0);
    }

    for i in 0..result.len() {
        result[i] = tmp.clone();
        tmp += step.clone();
    }

    if include_end {
        result[last_i] = Complex::from(end.clone());
    }

    result
}
/*
     Convolves a kernel with a data vector
     three modes are possible: full, same, valid
 */
pub fn convolve_complex(data: &Vec<Complex<f64>>, kernel: &Vec<Complex<f64>>, conv_mode: &ConvolutionMode) -> Vec<Complex<f64>> {
    let mut data_clone = data.clone();
    let mut kernel_clone = kernel.clone();

    if kernel.len() < data.len() {
        data_clone = kernel.clone();
        kernel_clone = data.clone();
    }

    let mut filter_padded: Vec<Complex<f64>> = kernel_clone.clone();
    let data_rev: Vec<Complex<f64>> = data_clone.to_vec().iter().rev().cloned().collect();
    let n: usize;
    let padding: usize;

    match conv_mode {
        ConvolutionMode::FULL => {
            n = data.len() + kernel.len() - 1;
            padding = n - kernel_clone.len();

            for _i in 0..padding {
                filter_padded.insert(0, Complex::new(0.0, 0.0));
                filter_padded.push(Complex::new(0.0, 0.0));
            }
        }
        ConvolutionMode::SAME => {
            n = max(data.len(), kernel.len());
            padding = n - kernel_clone.len() + 1;

            for _i in 0..padding {
                filter_padded.insert(0, Complex::new(0.0, 0.0));
                filter_padded.push(Complex::new(0.0, 0.0));
            }
        }
        ConvolutionMode::VALID => {
            n = max(data.len(), kernel.len()) - min(data.len(), kernel.len()) + 1;
        }
    }

    let mut result: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    let mut sum: Complex<f64>;
    let mut d;

    for j in 0..result.len() {
        sum = Complex::new(0.0, 0.0);

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
pub fn convolve_2d_complex(kernel: &Vec<Complex<f64>>, data: &Vec<Vec<Complex<f64>>>, conv_mode: &ConvolutionMode) -> Vec<Vec<Complex<f64>>> {
    let mut convolved: Vec<Vec<Complex<f64>>> = Vec::new();

    for (_i, row) in data.iter().enumerate() {
        convolved.push(convolve_complex(kernel, row, conv_mode));
    }

    convolved
}

pub fn convolve_3d_complex(kernel: &Vec<Complex<f64>>, data: &Vec<Vec<Vec<Complex<f64>>>>, conv_mode: &ConvolutionMode) -> Vec<Vec<Vec<Complex<f64>>>> {
    let mut convolved: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

    for (_i, row) in data.iter().enumerate() {
        convolved.push(convolve_2d_complex(kernel, row, conv_mode));
    }

    convolved
}

/*
    Integrates a vector using the rectangle integration method.
 */
pub fn integrate_complex(x: &Vec<Complex<f64>>, y: &Vec<Complex<f64>>, scale: &f64) -> Vec<Complex<f64>> {
    let step: f64 = x[1].re.clone() - x[0].re.clone();
    let mut integral: Vec<Complex<f64>> = Vec::new();
    let mut integral_psi_scaled: Vec<Complex<f64>> = Vec::new();
    let mut sum: Complex<f64>;

    let start = 0.0;
    let range: f64 = x[x.len() - 1].re.clone() - (x[0].re.clone());
    let end: f64 = (scale * range) + 1.0;
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
pub fn get_coef_complex(a: &Vec<Complex<f64>>, scale: &f64) -> Vec<Complex<f64>> {
    let mut diff: Vec<Complex<f64>> = Vec::new();
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
pub fn get_coef_2d_complex(a: &Vec<Vec<Complex<f64>>>, scale: &f64) -> Vec<Vec<Complex<f64>>> {
    let mut diff: Vec<Vec<Complex<f64>>> = Vec::new();
    let mut vec: Vec<Complex<f64>>;

    for i in 0..(a.len()) {
        vec = get_coef_complex(&a[i], scale);
        diff.push(vec);
    }

    diff
}

pub fn get_coef_3d_complex(a: &Vec<Vec<Vec<Complex<f64>>>>, scale: &f64) -> Vec<Vec<Vec<Complex<f64>>>> {
    let mut diff: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
    let mut vec: Vec<Vec<Complex<f64>>>;

    for i in 0..(a.len()) {
        vec = get_coef_2d_complex(&a[i], scale);
        diff.push(vec);
    }

    diff
}
