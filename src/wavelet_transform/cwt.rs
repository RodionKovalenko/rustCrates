use num_traits::real::Real;
use crate::utils::array::{arange, convolve, linspace};
use crate::utils::convolution_modes::ConvolutionMode;
use crate::wavelet_transform::cwt_type_resolver::transform_by_type;
use crate::wavelet_transform::cwt_types::ContiousWaletetType;

pub fn cwt_1d(data: Vec<f64>, scales: Vec<f64>, cw_type: &ContiousWaletetType, sampling_period: &f64) -> Vec<Vec<f64>> {
    let mut wavefun_result: Vec<Vec<f64>>;
    let mut wavelets: Vec<Vec<f64>> = Vec::new();
    let mut integral_scaled: Vec<f64> = Vec::new();
    let mut convolved = Vec::new();
    let mut coef: Vec<f64> = Vec::new();
    let mut index: f64 = 0.0;
    let mut limit_down;
    let mut limit_up;

    for s in 0..scales.len() {
        let scale = scales[s.clone()].clone();
        wavefun_result = wavefun(10, scale, &cw_type);
        integral_scaled = integrate(&wavefun_result[0], &wavefun_result[1], &scale);
        convolved = convolve(&integral_scaled, &data, &ConvolutionMode::FULL);
        coef = get_coef(&convolved, &scale);

        index = (coef.len() - data.len()) as f64 / 2.0;
        limit_down = index.floor() as usize;
        limit_up = index.ceil() as usize;

        wavelets.push(coef[limit_down..(coef.len() - limit_up)].to_vec());
    }

    wavelets
}

pub fn wavefun(precision: i32, scale: f64, cw_type: &ContiousWaletetType) -> Vec<Vec<f64>> {
    let mut y: Vec<f64> = Vec::new();
    let mut x_y_vec = Vec::new();
    let start = -8.0;
    let end = 8.0;
    let two_i32 = 2.0 as i32;
    let n_points = two_i32.pow(precision.clone() as u32);

    let x: Vec<f64> = linspace(&start, &end, &n_points, true);
    for i in 0..x.len() {
        y.push(transform_by_type(&x[i], &1, &cw_type));
    }

    x_y_vec.push(x);
    x_y_vec.push(y);

    x_y_vec
}

pub fn integrate(x: &Vec<f64>, y: &Vec<f64>, scale: &f64) -> Vec<f64> {
    let step = x[1].clone() - x[0].clone();
    let mut integral: Vec<f64> = Vec::new();
    let mut integral_psi_scaled: Vec<f64> = Vec::new();
    let mut sum: f64;

    let start = 0.0;
    let range = x[x.len() - 1].clone() - (x[0].clone());
    let end = ((scale * range) + 1.0);
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