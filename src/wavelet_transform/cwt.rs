use std::cmp::{Ordering};
use crate::utils::array::{convolve, convolve_2d, convolve_3d, get_coef, get_coef_2d, get_coef_3d, integrate, linspace};
use crate::utils::convolution_modes::ConvolutionMode;
use crate::wavelet_transform::cwt_type_resolver::transform_by_type;
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;
use crate::wavelet_transform::fft::fft_real1_d;

pub fn cwt_1d(data: &Vec<f64>, scales: &Vec<f64>, cw_type: &ContinuousWaletetType, sampling_period: &f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let wavefun_result: Vec<Vec<f64>> = wavefun(&10, &cw_type);
    let freqencies: Vec<f64> = scale_to_frequency(scales, &wavefun_result, sampling_period);

    let mut wavelets: Vec<Vec<f64>> = Vec::new();
    let mut integral_scaled: Vec<f64>;
    let mut convolved: Vec<f64>;
    let mut coef: Vec<f64>;
    let mut index: f64;
    let mut limit_down: usize;
    let mut limit_up: usize;

    for s in 0..scales.len() {
        let scale = &scales[s];
        integral_scaled = integrate(&wavefun_result[0], &wavefun_result[1], scale);
        convolved = convolve(data, &integral_scaled, &ConvolutionMode::FULL);
        coef = get_coef(&convolved, scale);

        index = (coef.len() - data.len()) as f64 / 2.0;
        limit_down = index.floor() as usize;
        limit_up = index.ceil() as usize;

        wavelets.push(coef[limit_down..(coef.len() - limit_up)].to_vec());
    }

    (wavelets, freqencies)
}

pub fn cwt_2d(data: &Vec<Vec<f64>>, scales: &Vec<f64>, cw_type: &ContinuousWaletetType, sampling_period: &f64) -> (Vec<Vec<Vec<f64>>>, Vec<f64>) {
    let wavefun_result: Vec<Vec<f64>> = wavefun(&10, &cw_type);
    let freqencies: Vec<f64> = scale_to_frequency(scales, &wavefun_result, sampling_period);

    let mut wavelets: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut integral_scaled: Vec<f64>;
    let mut convolved: Vec<Vec<f64>>;
    let mut coef: Vec<Vec<f64>>;
    let mut row_slice: Vec<Vec<f64>>;
    let mut index: f64;
    let mut limit_down: usize;
    let mut limit_up: usize;

    for s in 0..scales.len() {
        let scale = &scales[s];
        integral_scaled = integrate(&wavefun_result[0], &wavefun_result[1], scale);
        convolved = convolve_2d(&integral_scaled, data, &ConvolutionMode::FULL);

        coef = get_coef_2d(&convolved, scale);
        index = (coef[0].len() - data[0].len()) as f64 / 2.0;

        limit_down = index.floor() as usize;
        limit_up = index.ceil() as usize;
        row_slice = Vec::new();

        for i in 0..coef.len() {
            row_slice.push(coef[i][limit_down..(coef[i].len() - limit_up)].to_vec());
        }

        wavelets.push(row_slice);
    }

    (wavelets, freqencies)
}

pub fn cwt_3d(data: &Vec<Vec<Vec<f64>>>, scales: &Vec<f64>, cw_type: &ContinuousWaletetType, sampling_period: &f64) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>) {
    let wavefun_result: Vec<Vec<f64>> = wavefun(&10, &cw_type);
    let freqencies: Vec<f64> = scale_to_frequency(scales, &wavefun_result, sampling_period);
    let mut wavelets: Vec<Vec<Vec<Vec<f64>>>> = Vec::new();
    let mut integral_scaled: Vec<f64>;
    let mut convolved: Vec<Vec<Vec<f64>>>;
    let mut coef: Vec<Vec<Vec<f64>>>;
    let mut row_slice: Vec<Vec<Vec<f64>>>;
    let mut record_slice: Vec<Vec<f64>>;
    let mut index: f64;
    let mut limit_down: usize;
    let mut limit_up: usize;

    for s in 0..scales.len() {
        let scale = &scales[s];
        integral_scaled = integrate(&wavefun_result[0], &wavefun_result[1], scale);
        convolved = convolve_3d(&integral_scaled, data, &ConvolutionMode::FULL);

        coef = get_coef_3d(&convolved, scale);
        index = (coef[0][0].len() - data[0][0].len()) as f64 / 2.0;

        limit_down = index.floor() as usize;
        limit_up = index.ceil() as usize;
        row_slice = Vec::new();

        for i in 0..coef.len() {
            record_slice = Vec::new();

            for j in 0..coef[i].len() {
                record_slice.push(coef[i][j][limit_down..(coef[i][j].len() - limit_up)].to_vec());
            }
            row_slice.push(record_slice);
        }

        wavelets.push(row_slice);
    }

    (wavelets, freqencies)
}

pub fn wavefun(precision: &i32, cw_type: &ContinuousWaletetType) -> Vec<Vec<f64>> {
    let mut y: Vec<f64> = Vec::new();
    let mut x_y_vec = Vec::new();
    let start = -8.0;
    let end = 8.0;
    let two_i32 = 2.0 as i32;
    let n_points = two_i32.pow(precision.clone() as u32);

    let x: Vec<f64> = linspace(&start, &end, &n_points, true);
    for i in 0..x.len() {
        y.push(transform_by_type(&x[i], &1.0, &cw_type));
    }

    x_y_vec.push(x);
    x_y_vec.push(y);

    x_y_vec
}


fn get_central_frequency(wavefun: &Vec<Vec<f64>>) -> f64 {
    let (x, y) = (&wavefun[0], &wavefun[1]);
    let central_frequency: f64;
    let domain: f64 = &x[x.len() - 1] - &x[0];
    let fft = fft_real1_d(&y);

    let abs_fft: Vec<f64> = fft.iter().skip(1).map(|c| c.norm()).collect();

    // Find the index of the maximum absolute value
    let max_index_option = abs_fft.iter().enumerate().max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(Ordering::Equal)).map(|(i, _)| i);

    // Add 2 to the index (since we skipped the first element)
    let mut max_index = match max_index_option {
        Some(index) => index + 2,
        None => {
            println!("No maximum value found or the list is empty.");
            0 // default value when no maximum is found
        }
    };

    if max_index > (y.len() / 2) {
        max_index = y.len() - &max_index + 2;
    }

    central_frequency = 1.0 / (domain / (&max_index - 1) as f64);

    central_frequency
}

fn scale_to_frequency(scales: &Vec<f64>, wavefun: &Vec<Vec<f64>>, sampling_period: &f64) -> Vec<f64> {
    let mut freqencies: Vec<f64> = Vec::new();
    let central_frequency: f64 = get_central_frequency(wavefun);

    for i in 0..scales.len() {
        freqencies.push((&central_frequency / &scales[i]) / sampling_period);
    }

    freqencies
}

pub fn scale_to_frequency_by_cwt(scales: &Vec<f64>, cwt_type: &ContinuousWaletetType, sampling_period: &f64) -> Vec<f64> {
    let wavefun: Vec<Vec<f64>> = wavefun(&10, &cwt_type);
    let mut freqencies: Vec<f64> = Vec::new();
    let central_frequency: f64 = get_central_frequency(&wavefun);

    for i in 0..scales.len() {
        freqencies.push((&central_frequency / &scales[i]) / sampling_period);
    }

    freqencies
}

pub fn frequency_to_scale_by_cwt(frequencies: &Vec<f64>, cwt_type: &ContinuousWaletetType, sampling_period: &f64) -> Vec<f64> {
    let wavefun: Vec<Vec<f64>> = wavefun(&10, &cwt_type);
    let mut scales: Vec<f64> = Vec::new();
    let central_frequency: f64 = get_central_frequency(&wavefun);

    for i in 0..frequencies.len() {
        scales.push(&central_frequency / &frequencies[i] * sampling_period);
    }

    scales
}