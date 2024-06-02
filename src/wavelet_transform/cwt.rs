use std::cmp::{Ordering};
use crate::utils::array::{convolve, get_coef, integrate, linspace};
use crate::utils::convolution_modes::ConvolutionMode;
use crate::wavelet_transform::cwt_type_resolver::transform_by_type;
use crate::wavelet_transform::cwt_type_resolver::get_wavelet_range;
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;
use crate::wavelet_transform::fft::fft_real1_d;
use crate::utils::data_converter::{convert_to_f64_1d, convert_to_f64_2d, convert_to_f64_3d, convert_to_f64_4d, convert_to_f64_5d};
use crate::utils::num_trait::{Array, ArrayType};
use crate::wavelet_transform::cwt_complex::{cwt_complex, CWTComplex};

pub fn cwt_1d<T: ArrayType>(data: &T, scales: &Vec<f64>, cw_type: &ContinuousWaletetType, sampling_period: &f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let data_f64: Vec<f64> = convert_to_f64_1d(data);
    let wavefun_result: Vec<Vec<f64>> = wavefun(&10, &cw_type);
    let freqencies: Vec<f64> = scale_to_frequency(scales, &wavefun_result, sampling_period);

    let mut wavelets: Vec<Vec<f64>> = Vec::new();
    for s in 0..scales.len() {
        wavelets.push(get_wavelet(&data_f64, &wavefun_result, &scales[s]));
    }

    (wavelets, freqencies)
}

pub fn cwt_2d<T: ArrayType>(data: &T, scales: &Vec<f64>, cw_type: &ContinuousWaletetType, sampling_period: &f64) -> (Vec<Vec<Vec<f64>>>, Vec<f64>) {
    let data_f64 = convert_to_f64_2d(data);
    let wavefun_result: Vec<Vec<f64>> = wavefun(&10, &cw_type);
    let freqencies: Vec<f64> = scale_to_frequency(scales, &wavefun_result, sampling_period);

    let mut wavelets: Vec<Vec<Vec<f64>>> = Vec::new();

    for s in 0..scales.len() {
        wavelets.push(Vec::new());

        for i in 0..data_f64.len() {
            wavelets[s].push(get_wavelet(&data_f64[i], &wavefun_result, &scales[s]));
        }
    }

    (wavelets, freqencies)
}

pub fn cwt_3d<T: ArrayType>(data: &T, scales: &Vec<f64>, cw_type: &ContinuousWaletetType, sampling_period: &f64) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<f64>) {
    let data_f64 = convert_to_f64_3d(data);
    let wavefun_result: Vec<Vec<f64>> = wavefun(&10, &cw_type);
    let freqencies: Vec<f64> = scale_to_frequency(scales, &wavefun_result, sampling_period);
    let mut wavelets: Vec<Vec<Vec<Vec<f64>>>> = Vec::new();

    for s in 0..scales.len() {
        wavelets.push(Vec::new());

        for i in 0..data_f64.len() {
            wavelets[s].push(Vec::new());

            for j in 0..data_f64[i].len() {
                wavelets[s][i].push(get_wavelet(&data_f64[i][j], &wavefun_result, &scales[s]));
            }
        }
    }

    (wavelets, freqencies)
}

pub fn cwt_4d<T: ArrayType>(data: &T, scales: &Vec<f64>, cw_type: &ContinuousWaletetType, sampling_period: &f64) -> (Vec<Vec<Vec<Vec<Vec<f64>>>>>, Vec<f64>) {
    let data_f64 = convert_to_f64_4d(data);
    let wavefun_result: Vec<Vec<f64>> = wavefun(&10, &cw_type);
    let freqencies: Vec<f64> = scale_to_frequency(scales, &wavefun_result, sampling_period);
    let mut wavelets: Vec<Vec<Vec<Vec<Vec<f64>>>>> = Vec::new();

    for s in 0..scales.len() {
        wavelets.push(Vec::new());

        for i in 0..data_f64.len() {
            wavelets[s].push(Vec::new());

            for j in 0..data_f64[i].len() {
                wavelets[s][i].push(Vec::new());

                for k in 0..data_f64[i][j].len() {
                    wavelets[s][i][j].push(get_wavelet(&data_f64[i][j][k], &wavefun_result, &scales[s]));
                }
            }
        }
    }

    (wavelets, freqencies)
}

pub fn cwt_5d<T: ArrayType>(data: &T, scales: &Vec<f64>, cw_type: &ContinuousWaletetType, sampling_period: &f64) -> (Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>, Vec<f64>) {
    let data_f64 = convert_to_f64_5d(data);
    let wavefun_result: Vec<Vec<f64>> = wavefun(&10, &cw_type);
    let freqencies: Vec<f64> = scale_to_frequency(scales, &wavefun_result, sampling_period);
    let mut wavelets: Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>> = Vec::new();

    for s in 0..scales.len() {
        wavelets.push(Vec::new());

        for i in 0..data_f64.len() {
            wavelets[s].push(Vec::new());

            for j in 0..data_f64[i].len() {
                wavelets[s][i].push(Vec::new());

                for k in 0..data_f64[i][j].len() {
                    wavelets[s][i][j].push(Vec::new());

                    for p in 0..data_f64[i][j][k].len() {
                        wavelets[s][i][j][k].push(get_wavelet(&data_f64[i][j][k][p], &wavefun_result, &scales[s]));
                    }
                }
            }
        }
    }

    (wavelets, freqencies)
}

pub fn get_wavelet(data: &Vec<f64>, wavefun_result: &Vec<Vec<f64>>, scale: &f64) -> Vec<f64> {
    let integral_scaled: Vec<f64>;
    let convolved: Vec<f64>;
    let coef: Vec<f64>;
    let index: f64;
    let limit_down: usize;
    let limit_up: usize;

    integral_scaled = integrate(&wavefun_result[0], &wavefun_result[1], scale);
    convolved = convolve(data, &integral_scaled, &ConvolutionMode::FULL);
    coef = get_coef(&convolved, scale);

    index = (coef.len() - data.len()) as f64 / 2.0;
    limit_down = index.floor() as usize;
    limit_up = index.ceil() as usize;

    coef[limit_down..(coef.len() - limit_up)].to_vec()
}

pub fn wavefun(precision: &i32, cw_type: &ContinuousWaletetType) -> Vec<Vec<f64>> {
    let mut y: Vec<f64> = Vec::new();
    let mut x_y_vec = Vec::new();
    let range = get_wavelet_range(&cw_type);
    let start = range.0;
    let end = range.1;

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

type ArrayWithFrequencies = (Array, Vec<f64>);

pub fn cwt<T: ArrayType>(data: &T, wavelet: &mut CWTComplex) -> Option<ArrayWithFrequencies> {
    let num_dim = data.dimension();
    let cw_type = &wavelet.cw_type;
    let scales = &wavelet.scales;
    let sampling_period = &wavelet.sampling_period;

    match cw_type {
        ContinuousWaletetType::CMOR |
        ContinuousWaletetType::SHAN |
        ContinuousWaletetType::FBSP |
        ContinuousWaletetType::CGAU1 |
        ContinuousWaletetType::CGAU2 |
        ContinuousWaletetType::CGAU3 |
        ContinuousWaletetType::CGAU4 |
        ContinuousWaletetType::CGAU5 |
        ContinuousWaletetType::CGAU6 |
        ContinuousWaletetType::CGAU7 |
        ContinuousWaletetType::CGAU8 => {
            let (wavelets, frequencies) = cwt_complex(data, wavelet).unwrap();
            Some((wavelets, frequencies))
        }
        _ => {
            match num_dim {
                1 => {
                    let (wavelets, frequencies) = cwt_1d(data, scales, cw_type, sampling_period);
                    Some((Array::Array2D(wavelets), frequencies))
                }
                2 => {
                    let (wavelets, frequencies) = cwt_2d(data, scales, cw_type, sampling_period);
                    Some((Array::Array3D(wavelets), frequencies))
                }
                3 => {
                    let (wavelets, frequencies) = cwt_3d(data, scales, cw_type, sampling_period);
                    Some((Array::Array4D(wavelets), frequencies))
                }
                4 => {
                    let (wavelets, frequencies) = cwt_4d(data, scales, cw_type, sampling_period);
                    Some((Array::Array5D(wavelets), frequencies))
                }
                5 => {
                    let (wavelets, frequencies) = cwt_5d(data, scales, cw_type, sampling_period);
                    Some((Array::Array6D(wavelets), frequencies))
                }
                _ => {
                    None
                }
            }
        }
    }
}
