use crate::neural_networks::utils::matrix::transpose;
use crate::utils::array_complex::{convolve_complex, get_coef_complex, integrate_complex, linspace_complex};
use crate::utils::convolution_modes::ConvolutionMode;
use crate::utils::data_converter::{convert_to_c_f64_1d, convert_to_c_f64_2d, convert_to_c_f64_3d, convert_to_c_f64_4d, convert_to_c_f64_5d};
use crate::utils::num_trait::{Array, ArrayType};
use crate::wavelet_transform::cwt_type_resolver::get_wavelet_range;
use crate::wavelet_transform::cwt_type_resolver::transform_by_complex_type;
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;
use crate::wavelet_transform::fft::fft_1d;
use core::fmt::Debug;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CWTComplex {
    pub scales: Vec<f64>,
    pub cw_type: ContinuousWaletetType,
    pub sampling_period: f64,
    pub m: f64,
    pub fb: f64,
    pub fc: f64,
    pub frequencies: Vec<f64>,
}

impl Default for CWTComplex {
    fn default() -> Self {
        CWTComplex {
            scales: vec![],
            cw_type: ContinuousWaletetType::MORL, // or any sensible default
            sampling_period: 1.0,
            m: 1.0,
            fb: 1.0,
            fc: 1.0,
            frequencies: vec![],
        }
    }
}

pub fn cwt_c_1d<T: ArrayType>(data: &T, wavelet: &CWTComplex) -> (Vec<Vec<Complex<f64>>>, Vec<f64>) {
    let scales = &wavelet.scales;
    let sampling_period = &wavelet.sampling_period;

    let data_f64: Vec<Complex<f64>> = convert_to_c_f64_1d(data);
    let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, wavelet);
    let freqencies: Vec<f64> = scale_to_frequency_complex(scales, &wavefun_result, sampling_period);

    let mut wavelets: Vec<Vec<Complex<f64>>> = Vec::new();
    for s in 0..scales.len() {
        wavelets.push(get_wavelet_complex(&data_f64, &wavefun_result, &scales[s]));
    }

    (wavelets, freqencies)
}

pub fn cwt_2d<T: ArrayType>(data: &T, wavelet: &CWTComplex) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<f64>) {
    let scales = &wavelet.scales;
    let sampling_period = &wavelet.sampling_period;

    let data_f64 = convert_to_c_f64_2d(data);

    let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, wavelet);
    let freqencies: Vec<f64> = scale_to_frequency_complex(scales, &wavefun_result, sampling_period);

    let mut wavelets: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

    for s in 0..scales.len() {
        wavelets.push(Vec::new());

        for i in 0..data_f64.len() {
            wavelets[s].push(get_wavelet_complex(&data_f64[i], &wavefun_result, &scales[s]));
        }
    }

    (wavelets, freqencies)
}

pub fn cwt_2d_full<T: ArrayType>(data: &T, wavelet: &CWTComplex) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<f64>) {
    let scales = &wavelet.scales;
    let sampling_period = &wavelet.sampling_period;

    let data_f64 = convert_to_c_f64_2d(data);

    let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, wavelet);
    let freqencies: Vec<f64> = scale_to_frequency_complex(scales, &wavefun_result, sampling_period);

    let mut wavelets: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

    for s in 0..scales.len() {
        // Row-wise CWT
        let mut intermediate: Vec<Vec<Complex<f64>>> = Vec::new();
        for row in &data_f64 {
            intermediate.push(get_wavelet_complex(row, &wavefun_result, &scales[s]));
        }

        // Column-wise CWT
        let intermediate_transposed = transpose(&intermediate);
        let mut result_2d: Vec<Vec<Complex<f64>>> = Vec::new();
        for col in &intermediate_transposed {
            result_2d.push(get_wavelet_complex(col, &wavefun_result, &scales[s]));
        }

        wavelets.push(transpose(&result_2d)); // Restore original row-column orientation
    }

    (wavelets, freqencies)
}

pub fn cwt_3d<T: ArrayType>(data: &T, wavelet: &CWTComplex) -> (Vec<Vec<Vec<Vec<Complex<f64>>>>>, Vec<f64>) {
    let scales = &wavelet.scales;
    let sampling_period = &wavelet.sampling_period;

    let data_f64 = convert_to_c_f64_3d(data);
    let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, wavelet);
    let freqencies: Vec<f64> = scale_to_frequency_complex(scales, &wavefun_result, sampling_period);
    let mut wavelets: Vec<Vec<Vec<Vec<Complex<f64>>>>> = Vec::new();

    for s in 0..scales.len() {
        wavelets.push(Vec::new());

        for i in 0..data_f64.len() {
            wavelets[s].push(Vec::new());

            for j in 0..data_f64[i].len() {
                wavelets[s][i].push(get_wavelet_complex(&data_f64[i][j], &wavefun_result, &scales[s]));
            }
        }
    }

    (wavelets, freqencies)
}

pub fn cwt_4d<T: ArrayType>(data: &T, wavelet: &CWTComplex) -> (Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>, Vec<f64>) {
    let scales = &wavelet.scales;
    let sampling_period = &wavelet.sampling_period;

    let data_f64 = convert_to_c_f64_4d(data);
    let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, wavelet);
    let freqencies: Vec<f64> = scale_to_frequency_complex(scales, &wavefun_result, sampling_period);
    let mut wavelets: Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>> = Vec::new();

    for s in 0..scales.len() {
        wavelets.push(Vec::new());

        for i in 0..data_f64.len() {
            wavelets[s].push(Vec::new());

            for j in 0..data_f64[i].len() {
                wavelets[s][i].push(Vec::new());

                for k in 0..data_f64[i][j].len() {
                    wavelets[s][i][j].push(get_wavelet_complex(&data_f64[i][j][k], &wavefun_result, &scales[s]));
                }
            }
        }
    }

    (wavelets, freqencies)
}

pub fn cwt_5d<T: ArrayType>(data: &T, wavelet: &CWTComplex) -> (Vec<Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>>, Vec<f64>) {
    let scales = &wavelet.scales;
    let sampling_period = &wavelet.sampling_period;

    let data_f64 = convert_to_c_f64_5d(data);
    let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, wavelet);
    let freqencies: Vec<f64> = scale_to_frequency_complex(scales, &wavefun_result, sampling_period);
    let mut wavelets: Vec<Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>> = Vec::new();

    for s in 0..scales.len() {
        wavelets.push(Vec::new());

        for i in 0..data_f64.len() {
            wavelets[s].push(Vec::new());

            for j in 0..data_f64[i].len() {
                wavelets[s][i].push(Vec::new());

                for k in 0..data_f64[i][j].len() {
                    wavelets[s][i][j].push(Vec::new());

                    for p in 0..data_f64[i][j][k].len() {
                        wavelets[s][i][j][k].push(get_wavelet_complex(&data_f64[i][j][k][p], &wavefun_result, &scales[s]));
                    }
                }
            }
        }
    }

    (wavelets, freqencies)
}

pub fn get_wavelet_complex(data: &Vec<Complex<f64>>, wavefun_result: &Vec<Vec<Complex<f64>>>, scale: &f64) -> Vec<Complex<f64>> {
    let integral_scaled: Vec<Complex<f64>>;
    let convolved: Vec<Complex<f64>>;
    let coef: Vec<Complex<f64>>;
    let index: f64;
    let limit_down: usize;
    let limit_up: usize;

    integral_scaled = integrate_complex(&wavefun_result[0], &wavefun_result[1], scale);
    convolved = convolve_complex(data, &integral_scaled, &ConvolutionMode::FULL);

    coef = get_coef_complex(&convolved, scale);

    index = (coef.len() - data.len()) as f64 / 2.0;

    limit_down = index.floor() as usize;
    limit_up = index.ceil() as usize;

    coef[limit_down..(coef.len() - limit_up)].to_vec()
}

pub fn get_wavelet_derivative(
    data: &Vec<Complex<f64>>,
    wavefun_result: &Vec<Vec<Complex<f64>>>,
    scale: &f64,
    grad_output: &Vec<Complex<f64>>, // <- incoming gradient from next layer
) -> Vec<Complex<f64>> {
    // Forward pass to get convolution parameters
    let integral_scaled = integrate_complex(&wavefun_result[0], &wavefun_result[1], scale);
    let convolved = convolve_complex(data, &integral_scaled, &ConvolutionMode::FULL);
    let coef = get_coef_complex(&convolved, scale);

    // Calculate slicing indices used in forward pass
    let index = (coef.len() - data.len()) as f64 / 2.0;
    let limit_down = index.floor() as usize;

    // Adjoint of coefficient slicing (pad with zeros)
    let mut full_grad = vec![Complex::new(0.0, 0.0); coef.len()];
    full_grad.splice(limit_down..(limit_down + grad_output.len()), grad_output.clone());

    // Adjoint of finite difference operation
    let scaled_sqrt = -scale.sqrt();
    let mut grad_convolved = vec![Complex::new(0.0, 0.0); convolved.len()];
    for i in 0..full_grad.len() {
        if i < grad_convolved.len() - 1 {
            grad_convolved[i] += full_grad[i] * scaled_sqrt;
            grad_convolved[i + 1] -= full_grad[i] * scaled_sqrt;
        }
    }

    // Adjoint of convolution (cross-correlation)
    convolve_complex(&grad_convolved, &integral_scaled, &ConvolutionMode::VALID)
}

pub fn get_wavelet_derivative_full(input: &Vec<Vec<Complex<f64>>>, wavefun_result: &Vec<Vec<Complex<f64>>>, scale: &f64, grad_output: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let mut grad_rows: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); input[0].len()]; input.len()];

    // Step 1: Backprop through column-wise CWT
    let grad_output_t = transpose(grad_output);
    let input_t = transpose(input);
    let mut grad_intermediate_t: Vec<Vec<Complex<f64>>> = Vec::new();

    for (i, col) in input_t.iter().enumerate() {
        grad_intermediate_t.push(get_wavelet_derivative(col, wavefun_result, scale, &grad_output_t[i]));
    }

    let grad_intermediate = transpose(&grad_intermediate_t); // shape: same as row-CWT output

    // Step 2: Backprop through row-wise CWT
    for (i, row) in input.iter().enumerate() {
        grad_rows[i] = get_wavelet_derivative(row, wavefun_result, scale, &grad_intermediate[i]);
    }

    grad_rows
}

pub fn wavefun_complex(precision: &i32, wavelet: &CWTComplex) -> Vec<Vec<Complex<f64>>> {
    let cw_type = &wavelet.cw_type;
    let mut y: Vec<Complex<f64>> = Vec::new();
    let mut x_y_vec: Vec<Vec<Complex<f64>>> = Vec::new();
    let range = get_wavelet_range(cw_type);
    let start = range.0;
    let end = range.1;

    let two_i32 = 2.0 as i32;
    let n_points = two_i32.pow(precision.clone() as u32);

    let x: Vec<Complex<f64>> = linspace_complex(&start, &end, &n_points, true);
    for i in 0..x.len() {
        y.push(transform_by_complex_type(&x[i], &wavelet));
    }

    x_y_vec.push(x);
    x_y_vec.push(y);

    x_y_vec
}

fn get_central_frequency_complex(wavefun: &Vec<Vec<Complex<f64>>>) -> f64 {
    let (x, y) = (&wavefun[0], &wavefun[1]);
    let central_frequency: f64;
    let domain: f64 = &x[x.len() - 1].re - &x[0].re;
    let fft = fft_1d(&y);

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

fn scale_to_frequency_complex(scales: &Vec<f64>, wavefun: &Vec<Vec<Complex<f64>>>, sampling_period: &f64) -> Vec<f64> {
    let mut freqencies: Vec<f64> = Vec::new();
    let central_frequency: f64 = get_central_frequency_complex(wavefun);

    for i in 0..scales.len() {
        freqencies.push((&central_frequency / &scales[i]) * sampling_period);
    }

    freqencies
}

pub fn scale_to_frequency_by_cwt(scales: &Vec<f64>, wavelet: &CWTComplex) -> Vec<f64> {
    let sampling_period = &wavelet.sampling_period;
    let wavefun: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, &wavelet);
    let mut freqencies: Vec<f64> = Vec::new();
    let central_frequency: f64 = get_central_frequency_complex(&wavefun);

    for i in 0..scales.len() {
        freqencies.push((&central_frequency / &scales[i]) * sampling_period);
    }

    freqencies
}

pub fn frequency_to_scale_by_cwt(frequencies: &Vec<f64>, wavelet: &CWTComplex) -> Vec<f64> {
    let sampling_period = &wavelet.sampling_period;

    let wavefun: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, wavelet);
    let mut scales: Vec<f64> = Vec::new();
    let central_frequency: f64 = get_central_frequency_complex(&wavefun);

    for i in 0..frequencies.len() {
        scales.push(&central_frequency / &frequencies[i] * sampling_period);
    }

    scales
}

type ArrayWithFrequencies = (Array, Vec<f64>);

pub fn cwt_complex<T: ArrayType>(data: &T, wavelet: &CWTComplex) -> Option<ArrayWithFrequencies> {
    let num_dim = data.dimension();

    match num_dim {
        1 => {
            let (wavelets, frequencies) = cwt_c_1d(data, wavelet);
            Some((Array::ArrayC2D(wavelets), frequencies))
        }
        2 => {
            let (wavelets, frequencies) = cwt_2d(data, wavelet);
            Some((Array::ArrayC3D(wavelets), frequencies))
        }
        3 => {
            let (wavelets, frequencies) = cwt_3d(data, wavelet);
            Some((Array::ArrayC4D(wavelets), frequencies))
        }
        4 => {
            let (wavelets, frequencies) = cwt_4d(data, wavelet);
            Some((Array::ArrayC5D(wavelets), frequencies))
        }
        5 => {
            let (wavelets, frequencies) = cwt_5d(data, wavelet);
            Some((Array::ArrayC6D(wavelets), frequencies))
        }
        _ => None,
    }
}
