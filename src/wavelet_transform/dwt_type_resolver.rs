use crate::wavelet_transform::dwt_coeffients::*;
use crate::wavelet_transform::dwt_inverse_coeffients::*;
use crate::wavelet_transform::dwt_types::DiscreteWaveletType;
use std::f64::consts::TAU;

// Return a Low Pass Filter, a filter of moving averages for a specific discrete wavelet type
pub fn get_low_pass_filter(dw_type: &DiscreteWaveletType) -> Vec<f64> {
    match dw_type {
        DiscreteWaveletType::DB1 => DB_1.to_vec(),
        DiscreteWaveletType::DB2 => DB_2.to_vec(),
        DiscreteWaveletType::DB3 => DB_3.to_vec(),
        DiscreteWaveletType::DB4 => DB_4.to_vec(),
        DiscreteWaveletType::DB5 => DB_5.to_vec(),
        DiscreteWaveletType::DB6 => DB_6.to_vec(),
        DiscreteWaveletType::DB7 => DB_7.to_vec(),
        DiscreteWaveletType::DB8 => DB_8.to_vec(),
        DiscreteWaveletType::DB9 => DB_9.to_vec(),
        DiscreteWaveletType::DB10 => DB_10.to_vec(),
        DiscreteWaveletType::DB11 => DB_11.to_vec(),
        DiscreteWaveletType::DB12 => DB_12.to_vec(),
        DiscreteWaveletType::DB13 => DB_13.to_vec(),
        DiscreteWaveletType::DB14 => DB_14.to_vec(),
        DiscreteWaveletType::DB15 => DB_15.to_vec(),
        DiscreteWaveletType::DB16 => DB_16.to_vec(),
        DiscreteWaveletType::DB17 => DB_17.to_vec(),
        DiscreteWaveletType::DB18 => DB_18.to_vec(),
        DiscreteWaveletType::DB19 => DB_19.to_vec(),
        DiscreteWaveletType::DB20 => DB_20.to_vec(),
        DiscreteWaveletType::DB21 => DB_21.to_vec(),
        DiscreteWaveletType::DB22 => DB_22.to_vec(),
        DiscreteWaveletType::DB23 => DB_23.to_vec(),
        DiscreteWaveletType::DB24 => DB_24.to_vec(),
        DiscreteWaveletType::DB25 => DB_25.to_vec(),
        DiscreteWaveletType::DB26 => DB_26.to_vec(),
        DiscreteWaveletType::DB27 => DB_27.to_vec(),
        DiscreteWaveletType::DB28 => DB_28.to_vec(),
        DiscreteWaveletType::DB29 => DB_29.to_vec(),
        DiscreteWaveletType::DB30 => DB_30.to_vec(),
        DiscreteWaveletType::DB31 => DB_31.to_vec(),
        DiscreteWaveletType::DB32 => DB_32.to_vec(),
        DiscreteWaveletType::DB33 => DB_33.to_vec(),
        DiscreteWaveletType::DB34 => DB_34.to_vec(),
        DiscreteWaveletType::DB35 => DB_35.to_vec(),
        DiscreteWaveletType::DB36 => DB_36.to_vec(),
        DiscreteWaveletType::DB37 => DB_37.to_vec(),
        DiscreteWaveletType::DB38 => DB_38.to_vec(),
        DiscreteWaveletType::SYM2 => SYM_2.to_vec(),
        DiscreteWaveletType::SYM3 => SYM_3.to_vec(),
        DiscreteWaveletType::SYM4 => SYM_4.to_vec(),
        DiscreteWaveletType::SYM5 => SYM_5.to_vec(),
        DiscreteWaveletType::SYM6 => SYM_6.to_vec(),
        DiscreteWaveletType::SYM7 => SYM_7.to_vec(),
        DiscreteWaveletType::SYM8 => SYM_8.to_vec(),
        DiscreteWaveletType::SYM9 => SYM_9.to_vec(),
        DiscreteWaveletType::SYM10 => SYM_10.to_vec(),
        DiscreteWaveletType::SYM11 => SYM_11.to_vec(),
        DiscreteWaveletType::SYM12 => SYM_12.to_vec(),
        DiscreteWaveletType::SYM13 => SYM_13.to_vec(),
        DiscreteWaveletType::SYM14 => SYM_14.to_vec(),
        DiscreteWaveletType::SYM15 => SYM_15.to_vec(),
        DiscreteWaveletType::SYM16 => SYM_16.to_vec(),
        DiscreteWaveletType::SYM17 => SYM_17.to_vec(),
        DiscreteWaveletType::SYM18 => SYM_18.to_vec(),
        DiscreteWaveletType::SYM19 => SYM_19.to_vec(),
        DiscreteWaveletType::SYM20 => SYM_20.to_vec(),
        DiscreteWaveletType::COIF1 => COIF_1.to_vec(),
        DiscreteWaveletType::COIF2 => COIF_2.to_vec(),
        DiscreteWaveletType::COIF3 => COIF_3.to_vec(),
        DiscreteWaveletType::COIF4 => COIF_4.to_vec(),
        DiscreteWaveletType::COIF6 => COIF_6.to_vec(),
        DiscreteWaveletType::COIF7 => COIF_7.to_vec(),
        DiscreteWaveletType::COIF8 => COIF_8.to_vec(),
        DiscreteWaveletType::COIF9 => COIF_9.to_vec(),
        DiscreteWaveletType::COIF10 => COIF_10.to_vec(),
        DiscreteWaveletType::COIF11 => COIF_11.to_vec(),
        DiscreteWaveletType::COIF12 => COIF_12.to_vec(),
        DiscreteWaveletType::COIF13 => COIF_13.to_vec(),
        DiscreteWaveletType::COIF14 => COIF_14.to_vec(),
        DiscreteWaveletType::COIF15 => COIF_15.to_vec(),
        DiscreteWaveletType::COIF16 => COIF_16.to_vec(),
        DiscreteWaveletType::COIF17 => COIF_17.to_vec(),
        DiscreteWaveletType::BIOR10 => BIOR_1_0.to_vec(),
        DiscreteWaveletType::BIOR11 => BIOR_1_1.to_vec(),
        DiscreteWaveletType::BIOR13 => BIOR_1_3.to_vec(),
        DiscreteWaveletType::BIOR15 => BIOR_1_5.to_vec(),
        DiscreteWaveletType::BIOR20 => BIOR_2_0.to_vec(),
        DiscreteWaveletType::BIOR22 => BIOR_2_2.to_vec(),
        DiscreteWaveletType::BIOR24 => BIOR_2_4.to_vec(),
        DiscreteWaveletType::BIOR26 => BIOR_2_6.to_vec(),
        DiscreteWaveletType::BIOR28 => BIOR_2_8.to_vec(),
        DiscreteWaveletType::BIOR30 => BIOR_3_0.to_vec(),
        DiscreteWaveletType::BIOR31 => BIOR_3_1.to_vec(),
        DiscreteWaveletType::BIOR33 => BIOR_3_3.to_vec(),
        DiscreteWaveletType::BIOR35 => BIOR_3_5.to_vec(),
        DiscreteWaveletType::BIOR37 => BIOR_3_7.to_vec(),
        DiscreteWaveletType::BIOR39 => BIOR_3_9.to_vec(),
        DiscreteWaveletType::BIOR40 => BIOR_4_0.to_vec(),
        DiscreteWaveletType::BIOR44 => BIOR_4_4.to_vec(),
        DiscreteWaveletType::BIOR50 => BIOR_5_0.to_vec(),
        DiscreteWaveletType::BIOR55 => BIOR_5_5.to_vec(),
        DiscreteWaveletType::BIOR60 => BIOR_6_0.to_vec(),
        DiscreteWaveletType::BIOR68 => BIOR_6_8.to_vec(),
        DiscreteWaveletType::DMEY => DMEY.to_vec(),
        DiscreteWaveletType::OrthogonalParameterized { nums } => {
            let (low_pass, _) = generate_orthogonal_wavelet_filters(nums);
            low_pass
        }
    }
}

// Return a High Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_high_pass_filter(dw_type: &DiscreteWaveletType) -> Vec<f64> {
    let low_pass_filter = get_high_pass_filter_non_symmetric(&dw_type);
    let mut high_pass_filter: Vec<f64> = Vec::new();

    let mut v: f64;
    for (i, el) in low_pass_filter.iter().rev().enumerate() {
        v = el.clone();

        if i % 2 == 1 {
            v *= -1.0;
        }

        high_pass_filter.push(v);
    }

    high_pass_filter
}

// Return a Inverse Low Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_inverse_low_pass_filter(dw_type: &DiscreteWaveletType) -> Vec<f64> {
    let low_pass_filter = get_high_pass_filter_non_symmetric(&dw_type);
    let mut high_pass_filter = Vec::new();

    let mut v: f64;
    for (_i, el) in low_pass_filter.iter().rev().enumerate() {
        v = el.clone();

        high_pass_filter.push(v);
    }

    high_pass_filter
}

// Return a Inverse High Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_inverse_high_pass_filter(dw_type: &DiscreteWaveletType) -> Vec<f64> {
    let high_pass_filter_coef;
    let mut high_pass_filter: Vec<f64> = Vec::new();
    let mut is_default: bool = false;

    high_pass_filter_coef = match dw_type {
        DiscreteWaveletType::BIOR11
        | DiscreteWaveletType::BIOR13
        | DiscreteWaveletType::BIOR15
        | DiscreteWaveletType::BIOR22
        | DiscreteWaveletType::BIOR24
        | DiscreteWaveletType::BIOR26
        | DiscreteWaveletType::BIOR28
        | DiscreteWaveletType::BIOR31
        | DiscreteWaveletType::BIOR33
        | DiscreteWaveletType::BIOR35
        | DiscreteWaveletType::BIOR37
        | DiscreteWaveletType::BIOR39
        | DiscreteWaveletType::BIOR44
        | DiscreteWaveletType::BIOR55
        | DiscreteWaveletType::BIOR68 => get_low_pass_filter(dw_type),
        DiscreteWaveletType::OrthogonalParameterized { nums } => {
            let (low_pass, _) = generate_orthogonal_wavelet_filters(nums);
            low_pass
        }
        _ => {
            is_default = true;
            get_high_pass_filter(&dw_type)
        }
    };

    if is_default {
        let mut v: f64;
        for (_i, el) in high_pass_filter_coef.iter().rev().enumerate() {
            v = el.clone();

            high_pass_filter.push(v);
        }
    } else {
        fill_array_and_negate_odd(&high_pass_filter_coef, &mut high_pass_filter);
    }

    high_pass_filter
}

pub fn get_high_pass_filter_non_symmetric(dw_type: &DiscreteWaveletType) -> Vec<f64> {
    let high_pass_filter: Vec<f64> = match dw_type {
        DiscreteWaveletType::BIOR11 => INVERSE_BIOR_1_1.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR13 => INVERSE_BIOR_1_3.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR15 => INVERSE_BIOR_1_5.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR22 => INVERSE_BIOR_2_2.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR24 => INVERSE_BIOR_2_4.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR26 => INVERSE_BIOR_2_6.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR28 => INVERSE_BIOR_2_8.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR31 => INVERSE_BIOR_3_1.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR33 => INVERSE_BIOR_3_3.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR35 => INVERSE_BIOR_3_5.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR37 => INVERSE_BIOR_3_7.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR39 => INVERSE_BIOR_3_9.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR44 => INVERSE_BIOR_4_4.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR55 => INVERSE_BIOR_5_5.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::BIOR68 => INVERSE_BIOR_6_8.to_vec().into_iter().rev().collect(),
        DiscreteWaveletType::OrthogonalParameterized { nums } => {
            let (low_pass, _) = generate_orthogonal_wavelet_filters(nums);
            low_pass
        }
        _ => get_low_pass_filter(&dw_type),
    };

    high_pass_filter
}

/// Generate orthogonal wavelet filters of length 2^n using iterative rotations.
/// `n` is the number of iteration steps (final filter length = 2^n).
/// Returns (low_pass, high_pass) filters as Vec<f64>.
pub fn generate_orthogonal_wavelet_filters(n: &usize) -> (Vec<f64>, Vec<f64>) {
    assert!(*n > 0, "At least one rotation angle required");

    let thetas: Vec<f64> = (0..*n)
        .map(|i| TAU * (i as f64) / (*n as f64)) // evenly spaced angles
        .collect();

    fn rotate_pair(v0: f64, v1: f64, theta: f64) -> (f64, f64) {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        (cos_t * v0 - sin_t * v1, sin_t * v0 + cos_t * v1)
    }

    // Initialize base wavelet psi(2)
    let mut psi = vec![thetas[0].sin(), thetas[0].cos()];

    // Iteratively upshift and rotate for each subsequent theta
    for &theta in &thetas[1..] {
        // Upshift: zeros at even indices, previous psi values at odd indices
        let mut tmp = vec![0.0; psi.len() * 2];
        for (i, &val) in psi.iter().enumerate() {
            tmp[2 * i + 1] = val;
        }

        // Rotate every pair by R(theta)
        let mut rotated = vec![0.0; tmp.len()];
        for i in 0..(tmp.len() / 2) {
            let idx = 2 * i;
            let (r0, r1) = rotate_pair(tmp[idx], tmp[idx + 1], theta);
            rotated[idx] = r0;
            rotated[idx + 1] = r1;
        }

        psi = rotated;

        // Normalize after every iteration to maintain numeric stability
        let norm = psi.iter().map(|x| x * x).sum::<f64>().sqrt();
        for coeff in psi.iter_mut() {
            *coeff /= norm;
        }
    }

    // Construct high-pass filter as reversed and alternating sign low-pass filter
    let len = psi.len();
    let high_pass = (0..len)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * psi[len - 1 - i]
        })
        .collect::<Vec<_>>();

    (psi, high_pass)
}

pub fn fill_array_and_negate_odd(filter_coef: &Vec<f64>, data: &mut Vec<f64>) {
    let mut v: f64;

    for (i, el) in filter_coef.iter().enumerate() {
        v = el.clone();

        if i % 2 != 1 {
            v *= -1.0;
        }

        data.push(v);
    }
}
