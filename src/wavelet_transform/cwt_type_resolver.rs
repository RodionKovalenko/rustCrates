use std::f32::consts::{E, PI};
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;

pub fn get_mexh_constant(sigma: &f64) -> f64 {
    let mex_h_1: f64 = 3.0 * sigma;

    2.0 / (mex_h_1.powf(1.0 / 2.0) * (PI as f64).powf(1.0 / 4.0))
}

pub fn transform_by_type(v: &f64, sigma: &f64, cw_type: &ContinuousWaletetType) -> f64 {
    match cw_type {
        ContinuousWaletetType::MORL => morl(v, sigma),
        ContinuousWaletetType::CMOR => cmorl(v),
        ContinuousWaletetType::MEXH => mexh(v, sigma),
        ContinuousWaletetType::FBSP => fbsp(v),
        ContinuousWaletetType::SHAN => shan(v),
        ContinuousWaletetType::GAUS1 => gauss1(v),
        ContinuousWaletetType::GAUS2 => gauss2(v),
        ContinuousWaletetType::GAUS3 => gauss3(v),
        ContinuousWaletetType::GAUS4 => gauss4(v),
        ContinuousWaletetType::GAUS5 => gauss5(v),
        ContinuousWaletetType::GAUS6 => gauss6(v),
        ContinuousWaletetType::GAUS7 => gauss7(v),
        ContinuousWaletetType::GAUS8 => gauss8(v),
        ContinuousWaletetType::CGAU1 => gauss1(v),
        ContinuousWaletetType::CGAU2 => gauss1(v),
        ContinuousWaletetType::CGAU3 => gauss1(v),
        ContinuousWaletetType::CGAU4 => gauss1(v),
        ContinuousWaletetType::CGAU5 => gauss1(v),
        ContinuousWaletetType::CGAU6 => gauss1(v),
        ContinuousWaletetType::CGAU7 => gauss1(v),
        ContinuousWaletetType::CGAU8 => gauss1(v)
    }
}

pub fn get_wavelet_range(cw_type: &ContinuousWaletetType) -> (f64, f64) {
    match cw_type {
        ContinuousWaletetType::MORL
        | ContinuousWaletetType::CMOR
        | ContinuousWaletetType::MEXH
        | ContinuousWaletetType::FBSP
        | ContinuousWaletetType::SHAN => (-8.0, 8.0),
        _ => (-5.0, 5.0),
    }
}

pub fn morl(t: &f64, &_sigma: &f64) -> f64 {
    let exp_term: f64 = (E as f64).powf(-0.5 * t.powf(2.0));
    let cos_term: f64 = (5.0 * t).cos();

    let t = exp_term * cos_term;
    t
}


pub fn cmorl(_v: &f64) -> f64 {
    let t = 0.0;

    t
}

pub fn mexh(v: &f64, sigma: &f64) -> f64 {
    let t = get_mexh_constant(sigma) * ((1.0 - (v / (sigma)).powf(2.0)) * (E as f64).powf(-0.5 * (v / sigma.clone()).powf(2.0)));

    t
}

pub fn fbsp(_v: &f64) -> f64 {
    let t = 0.0;

    t
}

pub fn shan(_v: &f64) -> f64 {
    let t = 0.0;

    t
}

pub fn gauss1(v: &f64) -> f64 {
    -2.0 * v * (E as f64).powf(-v.powf(2.0)) / ((PI / 2.0).powf(1.0 / 4.0)) as f64
}

pub fn gauss2(v: &f64) -> f64 {
    -2.0 * (2.0 * v.powf(2.0) - 1.0) * (E as f64).powf(-v.powf(2.0)) / (3.0 * (PI as f64 / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss3(v: &f64) -> f64 {
    -4.0 * (-2.0 * v.powf(3.0) + 3.0 * v) * (-v.powf(2.0)).exp() / (15.0 * (PI as f64 / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss4(v: &f64) -> f64 {
    4.0 * (-12.0 * v.powf(2.0) + 4.0 * v.powf(4.0) + 3.0) * (-v.powf(2.0)).exp() / (105.0 * (PI as f64 / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss5(v: &f64) -> f64 {
    8.0 * (-4.0 * v.powf(5.0) + 20.0 * v.powf(3.0) - 15.0 * v) * (-v.powf(2.0)).exp() / (105.0 * 9.0 * (PI as f64 / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss6(v: &f64) -> f64 {
    -8.0 * (8.0 * v.powf(6.0) - 60.0 * v.powf(4.0) + 90.0 * v.powf(2.0) - 15.0) * (-v.powf(2.0)).exp() / (105.0 * 9.0 * 11.0 * (PI as f64 / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss7(v: &f64) -> f64 {
    -16.0 * (-8.0 * v.powf(7.0) + 84.0 * v.powf(5.0) - 210.0 * v.powf(3.0) + 105.0 * v) * (-v.powf(2.0)).exp() / (105.0 * 9.0 * 11.0 * 13.0 * (PI as f64 / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss8(v: &f64) -> f64 {
    (16.0 * (16.0 * v.powf(8.0) - 224.0 * v.powf(6.0) + 840.0 * v.powf(4.0) - 840.0 * v.powf(2.0) + 105.0) * (-v.powf(2.0)).exp()) / (105.0 * 9.0 * 11.0 * 13.0 * 15.0 * (PI as f64 / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}