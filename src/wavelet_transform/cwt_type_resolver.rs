use std::f32::consts::{E, PI};
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;

pub fn get_mexh_constant(sigma: &i32) -> f64 {
    let mex_h_1: f64 = (3 * sigma) as f64;

    2.0 / (mex_h_1.powf(1.0/2.0) * (PI as f64).powf(1.0/4.0))
}

pub fn transform_by_type(v: &f64, sigma: &i32, cw_type: &ContinuousWaletetType) -> f64 {
    match cw_type {
        ContinuousWaletetType::MORL => morl(v),
        ContinuousWaletetType::CMOR => cmorl(v),
        ContinuousWaletetType::MEXH => mexh(v, &sigma),
        ContinuousWaletetType::FBSP => fbsp(v),
        ContinuousWaletetType::SHAN => shan(v),
        ContinuousWaletetType::GAUS1 => gauss1(v),
        ContinuousWaletetType::GAUS2 => gauss1(v),
        ContinuousWaletetType::GAUS3 => gauss1(v),
        ContinuousWaletetType::GAUS4 => gauss1(v),
        ContinuousWaletetType::GAUS5 => gauss1(v),
        ContinuousWaletetType::GAUS6 => gauss1(v),
        ContinuousWaletetType::GAUS7 => gauss1(v),
        ContinuousWaletetType::GAUS8 => gauss1(v),
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

pub fn morl(_v: &f64) -> f64 {
    let t = 0.0;

    t
}

pub fn cmorl(_v: &f64) -> f64 {
    let t = 0.0;

    t
}

pub fn mexh(v: &f64, sigma: &i32) -> f64 {

    let sigma_f = sigma.clone() as f64;

    let t = get_mexh_constant(sigma) * ((1.0 - (v / (sigma_f) ).powf(2.0)) * (E as f64).powf(-0.5 * (v / sigma_f.clone()).powf(2.0)));

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


pub fn gauss1(_v: &f64) -> f64 {
    let t = 0.0;

    t
}
