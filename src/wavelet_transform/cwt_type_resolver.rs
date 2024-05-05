use std::f32::consts::{E, PI};
use crate::wavelet_transform::cwt_types::ContiousWaletetType;

pub fn get_mexh_constant(sigma: &i32) -> f64 {
    let mex_h_1: f64 = (3 * sigma) as f64;

    2.0 / (mex_h_1.powf(1.0/2.0) * (PI as f64).powf(1.0/4.0))
}

pub fn transform_by_type(v: &f64, sigma: &i32, cw_type: &ContiousWaletetType) -> f64 {
    match cw_type {
        ContiousWaletetType::MORL => morl(v),
        ContiousWaletetType::CMOR => cmorl(v),
        ContiousWaletetType::MEXH => mexh(v, &sigma),
        ContiousWaletetType::FBSP => fbsp(v),
        ContiousWaletetType::SHAN => shan(v),
        ContiousWaletetType::GAUS1 => gauss1(v),
        ContiousWaletetType::GAUS2 => gauss1(v),
        ContiousWaletetType::GAUS3 => gauss1(v),
        ContiousWaletetType::GAUS4 => gauss1(v),
        ContiousWaletetType::GAUS5 => gauss1(v),
        ContiousWaletetType::GAUS6 => gauss1(v),
        ContiousWaletetType::GAUS7 => gauss1(v),
        ContiousWaletetType::GAUS8 => gauss1(v),
        ContiousWaletetType::CGAU1 => gauss1(v),
        ContiousWaletetType::CGAU2 => gauss1(v),
        ContiousWaletetType::CGAU3 => gauss1(v),
        ContiousWaletetType::CGAU4 => gauss1(v),
        ContiousWaletetType::CGAU5 => gauss1(v),
        ContiousWaletetType::CGAU6 => gauss1(v),
        ContiousWaletetType::CGAU7 => gauss1(v),
        ContiousWaletetType::CGAU8 => gauss1(v),
        _ => 0.0
    }
}

pub fn morl(v: &f64) -> f64 {
    let t = 0.0;

    t
}

pub fn cmorl(v: &f64) -> f64 {
    let t = 0.0;

    t
}

pub fn mexh(v: &f64, sigma: &i32) -> f64 {

    let sigma_f = sigma.clone() as f64;

    let mut t = get_mexh_constant(sigma) * ((1.0 - (v / (sigma_f) ).powf(2.0)) * (E as f64).powf(-0.5 * (v / sigma_f.clone()).powf(2.0)));

    t
}

pub fn fbsp(v: &f64) -> f64 {
    let t = 0.0;

    t
}

pub fn shan(v: &f64) -> f64 {
    let t = 0.0;

    t
}


pub fn gauss1(v: &f64) -> f64 {
    let t = 0.0;

    t
}
