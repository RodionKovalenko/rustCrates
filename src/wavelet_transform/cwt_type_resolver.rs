use crate::wavelet_transform::cwt_complex::CWTComplex;
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;
use num_complex::Complex;
use std::f64::consts::{E, PI};

pub fn get_mexh_constant(sigma: &f64) -> f64 {
    let mex_h_1: f64 = 3.0 * sigma;

    2.0 / (mex_h_1.powf(1.0 / 2.0) * PI.powf(1.0 / 4.0))
}

pub fn transform_by_type(v: &f64, sigma: &f64, cw_type: &ContinuousWaletetType) -> f64 {
    match cw_type {
        ContinuousWaletetType::MORL => morl(v, sigma),
        ContinuousWaletetType::MEXH => mexh(v, sigma),
        ContinuousWaletetType::GAUS1 => gauss1(v),
        ContinuousWaletetType::GAUS2 => gauss2(v),
        ContinuousWaletetType::GAUS3 => gauss3(v),
        ContinuousWaletetType::GAUS4 => gauss4(v),
        ContinuousWaletetType::GAUS5 => gauss5(v),
        ContinuousWaletetType::GAUS6 => gauss6(v),
        ContinuousWaletetType::GAUS7 => gauss7(v),
        ContinuousWaletetType::GAUS8 => gauss8(v),
        _ => 0.0,
    }
}

pub fn transform_by_complex_type(v: &Complex<f64>, wavelet: &CWTComplex) -> Complex<f64> {
    let cw_type = &wavelet.cw_type;

    match cw_type {
        ContinuousWaletetType::CMOR => cmorl(v, wavelet),
        ContinuousWaletetType::FBSP => fbsp(v, wavelet),
        ContinuousWaletetType::SHAN => shan(v, wavelet),
        ContinuousWaletetType::CGAU1 => cgauss1(v),
        ContinuousWaletetType::CGAU2 => cgauss2(v),
        ContinuousWaletetType::CGAU3 => cgauss3(v),
        ContinuousWaletetType::CGAU4 => cgauss4(v),
        ContinuousWaletetType::CGAU5 => cgauss5(v),
        ContinuousWaletetType::CGAU6 => cgauss6(v),
        ContinuousWaletetType::CGAU7 => cgauss7(v),
        ContinuousWaletetType::CGAU8 => cgauss8(v),
        _ => Complex::new(0.0, 0.0),
    }
}

pub fn get_wavelet_range(cw_type: &ContinuousWaletetType) -> (f64, f64) {
    match cw_type {
        ContinuousWaletetType::MORL | ContinuousWaletetType::CMOR | ContinuousWaletetType::MEXH => (-8.0, 8.0),
        ContinuousWaletetType::SHAN | ContinuousWaletetType::FBSP => (-20.0, 20.0),
        _ => (-5.0, 5.0),
    }
}

pub fn morl(t: &f64, &_sigma: &f64) -> f64 {
    let exp_term: f64 = E.powf(-0.5 * t.powf(2.0));
    let cos_term: f64 = (5.0 * t).cos();

    let t = exp_term * cos_term;
    t
}

pub fn mexh(v: &f64, sigma: &f64) -> f64 {
    let t = get_mexh_constant(sigma) * ((1.0 - (v / (sigma)).powf(2.0)) * E.powf(-0.5 * (v / sigma.clone()).powf(2.0)));

    t
}

pub fn fbsp(v: &Complex<f64>, wavelet: &CWTComplex) -> Complex<f64> {
    let re = v.re;

    let fb = &wavelet.fb;
    let fc = &wavelet.fc;
    let m = &wavelet.m;

    if re != 0.0 {
        let r = (2.0 * PI * fc * re).cos() * fb.sqrt() * (((PI * re * fb / m).sin()) / (PI * re * (fb / m))).powf(m.clone());
        let i = -1.0 * ((2.0 * PI * fc * re).sin() * fb.sqrt() * (((PI * re * fb / m).sin()) / (PI * re * (fb / m))).powf(m.clone()));

        Complex::new(r, i)
    } else {
        let cos_val = (2.0 * PI * fc * re).cos();
        let sin_val = (2.0 * PI * fc * re).sin();
        let sqrt_fb = fb.sqrt();

        let r = cos_val * sqrt_fb;
        let i = -1.0 * (sin_val * sqrt_fb);

        Complex::new(r, i)
    }
}

pub fn shan(v: &Complex<f64>, wavelet: &CWTComplex) -> Complex<f64> {
    let re = v.re;
    let fb = &wavelet.fb;
    let fc = &wavelet.fc;

    let mut r = (2.0 * PI * fc * re).cos() * fb.sqrt();
    let mut i = (2.0 * PI * fc * re).sin() * fb.sqrt();

    if r != 0.0 {
        let sin_val = (PI * fb * re).sin();
        let normalizing_c = re * fb * PI;

        r *= sin_val / normalizing_c;
        i *= sin_val / normalizing_c;

        return Complex::new(r, -1.0 * i);
    }

    Complex::new(r, -1.0 * i)
}

pub fn cmorl(v: &Complex<f64>, wavelet: &CWTComplex) -> Complex<f64> {
    let fb = &wavelet.fb;
    let fc = &wavelet.fc;

    let re = v.re;
    let r = (2.0 * PI * fc * re).cos() * E.powf(-re.powf(2.0) / fb) / (PI * fb).sqrt();
    let i = (2.0 * PI * fc * re).sin() * E.powf(-re.powf(2.0) / fb) / (PI * fb).sqrt();

    Complex::new(r, -1.0 * i)
}

pub fn cmor_derivative(input_batch: &Vec<Vec<Vec<Complex<f64>>>>, wavelet: &CWTComplex) -> Vec<Vec<Vec<Complex<f64>>>> {
    let analytical_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = input_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|value| {
                            let x = value.re;
                            let fb = wavelet.fb;
                            let fc = wavelet.fc;

                            let norm = 1.0 / (PI * fb).sqrt();
                            let exp_term = E.powf(-x * x / fb);
                            let common = exp_term * norm;

                            let re = (-2.0 * x / fb) * common; // real part of gradient
                            let im = (2.0 * PI * fc) * common; // imaginary part

                            let exp_cos = (2.0 * PI * fc * x).cos();
                            let exp_sin = (2.0 * PI * fc * x).sin();

                            let grad = Complex::new(-exp_sin * im + exp_cos * re,  -1.0 * (exp_cos * im + exp_sin * re));
                            grad
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    analytical_gradient_batch
}

pub fn cgauss1(v: &Complex<f64>) -> Complex<f64> {
    let re = v.re;
    let common_factor = (PI / 2.0).powf(1.0 / 4.0) * 2.0_f64.powf(1.0 / 2.0);

    let r: f64 = ((-2.0 * re * re.cos() - re.sin()) * E.powf(-re.powf(2.0))) / common_factor;
    let i: f64 = -1.0 * ((2.0 * re * re.sin() - re.cos()) * E.powf(-re.powf(2.0))) / common_factor;

    Complex::new(r, i)
}

pub fn cgauss2(v: &Complex<f64>) -> Complex<f64> {
    let re = v.re;
    let common_factor = (PI / 2.0).powf(1.0 / 4.0) * 10.0_f64.powf(1.0 / 2.0);

    let r: f64 = ((4.0 * re.powf(2.0) * re.cos() + 4.0 * re * re.sin() - 3.0 * re.cos()) * E.powf(-re.powf(2.0))) / common_factor;
    let i: f64 = -1.0 * (-4.0 * re.powf(2.0) * re.sin() + 4.0 * re * re.cos() + 3.0 * re.sin()) * E.powf(-re.powf(2.0)) / common_factor;

    Complex::new(r, i)
}

pub fn cgauss3(v: &Complex<f64>) -> Complex<f64> {
    let re = v.re;
    let common_factor = (PI / 2.0).powf(1.0 / 4.0) * 76.0_f64.powf(1.0 / 2.0);

    let r: f64 = ((-8.0 * re.powf(3.0) * re.cos() - 12.0 * re.powf(2.0) * re.sin() + 18.0 * re * re.cos() + 7.0 * re.sin()) * E.powf(-re.powf(2.0))) / common_factor;
    let i: f64 = -1.0 * ((8.0 * re.powf(3.0) * re.sin() - 12.0 * re.powf(2.0) * re.cos() - 18.0 * re * re.sin() + 7.0 * re.cos()) * E.powf(-re.powf(2.0))) / common_factor;

    Complex::new(r, i)
}

pub fn cgauss4(v: &Complex<f64>) -> Complex<f64> {
    let re = v.re;
    let common_factor = (PI / 2.0).powf(1.0 / 4.0) * 764.0_f64.powf(1.0 / 2.0);

    let r: f64 = ((16.0 * re.powf(4.0) * re.cos() + 32.0 * re.powf(3.0) * re.sin() - 72.0 * re.powf(2.0) * re.cos() - 56.0 * re * re.sin() + 25.0 * re.cos()) * E.powf(-re.powf(2.0))) / common_factor;
    let i: f64 = -1.0 * ((-16.0 * re.powf(4.0) * re.sin() + 32.0 * re.powf(3.0) * re.cos() + 72.0 * re.powf(2.0) * re.sin() - 56.0 * re * re.cos() - 25.0 * re.sin()) * E.powf(-re.powf(2.0))) / common_factor;

    Complex::new(r, i)
}

pub fn cgauss5(v: &Complex<f64>) -> Complex<f64> {
    let re = v.re;
    let common_factor = (PI / 2.0).powf(1.0 / 4.0) * 9496.0_f64.powf(1.0 / 2.0);

    let r: f64 = ((-32.0 * re.powf(5.0) * re.cos() - 80.0 * re.powf(4.0) * re.sin() + 240.0 * re.powf(3.0) * re.cos() + 280.0 * re.powf(2.0) * re.sin() - 250.0 * re * re.cos() - 81.0 * re.sin()) * E.powf(-re.powf(2.0))) / common_factor;
    let i: f64 = -1.0 * ((32.0 * re.powf(5.0) * re.sin() - 80.0 * re.powf(4.0) * re.cos() - 240.0 * re.powf(3.0) * re.sin() + 280.0 * re.powf(2.0) * re.cos() + 250.0 * re * re.sin() - 81.0 * re.cos()) * E.powf(-re.powf(2.0))) / common_factor;

    Complex::new(r, i)
}

pub fn cgauss6(v: &Complex<f64>) -> Complex<f64> {
    let re = v.re;
    let common_factor = (PI / 2.0).powf(1.0 / 4.0) * 140152.0_f64.powf(1.0 / 2.0);

    let r: f64 = ((64.0 * re.powf(6.0) * re.cos() + 192.0 * re.powf(5.0) * re.sin() - 720.0 * re.powf(4.0) * re.cos() - 1120.0 * re.powf(3.0) * re.sin() + 1500.0 * re.powf(2.0) * re.cos() + 972.0 * re * re.sin() - 331.0 * re.cos()) * E.powf(-re.powf(2.0))) / common_factor;
    let i = -1.0 * ((-64.0 * re.powf(6.0) * re.sin() + 192.0 * re.powf(5.0) * re.cos() + 720.0 * re.powf(4.0) * re.sin() - 1120.0 * re.powf(3.0) * re.cos() - 1500.0 * re.powf(2.0) * re.sin() + 972.0 * re * re.cos() + 331.0 * re.sin()) * E.powf(-re.powf(2.0))) / common_factor;

    Complex::new(r, i)
}

pub fn cgauss7(v: &Complex<f64>) -> Complex<f64> {
    let re = v.re;
    let common_factor = (PI / 2.0).powf(1.0 / 4.0) * 2390480.0_f64.powf(1.0 / 2.0);

    let r = ((-128.0 * re.powf(7.0) * re.cos() - 448.0 * re.powf(6.0) * re.sin() + 2016.0 * re.powf(5.0) * re.cos() + 3920.0 * re.powf(4.0) * re.sin() - 7000.0 * re.powf(3.0) * re.cos() - 6804.0 * re.powf(2.0) * re.sin() + 4634.0 * re * re.cos() + 1303.0 * re.sin()) * E.powf(-re.powf(2.0)))
        / common_factor;
    let i = -1.0 * ((128.0 * re.powf(7.0) * re.sin() - 448.0 * re.powf(6.0) * re.cos() - 2016.0 * re.powf(5.0) * re.sin() + 3920.0 * re.powf(4.0) * re.cos() + 7000.0 * re.powf(3.0) * re.sin() - 6804.0 * re.powf(2.0) * re.cos() - 4634.0 * re * re.sin() + 1303.0 * re.cos()) * E.powf(-re.powf(2.0)))
        / common_factor;

    Complex::new(r, i)
}

pub fn cgauss8(v: &Complex<f64>) -> Complex<f64> {
    let re = v.re;
    let common_factor = (PI / 2.0).powf(1.0 / 4.0) * 46206736.0_f64.powf(1.0 / 2.0);

    let r = ((256.0 * re.powf(8.0) * re.cos() + 1024.0 * re.powf(7.0) * re.sin() - 5376.0 * re.powf(6.0) * re.cos() - 12544.0 * re.powf(5.0) * re.sin() + 28000.0 * re.powf(4.0) * re.cos() + 36288.0 * re.powf(3.0) * re.sin() - 37072.0 * re.powf(2.0) * re.cos() - 20848.0 * re * re.sin()
        + 5937.0 * re.cos())
        * E.powf(-re.powf(2.0)))
        / common_factor;
    let i = -1.0
        * ((-256.0 * re.powf(8.0) * re.sin() + 1024.0 * re.powf(7.0) * re.cos() + 5376.0 * re.powf(6.0) * re.sin() - 12544.0 * re.powf(5.0) * re.cos() - 28000.0 * re.powf(4.0) * re.sin() + 36288.0 * re.powf(3.0) * re.cos() + 37072.0 * re.powf(2.0) * re.sin()
            - 20848.0 * re * re.cos()
            - 5937.0 * re.sin())
            * E.powf(-re.powf(2.0)))
        / common_factor;
    Complex::new(r, i)
}

pub fn gauss1(v: &f64) -> f64 {
    -2.0 * v * E.powf(-v.powf(2.0)) / ((PI / 2.0).powf(1.0 / 4.0))
}

pub fn gauss2(v: &f64) -> f64 {
    -2.0 * (2.0 * v.powf(2.0) - 1.0) * E.powf(-v.powf(2.0)) / (3.0 * (PI / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss3(v: &f64) -> f64 {
    -4.0 * (-2.0 * v.powf(3.0) + 3.0 * v) * (-v.powf(2.0)).exp() / (15.0 * (PI / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss4(v: &f64) -> f64 {
    4.0 * (-12.0 * v.powf(2.0) + 4.0 * v.powf(4.0) + 3.0) * (-v.powf(2.0)).exp() / (105.0 * (PI / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss5(v: &f64) -> f64 {
    8.0 * (-4.0 * v.powf(5.0) + 20.0 * v.powf(3.0) - 15.0 * v) * (-v.powf(2.0)).exp() / (105.0 * 9.0 * (PI / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss6(v: &f64) -> f64 {
    -8.0 * (8.0 * v.powf(6.0) - 60.0 * v.powf(4.0) + 90.0 * v.powf(2.0) - 15.0) * (-v.powf(2.0)).exp() / (105.0 * 9.0 * 11.0 * (PI / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss7(v: &f64) -> f64 {
    -16.0 * (-8.0 * v.powf(7.0) + 84.0 * v.powf(5.0) - 210.0 * v.powf(3.0) + 105.0 * v) * (-v.powf(2.0)).exp() / (105.0 * 9.0 * 11.0 * 13.0 * (PI / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}

pub fn gauss8(v: &f64) -> f64 {
    (16.0 * (16.0 * v.powf(8.0) - 224.0 * v.powf(6.0) + 840.0 * v.powf(4.0) - 840.0 * v.powf(2.0) + 105.0) * (-v.powf(2.0)).exp()) / (105.0 * 9.0 * 11.0 * 13.0 * 15.0 * (PI / 2.0).powf(1.0 / 2.0)).powf(1.0 / 2.0)
}