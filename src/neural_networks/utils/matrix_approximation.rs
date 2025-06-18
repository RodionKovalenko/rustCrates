use std::f64::consts::PI;

use num::Complex;
use rand::Rng;
use rand_distr::{Distribution, Normal};

// Feature map using real+imag parts and kernel
pub fn phi_stable(x: &[Complex<f64>], w: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let x_concat: Vec<f64> = x.iter().flat_map(|c| [c.re, c.im]).collect();
    let norm_x_sq: f64 = x_concat.iter().map(|xi| xi * xi).sum();
    let r = w.len();
    let mut features = Vec::with_capacity(r);

    for i in 0..r {
        let dot = w[i].iter().zip(&x_concat).map(|(wi, xi)| wi * xi).sum::<f64>();
        let val = (dot + b[i]).exp() * (-norm_x_sq / 2.0).exp();
        features.push(val);
    }

    features
}

pub fn generate_random_features(r: usize, d: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).expect("no distribution");

    // w shape: r x 2d
    let w: Vec<Vec<f64>> = (0..r).map(|_| (0..2 * d).map(|_| normal.sample(&mut rng)).collect()).collect();

    // b shape: r, uniform from 0 to 2pi
    let b: Vec<f64> = (0..r).map(|_| rng.random_range(0.0..(2.0 * PI))).collect();

    (w, b)
}

fn mul_complex_real(a: Complex<f64>, b: f64) -> Complex<f64> {
    Complex { re: a.re * b, im: a.im * b }
}

// Main Performer-style approximate attention with Option 3
pub fn softmax_re_qk_approx_stable(q: &[Vec<Complex<f64>>], k: &[Vec<Complex<f64>>], v: &[Vec<Complex<f64>>], r: usize) -> Vec<Vec<Complex<f64>>> {
    let d_k = k[0].len() as f64;
    let scale = 1.0 / d_k.sqrt();
    let d = k[0].len();

    let (w, b) = generate_random_features(r, d);

    let phi_q: Vec<Vec<f64>> = q
        .iter()
        .map(|vec_q| {
            let scaled_q: Vec<Complex<f64>> = vec_q.iter().map(|c| *c * scale).collect();
            phi_stable(&scaled_q, &w, &b)
        })
        .collect();

    let phi_k: Vec<Vec<f64>> = k
        .iter()
        .map(|vec_k| {
            let scaled_k: Vec<Complex<f64>> = vec_k.iter().map(|c| *c * scale).collect();
            phi_stable(&scaled_k, &w, &b)
        })
        .collect();

    let n = q.len();
    let m = k.len();
    let d_v = v[0].len();
    let mut result = vec![vec![Complex::new(0.0, 0.0); d_v]; n];

    for i in 0..n {
        let mut weighted_sum = vec![Complex::new(0.0, 0.0); d_v];
        let mut weight_total = 0.0;

        for j in 0..m {
            // Real-valued attention weight
            let weight = phi_q[i].iter().zip(&phi_k[j]).map(|(a, b)| a * b).sum::<f64>();

            weight_total += weight;

            for dv in 0..d_v {
                weighted_sum[dv] += mul_complex_real(v[j][dv], weight);
            }
        }

        let norm = weight_total.max(1e-8);
        for dv in 0..d_v {
            result[i][dv] = weighted_sum[dv] / norm;
        }
    }

    result
}
