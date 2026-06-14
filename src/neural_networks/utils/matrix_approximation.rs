use std::f64::consts::PI;

use num::Complex;
use rand::RngExt;
use rand_distr::{Distribution, Normal};

/// Computes the Hermitian inner product a^H b = sum conj(a_i) * b_i
fn hermitian_dot(a: &[Complex<f64>], b: &[Complex<f64>]) -> Complex<f64> {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    a.iter().zip(b.iter()).map(|(ai, bi)| ai.conj() * *bi).fold(Complex::new(0.0, 0.0), |acc, v| acc + v)
}

/// Squared L2-norm: sum |x_i|^2
fn complex_norm_sq(x: &[Complex<f64>]) -> f64 {
    x.iter().map(|c| c.norm_sqr()).sum()
}

/// phi(x) = exp(omega^H x - ||x||^2 / 2)
pub fn phi(x: &[Complex<f64>], omega: &[Complex<f64>]) -> Complex<f64> {
    assert_eq!(x.len(), omega.len());
    let dot = hermitian_dot(omega, x); // now equals omega^H x
    let norm_sq = complex_norm_sq(x);
    (dot - Complex::new(norm_sq / 2.0, 0.0)).exp()
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

fn herm_dot_real(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm_sq_real(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

pub fn phi_real(x: &[f64], omega: &[f64]) -> f64 {
    let dot = herm_dot_real(omega, x);
    let n2 = norm_sq_real(x);
    (dot - n2 * 0.5).exp()
}
