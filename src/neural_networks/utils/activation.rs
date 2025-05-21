use num::{Complex, Float};
use rayon::prelude::*;
use std::f64::consts::E;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::marker::Copy;
use std::ops::{Add, Div, Mul};

use crate::neural_networks::network_components::layer::ActivationType;

/// SELU hyperparameters
pub const LAMBDA: f64 = 1.050700987355480493419334985294598;
pub const ALPHA: f64 = 1.673263242354377284817042991671750;
pub const EPSILON: f64 = 0.000000000000001;

// Define a trait for activation functions
pub trait RealActivation: Float + Copy {
    fn erf(self) -> Self; // Add erf method if required
}

// Implement the trait for f32 and f64
impl RealActivation for f32 {
    fn erf(self) -> Self {
        // Constants for approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if self < 0.0 { -1.0 } else { 1.0 };
        let x = self.abs();

        // A&S formula 7.1.26
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (E.powf((-x * x) as f64)) as f32;

        sign * y
    }
}

impl RealActivation for f64 {
    fn erf(self) -> Self {
        // Constants for approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if self < 0.0 { -1.0 } else { 1.0 };
        let x = self.abs();

        // A&S formula 7.1.26
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (E.powf(-x * x));

        sign * y
    }
}

// Define activation functions
fn sigmoid<T: RealActivation>(z: T) -> T {
    T::one() / (T::one() + (-z).exp())
}

fn tanh<T: RealActivation>(z: T) -> T {
    let exp_z = T::exp(z);
    let exp_neg_z = T::exp(-z);
    (exp_z - exp_neg_z) / (exp_z + exp_neg_z)
}

fn relu<T: RealActivation>(z: T) -> T {
    if z > T::zero() {
        z
    } else {
        T::zero()
    }
}

fn leaky_relu<T: RealActivation>(z: T) -> T {
    if z > T::zero() {
        z
    } else {
        T::from(0.01).unwrap() * z // You can adjust the slope
    }
}

fn elu<T: RealActivation>(z: T, alpha: T) -> T {
    if z > T::zero() {
        z
    } else {
        alpha * (T::exp(z) - T::one()) // Use T's exp method
    }
}

fn selu<T: RealActivation>(z: T, scale: T, alpha: T) -> T {
    if z > T::zero() {
        scale * z
    } else {
        scale * (alpha * (-z).exp() - alpha) // Use T's exp method
    }
}

fn gelu<T: RealActivation>(z: T) -> T {
    let cdf = T::one() / T::from(2.0).unwrap() * (T::one() + (T::from(2.0 / PI).unwrap() * z).erf());
    z * cdf
}

fn softsign<T: RealActivation>(z: T) -> T {
    z / (T::one() + z.abs())
}

fn softplus<T: RealActivation>(z: T) -> T {
    (T::exp(z) + T::one()).ln() // Use T's exp method
}

// Main activation function
pub fn activate_output<T>(data: &Vec<Vec<T>>, activation: ActivationType) -> Vec<Vec<T>>
where
    T: RealActivation + std::marker::Copy,
{
    match activation {
        ActivationType::SIGMOID => data.iter().map(|row| row.iter().map(|&x| sigmoid(x)).collect()).collect(),
        ActivationType::TANH => data.iter().map(|row| row.iter().map(|&x| tanh(x)).collect()).collect(),
        ActivationType::LINEAR => data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(), // Linear is identity
        ActivationType::RELU => data.iter().map(|row| row.iter().map(|&x| relu(x)).collect()).collect(),
        ActivationType::LEAKYRELU => data.iter().map(|row| row.iter().map(|&x| leaky_relu(x)).collect()).collect(),
        ActivationType::ELU => data.iter().map(|row| row.iter().map(|&x| elu(x, T::one())).collect()).collect(),             // Assuming alpha = 1.0
        ActivationType::SELU => data.iter().map(|row| row.iter().map(|&x| selu(x, T::one(), T::one())).collect()).collect(), // Assuming scale = 1.0, alpha = 1.0
        ActivationType::GELU => data.iter().map(|row| row.iter().map(|&x| gelu(x)).collect()).collect(),
        ActivationType::SOFTSIGN => data.iter().map(|row| row.iter().map(|&x| softsign(x)).collect()).collect(),
        ActivationType::SOFTPLUS => data.iter().map(|row| row.iter().map(|&x| softplus(x)).collect()).collect(),
        ActivationType::PROBIT => data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(), // Just return the value as is
        ActivationType::RANDOM => data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(), // Just return the value as is
        ActivationType::SOFTMAX => unimplemented!(),                                                 // Handle separately if needed
    }
}

// Implement activation functions for Complex<f64>
pub fn sigmoid_complex(z: Complex<f64>) -> Complex<f64> {
    let exp_neg_z = (-z).exp();
    Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) + exp_neg_z)
}

pub fn tanh_complex(z: Complex<f64>) -> Complex<f64> {
    let exp_z = z.exp();
    let exp_neg_z = (-z).exp();
    (exp_z - exp_neg_z) / (exp_z + exp_neg_z)
}

fn relu_complex(z: Complex<f64>) -> Complex<f64> {
    Complex::new(z.re.max(0.0), z.im) // Keep the imaginary part unchanged
}

fn leaky_relu_complex(z: Complex<f64>, slope: f64) -> Complex<f64> {
    if z.re > 0.0 {
        z
    } else {
        Complex::new(slope * z.re, z.im)
    }
}

fn elu_complex(z: Complex<f64>, alpha: f64) -> Complex<f64> {
    if z.re > 0.0 {
        z
    } else {
        Complex::new(alpha * (z.exp().re - 1.0), z.im)
    }
}
fn selu_complex(z: Complex<f64>) -> Complex<f64> {
    if z.re >= 0.0 {
        Complex::new(LAMBDA, 0.0) * z
    } else {
        Complex::new(LAMBDA * ALPHA, 0.0) * (z.exp() - Complex::new(1.0, 0.0))
    }
}

// pub fn gelu_complex(z: Complex<f64>) -> Complex<f64> {
//     // Precompute sqrt(2 / pi)
//     let sqrt_2_over_pi = SQRT_2 / PI.sqrt();

//     // Compute the argument inside tanh: z + 0.044715 * z^3
//     let f_z = sqrt_2_over_pi * (z + Complex::new(0.044715, 0.0) * z.powf(3.0));

//     // Compute tanh(f(z))
//     let tanh_f_z = f_z.tanh();

//     // Return the GELU activation: 0.5 * z * (1 + tanh(f(z)))
//     0.5 * z * (Complex::new(1.0, 0.0) + tanh_f_z)
// }

pub fn gelu_complex(z: Complex<f64>) -> Complex<f64> {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();

    let z_cubed = z.powi(3);
    let f_z = sqrt_2_over_pi * (z + Complex::new(0.044715, 0.0) * z_cubed);

    // Clamping to prevent numerical instability
    let real_part = f_z.re.max(-30.0).min(30.0);
    let im_part = f_z.im.max(-30.0).min(30.0);
    let clamped_f_z = Complex::new(real_part, im_part);
    let tanh_f_z = clamped_f_z.tanh();

    if tanh_f_z.re.is_nan() || tanh_f_z.im.is_nan() {
        panic!("NaN detected in tanh(f(z)), z: {:?}, f(z): {:?}, clamped f(z): {:?}", z, f_z, clamped_f_z);
    }

    0.5 * z * (Complex::new(1.0, 0.0) + tanh_f_z)
}

pub fn erf_complex(z: Complex<f64>) -> Complex<f64> {
    let sqrt_pi = PI.sqrt();
    let two_over_sqrt_pi = 2.0 / sqrt_pi;

    // Start with the first term of the series
    let mut term = z;
    let mut sum = term;
    let mut n = 1.0;

    // Series expansion with convergence check
    for _ in 1..100 {
        // Allow up to 100 iterations if necessary
        n += 1.0;
        term *= -z * z / n; // (-1)^n * z^(2n+1) / n!
        let delta = term / (2.0 * n + 1.0); // Current term
        sum += delta;

        // Convergence check: Stop if the term becomes very small
        if delta.norm() < 1e-12 {
            break;
        }
    }

    two_over_sqrt_pi * sum
}

pub fn softsign_complex(z: Complex<f64>) -> Complex<f64> {
    Complex::new(z.re / (1.0 + z.re.abs()), z.im / (1.0 + z.im.abs()))
}

fn softplus_complex(z: Complex<f64>) -> Complex<f64> {
    (z.exp() + Complex::new(1.0, 0.0)).ln()
}

// Main activation function for complex numbers
pub fn activate_output_complex(data: &Vec<Vec<Complex<f64>>>, activation: ActivationType) -> Vec<Vec<Complex<f64>>> {
    match activation {
        ActivationType::SIGMOID => data.iter().map(|row| row.iter().map(|&x| sigmoid_complex(x)).collect()).collect(),
        ActivationType::TANH => data.iter().map(|row| row.iter().map(|&x| tanh_complex(x)).collect()).collect(),
        ActivationType::LINEAR => data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(), // Linear is identity
        ActivationType::RELU => data.iter().map(|row| row.iter().map(|&x| relu_complex(x)).collect()).collect(),
        ActivationType::LEAKYRELU => data.iter().map(|row| row.iter().map(|&x| leaky_relu_complex(x, 0.01)).collect()).collect(),
        ActivationType::ELU => data.iter().map(|row| row.iter().map(|&x| elu_complex(x, 1.0)).collect()).collect(), // Assuming alpha = 1.0
        ActivationType::SELU => data.iter().map(|row| row.iter().map(|&x| selu_complex(x)).collect()).collect(),    // Assuming scale = 1.0, alpha = 1.0
        ActivationType::GELU => data.iter().map(|row| row.iter().map(|&x| gelu_complex(x)).collect()).collect(),
        ActivationType::SOFTSIGN => data.iter().map(|row| row.iter().map(|&x| softsign_complex(x)).collect()).collect(),
        ActivationType::SOFTPLUS => data.iter().map(|row| row.iter().map(|&x| softplus_complex(x)).collect()).collect(),
        ActivationType::PROBIT => data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(), // Just return the value as is
        ActivationType::RANDOM => data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(), // Just return the value as is
        _ => vec![],
    }
}

// Main activation function for complex numbers with padding support
pub fn activate_output_complex_padding(data: &Vec<Vec<Complex<f64>>>, activation: ActivationType, padding_mask: &Vec<u32>) -> Vec<Vec<Complex<f64>>> {
    match activation {
        ActivationType::SIGMOID => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { sigmoid_complex(x) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::TANH => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { tanh_complex(x) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::LINEAR => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { x } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::RELU => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { relu_complex(x) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::LEAKYRELU => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { leaky_relu_complex(x, 0.01) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::ELU => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { elu_complex(x, 1.0) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::SELU => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { selu_complex(x) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::GELU => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { gelu_complex(x) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::SOFTSIGN => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { softsign_complex(x) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::SOFTPLUS => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { softplus_complex(x) } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::PROBIT => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { x } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        ActivationType::RANDOM => data.iter().enumerate().map(|(row_ind, row)| row.iter().map(|&x| if padding_mask[row_ind] != 0 { x } else { Complex::new(0.0, 0.0) }).collect()).collect(),
        _ => vec![],
    }
}

pub fn gauss<T, V, G>(v: &T, m: &V, var: &G) -> f64
where
    T: Debug + Clone + Into<f64> + Mul<Output = T> + Add<Output = T> + Div<Output = T>,
    V: Debug + Clone + Into<f64> + Mul<Output = V> + Add<Output = V> + Div<Output = V>,
    G: Debug + Clone + Into<f64> + Mul<Output = G> + Add<Output = G> + Div<Output = G>,
{
    let value = v.clone().into();
    let mean = m.clone().into();
    let sigma = var.clone().into().sqrt();

    return (1.0 / (sigma * (2.0 * PI).sqrt())) * E.powf(-1.0 * (value - mean).powf(2.0) / (2.0 * var.clone().into()));
}

pub fn guass_d1<T, V, G>(v: &Vec<T>, m: &V, s: &G) -> Vec<f64>
where
    T: Debug + Clone + Into<f64> + Mul<Output = T> + Add<Output = T> + Div<Output = T>,
    V: Debug + Clone + Into<f64> + Mul<Output = V> + Add<Output = V> + Div<Output = V>,
    G: Debug + Clone + Into<f64> + Mul<Output = G> + Add<Output = G> + Div<Output = G>,
{
    let mean = m.clone().into();
    let sigma = s.clone().into();

    let mut result = vec![0.0; v.len()];

    for i in 0..v.len() {
        result[i] = gauss(&v[i], &mean, &sigma);
    }

    result
}

pub fn guass_d2<T, V, G>(v: &Vec<Vec<T>>, m: &V, s: &G) -> Vec<Vec<f64>>
where
    T: Debug + Clone + Into<f64> + Mul<Output = T> + Add<Output = T> + Div<Output = T>,
    V: Debug + Clone + Into<f64> + Mul<Output = V> + Add<Output = V> + Div<Output = V>,
    G: Debug + Clone + Into<f64> + Mul<Output = G> + Add<Output = G> + Div<Output = G>,
{
    let mean = m.clone().into();
    let sigma = s.clone().into();

    let mut result: Vec<Vec<f64>> = vec![vec![0.0; v[0].len()]; v.len()];

    for i in 0..v.len() {
        result[i] = guass_d1(&v[i], &mean, &sigma);
    }

    result
}
pub fn softmax_complex_norm(input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<f64>> {
    input
        .par_iter() // Parallel iterator over rows of the input
        .map(|row| softmax_row_norm(row))
        .collect()
}

pub fn softmax_complex_real(input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<f64>> {
    input
        .par_iter() // Parallel iterator over rows of the input
        .map(|row| softmax_row_real(row))
        .collect()
}

pub fn softmax_complex_padding_norm(input: &Vec<Vec<Complex<f64>>>, padding_mask: &Vec<u32>) -> Vec<Vec<f64>> {
    // println!("softmax input len {}, {}", input.len(), input[0].len());
    // println!("padding_mask len {}", padding_mask.len());
    // println!("padding mask: {:?}", &padding_mask);

    input
        .par_iter()
        .enumerate() // Parallel iterator over rows of the input
        .map(|(row_ind, row)| {
            if padding_mask.len() <= row_ind {
                panic!("Row mask is smaller than the index: {}, {}", padding_mask.len(), row_ind);
            }
            if padding_mask[row_ind] == 0 {
                return vec![0.0; row.len()];
            }

            softmax_row_norm(row)
        })
        .collect() // Collect the results into a Vec<Vec<Complex<f64>>>
}

pub fn softmax_complex_padding_real(input: &Vec<Vec<Complex<f64>>>, padding_mask: &Vec<u32>) -> Vec<Vec<f64>> {
    // println!("softmax input len {}, {}", input.len(), input[0].len());
    // println!("padding_mask len {}", padding_mask.len());
    // println!("padding mask: {:?}", &padding_mask);

    input
        .par_iter()
        .enumerate() // Parallel iterator over rows of the input
        .map(|(row_ind, row)| {
            if padding_mask.len() <= row_ind {
                panic!("Row mask is smaller than the index: {}, {}", padding_mask.len(), row_ind);
            }
            if padding_mask[row_ind] == 0 {
                return vec![0.0; row.len()];
            }

            softmax_row_real(row)
        })
        .collect() // Collect the results into a Vec<Vec<Complex<f64>>>
}

pub fn softmax_last_row(input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<f64>> {
    // Softmax function to scale attention scores to probability values
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; input[0].len()]; input.len()];

    // Get the last row from the input
    let last_row = &input[input.len() - 1];

    result[input.len() - 1] = softmax_row_norm(&last_row);

    result
}

pub fn softmax_row_norm(input: &Vec<Complex<f64>>) -> Vec<f64> {
    let max_norm = input.iter().map(|c| c.norm()).fold(f64::NEG_INFINITY, f64::max);

    let exps: Vec<f64> = input.iter().map(|c| (c.norm() - max_norm).exp()).collect();

    let sum: f64 = exps.iter().sum();

    exps.into_iter().map(|x| x / sum).collect()
}

pub fn softmax_row_real(input: &Vec<Complex<f64>>) -> Vec<f64> {
    let max_norm = input.iter().map(|c| c.re).fold(f64::NEG_INFINITY, f64::max);

    let exps: Vec<f64> = input.iter().map(|c| (c.re - max_norm).exp()).collect();

    let sum: f64 = exps.iter().sum();

    exps.iter().map(|x| x / sum).collect()
}

pub fn log_softmax_row(input: &Vec<Complex<f64>>) -> Vec<f64> {
    // Extract real parts only
    let real_parts: Vec<f64> = input.iter().map(|c| c.re).collect();

    // Numerical stability: subtract max real part
    let max_re = input.iter().map(|c| c.re).fold(f64::NEG_INFINITY, f64::max);

    let shifted_exps: Vec<f64> = real_parts.iter().map(|&x| (x - max_re).exp()).collect();

    let sum_exp: f64 = shifted_exps.iter().sum();

    let log_sum_exp = sum_exp.ln() + max_re;

    let result: Vec<f64> = real_parts.iter().map(|&x| x - log_sum_exp).collect();

    result
}
