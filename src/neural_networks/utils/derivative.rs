use crate::neural_networks::network_components::{layer::ActivationType, linear_layer::LinearLayer};
use num::abs;
use num_complex::Complex;
use std::f64::consts::PI;

use super::activation::{erf_complex, softmax_row};

fn sigmoid_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    let sigmoid_value = 1.0 / (1.0 + (-z.re).exp());
    Complex::new(sigmoid_value * (1.0 - sigmoid_value), 0.0)
}

fn tanh_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    let tanh_value = z.re.tanh();
    Complex::new(1.0 - tanh_value * tanh_value, 0.0)
}

fn relu_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    if z.re > 0.0 {
        Complex::new(1.0, 0.0)
    } else {
        Complex::new(0.0, 0.0)
    }
}

fn leaky_relu_derivative_complex(z: Complex<f64>, alpha: f64) -> Complex<f64> {
    if z.re > 0.0 {
        Complex::new(1.0, 0.0)
    } else {
        Complex::new(alpha, 0.0)
    }
}

fn elu_derivative_complex(z: Complex<f64>, alpha: f64) -> Complex<f64> {
    if z.re > 0.0 {
        Complex::new(1.0, 0.0)
    } else {
        Complex::new(alpha * (-z.re).exp(), 0.0)
    }
}

fn gelu_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    let sqrt_2 = (2.0_f64).sqrt();
    let sqrt_2pi = (2.0 * PI).sqrt();

    // Compute CDF: 0.5 * (1 + erf(z / sqrt(2)))
    let scaled_z = z / Complex::new(sqrt_2, 0.0);
    let erf_result = erf_complex(scaled_z);
    let cdf = Complex::new(0.5, 0.0) * (Complex::new(1.0, 0.0) + erf_result);

    // Compute PDF: (1 / sqrt(2 * pi)) * exp(-z^2 / 2)
    let pdf = (Complex::new(1.0, 0.0) / Complex::new(sqrt_2pi, 0.0)) * (-z * z / Complex::new(2.0, 0.0)).exp();

    // Derivative: CDF + z * PDF
    cdf + z * pdf
}

fn softsign_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    let denom = (1.0 + z.re.abs()).powi(2);
    Complex::new(1.0 / denom, 0.0)
}

fn softplus_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    Complex::new(1.0 / (1.0 + (-z.re).exp()), 0.0)
}

// Function to compute the derivative (Jacobian) of softmax for a matrix of complex numbers
fn softmax_derivative_complex(data: &Vec<Complex<f64>>) -> Vec<Vec<Complex<f64>>> {
    let s = softmax_row(data); // Get the softmax values for the vector

    let mut jacobian: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); s.len()]; s.len()];

    // Loop through each pair of indices (i, j)
    for i in 0..s.len() {
        for j in 0..s.len() {
            if i == j {
                // Diagonal elements: s_i * (1 - s_i)
                jacobian[i][j] = s[i] * (Complex::new(1.0, 0.0) - s[i]);
            } else {
                // Off-diagonal elements: -s_i * s_j
                jacobian[i][j] = -s[i] * s[j];
            }
        }
    }

    jacobian
}

pub fn get_ada_grad_optimizer(gradients: &Vec<Vec<f64>>) -> f64 {
    // https://mlfromscratch.com/optimizers-explained/#/
    // sum must not be null, because of sqrt
    let mut sum = 0.0000000005;

    for i in 0..gradients.len() {
        for j in 0..gradients[0].len() {
            sum += gradients[i][j] * gradients[i][j];
        }
    }

    sum.sqrt()
}

pub fn get_r_rms_prop(gradients: &Vec<Vec<f64>>, b1: f64, r: f64) -> f64 {
    // https://mlfromscratch.com/optimizers-explained/#/
    // sum must not be null, because of sqrt
    let mut sum = 0.0000005;

    for i in 0..gradients.len() {
        for j in 0..gradients[0].len() {
            sum += b1 * r + (1.0 - b1) * gradients[i][j] * gradients[i][j];
        }
    }

    sum
}

pub fn get_r_rms_value(gradient: &f64, b2: f64, v1: f64) -> f64 {
    // https://mlfromscratch.com/optimizers-explained/#/
    // sum must not be null, because of sqrt
    let mut sum = 0.0000005;

    sum += b2 * v1 + (1.0 - b2) * gradient * gradient;
    abs(sum)
}

pub fn get_adam(gradients: &Vec<Vec<f64>>, b1: f64, r: f64) -> f64 {
    // https://mlfromscratch.com/optimizers-explained/#/
    // sum must not be null, because of sqrt
    let mut sum = 0.0000005;

    for i in 0..gradients.len() {
        for j in 0..gradients[0].len() {
            sum += b1 * r + (1.0 - b1) * gradients[i][j];
        }
    }

    sum
}

pub fn get_adam_value(gradient: &f64, b1: f64, m1: f64) -> f64 {
    // https://mlfromscratch.com/optimizers-explained/#/
    // sum must not be null, because of sqrt
    let mut sum = 0.0000005;

    sum += b1 * m1 + (1.0 - b1) * gradient;

    sum
}

pub fn numerical_gradient<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64, linear_layer: &mut LinearLayer) -> Vec<Vec<Vec<Complex<f64>>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>, &mut LinearLayer) -> Complex<f64>,
{
    let mut grad_batch =  input.clone();

    for batch in 0..input.len() {
        for seq in 0..input[batch].len() {
            for dim in 0..input[batch][seq].len() {
                // Perturb input by epsilon
                let mut input_plus = input.clone();
                input_plus[batch][seq][dim] += epsilon;

                let mut input_minus = input.clone();
                input_minus[batch][seq][dim] -= epsilon;

                // Compute numerical gradient
                let loss_plus = f(&input_plus, linear_layer);
                let loss_minus = f(&input_minus, linear_layer);

                grad_batch[batch][seq][dim] = (loss_plus - loss_minus) / (2.0 * epsilon);
            }
        }
    }

    grad_batch
}

pub fn get_gradient_complex(data: &Vec<Vec<Complex<f64>>>, activation: ActivationType) -> Vec<Vec<Complex<f64>>> {
    match activation {
        ActivationType::SIGMOID => data.iter().map(|row| row.iter().map(|&x| sigmoid_derivative_complex(x)).collect()).collect(),
        ActivationType::TANH => data.iter().map(|row| row.iter().map(|&x| tanh_derivative_complex(x)).collect()).collect(),
        ActivationType::LINEAR => data.iter().map(|row| row.iter().map(|_| Complex::new(1.0, 0.0)).collect()).collect(), // Linear is identity
        ActivationType::RELU => data.iter().map(|row| row.iter().map(|&x| relu_derivative_complex(x)).collect()).collect(),
        ActivationType::LEAKYRELU => data.iter().map(|row| row.iter().map(|&x| leaky_relu_derivative_complex(x, 0.01)).collect()).collect(),
        ActivationType::ELU => data.iter().map(|row| row.iter().map(|&x| elu_derivative_complex(x, 1.0)).collect()).collect(), // Assuming alpha = 1.0
        ActivationType::SELU => data.iter().map(|row| row.iter().map(|&x| elu_derivative_complex(x, 1.0)).collect()).collect(), // Assuming scale = 1.0, alpha = 1.0
        ActivationType::GELU => data.iter().map(|row| row.iter().map(|&x| gelu_derivative_complex(x)).collect()).collect(),
        ActivationType::SOFTSIGN => data.iter().map(|row| row.iter().map(|&x| softsign_derivative_complex(x)).collect()).collect(),
        ActivationType::SOFTPLUS => data.iter().map(|row| row.iter().map(|&x| softplus_derivative_complex(x)).collect()).collect(),
        ActivationType::PROBIT => data.iter().map(|row| row.iter().map(|_| Complex::new(1.0, 0.0)).collect()).collect(), // Just return the value as is
        ActivationType::RANDOM => data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(),                     // Just return the value as is
        ActivationType::SOFTMAX => softmax_derivative_complex(&data[0]),
    }
}
