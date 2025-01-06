use num_complex::Complex;
use std::f64::consts::PI;
use crate::neural_networks::network_components::layer::ActivationType;

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
    let pdf = (Complex::new(1.0, 0.0) / Complex::new(sqrt_2pi, 0.0))
        * (-z * z / Complex::new(2.0, 0.0)).exp();

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
    let s = softmax_row(data);  // Get the softmax values for the vector

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
        ActivationType::RANDOM => data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(), // Just return the value as is
        ActivationType::SOFTMAX => softmax_derivative_complex(&data[0]),
    }
}
