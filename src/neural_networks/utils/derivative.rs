use std::f64::consts::{PI, SQRT_2};

use crate::neural_networks::network_components::layer::ActivationType;
use nalgebra::ComplexField;
use num::abs;
use num_complex::Complex;

use super::activation::{softmax_row, ALPHA, LAMBDA};

pub fn sigmoid_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    z * (Complex::new(1.0, 0.0) - z)
}

fn tanh_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    Complex::new(1.0, 0.0) - (z * z) // 1 - tanh^2(z)
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

// Derivative of SELU for complex numbers
fn selu_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    if z.re >= 0.0 {
        Complex::new(LAMBDA, 0.0)
    } else {
        Complex::new(LAMBDA * ALPHA, 0.0) * z.exp()
    }
}

pub fn gelu_derivative_complex(inactivated_input: Complex<f64>) -> Complex<f64> {
    // Compute sqrt(2 / pi)
    let sqrt_2_over_pi = SQRT_2 / PI.sqrt();

    // f(x) = sqrt(2 / pi) * (x + 0.044715 * x^3)
    let f_x = sqrt_2_over_pi * (inactivated_input + Complex::new(0.044715, 0.0) * inactivated_input.powf(3.0));

    // tanh(f(x)) and sech(f(x))
    let tanh_f_x = f_x.tanh();
    let sech_f_x = 1.0 / f_x.cosh();
    let sech_f_x_squared = sech_f_x.powi(2);

    // f'(x) = sqrt(2 / pi) * (1 + 0.134145 * x^2)
    let f_prime_x = sqrt_2_over_pi * (Complex::new(1.0, 0.0) + Complex::new(0.134145, 0.0) * inactivated_input.powf(2.0));

    // GELU'(x) = 0.5 * (1 + tanh(f(x))) + 0.5 * x * sech^2(f(x)) * f'(x)
    0.5 * (Complex::new(1.0, 0.0) + tanh_f_x) + 0.5 * inactivated_input * sech_f_x_squared * f_prime_x
}

pub fn softsign_derivative_complex(z: Complex<f64>) -> Complex<f64> {
    let real_part = z.re / (1.0 + z.re.abs()).powf(2.0);
    let imag_part = z.im / (1.0 + z.im.abs()).powf(2.0);

    Complex::new(real_part.abs() + imag_part.abs(), 0.0)
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

pub fn numerical_gradient_input<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<Complex<f64>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>) -> Complex<f64>,
{
    let mut grad_batch = vec![vec![Complex::new(0.0, 0.0); input[0][0].len()]; input[0].len()];

    for batch in 0..input.len() {
        for seq in 0..input[batch].len() {
            for dim in 0..input[batch][seq].len() {
                // Perturb input by epsilon
                let mut input_plus = input.clone();
                input_plus[batch][seq][dim] += epsilon;

                let mut input_minus = input.clone();
                input_minus[batch][seq][dim] -= epsilon;

                // Compute numerical gradient
                let loss_plus = f(&input_plus);
                let loss_minus = f(&input_minus);

                grad_batch[seq][dim] += (loss_plus - loss_minus) / (2.0 * epsilon);
            }
        }
    }

    grad_batch
}

pub fn numerical_gradient_weights<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>, epsilon: f64) -> Vec<Vec<Complex<f64>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>, &Vec<Vec<Complex<f64>>>) -> Complex<f64>,
{
    let mut grad_batch = vec![vec![Complex::new(0.0, 0.0); weights[0].len()]; weights.len()];

    for row in 0..weights.len() {
        for col in 0..weights[row].len() {
            // Perturb input by epsilon
            let mut weights_plus = weights.clone();
            weights_plus[row][col] += epsilon;

            let mut weights_minus = weights.clone();
            weights_minus[row][col] -= epsilon;

            // Compute numerical gradient
            let loss_plus = f(&input, &weights_plus);
            let loss_minus = f(&input, &weights_minus);

            grad_batch[row][col] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }
    }

    grad_batch
}

pub fn numerical_gradient_bias<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>, epsilon: f64) -> Vec<Complex<f64>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>, &Vec<Complex<f64>>) -> Complex<f64>,
{
    let mut grad_batch = vec![Complex::new(0.0, 0.0); bias.len()];

    for row in 0..bias.len() {
        // Perturb input by epsilon
        let mut bias_plus = bias.clone();
        bias_plus[row] += epsilon;

        let mut bias_minus = bias.clone();
        bias_minus[row] -= epsilon;

        // Compute numerical gradient
        let loss_plus = f(&input, &bias_plus);
        let loss_minus = f(&input, &bias_minus);

        grad_batch[row] = (loss_plus - loss_minus) / (2.0 * epsilon);
    }

    grad_batch
}

pub fn numerical_gradient_input_batch<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<Vec<Complex<f64>>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>) -> Complex<f64>,
{
    let mut grad_batch = vec![vec![vec![Complex::new(0.0, 0.0); input[0][0].len()]; input[0].len()]; input.len()];

    for batch in 0..input.len() {
        for seq in 0..input[batch].len() {
            for dim in 0..input[batch][seq].len() {
                // Perturb input by epsilon
                let mut input_plus = input.clone();
                input_plus[batch][seq][dim] += epsilon;

                let mut input_minus = input.clone();
                input_minus[batch][seq][dim] -= epsilon;

                // Compute numerical gradient
                let loss_plus = f(&input_plus);
                let loss_minus = f(&input_minus);

                grad_batch[batch][seq][dim] = (loss_plus - loss_minus) / (2.0 * epsilon);
            }
        }
    }

    grad_batch
}

pub fn numerical_gradient_weights_without_loss<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>, epsilon: f64) -> Vec<Vec<Vec<Complex<f64>>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>, &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Vec<Complex<f64>>>>,
{
    let mut grad_batch = vec![vec![vec![Complex::new(0.0, 0.0); weights[0].len()]; weights.len()]; input.len()];

    for row in 0..weights.len() {
        for col in 0..weights[row].len() {
            // Perturb input by epsilon
            let mut weights_plus = weights.clone();
            weights_plus[row][col] += epsilon;

            let mut weights_minus = weights.clone();
            weights_minus[row][col] -= epsilon;

            // Compute numerical gradient
            let loss_plus = f(&input, &weights_plus);
            let loss_minus = f(&input, &weights_minus);

            for batch_ind in 0..input.len() {
                for (seq_ind, _input_vec) in input[batch_ind].iter().enumerate() {
                    let gradient: Complex<f64> = (loss_plus[batch_ind][seq_ind][col] - loss_minus[batch_ind][seq_ind][col]) / (2.0 * epsilon);
                    grad_batch[batch_ind][row][col] += gradient;
                }
            }
        }
    }

    grad_batch
}

pub fn numerical_gradient_weights_multiple_layers_without_loss<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>, output: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<Vec<Complex<f64>>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>, &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Vec<Complex<f64>>>>,
{
    let mut grad_batch = vec![vec![vec![Complex::new(0.0, 0.0); weights[0].len()]; weights.len()]; input.len()];

    for row in 0..weights.len() {
        for col in 0..weights[row].len() {
            // Perturb input by epsilon
            let mut weights_plus = weights.clone();
            weights_plus[row][col] += epsilon;

            let mut weights_minus = weights.clone();
            weights_minus[row][col] -= epsilon;

            // Compute numerical gradient
            let loss_plus = f(&input, &weights_plus);
            let loss_minus = f(&input, &weights_minus);

            for batch_ind in 0..output.len() {
                for (seq_ind, _input_vec) in output[batch_ind].iter().enumerate() {
                    for (dim_ind, _value) in _input_vec.iter().enumerate() {
                        let gradient: Complex<f64> = (loss_plus[batch_ind][seq_ind][dim_ind] - loss_minus[batch_ind][seq_ind][dim_ind]) / (2.0 * epsilon);
                        grad_batch[batch_ind][row][col] += gradient;
                    }
                }
            }
        }
    }

    grad_batch
}

pub fn numerical_gradient_bias_without_loss<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Complex<f64>>, epsilon: f64) -> Vec<Complex<f64>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>, &Vec<Complex<f64>>) -> Vec<Vec<Vec<Complex<f64>>>>,
{
    let mut grad_batch = vec![Complex::new(0.0, 0.0); bias.len()];

    for row in 0..bias.len() {
        // Perturb input by epsilon
        let mut bias_plus = bias.clone();
        bias_plus[row] += epsilon;

        let mut bias_minus = bias.clone();
        bias_minus[row] -= epsilon;

        // Compute numerical gradient
        let loss_plus = f(&input, &bias_plus);
        let loss_minus = f(&input, &bias_minus);

        // Sum all elements of the loss to obtain a proper gradient estimation
        let sum_loss_plus: Complex<f64> = loss_plus.iter().flat_map(|batch| batch.iter()).flat_map(|seq| seq.iter()).sum();

        let sum_loss_minus: Complex<f64> = loss_minus.iter().flat_map(|batch| batch.iter()).flat_map(|seq| seq.iter()).sum();

        // Compute numerical gradient
        let gradient = (sum_loss_plus - sum_loss_minus) / (Complex::new(2.0 * epsilon, 0.0));
        grad_batch[row] = gradient;
    }

    grad_batch
}

pub fn numerical_gradient_input_batch_jacobi_without_loss<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<Vec<Complex<f64>>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>>,
{
    let mut grad_batch = vec![vec![vec![Complex::new(0.0, 0.0); input[0][0].len()]; input[0].len()]; input.len()];

    for batch in 0..input.len() {
        for seq in 0..input[batch].len() {
            for dim_i in 0..input[batch][seq].len() {
                for dim_j in 0..input[batch][seq].len() {
                    // Perturb input by epsilon
                    let mut input_plus = input.clone();
                    input_plus[batch][seq][dim_j] += epsilon;

                    let mut input_minus = input.clone();
                    input_minus[batch][seq][dim_j] -= epsilon;

                    // Compute numerical gradient
                    let loss_plus = f(&input_plus);
                    let loss_minus = f(&input_minus);

                    let gradient: Complex<f64> = (loss_plus[batch][seq][dim_i] - loss_minus[batch][seq][dim_i]) / (2.0 * epsilon);
                    grad_batch[batch][seq][dim_i] += gradient;
                }
            }
        }
    }

    grad_batch
}

pub fn numerical_gradient_input_batch_without_loss<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<Vec<Complex<f64>>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>>,
{
    let mut grad_batch = vec![vec![vec![Complex::new(0.0, 0.0); input[0][0].len()]; input[0].len()]; input.len()];

    for batch in 0..input.len() {
        for seq in 0..input[batch].len() {
            for dim_i in 0..input[batch][seq].len() {
                // Perturb input by epsilon
                let mut input_plus = input.clone();
                input_plus[batch][seq][dim_i] += epsilon;

                let mut input_minus = input.clone();
                input_minus[batch][seq][dim_i] -= epsilon;

                // Compute numerical gradient
                let loss_plus = f(&input_plus);
                let loss_minus = f(&input_minus);

                let sum_loss_plus: Complex<f64> = loss_plus.iter().flat_map(|batch| batch.iter()).flat_map(|seq| seq.iter()).sum();

                let sum_loss_minus: Complex<f64> = loss_minus.iter().flat_map(|batch| batch.iter()).flat_map(|seq| seq.iter()).sum();

                // Compute numerical gradient
                let gradient = (sum_loss_plus - sum_loss_minus) / (Complex::new(2.0 * epsilon, 0.0));
                grad_batch[batch][seq][dim_i] = gradient;
            }
        }
    }

    grad_batch
}

pub fn test_gradient_batch_error(numerical_grad_batch: &Vec<Vec<Vec<Complex<f64>>>>, analytical_grad_batch: &Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) {
    for (gradient_numerical, gradient_analytical) in numerical_grad_batch.iter().zip(analytical_grad_batch) {
        test_gradient_error_2d(gradient_numerical, gradient_analytical, epsilon);
    }
}

pub fn test_gradient_error_2d(numerical_grad: &Vec<Vec<Complex<f64>>>, analytical_grad: &Vec<Vec<Complex<f64>>>, epsilon: f64) {
    for (row_numerical, row_analytical) in numerical_grad.iter().zip(analytical_grad) {
        test_gradient_error_1d(row_numerical, row_analytical, epsilon);
    }
}

pub fn test_gradient_error_1d(numerical_grad: &Vec<Complex<f64>>, analytical_grad: &Vec<Complex<f64>>, epsilon: f64) {
    for (val_numerical, val_analytical) in numerical_grad.iter().zip(analytical_grad) {
        let abs_diff = (val_numerical - val_analytical).abs();
        // take the largest value out of (val_numerical, val_analytical, epsilon)
        let max_val = val_numerical.abs().max(val_analytical.abs()).max(epsilon);
        let rel_diff = abs_diff / max_val;

        if rel_diff > epsilon {
            println!("Gradient mismatch: numerical = {:.12}, analytical = {:.12}, abs_diff = {:.12}, rel_diff = {:.12}", val_numerical, val_analytical, abs_diff, rel_diff);
        }

        assert!(rel_diff < epsilon);
    }
}

pub fn get_gradient_complex(activated_data: &Vec<Vec<Complex<f64>>>, input_data: &Vec<Vec<Complex<f64>>>, activation: ActivationType) -> Vec<Vec<Complex<f64>>> {
    match activation {
        ActivationType::SIGMOID => activated_data.iter().map(|row| row.iter().map(|&x| sigmoid_derivative_complex(x)).collect()).collect(),
        ActivationType::TANH => activated_data.iter().map(|row| row.iter().map(|&x| tanh_derivative_complex(x)).collect()).collect(),
        ActivationType::LINEAR => activated_data.iter().map(|row| row.iter().map(|_| Complex::new(1.0, 0.0)).collect()).collect(), // Linear is identity
        ActivationType::RELU => activated_data.iter().map(|row| row.iter().map(|&x| relu_derivative_complex(x)).collect()).collect(),
        ActivationType::LEAKYRELU => activated_data.iter().map(|row| row.iter().map(|&x| leaky_relu_derivative_complex(x, 0.01)).collect()).collect(),
        ActivationType::ELU => activated_data.iter().map(|row| row.iter().map(|&x| elu_derivative_complex(x, 1.0)).collect()).collect(), // Assuming alpha = 1.0
        ActivationType::SELU => activated_data.iter().map(|row| row.iter().map(|&x| selu_derivative_complex(x)).collect()).collect(),    // Assuming scale = 1.0, alpha = 1.0
        ActivationType::GELU => input_data.iter().map(|row| row.iter().map(|&x| gelu_derivative_complex(x)).collect()).collect(),
        ActivationType::SOFTSIGN => input_data.iter().map(|row| row.iter().map(|&x| softsign_derivative_complex(x)).collect()).collect(),
        ActivationType::SOFTPLUS => activated_data.iter().map(|row| row.iter().map(|&x| softplus_derivative_complex(x)).collect()).collect(),
        ActivationType::PROBIT => activated_data.iter().map(|row| row.iter().map(|_| Complex::new(1.0, 0.0)).collect()).collect(), // Just return the value as is
        ActivationType::RANDOM => activated_data.iter().map(|row| row.iter().map(|&x| x).collect()).collect(),                     // Just return the value as is
        ActivationType::SOFTMAX => softmax_derivative_complex(&activated_data[0]),
    }
}
