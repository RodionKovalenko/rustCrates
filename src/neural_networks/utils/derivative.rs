use crate::neural_networks::network_components::layer::ActivationType;
use num::abs;
use num_complex::{Complex, ComplexFloat};

use super::activation::{ALPHA, LAMBDA};

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
    use std::f64::consts::PI;

    let sqrt_2_over_pi = (2.0 / PI).sqrt();

    // f(x) = sqrt(2 / pi) * (x + 0.044715 * x^3)
    let x3 = inactivated_input.powi(3);
    let f_x = sqrt_2_over_pi * (inactivated_input + Complex::new(0.044715, 0.0) * x3);

    // Numerically stable tanh approximation
    let tanh_f_x = if f_x.norm() > 20.0 {
        Complex::new(1.0, 0.0) // tanh saturates for large values
    } else {
        f_x.tanh()
    };

    // Compute sech^2(f_x) safely: sech^2(x) = 4 / (exp(2x) + 2 + exp(-2x))
    let sech_f_x_squared = if f_x.norm() > 20.0 {
        Complex::new(0.0, 0.0) // sech^2(x) approaches 0 for large x
    } else {
        Complex::new(1.0, 0.0) - tanh_f_x.powi(2)
    };

    // f'(x) = sqrt(2 / pi) * (1 + 0.134145 * x^2)
    let f_prime_x = sqrt_2_over_pi * (Complex::new(1.0, 0.0) + Complex::new(0.134145, 0.0) * inactivated_input.powi(2));

    // GELU'(x) = 0.5 * (1 + tanh(f(x))) + 0.5 * x * sech^2(f(x)) * f'(x)
    let gelu_derivative = Complex::new(0.5, 0.0) * (Complex::new(1.0, 0.0) + tanh_f_x) + Complex::new(0.5, 0.0) * inactivated_input * sech_f_x_squared * f_prime_x;

    // Smoothly clamp extreme values instead of hard zeroing
    if gelu_derivative.norm() > 1e6 {
        return gelu_derivative / gelu_derivative.norm() * Complex::new(1e6, 0.0);
    }
    if gelu_derivative.norm() < 1e-12 {
        return Complex::new(1e-12, 0.0);
    }

    gelu_derivative
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
fn softmax_derivative_complex(data: &Vec<f64>) -> Vec<Vec<f64>> {
    let mut jacobian: Vec<Vec<f64>> = vec![vec![0.0; data.len()]; data.len()];

    // Loop through each pair of indices (i, j)
    for i in 0..data.len() {
        for j in 0..data.len() {
            if i == j {
                // Diagonal elements: s_i * (1 - s_i)
                jacobian[i][j] = data[i] * (1.0 - data[i]);
            } else {
                // Off-diagonal elements: -s_i * s_j
                jacobian[i][j] = -data[i] * data[j];
            }
        }
    }

    jacobian
}

pub fn softmax_derivative_complex_matrix(softmax_values: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let num_rows = softmax_values.len();
    let num_cols = softmax_values[0].len();

    // 3D tensor to hold Jacobian matrices for each row
    let mut derivative: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; num_cols]; num_cols]; num_rows];

    for i in 0..num_rows {
        derivative[i] = softmax_derivative_complex(&softmax_values[i]);
    }

    // println!("original softmax derivative 3d: {:?} ", &derivative);
    let mut grouped_derivated: Vec<Vec<f64>> = vec![vec![0.0; num_cols]; num_rows];

    for i in 0..num_rows {
        for j in 0..num_cols {
            for k in 0..num_cols {
                if j == k {
                    grouped_derivated[i][j] += derivative[i][j][k];
                }
            }
        }
    }

    grouped_derivated
}

pub fn softmax_derivative_complex_jacobian(softmax_values: &Vec<Vec<f64>>) -> Vec<Vec<Vec<f64>>> {
    let num_rows = softmax_values.len();
    let num_cols = softmax_values[0].len();

    // 3D tensor to hold Jacobian matrices for each row
    let mut derivative: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; num_cols]; num_cols]; num_rows];

    for i in 0..num_rows {
        derivative[i] = softmax_derivative_complex(&softmax_values[i]);
    }

    //println!("original softmax derivative 3d: {:?} ", &derivative);

    derivative
}

pub fn reduce_softmax_jacobian(softmax_jacobian: &Vec<Vec<Vec<Complex<f64>>>>, // [num_rows, num_cols, num_cols]
) -> Vec<Vec<Complex<f64>>> {
    // Returns [num_rows, num_cols] (2D matrix)
    let num_rows = softmax_jacobian.len();
    let num_cols = softmax_jacobian[0].len();

    let mut reduced_jacobian = vec![vec![Complex::new(0.0, 0.0); num_cols]; num_rows];

    for i in 0..num_rows {
        for j in 0..num_cols {
            for k in 0..num_cols {
                reduced_jacobian[i][j] += softmax_jacobian[i][j][k]; // Summing over `k`
            }
        }
    }

    reduced_jacobian
}

pub fn backpropagate_softmax(softmax_jacobian: &Vec<Vec<Vec<Complex<f64>>>>, dl_ds: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    // Returns dL/dZ (gradient w.r.t. input Z)
    let num_rows = dl_ds.len();
    let num_cols = dl_ds[0].len();

    let mut dl_dz = vec![vec![Complex::new(0.0, 0.0); num_cols]; num_rows];

    for i in 0..num_rows {
        for j in 0..num_cols {
            for k in 0..num_cols {
                dl_dz[i][j] += softmax_jacobian[i][j][k] * dl_ds[i][k]; // Matrix-vector multiplication
            }
        }
    }

    dl_dz
}

pub fn backpropagate_softmax_masked(softmax_jacobian: &Vec<Vec<Vec<f64>>>, dl_ds: &Vec<Vec<Complex<f64>>>, padding_mask: &Vec<u32>) -> Vec<Vec<Complex<f64>>> {
    let num_rows = dl_ds.len();
    let num_cols = dl_ds[0].len();

    let mut dl_dz = vec![vec![Complex::new(0.0, 0.0); num_cols]; num_rows];

    for i in 0..num_rows {
        if padding_mask[i] == 0 {
            // Skip updating gradients for masked positions (zero out dl_dz[i])
            continue;
        }
        for j in 0..num_cols {
            for k in 0..num_cols {
                dl_dz[i][j] += softmax_jacobian[i][j][k] * dl_ds[i][k];
            }
        }
    }

    dl_dz
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
        _ => vec![],
    }
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

pub fn numerical_gradient_input_f64<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<f64>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>) -> Complex<f64>,
{
    let mut grad_batch = vec![vec![0.0; input[0][0].len()]; input[0].len()];

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

                grad_batch[seq][dim] += (loss_plus.re - loss_minus.re) / (2.0 * epsilon);
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

            grad_batch[row][col] += (loss_plus - loss_minus) / (2.0 * epsilon);
        }
    }

    grad_batch
}

pub fn numerical_gradient_weights_batch<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, weights: &Vec<Vec<Complex<f64>>>, epsilon: f64) -> Vec<Vec<Vec<Complex<f64>>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>, &Vec<Vec<Complex<f64>>>) -> Complex<f64>,
{
    let mut grad_batch = vec![vec![vec![Complex::new(0.0, 0.0); weights[0].len()]; weights.len()]; input.len()];

    for batch in 0..input.len() {
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

                grad_batch[batch][row][col] += (loss_plus - loss_minus) / (2.0 * epsilon);
            }
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

pub fn numerical_gradient_input_batch_softmax<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<Vec<f64>>>
where
    F: FnMut(&Vec<Vec<Vec<Complex<f64>>>>) -> Complex<f64>,
{
    let mut grad_batch = vec![vec![vec![0.0; input[0][0].len()]; input[0].len()]; input.len()];

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

                grad_batch[batch][seq][dim] = (loss_plus.re - loss_minus.re) / (2.0 * epsilon);
            }
        }
    }

    grad_batch
}


// should be < 1e-7
pub fn global_relative_error_l2(numerical_grad: &Vec<Vec<Vec<Complex<f64>>>>, analytical_grad: &Vec<Vec<Vec<Complex<f64>>>>) -> f64 {
    let mut diff_norm_sq = 0.0;
    let mut numerical_norm_sq = 0.0;
    let mut analytical_norm_sq = 0.0;

    for (batch_n, batch_a) in numerical_grad.iter().zip(analytical_grad.iter()) {
        for (seq_n, seq_a) in batch_n.iter().zip(batch_a.iter()) {
            for (dim_n, dim_a) in seq_n.iter().zip(seq_a.iter()) {
                let diff = dim_n - dim_a;
                diff_norm_sq += diff.re.powf(2.0); // Equivalent to |diff|^2
                numerical_norm_sq += dim_n.re.powf(2.0);
                analytical_norm_sq += dim_a.re.powf(2.0);
            }
        }
    }

    println!("\nabsolute error: {:?}", &diff_norm_sq);

    let diff_norm = diff_norm_sq.sqrt();
    let total_norm = numerical_norm_sq.sqrt() + analytical_norm_sq.sqrt();

    if total_norm == 0.0 {
        return 0.0; // or f64::INFINITY depending on your use case
    }

    let global_rel_error = diff_norm / total_norm;

    //assert!(global_rel_error < 1e-7);

    global_rel_error
}

pub fn global_relative_error_2d_l2(numerical_grad: &Vec<Vec<Complex<f64>>>, analytical_grad: &Vec<Vec<Complex<f64>>>) -> f64 {
    let mut diff_norm_sq = 0.0;
    let mut numerical_norm_sq = 0.0;
    let mut analytical_norm_sq = 0.0;

    for (seq_n, seq_a) in numerical_grad.iter().zip(analytical_grad.iter()) {
        for (dim_n, dim_a) in seq_n.iter().zip(seq_a.iter()) {
            let diff = dim_n - dim_a;
            diff_norm_sq += diff.norm_sqr(); // Equivalent to |diff|^2
            numerical_norm_sq += dim_n.norm_sqr();
            analytical_norm_sq += dim_a.norm_sqr();
        }
    }

    println!("absolute error: {:?}", &diff_norm_sq);

    let diff_norm = diff_norm_sq.sqrt();
    let total_norm = numerical_norm_sq.sqrt() + analytical_norm_sq.sqrt();

    if total_norm == 0.0 {
        return 0.0; // or f64::INFINITY depending on your use case
    }

    let global_rel_error = diff_norm / total_norm;

    //assert!(global_rel_error < 1e-7);

    global_rel_error
}

pub fn global_relative_error_2d_l2_f64(numerical_grad: &Vec<Vec<f64>>, analytical_grad: &Vec<Vec<f64>>) -> f64 {
    let mut diff_norm_sq = 0.0;
    let mut numerical_norm_sq = 0.0;
    let mut analytical_norm_sq = 0.0;

    for (seq_n, seq_a) in numerical_grad.iter().zip(analytical_grad.iter()) {
        for (dim_n, dim_a) in seq_n.iter().zip(seq_a.iter()) {
            let diff = dim_n - dim_a;
            diff_norm_sq += diff.powf(2.0); // Equivalent to |diff|^2
            numerical_norm_sq += dim_n.powf(2.0);
            analytical_norm_sq += dim_a.powf(2.0);
        }
    }

    println!("absolute error: {:?}", &diff_norm_sq);

    let diff_norm = diff_norm_sq.sqrt();
    let total_norm = numerical_norm_sq.sqrt() + analytical_norm_sq.sqrt();

    if total_norm == 0.0 {
        return 0.0; // or f64::INFINITY depending on your use case
    }

    let global_rel_error = diff_norm / total_norm;

    //assert!(global_rel_error < 1e-7);

    global_rel_error
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

pub fn numerical_gradient_weigts_transformer_multiple_layers_without_loss<F>(f: &mut F, weights: &Vec<Vec<Complex<f64>>>, output: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<Vec<Complex<f64>>>>
where
    F: FnMut(&Vec<Vec<Complex<f64>>>) -> Vec<Vec<Vec<Complex<f64>>>>,
{
    let mut grad_batch = vec![vec![vec![Complex::new(0.0, 0.0); weights[0].len()]; weights.len()]; output.len()];

    for row in 0..weights.len() {
        for col in 0..weights[row].len() {
            // Perturb input by epsilon
            let mut weights_plus = weights.clone();
            weights_plus[row][col] += epsilon;

            let mut weights_minus = weights.clone();
            weights_minus[row][col] -= epsilon;

            // Compute numerical gradient
            let loss_plus = f(&weights_plus);
            let loss_minus = f(&weights_minus);

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

pub fn numerical_gradient_input_batch_sum_without_loss<F>(f: &mut F, input: Vec<Vec<Vec<Complex<f64>>>>, epsilon: f64) -> Vec<Vec<Vec<Complex<f64>>>>
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
                let loss_plus: Vec<Vec<Vec<Complex<f64>>>> = f(&input_plus);
                let loss_minus: Vec<Vec<Vec<Complex<f64>>>> = f(&input_minus);

                // println!("output loss plus: {:?}", &loss_plus);
                // println!("output loss minus: {:?} \n\n", &loss_minus);

                let sum_loss_plus: Complex<f64> = loss_plus.iter().flat_map(|batch| batch.iter()).flat_map(|seq| seq.iter()).sum();
                let sum_loss_minus: Complex<f64> = loss_minus.iter().flat_map(|batch| batch.iter()).flat_map(|seq| seq.iter()).sum();
                let gradient = (sum_loss_plus - sum_loss_minus) / (2.0 * epsilon);

                //let gradient = (loss_plus[batch][seq][dim_i] - loss_minus[batch][seq][dim_i]) / (2.0 * epsilon);
                grad_batch[batch][seq][dim_i] = gradient;
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
                let loss_plus: Vec<Vec<Vec<Complex<f64>>>> = f(&input_plus);
                let loss_minus: Vec<Vec<Vec<Complex<f64>>>> = f(&input_minus);

                // println!("output loss plus: {:?}", &loss_plus);
                // println!("output loss minus: {:?} \n\n", &loss_minus);

                let gradient = (loss_plus[batch][seq][dim_i] - loss_minus[batch][seq][dim_i]) / (2.0 * epsilon);
                grad_batch[batch][seq][dim_i] = gradient;
            }
        }
    }

    grad_batch
}

pub fn numerical_gradient_check<F>(f: F, z: &Vec<Vec<Complex<f64>>>, epsilon: f64) -> Vec<Vec<Complex<f64>>>
where
    F: Fn(&Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>>,
{
    let num_rows = z.len();
    let num_cols = z[0].len();
    let mut numerical_gradient = vec![vec![Complex::new(0.0, 0.0); num_cols]; num_rows];

    for i in 0..num_rows {
        for j in 0..num_cols {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();

            // Perturb z[i][j] by +epsilon
            z_plus[i][j] += Complex::new(epsilon, 0.0);
            let f_plus = f(&z_plus);

            // Perturb z[i][j] by -epsilon
            z_minus[i][j] -= Complex::new(epsilon, 0.0);
            let f_minus = f(&z_minus);

            // Compute the numerical gradient
            numerical_gradient[i][j] = (f_plus[i][j] - f_minus[i][j]) / Complex::new(2.0 * epsilon, 0.0);
        }
    }

    numerical_gradient
}

pub fn numerical_gradient_check_f64<F>(f: F, z: &Vec<Vec<Complex<f64>>>, epsilon: f64) -> Vec<Vec<f64>>
where
    F: Fn(&Vec<Vec<Complex<f64>>>) -> Vec<Vec<f64>>,
{
    let num_rows = z.len();
    let num_cols = z[0].len();
    let mut numerical_gradient = vec![vec![0.0; num_cols]; num_rows];

    for i in 0..num_rows {
        for j in 0..num_cols {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();

            // Perturb z[i][j] by +epsilon
            z_plus[i][j] += Complex::new(epsilon, 0.0);
            let f_plus = f(&z_plus);

            // Perturb z[i][j] by -epsilon
            z_minus[i][j] -= Complex::new(epsilon, 0.0);
            let f_minus = f(&z_minus);

            // Compute the numerical gradient
            numerical_gradient[i][j] = (f_plus[i][j] - f_minus[i][j]) / (2.0 * epsilon);
        }
    }

    numerical_gradient
}

pub fn compute_relative_errors(numerical: &Vec<Vec<Vec<Complex<f64>>>>, analytical: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<f64>>> {
    let epsilon_tol = 1e-8;
    let mut rel_errors = vec![vec![vec![0.0; numerical[0][0].len()]; numerical[0].len()]; numerical.len()];

    for batch in 0..numerical.len() {
        for seq in 0..numerical[batch].len() {
            for dim in 0..numerical[batch][seq].len() {
                let g_num = numerical[batch][seq][dim];
                let g_ana = analytical[batch][seq][dim];

                // Compute absolute difference (magnitude)
                let diff = g_num - g_ana;
                let abs_diff = diff.norm(); // sqrt(re^2 + im^2)

                // Sum of magnitudes
                let sum_mags = g_num.norm() + g_ana.norm();

                // Relative error
                let rel_error = abs_diff / sum_mags.max(epsilon_tol);
                rel_errors[batch][seq][dim] = rel_error;

                // Optionally print values for inspection
                println!("Batch {}, Seq {}, Dim {}: Num = {:?}, Ana = {:?}, RelError = {:.6e}", batch, seq, dim, g_num, g_ana, rel_error);
            }
        }
    }

    rel_errors
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

pub fn test_gradient_batch_error_f64(numerical_grad_batch: &Vec<Vec<Vec<f64>>>, analytical_grad_batch: &Vec<Vec<Vec<f64>>>, epsilon: f64) {
    for (gradient_numerical, gradient_analytical) in numerical_grad_batch.iter().zip(analytical_grad_batch) {
        test_gradient_error_2d_f64(gradient_numerical, gradient_analytical, epsilon);
    }
}

pub fn test_gradient_error_2d_f64(numerical_grad: &Vec<Vec<f64>>, analytical_grad: &Vec<Vec<f64>>, epsilon: f64) {
    for (row_numerical, row_analytical) in numerical_grad.iter().zip(analytical_grad) {
        test_gradient_error_1d_f64(row_numerical, row_analytical, epsilon);
    }
}

pub fn test_gradient_error_1d_f64(numerical_grad: &Vec<f64>, analytical_grad: &Vec<f64>, epsilon: f64) {
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

pub fn test_gradient_error_1d(numerical_grad: &Vec<Complex<f64>>, analytical_grad: &Vec<Complex<f64>>, epsilon: f64) {
    for (val_numerical, val_analytical) in numerical_grad.iter().zip(analytical_grad) {
        let abs_diff = (val_numerical.re - val_analytical.re).abs();
        // take the largest value out of (val_numerical, val_analytical, epsilon)
        let max_val = val_numerical.abs().max(val_analytical.abs()).max(epsilon);
        let rel_diff = abs_diff / max_val;

        if rel_diff > epsilon {
            println!("Gradient mismatch: numerical = {:.12}, analytical = {:.12}, abs_diff = {:.12}, rel_diff = {:.12}", val_numerical, val_analytical, abs_diff, rel_diff);
        }

        assert!(rel_diff < epsilon);
    }
}
