use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::calculate_adam_w_bias,
    matrix::{add_matrix_3d_c, clip_gradient_1d, is_nan_or_inf},
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalNormLayer {
    gamma: Vec<Complex<f64>>,
    beta: Vec<Complex<f64>>,
    epsilon: f64,
    pub learning_rate: f64,

    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    input_batch_before: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub previous_gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    normalized_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    mean_batch: Option<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    var_batch: Option<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    gradient: Option<Gradient>,
    #[serde(skip)]
    previous_gradient: Option<Gradient>,
    time_step: usize,
    #[serde(skip)]
    output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
}

impl NormalNormLayer {
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![Complex::new(1.0, 0.0); feature_dim],
            beta: vec![Complex::new(0.0, 0.0); feature_dim],
            epsilon,
            learning_rate,
            input_batch: None,
            input_batch_before: None,
            previous_gradient_input_batch: None,
            normalized_batch: None,
            mean_batch: None,
            var_batch: None,
            gradient: None,
            previous_gradient: None,
            output_batch: None,
            time_step: 0,
        }
    }

    pub fn normalize(&self, input: &Vec<Complex<f64>>) -> (Vec<Complex<f64>>, Complex<f64>, Complex<f64>) {
        let len: f64 = input.len() as f64;
        let mean: Complex<f64> = input.iter().sum::<Complex<f64>>() / len;

        let variance: Complex<f64> = input.iter().map(|x| (*x - mean).powu(2)).sum::<Complex<f64>>() / len;

        let stddev: Complex<f64> = (variance + Complex::new(self.epsilon, 0.0)).sqrt();

        let normalized: Vec<Complex<f64>> = input
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let val = ((*x - mean) / stddev) * self.gamma[i] + self.beta[i];
                Complex::new(val.re, 0.0)
            })
            .collect();

        (normalized, mean, variance)
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();
        let input_batch_before: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch_before();
        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
        let mut normalized_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
        let mut mean_batch: Vec<Vec<Complex<f64>>> = Vec::new();
        let mut var_batch: Vec<Vec<Complex<f64>>> = Vec::new();

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = add_matrix_3d_c(&input_batch, &input_batch_before);

        for input in input_batch.iter() {
            let mut norm_seq = Vec::new();
            let mut mean_seq = Vec::new();
            let mut var_seq = Vec::new();
            for vec in input.iter() {
                let (norm, mean, var) = self.normalize(vec);

                norm_seq.push(norm);
                mean_seq.push(mean);
                var_seq.push(var);
            }
            normalized_batch.push(norm_seq.clone());
            output_batch.push(norm_seq);
            mean_batch.push(mean_seq);
            var_batch.push(var_seq);
        }

        self.input_batch = Some(input_batch.clone());
        //self.input_batch = Some(layer_input.get_input_batch());
        self.input_batch_before = Some(input_batch_before.clone());
        self.normalized_batch = Some(normalized_batch);
        self.mean_batch = Some(mean_batch);
        self.var_batch = Some(var_batch);
        self.time_step = layer_input.get_time_step();
        self.output_batch = Some(output_batch.clone());
        self.previous_gradient_input_batch = Some(layer_input.get_previous_gradient_input_batch());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, grad_output: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found");
        let _input_batch_before = self.input_batch_before.as_ref().expect("Input batch before not found");
        let previous_gradient_input_batch = self.previous_gradient_input_batch.as_mut().expect("Previous gradient input batch not found");
        let normalized_batch = self.normalized_batch.as_ref().expect("Normalized batch not found");
        let mean_batch = self.mean_batch.as_ref().expect("Mean not found");
        let var_batch = self.var_batch.as_ref().expect("Variance not found");

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let feature_dim = input_batch[0][0].len();

        // Initialize the gradients
        let mut input_grads: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); feature_dim]; seq_len]; batch_size];
        let mut gamma_grad: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); feature_dim];
        let mut beta_grad: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); feature_dim];

        let n = feature_dim as f64;
        let eps = 1e-8;

        // println!("\n GRADIENT INPUT:  : {}, {}, {}", input_grads.len(), input_grads[0].len(), input_grads[0][0].len());
        // println!("previous gradient input batch: {:?}, {}, {}", &previous_gradient_input_batch.len(), previous_gradient_input_batch[0].len(), previous_gradient_input_batch[0][0].len());

        for b in 0..batch_size {
            //previous_gradient_input_batch[b] = transpose(&previous_gradient_input_batch[b]);
            for s in 0..seq_len {
                let mu: Complex<f64> = mean_batch[b][s];
                let var: Complex<f64> = var_batch[b][s] + eps;
                let std_inv: Complex<f64> = 1.0 / var.sqrt();
                let var_pow_minus_3_2: Complex<f64> = 1.0 / var.powf(1.5);

                // Precompute useful terms for mean/var gradients
                for f in 0..feature_dim {
                    let mut dvar_sum = Complex::new(0.0, 0.0);
                    let mut dmu_sum = Complex::new(0.0, 0.0);
                    let mut dx_minus_mu_sum = Complex::new(0.0, 0.0);
                    for d in 0..feature_dim {
                        let x: Complex<f64> = input_batch[b][s][d];
                        let x_hat: Complex<f64> = normalized_batch[b][s][d];
                        let dout: Complex<f64> = grad_output[b][s][d];

                        // ∂L/∂gamma and ∂L/∂beta
                        gamma_grad[d] += dout * x_hat;
                        beta_grad[d] += dout;

                        let dxhat: Complex<f64> = dout * self.gamma[f];
                        dvar_sum += dxhat * (x - mu) * (-0.5) * var_pow_minus_3_2;
                        dx_minus_mu_sum += -2.0 * (x - mu) / n;
                    }
                    for d in 0..feature_dim {
                        let dout: Complex<f64> = grad_output[b][s][d];
                        dmu_sum += dout * (-std_inv);
                    }

                    let dxhat: Complex<f64> = grad_output[b][s][f] * self.gamma[f];
                    let x: Complex<f64> = input_batch[b][s][f];
                    let _x_orig: Complex<f64> = _input_batch_before[b][s][f];
                    let _x_input = x - _x_orig;

                    let dmu = dmu_sum * self.gamma[f] + dvar_sum * dx_minus_mu_sum;

                    let gradient: Complex<f64> = (dxhat * std_inv) + (dvar_sum * (2.0 * (x - mu) / n)) + dmu / n;

                    for j in 0..feature_dim {
                        let _identity: f64 = if j == f { 1.0 } else { 0.0 };
                        input_grads[b][s][j] += Complex::new((gradient * _identity + gradient * previous_gradient_input_batch[b][s][j]).re, 0.0);
                    }
                }
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(input_grads);
        gradient.set_gradient_gamma(gamma_grad);
        gradient.set_gradient_beta(beta_grad);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.as_mut().expect("No gradient found in NormalNormLayer");
        let mut gradient_gamma = gradient.get_gradient_gamma();
        let mut gradient_beta = gradient.get_gradient_beta();

        let threshold = 1.0;
        clip_gradient_1d(&mut gradient_gamma, threshold);
        clip_gradient_1d(&mut gradient_beta, threshold);

        let batch_size = gradient_gamma.len() as f64;
        let learning_rate = self.learning_rate;

        let mut prev_m_gamma = vec![Complex::new(0.0, 0.0); gradient_gamma.len()];
        let mut prev_v_gamma = vec![Complex::new(0.0, 0.0); gradient_gamma.len()];

        let mut prev_m_beta = vec![Complex::new(0.0, 0.0); gradient_gamma.len()];
        let mut prev_v_beta = vec![Complex::new(0.0, 0.0); gradient_gamma.len()];

        if let Some(previous_gradient) = &mut self.previous_gradient {
            prev_m_gamma = previous_gradient.get_prev_m_gamma();
            prev_v_gamma = previous_gradient.get_prev_v_gamma();

            prev_m_beta = previous_gradient.get_prev_m_beta();
            prev_v_beta = previous_gradient.get_prev_v_beta();

            self.gamma = calculate_adam_w_bias(&self.gamma, &gradient.get_gradient_gamma(), &mut prev_m_gamma, &mut prev_v_gamma, learning_rate, gradient.get_time_step());
            self.beta = calculate_adam_w_bias(&self.beta, &gradient.get_gradient_beta(), &mut prev_m_beta, &mut prev_v_beta, learning_rate, gradient.get_time_step());
        } else {
            for (i, value) in self.gamma.iter_mut().enumerate() {
                if !is_nan_or_inf(&gradient_gamma[i]) {
                    *value -= self.learning_rate * (gradient_gamma[i] / batch_size);
                }
            }
            for (i, value) in self.beta.iter_mut().enumerate() {
                if !is_nan_or_inf(&gradient_beta[i]) {
                    *value -= self.learning_rate * (gradient_beta[i] / batch_size);
                }
            }
        }

        gradient.set_prev_m_gamma(prev_m_gamma);
        gradient.set_prev_v_gamma(prev_v_gamma);
        gradient.set_prev_m_beta(prev_m_beta);
        gradient.set_prev_v_beta(prev_v_beta);
        self.previous_gradient = Some(gradient.clone());
    }
}
