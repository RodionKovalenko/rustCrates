use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::calculate_adam_w_bias,
    matrix::{add_vectors, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_1d},
};

use super::{
    gradient_struct::{Gradient, GradientBatch},
    layer_input_struct::LayerInput,
    layer_output_struct::LayerOutput,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalNormLayer {
    pub gamma: Vec<Complex<f64>>,
    pub beta: Vec<Complex<f64>>,
    pub epsilon: f64,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub residual_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub previous_gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub normalized_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub mean_batch: Option<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub var_batch: Option<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,

    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub is_residual_input_present: bool,
}

impl NormalNormLayer {
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![Complex::new(1.0, 0.0); feature_dim],
            beta: vec![Complex::new(0.0, 0.0); feature_dim],
            epsilon,
            learning_rate,
            input_batch: None,
            residual_input_batch: None,
            previous_gradient_input_batch: None,
            normalized_batch: None,
            mean_batch: None,
            var_batch: None,
            gradient: None,
            padding_mask_batch: None,
            previous_gradient: None,
            is_residual_input_present: false,
            output_batch: None,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
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
                let val: Complex<f64> = ((*x - mean) / stddev) * self.gamma[i] + self.beta[i];
                //Complex::new(val.re, 0.0)
                val
            })
            .collect();

        (normalized, mean, variance)
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();
        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
        let mut normalized_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();
        let mut mean_batch: Vec<Vec<Complex<f64>>> = Vec::new();
        let mut var_batch: Vec<Vec<Complex<f64>>> = Vec::new();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

        self.batch_size = layer_input.get_batch_size();

        if self.gamma.len() != input_batch[0][0].len() {
            self.gamma = vec![Complex::new(1.0, 0.0); input_batch[0][0].len()];
            self.beta = vec![Complex::new(0.0, 0.0); input_batch[0][0].len()];
        }

        // let input_batch_before = vec![vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];
        // input_batch = add_matrix_3d_c(&input_batch, &input_batch_before);

        for (batch_idx, input) in input_batch.iter().enumerate() {
            let mut norm_seq = Vec::new();
            let mut mean_seq = Vec::new();
            let mut var_seq = Vec::new();
            let padding_mask = if batch_idx < padding_mask_batch.len() {
                &padding_mask_batch[batch_idx]
            } else {
                &vec![1u32; input.len()]
            };
            
            for (seq_idx, vec) in input.iter().enumerate() {
                let (norm, mean, var) = self.normalize(vec);
                
                // Apply padding mask: zero out padded positions
                let masked_norm = if seq_idx < padding_mask.len() && padding_mask[seq_idx] == 0 {
                    vec![Complex::new(0.0, 0.0); norm.len()]
                } else {
                    norm
                };

                norm_seq.push(masked_norm);
                mean_seq.push(mean);
                var_seq.push(var);
            }
            normalized_batch.push(norm_seq.clone());
            output_batch.push(norm_seq);
            mean_batch.push(mean_seq);
            var_batch.push(var_seq);
        }

        self.input_batch = Some(input_batch.clone());
        self.normalized_batch = Some(normalized_batch);
        self.mean_batch = Some(mean_batch);
        self.var_batch = Some(var_batch);
        self.padding_mask_batch = Some(padding_mask_batch);
        self.time_step = layer_input.get_time_step();
        self.output_batch = Some(output_batch.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found");
        let normalized_batch = self.normalized_batch.as_ref().expect("Normalized batch not found");
        let mean_batch = self.mean_batch.as_ref().expect("Mean not found");
        let var_batch = self.var_batch.as_ref().expect("Variance not found");

        let previous_gradient_batch = if !previous_gradient.get_gradient_input_batch().is_empty() {
            GradientBatch::Complex(previous_gradient.get_gradient_input_batch())
        } else {
            GradientBatch::Real(previous_gradient.get_gradient_input_batch_softmax())
        };
        
        let empty_mask = vec![];
        let padding_mask_batch = self.padding_mask_batch.as_ref().unwrap_or(&empty_mask);

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let feature_dim = input_batch[0][0].len();

        // Initialize the gradients
        let mut input_grads: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); feature_dim]; seq_len]; batch_size];
        let mut gamma_grad: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); feature_dim];
        let mut beta_grad: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); feature_dim];

        let n = feature_dim as f64;
        let eps = 1e-8;

        match previous_gradient_batch {
            GradientBatch::Complex(previous_gradient) => {
                for b in 0..batch_size {
                    let padding_mask = if b < padding_mask_batch.len() {
                        &padding_mask_batch[b]
                    } else {
                        &vec![1u32; seq_len]
                    };
                    
                    for s in 0..seq_len {
                        // Skip gradient computation for padded positions
                        if s < padding_mask.len() && padding_mask[s] == 0 {
                            continue;
                        }
                        
                        let mu: Complex<f64> = mean_batch[b][s];
                        let var: Complex<f64> = var_batch[b][s] + eps;
                        let std_inv: Complex<f64> = 1.0 / var.sqrt();
                        let var_pow_minus_3_2: Complex<f64> = 1.0 / var.powf(1.5);

                        for f in 0..feature_dim {
                            let x_hat: Complex<f64> = normalized_batch[b][s][f];
                            let dout: Complex<f64> = previous_gradient[b][s][f].conj();

                            // Accumulate gamma and beta gradients
                            gamma_grad[f] += dout * x_hat;
                            beta_grad[f] += dout;

                            let mut d_common_1 = Complex::new(0.0, 0.0);
                            let mut dmu_term_2 = Complex::new(0.0, 0.0);
                            let mut dmu_term_3 = Complex::new(0.0, 0.0);

                            // Compute sums over d features
                            for d in 0..feature_dim {
                                let x: Complex<f64> = input_batch[b][s][d];
                                let dout: Complex<f64> = previous_gradient[b][s][d].conj();

                                let dxhat: Complex<f64> = dout * self.gamma[f];
                                d_common_1 += dxhat * (x - mu);
                                dmu_term_2 += dout;
                                dmu_term_3 += (x - mu) / n;
                            }

                            let dvar_sum = d_common_1 * (-0.5) * var_pow_minus_3_2;
                            let dmu_sum = self.gamma[f] * (-std_inv) * dmu_term_2 + var_pow_minus_3_2 * d_common_1 * dmu_term_3;

                            let dxhat: Complex<f64> = previous_gradient[b][s][f].conj() * self.gamma[f];
                            let x: Complex<f64> = input_batch[b][s][f];

                            let gradient: Complex<f64> = (dxhat * std_inv) + (dvar_sum * ((2.0 * (x - mu)) / n)) + dmu_sum / n;

                            input_grads[b][s][f] = gradient.conj();
                        }
                    }
                }
            }
            GradientBatch::Real(_previous_gradient) => {
                panic!("Backward pass for real gradients not implemented for NormalNormLayer");
            }
        }

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            gamma_grad = add_vectors(&gamma_grad, &previous_gradient.get_gradient_gamma());
            beta_grad = add_vectors(&beta_grad, &previous_gradient.get_gradient_beta());
        }

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(input_grads);
        gradient.set_gradient_gamma(conjugate_1d(&gamma_grad));
        gradient.set_gradient_beta(conjugate_1d(&beta_grad));

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.as_mut().expect("No gradient found in NormalNormLayer");
        let mut gradient_gamma = gradient.get_gradient_gamma();
        let mut gradient_beta = gradient.get_gradient_beta();
        let input_batch = self.input_batch.as_ref().expect("no input batch in norm layer");

        let mut batch_size = input_batch.len() as f64;

        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }

        gradient_beta = average_vector_by_scalar(&gradient_beta, batch_size);
        gradient_gamma = average_vector_by_scalar(&gradient_gamma, batch_size);

        clip_all_gradients_by_global_norm_2d(&mut vec![], &mut gradient_gamma, self.global_norm, self.max_norm);
        clip_all_gradients_by_global_norm_2d(&mut vec![], &mut gradient_beta, self.global_norm, self.max_norm);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        let (mut prev_m_gamma, mut prev_v_gamma, mut prev_m_beta, mut prev_v_beta, mut prev_v_gamma_hat, mut prev_v_beta_hat) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_gamma(),
                previous_gradient.get_prev_v_gamma(),
                previous_gradient.get_prev_m_beta(),
                previous_gradient.get_prev_v_beta(),
                previous_gradient.get_prev_v_gamma_hat(),
                previous_gradient.get_prev_v_beta_hat(),
            )
        } else {
            // Initialize to zeros on first step
            (
                vec![Complex::new(0.0, 0.0); gradient_gamma.len()],
                vec![Complex::new(0.0, 0.0); gradient_gamma.len()],
                vec![Complex::new(0.0, 0.0); gradient_beta.len()],
                vec![Complex::new(0.0, 0.0); gradient_beta.len()],
                vec![Complex::new(0.0, 0.0); gradient_gamma.len()],
                vec![Complex::new(0.0, 0.0); gradient_beta.len()],
            )
        };

        calculate_adam_w_bias(
            &mut self.gamma,
            &gradient.get_gradient_gamma(),
            &mut prev_m_gamma,
            &mut prev_v_gamma,
            &mut prev_v_gamma_hat,
            learning_rate,
            time_step,
        );
        calculate_adam_w_bias(
            &mut self.beta,
            &gradient.get_gradient_beta(),
            &mut prev_m_beta,
            &mut prev_v_beta,
            &mut prev_v_beta_hat,
            learning_rate,
            time_step,
        );

        gradient.set_prev_m_gamma(prev_m_gamma);
        gradient.set_prev_v_gamma(prev_v_gamma);
        gradient.set_prev_m_beta(prev_m_beta);
        gradient.set_prev_v_beta(prev_v_beta);
        gradient.set_prev_v_gamma_hat(prev_v_gamma_hat);
        gradient.set_prev_v_beta_hat(prev_v_beta_hat);
        gradient.set_gradient_beta(gradient_beta.clone());
        gradient.set_gradient_gamma(gradient_gamma.clone());
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}
