use core::fmt::Debug;
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_components::gradient_struct::{Gradient, GradientBatch};
use crate::neural_networks::network_components::layer_input_struct::LayerInput;
use crate::neural_networks::network_components::layer_output_struct::LayerOutput;
use crate::neural_networks::utils::dtype::{r, Real, C, ONE, ZERO};
use crate::neural_networks::utils::matrix::normalize_bias;
use crate::neural_networks::utils::{
    adam_w::calculate_adam_w_bias,
    matrix::{add_vectors, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_1d},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalNormLayer {
    pub gamma: Vec<C>,
    pub beta: Vec<C>,
    pub epsilon: f64,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub residual_input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub previous_gradient_input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub normalized_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub mean_batch: Option<Vec<Vec<C>>>,
    #[serde(skip)]
    pub var_batch: Option<Vec<Vec<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,

    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub is_residual_input_present: bool,
}

impl NormalNormLayer {
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![C::new(ONE, ZERO); feature_dim],
            beta: vec![C::new(ZERO, ZERO); feature_dim],
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

    pub fn normalize(&self, input: &Vec<C>) -> (Vec<C>, C, C) {
        let len: Real = r(input.len() as f64);
        let mean: C = input.iter().sum::<C>() / len;

        let variance: C = input.iter().map(|x| (*x - mean).powu(2)).sum::<C>() / len;

        let stddev: C = (variance + C::new(r(self.epsilon), ZERO)).sqrt();

        let normalized: Vec<C> = input
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let val: C = ((*x - mean) / stddev) * self.gamma[i] + self.beta[i];
                //Complex::new(val.re, 0.0)
                val
            })
            .collect();

        (normalized, mean, variance)
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_ref = layer_input.get_input_batch_ref();

        let mut output_batch: Vec<Vec<Vec<C>>> = Vec::new();
        let mut normalized_batch: Vec<Vec<Vec<C>>> = Vec::new();
        let mut mean_batch: Vec<Vec<C>> = Vec::new();
        let mut var_batch: Vec<Vec<C>> = Vec::new();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

        self.batch_size = layer_input.get_batch_size();

        let feature_dim = input_batch_ref
            .expect("NormalNormLayer: input batch missing")
            .get(0)
            .and_then(|seq| seq.get(0))
            .map(|row| row.len())
            .unwrap_or(0);

        assert!(
            self.gamma.len() == self.beta.len() && self.gamma.len() == feature_dim,
            "NormalNormLayer: shape mismatch (gamma len = {}, beta len = {}, input feature_dim = {})",
            self.gamma.len(),
            self.beta.len(),
            feature_dim
        );

        // let input_batch_before = vec![vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];
        // input_batch = add_matrix_3d_c(&input_batch, &input_batch_before);

        let input_batch = input_batch_ref.expect("NormalNormLayer: input batch missing");
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

                let masked_norm = if seq_idx < padding_mask.len() && padding_mask[seq_idx] == 0 {
                    vec![C::new(ZERO, ZERO); norm.len()]
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

        if layer_input.get_calculate_gradient() {
            self.input_batch = Some(input_batch_ref.unwrap().to_vec());
            self.normalized_batch = Some(normalized_batch);

            self.mean_batch = Some(mean_batch);
            self.var_batch = Some(var_batch);
        } else {
            self.input_batch = None;
            self.normalized_batch = None;
            self.mean_batch = None;
            self.var_batch = None;
        }
        self.padding_mask_batch = Some(padding_mask_batch);
        self.time_step = layer_input.get_time_step();
        if layer_input.get_calculate_gradient() {
            self.output_batch = Some(output_batch.clone());
        } else {
            self.output_batch = None;
        }

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref();
        let normalized_batch = self.normalized_batch.as_ref();
        let mean_batch = self.mean_batch.as_ref().expect("Mean not found");
        let var_batch = self.var_batch.as_ref().expect("Variance not found");

        let previous_gradient_batch = if !previous_gradient.get_gradient_input_batch().is_empty() {
            Some(GradientBatch::Complex(previous_gradient.get_gradient_input_batch()))
        } else {
            Some(GradientBatch::Real(previous_gradient.get_gradient_input_batch_softmax()))
        };

        let empty_mask = vec![];
        let padding_mask_batch = self.padding_mask_batch.as_ref().unwrap_or(&empty_mask);

        let (batch_size, seq_len, feature_dim) = if let Some(input) = input_batch {
            (input.len(), input[0].len(), input[0][0].len())
        } else {
            panic!("Input batch not found");
        };

        assert!(
            self.gamma.len() == self.beta.len() && self.gamma.len() == feature_dim,
            "NormalNormLayer::backward: shape mismatch (gamma len = {}, beta len = {}, input feature_dim = {})",
            self.gamma.len(),
            self.beta.len(),
            feature_dim
        );

        // Initialize the gradients
        let mut input_grads: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); feature_dim]; seq_len]; batch_size];
        let mut gamma_grad: Vec<C> = vec![C::new(ZERO, ZERO); feature_dim];
        let mut beta_grad: Vec<C> = vec![C::new(ZERO, ZERO); feature_dim];

        let n: Real = r(feature_dim as f64);
        let eps: Real = r(1e-8);

        if let Some(previous_gradient_batch) = previous_gradient_batch {
            match previous_gradient_batch {
                GradientBatch::Complex(previous_gradient) => {
                    let input_batch = input_batch.expect("Input batch not found");
                    let normalized_batch = normalized_batch.expect("Normalized batch not found");
                    for b in 0..batch_size {
                        let padding_mask = if b < padding_mask_batch.len() { &padding_mask_batch[b] } else { &vec![1u32; seq_len] };

                        for s in 0..seq_len {
                            // Skip gradient computation for padded positions
                            if s < padding_mask.len() && padding_mask[s] == 0 {
                                continue;
                            }

                            let mu: C = mean_batch[b][s];
                            let var: C = var_batch[b][s] + C::new(eps, ZERO);
                            let var_sqrt = var.sqrt();
                            let std_inv: C = C::new(ONE, ZERO) / var_sqrt;
                            let var_pow_minus_3_2: C = C::new(ONE, ZERO) / (var * var_sqrt);

                            for f in 0..feature_dim {
                                let x_hat: C = normalized_batch[b][s][f];
                                let dout: C = previous_gradient[b][s][f].conj();

                                // Accumulate gamma and beta gradients
                                gamma_grad[f] += dout * x_hat;
                                beta_grad[f] += dout;

                                let mut d_common_1 = C::new(ZERO, ZERO);
                                let mut dmu_term_2 = C::new(ZERO, ZERO);
                                let mut dmu_term_3 = C::new(ZERO, ZERO);

                                // Compute sums over d features
                                for d in 0..feature_dim {
                                    let x: C = input_batch[b][s][d];
                                    let dout: C = previous_gradient[b][s][d].conj();

                                    let dxhat: C = dout * self.gamma[f];
                                    d_common_1 += dxhat * (x - mu);
                                    dmu_term_2 += dout;
                                    dmu_term_3 += (x - mu) / n;
                                }

                                let dvar_sum = d_common_1 * r(-0.5) * var_pow_minus_3_2;
                                let dmu_sum = self.gamma[f] * (-std_inv) * dmu_term_2 + var_pow_minus_3_2 * d_common_1 * dmu_term_3;

                                let dxhat: C = previous_gradient[b][s][f].conj() * self.gamma[f];
                                let x: C = input_batch[b][s][f];

                                let gradient: C = (dxhat * std_inv) + (dvar_sum * ((r(2.0) * (x - mu)) / n)) + dmu_sum / n;

                                input_grads[b][s][f] = gradient.conj();
                            }
                        }
                    }
                }
                GradientBatch::Real(_previous_gradient) => {
                    panic!("Backward pass for real gradients not implemented for NormalNormLayer");
                }
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

        let mut batch_size: Real = if self.batch_size > 0 {
            r(self.batch_size as f64)
        } else if let Some(input_batch) = &self.input_batch {
            r(input_batch.len() as f64)
        } else {
            ONE
        };

        if batch_size <= ZERO {
            batch_size = ONE;
        }

        gradient_beta = average_vector_by_scalar(&gradient_beta, batch_size);
        gradient_gamma = average_vector_by_scalar(&gradient_gamma, batch_size);

        clip_all_gradients_by_global_norm_2d(&mut vec![], &mut gradient_gamma, self.global_norm, self.max_norm);
        clip_all_gradients_by_global_norm_2d(&mut vec![], &mut gradient_beta, self.global_norm, self.max_norm);

        normalize_bias(&mut gradient_beta);
        normalize_bias(&mut gradient_gamma);

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
                vec![C::new(ZERO, ZERO); gradient_gamma.len()],
                vec![C::new(ZERO, ZERO); gradient_gamma.len()],
                vec![C::new(ZERO, ZERO); gradient_beta.len()],
                vec![C::new(ZERO, ZERO); gradient_beta.len()],
                vec![C::new(ZERO, ZERO); gradient_gamma.len()],
                vec![C::new(ZERO, ZERO); gradient_beta.len()],
            )
        };

        calculate_adam_w_bias(
            &mut self.gamma,
            &gradient_gamma,
            &mut prev_m_gamma,
            &mut prev_v_gamma,
            &mut prev_v_gamma_hat,
            learning_rate,
            time_step,
        );
        calculate_adam_w_bias(
            &mut self.beta,
            &gradient_beta,
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
