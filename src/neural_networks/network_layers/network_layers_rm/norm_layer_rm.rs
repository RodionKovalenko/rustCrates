use core::fmt::Debug;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::neural_networks::network_components::gradient_struct::Gradient;
use crate::neural_networks::network_components::layer_input_struct::LayerInput;
use crate::neural_networks::network_components::layer_output_struct::LayerOutput;
use crate::neural_networks::network_layers::default_layer::LayerInterface;
use crate::neural_networks::utils::dtype::{r, Real, C, ONE, ZERO};
use crate::neural_networks::utils::complex_layer_norm::{layer_norm_row, layer_norm_row_backward};
use crate::neural_networks::utils::matrix::normalize_bias;
use crate::neural_networks::utils::{
    adam_w::calculate_adam_w_bias,
    matrix::{add_vectors, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d}, matrix_rm::RowMajorMatrix,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalNormLayerRm {
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
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub normalized_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
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
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub batch_size: usize,
}

impl NormalNormLayerRm {
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![C::new(ONE, ZERO); feature_dim],
            beta: vec![C::new(ZERO, ZERO); feature_dim],
            epsilon,
            learning_rate,
            input_batch_rm: None,
            normalized_batch_rm: None,
            mean_batch: None,
            var_batch: None,
            gradient: None,
            padding_mask_batch: None,
            previous_gradient: None,
            output_batch_rm: None,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm = layer_input.get_input_batch_rm();

        let padding_mask_batch = layer_input.get_padding_mask_batch();
        self.batch_size = layer_input.get_batch_size();
        self.time_step = layer_input.get_time_step();

        let feature_dim = input_batch_rm[0].cols;
        if self.gamma.len() != feature_dim {
            self.gamma = vec![C::new(ONE, ZERO); feature_dim];
            self.beta = vec![C::new(ZERO, ZERO); feature_dim];
        }

        let (gamma, beta, epsilon) = (&self.gamma, &self.beta, self.epsilon);
        // One sample per task: (normalized output, row means, row variances).
        let per_sample: Vec<(RowMajorMatrix<C>, Vec<C>, Vec<C>)> = input_batch_rm.par_iter().enumerate().map(|(batch_idx, input_matrix)| {
            let seq_len = input_matrix.rows;
            let mut out = RowMajorMatrix::from_data(seq_len, input_matrix.cols, vec![C::new(ZERO, ZERO); seq_len * input_matrix.cols]);

            let all_valid;
            let padding_mask = if batch_idx < padding_mask_batch.len() {
                &padding_mask_batch[batch_idx]
            } else {
                all_valid = vec![1u32; seq_len];
                &all_valid
            };

            let mut mean_seq = Vec::with_capacity(seq_len);
            let mut var_seq = Vec::with_capacity(seq_len);

            for seq_idx in 0..seq_len {
                if seq_idx < padding_mask.len() && padding_mask[seq_idx] == 0 {
                    mean_seq.push(C::new(ZERO, ZERO));
                    var_seq.push(C::new(ZERO, ZERO));
                    continue;
                }

                let row = input_matrix.row_range(seq_idx);
                let (normalized, mean, variance) = layer_norm_row(&input_matrix.data[row.clone()], gamma, beta, epsilon);
                out.data[row].copy_from_slice(&normalized);
                let variance = C::new(variance, ZERO);

                mean_seq.push(mean);
                var_seq.push(variance);
            }

            (out, mean_seq, var_seq)
        }).collect();

        let mut output_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(per_sample.len());
        let mut mean_batch: Vec<Vec<C>> = Vec::with_capacity(per_sample.len());
        let mut var_batch: Vec<Vec<C>> = Vec::with_capacity(per_sample.len());
        for (out, mean_seq, var_seq) in per_sample {
            output_batch_rm.push(out);
            mean_batch.push(mean_seq);
            var_batch.push(var_seq);
        }

        if layer_input.get_calculate_gradient() {
            self.input_batch_rm = Some(input_batch_rm);
            self.normalized_batch_rm = Some(output_batch_rm.clone());
            self.mean_batch = Some(mean_batch);
            self.var_batch = Some(var_batch);
            self.output_batch_rm = Some(output_batch_rm.clone());
        } else {
            self.input_batch_rm = None;
            self.normalized_batch_rm = None;
            self.mean_batch = None;
            self.var_batch = None;
            self.output_batch_rm = None;
        }

        self.padding_mask_batch = Some(padding_mask_batch);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("NormalNormLayerRm: input_batch_rm missing");
        let mean_batch = self.mean_batch.as_ref().expect("NormalNormLayerRm: mean_batch missing");
        let var_batch = self.var_batch.as_ref().expect("NormalNormLayerRm: var_batch missing");

        let previous_gradient_rm = previous_gradient.get_gradient_input_batch_rm_cow();

        let empty_mask = vec![];
        let padding_mask_batch = self.padding_mask_batch.as_ref().unwrap_or(&empty_mask);

        let batch_size = input_batch_rm.len();
        let seq_len = input_batch_rm[0].rows;
        let feature_dim = input_batch_rm[0].cols;

        let (gamma, epsilon) = (&self.gamma, self.epsilon);
        let all_valid = vec![1u32; seq_len];

        // One sample per task: (input gradient, gamma gradient, beta gradient) of sample b.
        let per_sample: Vec<(RowMajorMatrix<C>, Vec<C>, Vec<C>)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let input = &input_batch_rm[b];
                let mut input_grad = RowMajorMatrix::from_data(input.rows, input.cols, vec![C::new(ZERO, ZERO); input.rows * input.cols]);
                let mut gamma_grad: Vec<C> = vec![C::new(ZERO, ZERO); feature_dim];
                let mut beta_grad: Vec<C> = vec![C::new(ZERO, ZERO); feature_dim];
                let padding_mask = if b < padding_mask_batch.len() { &padding_mask_batch[b] } else { &all_valid };

                for s in 0..input.rows {
                    if s < padding_mask.len() && padding_mask[s] == 0 {
                        continue;
                    }

                    let row = input.row_range(s);
                    let (row_input_grad, row_gamma_grad, row_beta_grad) =
                        layer_norm_row_backward(&input.data[row.clone()], mean_batch[b][s], var_batch[b][s].re, gamma, epsilon, &previous_gradient_rm[b].data[row.clone()]);
                    input_grad.data[row].copy_from_slice(&row_input_grad);
                    for f in 0..feature_dim {
                        gamma_grad[f] += row_gamma_grad[f];
                        beta_grad[f] += row_beta_grad[f];
                    }
                }
                (input_grad, gamma_grad, beta_grad)
            })
            .collect();

        // Sum the per-sample gamma/beta gradients in sample order.
        let mut input_grads_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
        let mut gamma_grad: Vec<C> = vec![C::new(ZERO, ZERO); feature_dim];
        let mut beta_grad: Vec<C> = vec![C::new(ZERO, ZERO); feature_dim];
        for (input_grad, sample_gamma_grad, sample_beta_grad) in per_sample {
            for f in 0..feature_dim {
                gamma_grad[f] += sample_gamma_grad[f];
                beta_grad[f] += sample_beta_grad[f];
            }
            input_grads_rm.push(input_grad);
        }

        if let Some(prev) = self.gradient.as_ref() {
            gamma_grad = add_vectors(&gamma_grad, &prev.get_gradient_gamma());
            beta_grad = add_vectors(&beta_grad, &prev.get_gradient_beta());
        }

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch_rm(input_grads_rm);
        gradient.set_gradient_gamma(gamma_grad);
        gradient.set_gradient_beta(beta_grad);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.as_mut().expect("No gradient found in NormalNormLayerRm");

        let mut gradient_gamma = gradient.get_gradient_gamma();
        let mut gradient_beta = gradient.get_gradient_beta();

        let mut batch_size: Real = if self.batch_size > 0 {
            r(self.batch_size as f64)
        } else if let Some(input_batch_rm) = &self.input_batch_rm {
            r(input_batch_rm.len() as f64)
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

impl LayerInterface for NormalNormLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        NormalNormLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        NormalNormLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        NormalNormLayerRm::update_parameters(self)
    }
}
