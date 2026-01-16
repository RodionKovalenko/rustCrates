use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_components::adaptive_pooling::adaptive_avg_pool1d_layer::CompressionMetadata;
use crate::neural_networks::utils::matrix::RowMajorMatrix;

#[derive(Debug, Clone)]
pub enum GradientBatch {
    Complex(Vec<Vec<Vec<Complex<f64>>>>),
    Real(Vec<Vec<Vec<f64>>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    gradient_weights_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_weights_2_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    gradient_input_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    gradient_input_batch_softmax: Option<Vec<Vec<Vec<f64>>>>,
    gradient_bias_batch: Option<Vec<Vec<Complex<f64>>>>,
    gradient_gamma_batch: Option<Vec<Vec<Complex<f64>>>>,
    gradient_beta_batch: Option<Vec<Vec<Complex<f64>>>>,

    gradient_weights: Option<Vec<Vec<Complex<f64>>>>,
    gradient_weights_vec_batch_1: Option<Vec<Vec<Complex<f64>>>>,
    gradient_weights_vec_batch_2: Option<Vec<Vec<Complex<f64>>>>,

    gradient_weights_vec_1: Option<Vec<Complex<f64>>>,
    gradient_weights_vec_2: Option<Vec<Complex<f64>>>,

    gradient_weights_2: Option<Vec<Vec<Complex<f64>>>>,
    gradient_input: Option<Vec<Vec<Complex<f64>>>>,
    gradient_bias: Option<Vec<Complex<f64>>>,
    gradient_gamma: Option<Vec<Complex<f64>>>,
    gradient_beta: Option<Vec<Complex<f64>>>,

    gradient_weights_q_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_weights_v_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_weights_k_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,

    gradient_bias_pos_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,

    gradient_weights_q: Option<Vec<Vec<Complex<f64>>>>,
    gradient_weights_v: Option<Vec<Vec<Complex<f64>>>>,
    gradient_weights_k: Option<Vec<Vec<Complex<f64>>>>,

    gradient_bias_pos: Option<Vec<Vec<Complex<f64>>>>,

    prev_m_weights_q: Option<Vec<Vec<Complex<f64>>>>,
    prev_m_weights_k: Option<Vec<Vec<Complex<f64>>>>,
    prev_m_weights_v: Option<Vec<Vec<Complex<f64>>>>,
    prev_m_bias_pos: Option<Vec<Vec<Complex<f64>>>>,

    prev_v_weights_q: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights_k: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights_v: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_bias_pos: Option<Vec<Vec<Complex<f64>>>>,

    prev_m_weights: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights: Option<Vec<Vec<Complex<f64>>>>,

    prev_m_weights_2: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights_2: Option<Vec<Vec<Complex<f64>>>>,

    prev_m_weights_vec_1: Option<Vec<Complex<f64>>>,
    prev_v_weights_vec_1: Option<Vec<Complex<f64>>>,

    prev_v_weights_vec_hat_1: Option<Vec<Complex<f64>>>,
    prev_v_weights_vec_hat_2: Option<Vec<Complex<f64>>>,

    prev_m_weights_vec_2: Option<Vec<Complex<f64>>>,
    prev_v_weights_vec_2: Option<Vec<Complex<f64>>>,

    prev_v_weights_hat: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights_hat_2: Option<Vec<Vec<Complex<f64>>>>,

    prev_v_weights_q_hat: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights_k_hat: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights_v_hat: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights_p_hat: Option<Vec<Vec<Complex<f64>>>>,

    prev_v_gamma_hat: Option<Vec<Complex<f64>>>,
    prev_v_beta_hat: Option<Vec<Complex<f64>>>,

    prev_m_bias: Option<Vec<Complex<f64>>>,
    prev_v_bias: Option<Vec<Complex<f64>>>,
    prev_v_bias_hat: Option<Vec<Complex<f64>>>,

    prev_m_gamma: Option<Vec<Complex<f64>>>,
    prev_v_gamma: Option<Vec<Complex<f64>>>,

    prev_m_beta: Option<Vec<Complex<f64>>>,
    prev_v_beta: Option<Vec<Complex<f64>>>,
    pooling_metadata: Option<CompressionMetadata>,

    time_step: Option<usize>,
    total_valid_tokens: Option<usize>,
}

impl Gradient {
    pub fn new_default() -> Self {
        Gradient {
            gradient_weights_batch: None,
            gradient_weights_2_batch: None,
            gradient_input_batch: None,
            gradient_input_batch_rm: None,
            gradient_input_batch_softmax: None,
            gradient_bias_batch: None,
            gradient_gamma_batch: None,
            gradient_beta_batch: None,

            gradient_weights: None,
            gradient_weights_vec_batch_1: None,
            gradient_weights_vec_batch_2: None,
            gradient_weights_vec_1: None,
            gradient_weights_vec_2: None,
            gradient_weights_2: None,

            prev_m_weights_vec_1: None,
            prev_v_weights_vec_1: None,
            prev_m_weights_vec_2: None,
            prev_v_weights_vec_2: None,
            prev_v_weights_vec_hat_1: None,
            prev_v_weights_vec_hat_2: None,

            gradient_input: None,
            gradient_bias: None,
            gradient_gamma: None,
            gradient_beta: None,

            gradient_weights_q_batch: None,
            gradient_weights_v_batch: None,
            gradient_weights_k_batch: None,
            gradient_bias_pos_batch: None,

            prev_m_bias_pos: None,
            prev_v_bias_pos: None,

            gradient_weights_q: None,
            gradient_weights_v: None,
            gradient_weights_k: None,
            gradient_bias_pos: None,
            time_step: None,
            total_valid_tokens: None,

            prev_m_weights: None,
            prev_v_weights: None,
            prev_m_weights_2: None,
            prev_v_weights_2: None,
            prev_m_bias: None,
            prev_v_bias: None,

            prev_v_weights_hat: None,
            prev_v_weights_hat_2: None,
            prev_v_weights_q_hat: None,
            prev_v_weights_k_hat: None,
            prev_v_weights_v_hat: None,
            prev_v_weights_p_hat: None,
            prev_v_bias_hat: None,
            prev_v_gamma_hat: None,
            prev_v_beta_hat: None,

            prev_m_beta: None,
            prev_v_beta: None,

            prev_m_gamma: None,
            prev_v_gamma: None,

            prev_m_weights_k: None,
            prev_m_weights_q: None,
            prev_m_weights_v: None,
            prev_v_weights_k: None,
            prev_v_weights_q: None,
            prev_v_weights_v: None,
            pooling_metadata: None,
        }
    }
    pub fn set_gradient_input_batch(&mut self, gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_input_batch = Some(gradient_input_batch);
    }

    pub fn set_gradient_input_batch_rm(&mut self, gradient_input_batch_rm: Vec<RowMajorMatrix<Complex<f64>>>) {
        self.gradient_input_batch_rm = Some(gradient_input_batch_rm);
    }
    pub fn set_gradient_input_batch_softmax(&mut self, gradient_input_batch: Vec<Vec<Vec<f64>>>) {
        self.gradient_input_batch_softmax = Some(gradient_input_batch);
    }
    pub fn set_gradient_weight_batch(&mut self, gradient_weight_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_weights_batch = Some(gradient_weight_batch);
    }
    pub fn set_gradient_weight_2_batch(&mut self, gradient_weight_2_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_weights_2_batch = Some(gradient_weight_2_batch);
    }
    pub fn set_gradient_bias_batch(&mut self, gradient_bias_batch: Vec<Vec<Complex<f64>>>) {
        self.gradient_bias_batch = Some(gradient_bias_batch);
    }
    pub fn set_gradient_gamma_batch(&mut self, gradient_gamma_batch: Vec<Vec<Complex<f64>>>) {
        self.gradient_gamma_batch = Some(gradient_gamma_batch);
    }
    pub fn set_gradient_beta_batch(&mut self, gradient_beta_batch: Vec<Vec<Complex<f64>>>) {
        self.gradient_beta_batch = Some(gradient_beta_batch);
    }

    pub fn set_gradient_input(&mut self, gradient_input: Vec<Vec<Complex<f64>>>) {
        self.gradient_input = Some(gradient_input);
    }
    pub fn set_gradient_weights(&mut self, gradient_weights: Vec<Vec<Complex<f64>>>) {
        self.gradient_weights = Some(gradient_weights);
    }
    pub fn set_gradient_weights_2(&mut self, gradient_weights_2: Vec<Vec<Complex<f64>>>) {
        self.gradient_weights_2 = Some(gradient_weights_2);
    }
    pub fn set_gradient_bias(&mut self, gradient_bias: Vec<Complex<f64>>) {
        self.gradient_bias = Some(gradient_bias);
    }
    pub fn set_gradient_gamma(&mut self, gradient_gamma: Vec<Complex<f64>>) {
        self.gradient_gamma = Some(gradient_gamma);
    }
    pub fn set_gradient_beta(&mut self, gradient_beta: Vec<Complex<f64>>) {
        self.gradient_beta = Some(gradient_beta);
    }
    pub fn set_pooling_metadata(&mut self, metadata: CompressionMetadata) {
        self.pooling_metadata = Some(metadata);
    }
    pub fn get_pooling_metadata(&self) -> Option<&CompressionMetadata> {
        self.pooling_metadata.as_ref()
    }

    pub fn set_gradient_weights_q_batch(&mut self, gradient_weights_q_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_weights_q_batch = Some(gradient_weights_q_batch);
    }
    pub fn set_gradient_weights_v_batch(&mut self, gradient_weights_v_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_weights_v_batch = Some(gradient_weights_v_batch);
    }
    pub fn set_gradient_bias_pos_batch(&mut self, bias_pos_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_bias_pos_batch = Some(bias_pos_batch);
    }

    pub fn set_gradient_bias_pos(&mut self, bias_pos: Vec<Vec<Complex<f64>>>) {
        self.gradient_bias_pos = Some(bias_pos);
    }
    pub fn set_gradient_weights_k_batch(&mut self, gradient_weights_k_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_weights_k_batch = Some(gradient_weights_k_batch);
    }

    pub fn set_gradient_q(&mut self, gradient_weights_q: Vec<Vec<Complex<f64>>>) {
        self.gradient_weights_q = Some(gradient_weights_q);
    }
    pub fn set_gradient_v(&mut self, gradient_weights_v: Vec<Vec<Complex<f64>>>) {
        self.gradient_weights_v = Some(gradient_weights_v);
    }
    pub fn set_gradient_k(&mut self, gradient_weights_k: Vec<Vec<Complex<f64>>>) {
        self.gradient_weights_k = Some(gradient_weights_k);
    }
    pub fn set_prev_m_weights(&mut self, prev_m_weights: Vec<Vec<Complex<f64>>>) {
        self.prev_m_weights = Some(prev_m_weights);
    }
    pub fn set_prev_v_weights(&mut self, prev_v_weights: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights = Some(prev_v_weights);
    }

    pub fn set_prev_m_weights_2(&mut self, prev_m_weights_2: Vec<Vec<Complex<f64>>>) {
        self.prev_m_weights_2 = Some(prev_m_weights_2);
    }
    pub fn set_prev_v_weights_2(&mut self, prev_v_weights_2: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_2 = Some(prev_v_weights_2);
    }
    pub fn set_gradient_weights_vec_batch_1(&mut self, gradient_weights_vec_1: Vec<Vec<Complex<f64>>>) {
        self.gradient_weights_vec_batch_1 = Some(gradient_weights_vec_1);
    }
    pub fn set_gradient_weights_vec_batch_2(&mut self, gradient_weights_vec_2: Vec<Vec<Complex<f64>>>) {
        self.gradient_weights_vec_batch_2 = Some(gradient_weights_vec_2);
    }
    pub fn set_gradient_weights_vec_1(&mut self, gradient_weights_vec_1: Vec<Complex<f64>>) {
        self.gradient_weights_vec_1 = Some(gradient_weights_vec_1);
    }
    pub fn set_gradient_weights_vec_2(&mut self, gradient_weights_vec_2: Vec<Complex<f64>>) {
        self.gradient_weights_vec_2 = Some(gradient_weights_vec_2);
    }

    pub fn set_prev_m_weights_vec_1(&mut self, prev_m_weights_vec_1: Vec<Complex<f64>>) {
        self.prev_m_weights_vec_1 = Some(prev_m_weights_vec_1);
    }
    pub fn set_prev_v_weights_vec_1(&mut self, prev_v_weights_vec_1: Vec<Complex<f64>>) {
        self.prev_v_weights_vec_1 = Some(prev_v_weights_vec_1);
    }

    pub fn set_prev_m_weights_vec_2(&mut self, prev_m_weights_vec_2: Vec<Complex<f64>>) {
        self.prev_m_weights_vec_2 = Some(prev_m_weights_vec_2);
    }
    pub fn set_prev_v_weights_vec_2(&mut self, prev_v_weights_vec_2: Vec<Complex<f64>>) {
        self.prev_v_weights_vec_2 = Some(prev_v_weights_vec_2);
    }

    pub fn set_prev_v_weights_vec_hat_1(&mut self, prev_v_weights_vec_hat_1: Vec<Complex<f64>>) {
        self.prev_v_weights_vec_hat_1 = Some(prev_v_weights_vec_hat_1);
    }
    pub fn set_prev_v_weights_vec_hat_2(&mut self, prev_v_weights_vec_hat_2: Vec<Complex<f64>>) {
        self.prev_v_weights_vec_hat_2 = Some(prev_v_weights_vec_hat_2);
    }

    pub fn set_prev_m_weights_q(&mut self, prev_m_weights_q: Vec<Vec<Complex<f64>>>) {
        self.prev_m_weights_q = Some(prev_m_weights_q);
    }
    pub fn set_prev_m_weights_k(&mut self, prev_m_weights_k: Vec<Vec<Complex<f64>>>) {
        self.prev_m_weights_k = Some(prev_m_weights_k);
    }
    pub fn set_prev_m_weights_v(&mut self, prev_m_weights_v: Vec<Vec<Complex<f64>>>) {
        self.prev_m_weights_v = Some(prev_m_weights_v);
    }
    pub fn set_prev_m_bias_pos(&mut self, prev_m_bias_pos: Vec<Vec<Complex<f64>>>) {
        self.prev_m_bias_pos = Some(prev_m_bias_pos);
    }

    pub fn set_prev_v_weights_q(&mut self, prev_v_weights_q: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_q = Some(prev_v_weights_q);
    }
    pub fn set_prev_v_weights_k(&mut self, prev_v_weights_k: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_k = Some(prev_v_weights_k);
    }
    pub fn set_prev_v_weights_v(&mut self, prev_v_weights_v: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_v = Some(prev_v_weights_v);
    }
    pub fn set_prev_v_bias_pos(&mut self, prev_v_bias_pos: Vec<Vec<Complex<f64>>>) {
        self.prev_v_bias_pos = Some(prev_v_bias_pos);
    }

    pub fn set_prev_m_bias(&mut self, prev_m_bias: Vec<Complex<f64>>) {
        self.prev_m_bias = Some(prev_m_bias);
    }
    pub fn set_prev_v_bias(&mut self, prev_v_bias: Vec<Complex<f64>>) {
        self.prev_v_bias = Some(prev_v_bias);
    }

    pub fn set_prev_m_beta(&mut self, prev_m_beta: Vec<Complex<f64>>) {
        self.prev_m_beta = Some(prev_m_beta);
    }
    pub fn set_prev_v_beta(&mut self, prev_v_beta: Vec<Complex<f64>>) {
        self.prev_v_beta = Some(prev_v_beta);
    }

    pub fn set_prev_v_bias_hat(&mut self, prev_v_bias_hat: Vec<Complex<f64>>) {
        self.prev_v_bias_hat = Some(prev_v_bias_hat);
    }

    pub fn set_prev_v_beta_hat(&mut self, prev_v_beta_hat: Vec<Complex<f64>>) {
        self.prev_v_beta_hat = Some(prev_v_beta_hat);
    }
    pub fn set_prev_v_gamma_hat(&mut self, prev_v_gamma_hat: Vec<Complex<f64>>) {
        self.prev_v_gamma_hat = Some(prev_v_gamma_hat);
    }

    pub fn set_prev_v_weights_hat(&mut self, prev_v_weights_hat: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_hat = Some(prev_v_weights_hat);
    }

    pub fn set_prev_v_weights_hat_2(&mut self, prev_v_weights_hat_2: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_hat_2 = Some(prev_v_weights_hat_2);
    }

    pub fn set_prev_v_weights_q_hat(&mut self, prev_v_weights_q_hat: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_q_hat = Some(prev_v_weights_q_hat);
    }
    pub fn set_prev_v_weights_k_hat(&mut self, prev_v_weights_k_hat: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_k_hat = Some(prev_v_weights_k_hat);
    }
    pub fn set_prev_v_weights_v_hat(&mut self, prev_v_weights_v_hat: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_v_hat = Some(prev_v_weights_v_hat);
    }
    pub fn set_prev_v_weights_p_hat(&mut self, prev_v_weights_p_hat: Vec<Vec<Complex<f64>>>) {
        self.prev_v_weights_p_hat = Some(prev_v_weights_p_hat);
    }

    pub fn set_prev_m_gamma(&mut self, prev_m_gamma: Vec<Complex<f64>>) {
        self.prev_m_gamma = Some(prev_m_gamma);
    }
    pub fn set_prev_v_gamma(&mut self, prev_v_gamma: Vec<Complex<f64>>) {
        self.prev_v_gamma = Some(prev_v_gamma);
    }

    pub fn set_time_step(&mut self, time_step: usize) {
        self.time_step = Some(time_step);
    }

    pub fn set_total_valid_tokens(&mut self, total_valid_tokens: usize) {
        self.total_valid_tokens = Some(total_valid_tokens);
    }

    pub fn get_gradient_input_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        if let Some(gradient_input_batch) = &self.gradient_input_batch {
            return gradient_input_batch.clone();
        }

        if let Some(gradient_input_batch_rm) = &self.gradient_input_batch_rm {
            return gradient_input_batch_rm.iter().map(|m| m.to_rows()).collect();
        }

        vec![]
    }

    pub fn get_gradient_input_batch_rm(&self) -> Vec<RowMajorMatrix<Complex<f64>>> {
        if let Some(gradient_input_batch_rm) = &self.gradient_input_batch_rm {
            return gradient_input_batch_rm.clone();
        }

        if let Some(gradient_input_batch) = &self.gradient_input_batch {
            if gradient_input_batch.is_empty() {
                return vec![];
            }

            let mut out = Vec::with_capacity(gradient_input_batch.len());
            for m in gradient_input_batch {
                match RowMajorMatrix::try_from_rows(m) {
                    Some(rm) => out.push(rm),
                    None => return vec![],
                }
            }

            return out;
        }

        vec![]
    }

    pub fn get_gradient_input_batch_rm_ref(&self) -> Option<&[RowMajorMatrix<Complex<f64>>]> {
        self.gradient_input_batch_rm.as_deref()
    }

    pub fn get_gradient_input_batch_ref(&self) -> Option<&[Vec<Vec<Complex<f64>>>]> {
        self.gradient_input_batch.as_deref()
    }
    pub fn get_gradient_input_batch_softmax(&self) -> Vec<Vec<Vec<f64>>> {
        self.gradient_input_batch_softmax.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_input_softmax(&self) -> Vec<Vec<f64>> {
        if let Some(gradient_input_batch_softmax) = self.gradient_input_batch_softmax.clone() {
            self.group_gradient_batch_f64(&gradient_input_batch_softmax)
        } else {
            vec![]
        }
    }
    pub fn get_gradient_weight_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.gradient_weights_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_weight_2_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.gradient_weights_2_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_bias_batch(&self) -> Vec<Vec<Complex<f64>>> {
        self.gradient_bias_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_gamma_batch(&self) -> Vec<Vec<Complex<f64>>> {
        self.gradient_gamma_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_beta_batch(&self) -> Vec<Vec<Complex<f64>>> {
        self.gradient_beta_batch.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_gradient_weights_q_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.gradient_weights_q_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_weights_v_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.gradient_weights_v_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_weights_k_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.gradient_weights_k_batch.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_gradient_bias_pos_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.gradient_bias_pos_batch.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_m_weights(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_m_weights.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_v_weights_hat_2(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_hat_2.clone().unwrap_or_else(|| vec![])
    }
    
    pub fn get_prev_m_weights_2(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_m_weights_2.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_2(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_2.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_vec_hat_1(&self) -> Vec<Complex<f64>> {
        self.prev_v_weights_vec_hat_1.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_vec_hat_2(&self) -> Vec<Complex<f64>> {
        self.prev_v_weights_vec_hat_2.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_gradient_weights_vec_batch_1(&self) -> Vec<Vec<Complex<f64>>> {
        self.gradient_weights_vec_batch_1.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_weights_vec_batch_2(&self) -> Vec<Vec<Complex<f64>>> {
        self.gradient_weights_vec_batch_2.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_m_weights_vec_1(&self) -> Vec<Complex<f64>> {
        self.prev_m_weights_vec_1.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_vec_1(&self) -> Vec<Complex<f64>> {
        self.prev_v_weights_vec_1.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_m_weights_vec_2(&self) -> Vec<Complex<f64>> {
        self.prev_m_weights_vec_2.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_vec_2(&self) -> Vec<Complex<f64>> {
        self.prev_v_weights_vec_2.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_m_weigths_q(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_m_weights_q.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_m_weigths_k(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_m_weights_k.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_m_weigths_v(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_m_weights_v.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_v_weigths_q(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_q.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_k(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_k.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_hat(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_hat.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_bias_hat(&self) -> Vec<Complex<f64>> {
        self.prev_v_bias_hat.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_v_beta_hat(&self) -> Vec<Complex<f64>> {
        self.prev_v_beta_hat.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_gamma_hat(&self) -> Vec<Complex<f64>> {
        self.prev_v_gamma_hat.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_v_weights_v(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_v.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_v_weights_q_hat(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_q_hat.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_k_hat(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_k_hat.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_v_hat(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_v_hat.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights_p_hat(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights_p_hat.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_m_bias_pos(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_m_bias_pos.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_bias_pos(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_bias_pos.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_m_bias(&self) -> Vec<Complex<f64>> {
        self.prev_m_bias.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_bias(&self) -> Vec<Complex<f64>> {
        self.prev_v_bias.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_time_step(&self) -> usize {
        self.time_step.clone().unwrap_or_else(|| 0)
    }

    pub fn get_total_valid_tokens(&self) -> usize {
        self.total_valid_tokens.clone().unwrap_or_else(|| 1)
    }

    pub fn get_prev_m_beta(&self) -> Vec<Complex<f64>> {
        self.prev_m_beta.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_beta(&self) -> Vec<Complex<f64>> {
        self.prev_v_beta.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_prev_m_gamma(&self) -> Vec<Complex<f64>> {
        self.prev_m_gamma.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_gamma(&self) -> Vec<Complex<f64>> {
        self.prev_v_gamma.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_gradient_weights_q(&self) -> Vec<Vec<Complex<f64>>> {
        if let Some(gradient_weight_q_batch) = self.gradient_weights_q_batch.clone() {
            self.group_gradient_batch(&gradient_weight_q_batch)
        } else {
            self.gradient_weights_q.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn get_gradient_weights_v(&self) -> Vec<Vec<Complex<f64>>> {
        if let Some(gradient_weight_v_batch) = self.gradient_weights_v_batch.clone() {
            self.group_gradient_batch(&gradient_weight_v_batch)
        } else {
            self.gradient_weights_v.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn get_gradient_weights_k(&self) -> Vec<Vec<Complex<f64>>> {
        if let Some(gradient_weight_k_batch) = self.gradient_weights_k_batch.clone() {
            self.group_gradient_batch(&gradient_weight_k_batch)
        } else {
            self.gradient_weights_k.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn get_gradient_bias_pos(&self) -> Vec<Vec<Complex<f64>>> {
        if let Some(gradient_bias_pos_batch) = self.gradient_bias_pos_batch.clone() {
            self.group_gradient_batch(&gradient_bias_pos_batch)
        } else {
            self.gradient_bias_pos.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn get_gradient_weights(&self) -> Vec<Vec<Complex<f64>>> {
        if let Some(gradient_weight_batch) = self.gradient_weights_batch.clone() {
            self.group_gradient_batch(&gradient_weight_batch)
        } else {
            self.gradient_weights.clone().unwrap_or_else(|| vec![])
        }
    }
    pub fn get_gradient_weights_2(&self) -> Vec<Vec<Complex<f64>>> {
        if let Some(gradient_weight_2_batch) = self.gradient_weights_2_batch.clone() {
            self.group_gradient_batch(&gradient_weight_2_batch)
        } else {
            self.gradient_weights_2.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn get_gradient_input(&self) -> Vec<Vec<Complex<f64>>> {
        if let Some(gradient_input_batch) = self.gradient_input_batch.clone() {
            self.group_gradient_batch(&gradient_input_batch)
        } else {
            self.gradient_input.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn get_gradient_bias(&self) -> Vec<Complex<f64>> {
        if let Some(gradient_bias_batch) = self.gradient_bias_batch.clone() {
            self.group_gradient_batch_bias(&gradient_bias_batch)
        } else {
            self.gradient_bias.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn get_gradient_weights_vec_1(&self) -> Vec<Complex<f64>> {
        if let Some(gradient_weights_vec_batch_1) = self.gradient_weights_vec_batch_1.clone() {
            self.group_gradient_batch_bias(&gradient_weights_vec_batch_1)
        } else {
            vec![]
        }
    }
    pub fn get_gradient_weights_vec_2(&self) -> Vec<Complex<f64>> {
        if let Some(gradient_weights_vec_batch_2) = self.gradient_weights_vec_batch_2.clone() {
            self.group_gradient_batch_bias(&gradient_weights_vec_batch_2)
        } else {
            vec![]
        }
    }

    pub fn get_gradient_gamma(&self) -> Vec<Complex<f64>> {
        if let Some(gradient_gamma_batch) = self.gradient_gamma_batch.clone() {
            self.group_gradient_batch_bias(&gradient_gamma_batch)
        } else {
            self.gradient_gamma.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn get_gradient_beta(&self) -> Vec<Complex<f64>> {
        if let Some(gradient_beta_batch) = self.gradient_beta_batch.clone() {
            self.group_gradient_batch_bias(&gradient_beta_batch)
        } else {
            self.gradient_beta.clone().unwrap_or_else(|| vec![])
        }
    }

    pub fn group_gradient_batch(&self, weight_gradients_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Complex<f64>>> {
        if weight_gradients_batch.is_empty() || weight_gradients_batch[0].is_empty() || weight_gradients_batch[0][0].is_empty() {
            return vec![];
        }

        let mut weight_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); weight_gradients_batch[0][0].len()]; weight_gradients_batch[0].len()];

        for weight_gradient_batch in weight_gradients_batch {
            for (row, w_gradient) in weight_gradient_batch.iter().enumerate() {
                for (col, gradient_value) in w_gradient.iter().enumerate() {
                    if col >= weight_gradients[row].len() {
                        continue;
                    }
                    weight_gradients[row][col] += gradient_value;
                }
            }
        }

        weight_gradients
    }
    pub fn group_gradient_batch_f64(&self, weight_gradients_batch: &Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
        if weight_gradients_batch.is_empty() || weight_gradients_batch[0].is_empty() || weight_gradients_batch[0][0].is_empty() {
            return vec![];
        }

        let mut weight_gradients: Vec<Vec<f64>> = vec![vec![0.0; weight_gradients_batch[0][0].len()]; weight_gradients_batch[0].len()];

        for weight_gradient_batch in weight_gradients_batch {
            for (row, w_gradient) in weight_gradient_batch.iter().enumerate() {
                for (col, gradient_value) in w_gradient.iter().enumerate() {
                    weight_gradients[row][col] += gradient_value;
                }
            }
        }

        weight_gradients
    }
    pub fn group_gradient_batch_bias(&self, bias_gradient_batch: &Vec<Vec<Complex<f64>>>) -> Vec<Complex<f64>> {
        if bias_gradient_batch.is_empty() || bias_gradient_batch[0].is_empty() {
            return vec![];
        }

        let mut bias_gradients: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); bias_gradient_batch[0].len()];

        for bias_gradient_batch in bias_gradient_batch {
            for (row, w_gradient) in bias_gradient_batch.iter().enumerate() {
                bias_gradients[row] += w_gradient;
            }
        }

        bias_gradients
    }

    pub fn group_array_batch(weight_gradients_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Complex<f64>>> {
        let mut weight_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); weight_gradients_batch[0][0].len()]; weight_gradients_batch[0].len()];

        for weight_gradient_batch in weight_gradients_batch {
            for (row, w_gradient) in weight_gradient_batch.iter().enumerate() {
                for (col, gradient_value) in w_gradient.iter().enumerate() {
                    weight_gradients[row][col] += gradient_value;
                }
            }
        }

        weight_gradients
    }
}
