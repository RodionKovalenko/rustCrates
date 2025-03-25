use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    gradient_weights_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_bias_batch: Option<Vec<Vec<Complex<f64>>>>,
    gradient_gamma_batch: Option<Vec<Vec<Complex<f64>>>>,
    gradient_beta_batch: Option<Vec<Vec<Complex<f64>>>>,

    gradient_weights: Option<Vec<Vec<Complex<f64>>>>,
    gradient_input: Option<Vec<Vec<Complex<f64>>>>,
    gradient_bias: Option<Vec<Complex<f64>>>,
    gradient_gamma: Option<Vec<Complex<f64>>>,
    gradient_beta: Option<Vec<Complex<f64>>>,

    gradient_weights_q_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_weights_v_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    gradient_weights_k_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,

    gradient_weights_q: Option<Vec<Vec<Complex<f64>>>>,
    gradient_weights_v: Option<Vec<Vec<Complex<f64>>>>,
    gradient_weights_k: Option<Vec<Vec<Complex<f64>>>>,

    prev_m_weights: Option<Vec<Vec<Complex<f64>>>>,
    prev_v_weights: Option<Vec<Vec<Complex<f64>>>>,

    prev_m_bias: Option<Vec<Complex<f64>>>,
    prev_v_bias: Option<Vec<Complex<f64>>>,

    prev_m_gamma: Option<Vec<Complex<f64>>>,
    prev_v_gamma: Option<Vec<Complex<f64>>>,

    prev_m_beta: Option<Vec<Complex<f64>>>,
    prev_v_beta: Option<Vec<Complex<f64>>>,

    time_step: Option<usize>,
}

impl Gradient {
    pub fn new_default() -> Self {
        Gradient {
            gradient_weights_batch: None,
            gradient_input_batch: None,
            gradient_bias_batch: None,
            gradient_gamma_batch: None,
            gradient_beta_batch: None,

            gradient_weights: None,
            gradient_input: None,
            gradient_bias: None,
            gradient_gamma: None,
            gradient_beta: None,

            gradient_weights_q_batch: None,
            gradient_weights_v_batch: None,
            gradient_weights_k_batch: None,

            gradient_weights_q: None,
            gradient_weights_v: None,
            gradient_weights_k: None,
            time_step: None,

            prev_m_weights: None,
            prev_v_weights: None,
            prev_m_bias: None,
            prev_v_bias: None,

            prev_m_beta: None,
            prev_v_beta: None,

            prev_m_gamma: None,
            prev_v_gamma: None,
        }
    }
    pub fn set_gradient_input_batch(&mut self, gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_input_batch = Some(gradient_input_batch);
    }
    pub fn set_gradient_weight_batch(&mut self, gradient_weight_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_weights_batch = Some(gradient_weight_batch);
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
    pub fn set_gradient_bias(&mut self, gradient_bias: Vec<Complex<f64>>) {
        self.gradient_bias = Some(gradient_bias);
    }
    pub fn set_gradient_gamma(&mut self, gradient_gamma: Vec<Complex<f64>>) {
        self.gradient_gamma = Some(gradient_gamma);
    }
    pub fn set_gradient_beta(&mut self, gradient_beta: Vec<Complex<f64>>) {
        self.gradient_beta = Some(gradient_beta);
    }

    pub fn set_gradient_weights_q_batch(&mut self, gradient_weights_q_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_weights_q_batch = Some(gradient_weights_q_batch);
    }
    pub fn set_gradient_weights_v_batch(&mut self, gradient_weights_v_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.gradient_weights_v_batch = Some(gradient_weights_v_batch);
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
    pub fn set_prev_m_gamma(&mut self, prev_m_gamma: Vec<Complex<f64>>) {
        self.prev_m_gamma = Some(prev_m_gamma);
    }
    pub fn set_prev_v_gamma(&mut self, prev_v_gamma: Vec<Complex<f64>>) {
        self.prev_v_gamma = Some(prev_v_gamma);
    }

    pub fn set_time_step(&mut self, time_step: usize) {
        self.time_step = Some(time_step);
    }


    pub fn get_gradient_input_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.gradient_input_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_gradient_weight_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.gradient_weights_batch.clone().unwrap_or_else(|| vec![])
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

    pub fn get_prev_m_weigths(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_m_weights.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_prev_v_weights(&self) -> Vec<Vec<Complex<f64>>> {
        self.prev_v_weights.clone().unwrap_or_else(|| vec![])
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

    pub fn get_gradient_weights(&self) -> Vec<Vec<Complex<f64>>> {
        if let Some(gradient_weight_batch) = self.gradient_weights_batch.clone() {
            self.group_gradient_batch(&gradient_weight_batch)
        } else {
            self.gradient_weights.clone().unwrap_or_else(|| vec![])
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
    pub fn group_gradient_batch_bias(&self, bias_gradient_batch: &Vec<Vec<Complex<f64>>>) -> Vec<Complex<f64>> {
        let mut bias_gradients: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); bias_gradient_batch[0].len()];

        for bias_gradient_batch in bias_gradient_batch {
            for (row, w_gradient) in bias_gradient_batch.iter().enumerate() {
                bias_gradients[row] += w_gradient;
            }
        }

        bias_gradients
    }
}
