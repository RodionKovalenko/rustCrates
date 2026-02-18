use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::layer::LayerEnum,
    utils::{
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        dtype::{r, C, ONE, ZERO},
        matrix::{add_vector, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose, multiply_complex, normalize_bias, normalize_gradients},
        shared_f32_matrix::SharedF32Matrix,
        weights_initializer::{initialize_weights_complex, initialize_weights_complex_only_real},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weights: Vec<Vec<C>>,
    pub learning_rate: f64,
    pub bias: Vec<C>,
    pub smoothing: f64,
    pub ema: f64,
    pub norm_layer: Option<LayerEnum>,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub is_complex: bool,

    #[serde(skip)]
    pub tied_weights: Option<SharedF32Matrix>,
    #[serde(skip)]
    pub tied_embedding_grad_by_token: Option<Arc<RwLock<HashMap<usize, Vec<C>>>>>,

    #[serde(skip)]
    pub gradients: Vec<Vec<C>>,
    #[serde(skip)]
    pub gradients_bias: Vec<Vec<C>>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub output_indices: Option<Vec<Vec<Vec<usize>>>>,
}

impl LinearLayer {
    pub fn set_tied_weights(&mut self, weights: SharedF32Matrix, grad_by_token: Arc<RwLock<HashMap<usize, Vec<C>>>>) {
        self.tied_weights = Some(weights);
        self.tied_embedding_grad_by_token = Some(grad_by_token);
        self.sync_from_tied_weights();
    }

    pub fn is_tied(&self) -> bool {
        self.tied_weights.is_some() && self.tied_embedding_grad_by_token.is_some()
    }

    fn sync_from_tied_weights(&mut self) {
        let Some(tied) = &self.tied_weights else {
            return;
        };

        let table = tied.read(); // vocab x in_dim (real)
        if table.is_empty() || table[0].is_empty() {
            return;
        }

        let vocab = table.len();
        let in_dim = table[0].len();
        let mut transposed: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); vocab]; in_dim];

        for v in 0..vocab {
            if table[v].len() != in_dim {
                return;
            }
            for i in 0..in_dim {
                transposed[i][v] = C::new(r(table[v][i] as f64), ZERO);
            }
        }

        self.weights = transposed;
    }

    fn write_back_to_tied_weights(&mut self) {
        let Some(tied) = &self.tied_weights else {
            return;
        };

        if self.weights.is_empty() || self.weights[0].is_empty() {
            return;
        }

        let in_dim = self.weights.len();
        let vocab = self.weights[0].len();
        let mut table = tied.write();

        if table.len() != vocab || table.first().map_or(true, |r| r.len() != in_dim) {
            *table = vec![vec![0.0; in_dim]; vocab];
        }

        for i in 0..in_dim {
            if self.weights[i].len() != vocab {
                return;
            }
            for v in 0..vocab {
                table[v][i] = self.weights[i][v].re as f32;
            }
        }
    }

    pub fn new(learning_rate: f64, rows: usize, cols: usize, is_complex: bool) -> Self {
        let mut weights: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let bias: Vec<C> = vec![C::new(ONE, ZERO); cols];

        if is_complex {
            initialize_weights_complex(rows, cols, &mut weights);
        } else {
            initialize_weights_complex_only_real(rows, cols, &mut weights);
        }

        Self {
            weights,
            bias,
            learning_rate,
            gradients: vec![],
            norm_layer: None,
            gradients_bias: vec![],
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            is_complex,
            tied_weights: None,
            tied_embedding_grad_by_token: None,
            output_indices: None,
        }
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        if self.tied_weights.is_some() {
            self.sync_from_tied_weights();
        }

        let input_batch: Vec<Vec<Vec<C>>> = input.get_input_batch();
        if input_batch.is_empty() && input.get_input_batch_rm_ref().is_some_and(|rm| !rm.is_empty()) {
            panic!("LinearLayer received RM input; use LinearLayerRm");
        }

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        self.input_batch = (!input_batch.is_empty()).then_some(input_batch.clone());

        let output_indices: Vec<Vec<Vec<usize>>> = vec![];

        let output_batch: Vec<Vec<Vec<C>>> = input_batch
            .par_iter()
            .map(|input_rows| {
                let mut out = multiply_complex(input_rows, &self.weights);
                add_vector(&mut out, &self.bias);
                out
            })
            .collect();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output.set_output_indices(output_indices);
        self.output_indices = Some(layer_output.get_output_indices());
        return layer_output;
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_vec = self.input_batch.as_ref().expect("No input batch found for linear layer backward pass");
        let mut gradient = Gradient::new_default();

        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let previous_gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient.get_gradient_input_batch();

        // Initialize gradients for weights and biases
        let batch_len = input_batch_vec.len();

        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];

        let input_batch = input_batch_vec;
        let weights_h = conjugate_transpose(&self.weights);

        let mut gradient_input_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        for batch_idx in 0..batch_len {
            let input_sample = &input_batch[batch_idx];
            let grad = &previous_gradient_input_batch[batch_idx];

            weight_gradients[batch_idx] = multiply_complex(&conjugate_transpose(input_sample), grad);

            for grad_row in grad.iter() {
                for (k, grad_val) in grad_row.iter().enumerate() {
                    bias_gradients[batch_idx][k] += grad_val;
                }
            }

            gradient_input_batch[batch_idx] = multiply_complex(grad, &weights_h);
        }

        gradient.set_gradient_input_batch(gradient_input_batch);

        gradient.set_gradient_input_batch_rm(vec![]);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
        let total_valid_tokens = gradient.get_total_valid_tokens();

        weight_gradients = average_matrix_by_scalar(&weight_gradients, r(total_valid_tokens as f64));
        bias_gradients = average_vector_by_scalar(&bias_gradients, r(total_valid_tokens as f64));

        clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        normalize_gradients(&mut weight_gradients);
        normalize_bias(&mut bias_gradients);

        // Merge embedding-side token gradients when tied.
        // Accumulator layout: token(vocab) -> [embedding_dim], while linear weights are [embedding_dim][vocab].
        if let Some(acc) = &self.tied_embedding_grad_by_token {
            let mut acc_lock = acc.write().expect("tied grad accumulator poisoned");
            let in_dim = weight_gradients.len();
            let vocab = weight_gradients.first().map(|r| r.len()).unwrap_or(0);

            for (token_idx, grad_vec) in acc_lock.drain() {
                if token_idx >= vocab {
                    continue;
                }
                let n = in_dim.min(grad_vec.len());
                for i in 0..n {
                    weight_gradients[i][token_idx] += grad_vec[i];
                }
            }
        }

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights, mut prev_v_weights_hat, mut prev_v_bias_hat) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_bias(),
                previous_gradient.get_prev_v_bias(),
                previous_gradient.get_prev_m_weights(),
                previous_gradient.get_prev_v_weights(),
                previous_gradient.get_prev_v_weights_hat(),
                previous_gradient.get_prev_v_bias_hat(),
            )
        } else {
            // Initialize to zeros on first step
            (
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
            )
        };

        calculate_adam_w_bias(&mut self.bias, &bias_gradients, &mut prev_m_bias, &mut prev_v_bias, &mut prev_v_bias_hat, learning_rate, time_step);
        calculate_adam_w(
            &mut self.weights,
            &weight_gradients,
            &mut prev_m_weights,
            &mut prev_v_weights,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
        );

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);
        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;

        if self.tied_weights.is_some() {
            self.write_back_to_tied_weights();
        }
    }
}
