use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    utils::{
        adam_w::calculate_adam_w,
        dtype::{r, Real, C, ZERO},
        matrix::average_matrix_by_scalar,
        weights_initializer::initialize_weights_complex_only_real,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexToLinearLayer {
    pub weights_1: Vec<Vec<C>>,
    pub weights_2: Vec<Vec<C>>,
    pub learning_rate: Real,
    pub smoothing: Real,
    pub ema: Real,
    pub global_norm: Real,
    pub max_norm: Real,
    pub previous_gradient: Option<Gradient>,

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
}

impl ComplexToLinearLayer {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut weights_1: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let mut weights_2: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];

        initialize_weights_complex_only_real(rows, cols, &mut weights_1);
        initialize_weights_complex_only_real(rows, cols, &mut weights_2);

        Self {
            weights_1,
            weights_2,
            learning_rate: r(learning_rate),
            gradients: vec![],
            gradients_bias: vec![],
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            smoothing: r(0.99),
            ema: ZERO,
            global_norm: ZERO,
            max_norm: ZERO,
        }
    }
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let has_vec_input = input.has_non_empty_input_batch();
        let has_rm_input = input.has_non_empty_input_batch_rm();

        // LayerInput can carry an RM cache even for Vec inputs. Only treat it as an error when
        // Vec input is absent but RM input is present.
        if !has_vec_input && has_rm_input {
            panic!("ComplexToLinearLayer (Vec) received RM-only input; use ComplexToLinearLayerRm");
        }

        if !has_vec_input {
            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch(vec![]);
            layer_output.set_output_batch_rm(vec![]);
            self.input_batch = None;
            return layer_output;
        }

        let input_batch: Vec<Vec<Vec<C>>> = input
            .get_input_batch_ref()
            .expect("ComplexToLinearLayer (Vec): expected Vec input")
            .to_vec();
        self.input_batch = Some(input_batch.clone());

        let in_f = self.weights_1.len();
        let out_f = self.weights_1[0].len();

        let output_batch: Vec<Vec<Vec<C>>> = input_batch
            .par_iter()
            .map(|input_rows| {
                input_rows
                    .iter()
                    .map(|input_row| {
                        let mut out_row = vec![C::new(ZERO, ZERO); out_f];
                        for f in 0..out_f {
                            let mut sum_real: Real = ZERO;
                            for k in 0..in_f {
                                let x = input_row[k];
                                sum_real += x.re * self.weights_1[k][f].re + x.im * self.weights_2[k][f].re;
                            }
                            out_row[f] = C::new(sum_real, ZERO);
                        }
                        out_row
                    })
                    .collect()
            })
            .collect();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output.set_output_batch_rm(vec![]);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let input_batch_vec = self.input_batch.as_ref().filter(|b| !b.is_empty());
        if input_batch_vec.is_none() {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_weight_2_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let has_prev_vec_grad = previous_gradient
            .get_gradient_input_batch_ref()
            .is_some_and(|b| !b.is_empty());
        let has_prev_rm_grad = previous_gradient
            .get_gradient_input_batch_rm_ref()
            .is_some_and(|b| !b.is_empty());

        // Gradients may also carry RM caches; only error when Vec gradients are absent.
        if !has_prev_vec_grad && has_prev_rm_grad {
            panic!("ComplexToLinearLayer (Vec) received RM-only gradient; use ComplexToLinearLayerRm");
        }

        let previous_gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient
            .get_gradient_input_batch_ref()
            .expect("ComplexToLinearLayer (Vec): expected Vec previous gradient")
            .to_vec();

        let batch_len = input_batch_vec.expect("vec batch").len();

        let in_f = self.weights_1.len();
        let out_f = self.weights_1[0].len();

        let mut grad_w1 = vec![vec![vec![C::new(ZERO, ZERO); out_f]; in_f]; batch_len];
        let mut grad_w2 = vec![vec![vec![C::new(ZERO, ZERO); out_f]; in_f]; batch_len];
        let mut gradient_input_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); in_f]; 0]; 0];

        if batch_len > 0 {
            let time = input_batch_vec.expect("vec batch")[0].len();
            gradient_input_batch = vec![vec![vec![C::new(ZERO, ZERO); in_f]; time]; batch_len];
        }

        if batch_len == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_weight_2_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        if previous_gradient_input_batch.is_empty() {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_weight_2_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        for batch_ind in 0..batch_len {
            let input_sample = &input_batch_vec.expect("vec batch")[batch_ind];
            let grad_sample = &previous_gradient_input_batch[batch_ind];

            assert_eq!(input_sample[0].len(), in_f);
            assert_eq!(grad_sample[0].len(), out_f);
            assert_eq!(input_sample.len(), grad_sample.len());

            let time = input_sample.len();
            for t in 0..time {
                let input_row = &input_sample[t];
                let g_row = &grad_sample[t];
                for f in 0..out_f {
                    let g = g_row[f].re;
                    for k in 0..in_f {
                        let x = input_row[k];

                        // input gradients
                        gradient_input_batch[batch_ind][t][k].re += g * self.weights_1[k][f].re;
                        gradient_input_batch[batch_ind][t][k].im += g * self.weights_2[k][f].re;

                        // weight gradients
                        grad_w1[batch_ind][k][f].re += x.re * g;
                        grad_w2[batch_ind][k][f].re += x.im * g;
                    }
                }
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(gradient_input_batch);
        gradient.set_gradient_input_batch_rm(vec![]);
        gradient.set_gradient_weight_batch(grad_w1);
        gradient.set_gradient_weight_2_batch(grad_w2);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");

        let mut weight_gradients_1: Vec<Vec<C>> = gradient.get_gradient_weights();
        let mut weight_gradients_2: Vec<Vec<C>> = gradient.get_gradient_weights_2();

        // Get total valid tokens from gradient
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in complex_to_linear_layer");
        let total_valid_tokens: Real = r(gradient.get_total_valid_tokens().max(1) as f64);

        weight_gradients_1 = average_matrix_by_scalar(&weight_gradients_1, total_valid_tokens);
        weight_gradients_2 = average_matrix_by_scalar(&weight_gradients_2, total_valid_tokens);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let (mut prev_m_weights_1, mut prev_v_weights_1, mut prev_v_weights_hat_1, mut prev_m_weights_2, mut prev_v_weights_2, mut prev_v_weights_hat_2) =
            if let Some(previous_gradient) = &mut self.previous_gradient {
                (
                    previous_gradient.get_prev_m_weights(),
                    previous_gradient.get_prev_v_weights(),
                    previous_gradient.get_prev_v_weights_hat(),
                    previous_gradient.get_prev_m_weights_2(),
                    previous_gradient.get_prev_v_weights_2(),
                    previous_gradient.get_prev_v_weights_hat_2(),
                )
            } else {
                // Initialize to zeros on first step
                (
                    vec![vec![C::new(ZERO, ZERO); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![C::new(ZERO, ZERO); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![C::new(ZERO, ZERO); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![C::new(ZERO, ZERO); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![C::new(ZERO, ZERO); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![C::new(ZERO, ZERO); self.weights_1[0].len()]; self.weights_1.len()],
                )
            };
        calculate_adam_w(
            &mut self.weights_1,
            &weight_gradients_1,
            &mut prev_m_weights_1,
            &mut prev_v_weights_1,
            &mut prev_v_weights_hat_1,
            learning_rate as f64,
            time_step,
        );
        calculate_adam_w(
            &mut self.weights_2,
            &weight_gradients_2,
            &mut prev_m_weights_2,
            &mut prev_v_weights_2,
            &mut prev_v_weights_hat_2,
            learning_rate as f64,
            time_step,
        );

        gradient.set_prev_m_weights(prev_m_weights_1);
        gradient.set_prev_v_weights(prev_v_weights_1);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat_1);

        gradient.set_prev_m_weights_2(prev_m_weights_2);
        gradient.set_prev_v_weights_2(prev_v_weights_2);
        gradient.set_prev_v_weights_hat_2(prev_v_weights_hat_2);

        gradient.set_gradient_weights(weight_gradients_1.clone());
        gradient.set_gradient_weights_2(weight_gradients_2.clone());

        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}
