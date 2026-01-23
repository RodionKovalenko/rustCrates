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
        let input_batch = input.get_input_batch(); // batch × time × features_in
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());

        let output_batch: Vec<Vec<Vec<C>>> = input_batch
            .par_iter() // over batch
            .map(|input| {
                // perform matrix multiplication for each time step
                let mut output = vec![vec![C::new(0.0, 0.0); self.weights_1[0].len()]; input.len()];
                for t in 0..input.len() {
                    for f in 0..self.weights_1[0].len() {
                        let mut sum_real = C::new(0.0, 0.0);
                        for k in 0..self.weights_1.len() {
                            sum_real += input[t][k].re * self.weights_1[k][f].re + input[t][k].im * self.weights_2[k][f].re;
                        }
                        output[t][f] = C::new(sum_real.re, 0.0);
                    }
                }
                output
            })
            .collect();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().unwrap();
        let prev_grad_batch = previous_gradient.get_gradient_input_batch();

        let batch = input_batch.len();
        let time = input_batch[0].len();
        let in_f = self.weights_1.len();
        let out_f = self.weights_1[0].len();

        let mut grad_input = vec![vec![vec![C::new(0.0, 0.0); in_f]; time]; batch];
        let mut grad_w1 = vec![vec![vec![C::new(0.0, 0.0); out_f]; in_f]; batch];
        let mut grad_w2 = vec![vec![vec![C::new(0.0, 0.0); out_f]; in_f]; batch];

        for b in 0..batch {
            for t in 0..time {
                for f in 0..out_f {
                    let g = prev_grad_batch[b][t][f].re;

                    for k in 0..in_f {
                        // input gradients
                        grad_input[b][t][k].re += g * self.weights_1[k][f].re;
                        grad_input[b][t][k].im += g * self.weights_2[k][f].re;

                        // weight gradients
                        grad_w1[b][k][f].re += input_batch[b][t][k].re * g;
                        grad_w2[b][k][f].re += input_batch[b][t][k].im * g;
                    }
                }
            }
        }

        // // Combine with previous stored gradients if needed
        // if let Some(prev_grad) = &self.gradient {
        //     grad_w1 = add_matrix_3d(&grad_w1, &prev_grad.get_gradient_weight_batch());
        //     grad_w2 = add_matrix_3d(&grad_w2, &prev_grad.get_gradient_weight_2_batch());
        // }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(grad_input.clone());
        gradient.set_gradient_weight_batch(grad_w1);
        gradient.set_gradient_weight_2_batch(grad_w2);

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
