use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::calculate_adam_w,
    matrix::{add_matrix_3d, average_matrix_by_scalar, normalize_gradients},
    weights_initializer::initialize_weights_complex_only_real,
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexToLinearLayer {
    pub weights_1: Vec<Vec<Complex<f64>>>,
    pub weights_2: Vec<Vec<Complex<f64>>>,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub gradients: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub gradients_bias: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl ComplexToLinearLayer {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut weights_1: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_2: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

        initialize_weights_complex_only_real(rows, cols, &mut weights_1);
        initialize_weights_complex_only_real(rows, cols, &mut weights_2);

        Self {
            weights_1,
            weights_2,
            learning_rate,
            gradients: vec![],
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
        }
    }
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = input.get_input_batch(); // batch × time × features_in
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());

        let output_batch: Vec<Vec<Vec<Complex<f64>>>> = input_batch
            .par_iter() // over batch
            .map(|input| {
                // perform matrix multiplication for each time step
                let mut output = vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; input.len()];

                println!("input len: {} {}", input.len(), input[0].len());
                println!("weights_1 len: {} {}", self.weights_1.len(), self.weights_1[0].len());

                for t in 0..input.len() {
                    for f in 0..self.weights_1.len() {
                        let mut sum_real = Complex::new(0.0, 0.0);
                        for k in 0..self.weights_1.len() {
                            sum_real += input[t][k].re * self.weights_1[k][f].re + input[t][k].im * self.weights_2[k][f].re;
                        }
                        output[t][f] = Complex::new(sum_real.re, 0.0);
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
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let previous_input_gradient = previous_gradient.get_gradient_input_batch();

        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_1.len()]; input_batch[0].len()]; input_batch.len()];
        let mut weight_gradients_1: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()]; input_batch.len()];
        let mut weight_gradients_2: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_2[0].len()]; self.weights_2.len()]; input_batch.len()];

        for (b, input) in input_batch.iter().enumerate() {
            let prev_gradient = &previous_input_gradient[b];

            for t in 0..input.len() {
                for f in 0..self.weights_1[0].len() {
                    for k in 0..self.weights_1.len() {
                        gradient_input_batch[b][t][k].re += prev_gradient[t][f].re * self.weights_1[k][f].re;
                        gradient_input_batch[b][t][k].im += prev_gradient[t][f].re * self.weights_2[k][f].re;

                        weight_gradients_1[b][k][f].re += input[t][k].re * prev_gradient[t][f].re;
                        weight_gradients_2[b][k][f].re += input[t][k].im * prev_gradient[t][f].re;
                    }
                }
            }
        }

        // Combine with previous stored gradients if needed
        if let Some(prev_grad) = &self.gradient {
            weight_gradients_1 = add_matrix_3d(&weight_gradients_1, &prev_grad.get_gradient_weight_batch());
            weight_gradients_2 = add_matrix_3d(&weight_gradients_2, &prev_grad.get_gradient_weight_2_batch());
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(gradient_input_batch.clone());
        gradient.set_gradient_weight_batch(weight_gradients_1);
        gradient.set_gradient_weight_2_batch(weight_gradients_2);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");

        let mut weight_gradients_1: Vec<Vec<Complex<f64>>> = gradient.get_gradient_weights();
        let mut weight_gradients_2: Vec<Vec<Complex<f64>>> = gradient.get_gradient_weights_2();

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = gradient.get_gradient_input_batch();
        let mut batch_size = input_batch.len() as f64;

        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }

        // Get sequence length from the first batch item
        let mut seq_len = 1.0;

        for input in input_batch.iter() {
            seq_len += input.len() as f64;
        }

        // Normalize by batch_size * seq_len to account for accumulation over both dimensions
        let normalization_factor = batch_size * seq_len;

        weight_gradients_1 = average_matrix_by_scalar(&weight_gradients_1, normalization_factor);
        weight_gradients_2 = average_matrix_by_scalar(&weight_gradients_2, normalization_factor);

        // normalize gradients
        normalize_gradients(&mut weight_gradients_1);
        normalize_gradients(&mut weight_gradients_2);

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
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                )
            };
        calculate_adam_w(
            &mut self.weights_1,
            &weight_gradients_1,
            &mut prev_m_weights_1,
            &mut prev_v_weights_1,
            &mut prev_v_weights_hat_1,
            learning_rate,
            time_step,
        );
        calculate_adam_w(
            &mut self.weights_2,
            &weight_gradients_2,
            &mut prev_m_weights_2,
            &mut prev_v_weights_2,
            &mut prev_v_weights_hat_2,
            learning_rate,
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
