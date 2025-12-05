use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::{calculate_adam_w, calculate_adam_w_bias},
    matrix::{add_matrix_2d_c, add_matrix_3d, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d},
    weights_initializer::initialize_weights_complex,
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexToLinearLayer {
    pub weights_1: Vec<Vec<Complex<f64>>>,
    pub weights_2: Vec<Vec<Complex<f64>>>,
    pub learning_rate: f64,
    pub bias: Vec<Complex<f64>>,
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
    pub fn new(learning_rate: f64, rows: usize, cols: usize) -> Self {
        let mut weights_1: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_2: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

        initialize_weights_complex(rows, cols, &mut weights_1);
        initialize_weights_complex(rows, cols, &mut weights_2);

        Self {
            weights_1,
            weights_2,
            bias,
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
                input
                    .iter() // over time steps
                    .map(|row| {
                        let features_in = row.len();
                        let features_out = self.weights_1[0].len();

                        (0..features_out)
                            .map(|o| {
                                let mut acc = self.bias[o];

                                for i in 0..features_in {
                                    let z = row[i];
                                    acc += z.re * self.weights_1[i][o].re + z.im * self.weights_2[i][o].re;
                                }

                                // println!("ComplexToLinearLayer forward output acc {:?}", acc);
                                acc
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let mut gradient = Gradient::new_default();

        let previous_input_gradient: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()]; input_batch.len()];
        let mut weight_gradients_2: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.bias.len()]; input_batch.len()];
        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        for (b, input) in input_batch.iter().enumerate() {
            let prev_gradient = &previous_input_gradient[b];

            for s in 0..input.len() {
                for f in 0..self.weights_1[0].len() {
                    // Bias gradient
                    bias_gradients[b][f] += prev_gradient[s][f];

                    for i in 0..self.weights_1.len() {
                        // Weight gradients
                        weight_gradients[b][i][f] += prev_gradient[s][f] * input[s][i].re;
                        weight_gradients_2[b][i][f] += prev_gradient[s][f] * input[s][i].im;

                        // Input gradient
                        gradient_input_batch[b][s][i] += Complex::new((prev_gradient[s][f] * self.weights_1[i][f].re).re, (prev_gradient[s][f] * self.weights_2[i][f].re).re);
                    }
                }
            }
        }

        gradient.set_gradient_input_batch(gradient_input_batch.clone());

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            // gradient_input_batch = add_matrix_3d(&gradient_input_batch, &previous_gradient.get_gradient_input_batch());
            weight_gradients = add_matrix_3d(&weight_gradients, &previous_gradient.get_gradient_weight_batch());
            weight_gradients_2 = add_matrix_3d(&weight_gradients_2, &previous_gradient.get_gradient_weight_2_batch());
            bias_gradients = add_matrix_2d_c(&bias_gradients, &previous_gradient.get_gradient_bias_batch());
        }
        //  println!("batch size in linear layer: {}", self.batch_size);

        gradient.set_gradient_input_batch(gradient_input_batch.clone());
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_weight_2_batch(weight_gradients_2);
        gradient.set_gradient_bias_batch(bias_gradients);

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
        let mut weight_gradients_2 = gradient.get_gradient_weights_2();
        let input_batch = gradient.get_gradient_input_batch();
        let mut batch_size = input_batch.len() as f64;

        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }

        weight_gradients = average_matrix_by_scalar(&weight_gradients, batch_size);
        weight_gradients_2 = average_matrix_by_scalar(&weight_gradients_2, batch_size);
        bias_gradients = average_vector_by_scalar(&bias_gradients, batch_size);

        clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);
        clip_all_gradients_by_global_norm_2d(&mut weight_gradients_2, &mut bias_gradients, self.global_norm, self.max_norm);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights, mut prev_m_weights_2, mut prev_v_weights_2, mut prev_v_weights_hat, mut prev_v_bias_hat) =
            if let Some(previous_gradient) = &mut self.previous_gradient {
                (
                    previous_gradient.get_prev_m_bias(),
                    previous_gradient.get_prev_v_bias(),
                    previous_gradient.get_prev_m_weights(),
                    previous_gradient.get_prev_v_weights(),
                    previous_gradient.get_prev_m_weights_2(),
                    previous_gradient.get_prev_v_weights_2(),
                    previous_gradient.get_prev_v_weights_hat(),
                    previous_gradient.get_prev_v_bias_hat(),
                )
            } else {
                // Initialize to zeros on first step
                (
                    vec![Complex::new(0.0, 0.0); self.bias.len()],
                    vec![Complex::new(0.0, 0.0); self.bias.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![vec![Complex::new(0.0, 0.0); self.weights_1[0].len()]; self.weights_1.len()],
                    vec![Complex::new(0.0, 0.0); self.bias.len()],
                )
            };
        calculate_adam_w_bias(
            &mut self.bias,
            &gradient.get_gradient_bias(),
            &mut prev_m_bias,
            &mut prev_v_bias,
            &mut prev_v_bias_hat,
            learning_rate,
            time_step,
        );
        calculate_adam_w(
            &mut self.weights_1,
            &gradient.get_gradient_weights(),
            &mut prev_m_weights,
            &mut prev_v_weights,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
        );

        calculate_adam_w(
            &mut self.weights_2,
            &weight_gradients_2,
            &mut prev_m_weights_2,
            &mut prev_v_weights_2,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
        );

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        gradient.set_prev_m_weights_2(prev_m_weights_2);
        gradient.set_prev_v_weights_2(prev_v_weights_2);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);
        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());
        self.previous_gradient = Some(gradient.clone());
    }
}
