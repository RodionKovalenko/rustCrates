use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::calculate_adam_w_bias,
    matrix::{add_matrix_2d_c, average_vector_by_scalar},
    weights_initializer::initialize_bias,
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexToLinearLayer {
    pub weights_1: Vec<Complex<f64>>,
    pub weights_2: Vec<Complex<f64>>,
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
    pub fn new(cols: usize, learning_rate: f64) -> Self {
        let mut weights_1: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); cols];
        let mut weights_2: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); cols];
        let mut bias: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); cols];

        initialize_bias(cols, &mut weights_1);
        initialize_bias(cols, &mut weights_2);
        initialize_bias(cols, &mut bias);

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
                        let mut output_row = Vec::new();
                        for i in 0..row.len() {
                            let z = row[i];
                            let linear_v = z.re * self.weights_1[i].re + z.im * self.weights_2[i].re + self.bias[i].re;
                            output_row.push(Complex::new(linear_v, 0.0));
                        }

                        // println!("ComplexToLinearLayer forward output acc {:?}", acc);
                        output_row
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
        let previous_input_gradient = previous_gradient.get_gradient_input_batch();

        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::with_capacity(input_batch.len());
        let mut weight_gradients_1: Vec<Vec<Complex<f64>>> = Vec::with_capacity(input_batch.len());
        let mut weight_gradients_2: Vec<Vec<Complex<f64>>> = Vec::with_capacity(input_batch.len());
        let mut bias_gradients: Vec<Vec<Complex<f64>>> = Vec::with_capacity(input_batch.len());

        for (b, input) in input_batch.iter().enumerate() {
            let prev_gradient = &previous_input_gradient[b];

            // Per-batch storage
            let mut grad_input_batch_b: Vec<Vec<Complex<f64>>> = Vec::with_capacity(input.len());
            let mut weight_grad_1 = vec![Complex::new(0.0, 0.0); self.weights_1.len()];
            let mut weight_grad_2 = vec![Complex::new(0.0, 0.0); self.weights_2.len()];
            let mut bias_grad_b = vec![Complex::new(0.0, 0.0); self.bias.len()];

            for s in 0..input.len() {
                let row_len = input[s].len();
                let mut grad_input_row = vec![Complex::new(0.0, 0.0); row_len];

                for f in 0..input[s].len() {
                    // Bias gradient
                    bias_grad_b[f] += prev_gradient[s][f];

                    // Weight gradients
                    weight_grad_1[f] += prev_gradient[s][f] * input[s][f].re;
                    weight_grad_2[f] += prev_gradient[s][f] * input[s][f].im;

                    // Input gradient
                    grad_input_row[f] += Complex::new((prev_gradient[s][f] * self.weights_1[f].re).re, (prev_gradient[s][f] * self.weights_2[f].re).re);
                }

                grad_input_batch_b.push(grad_input_row);
            }

            gradient_input_batch.push(grad_input_batch_b);
            weight_gradients_1.push(weight_grad_1);
            weight_gradients_2.push(weight_grad_2);
            bias_gradients.push(bias_grad_b);
        }

        // Combine with previous stored gradients if needed
        if let Some(prev_grad) = &self.gradient {
            weight_gradients_1 = add_matrix_2d_c(&weight_gradients_1, &prev_grad.get_gradient_weights_vec_batch_1());
            weight_gradients_2 = add_matrix_2d_c(&weight_gradients_2, &prev_grad.get_gradient_weights_vec_batch_2());
            bias_gradients = add_matrix_2d_c(&bias_gradients, &prev_grad.get_gradient_bias_batch());
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(gradient_input_batch.clone());
        gradient.set_gradient_weights_vec_batch_1(weight_gradients_1);
        gradient.set_gradient_weights_vec_batch_2(weight_gradients_2);
        gradient.set_gradient_bias_batch(bias_gradients);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");

        let mut weight_gradeints_vec_1: Vec<Complex<f64>> = gradient.get_gradient_weights_vec_1();
        let mut weight_gradients_vec_2: Vec<Complex<f64>> = gradient.get_gradient_weights_vec_2();
        let mut bias_gradients: Vec<Complex<f64>> = gradient.get_gradient_bias();

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = gradient.get_gradient_input_batch();
        let mut batch_size = input_batch.len() as f64;

        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }

        weight_gradeints_vec_1 = average_vector_by_scalar(&weight_gradeints_vec_1, batch_size);
        weight_gradients_vec_2 = average_vector_by_scalar(&weight_gradients_vec_2, batch_size);
        bias_gradients = average_vector_by_scalar(&bias_gradients, batch_size);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let (
            mut prev_m_bias,
            mut prev_v_bias,
            mut prev_v_bias_hat,
            mut prev_m_weights_vec_1,
            mut prev_v_weights_vec_1,
            mut prev_v_weights_vec_hat_1,
            mut prev_m_weights_vec_2,
            mut prev_v_weights_vec_2,
            mut prev_v_weights_vec_hat_2,
        ) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_bias(),
                previous_gradient.get_prev_v_bias(),
                previous_gradient.get_prev_v_bias_hat(),
                previous_gradient.get_prev_m_weights_vec_1(),
                previous_gradient.get_prev_v_weights_vec_1(),
                previous_gradient.get_prev_v_weights_vec_hat_1(),
                previous_gradient.get_prev_m_weights_vec_2(),
                previous_gradient.get_prev_v_weights_vec_2(),
                previous_gradient.get_prev_v_weights_vec_hat_2(),
            )
        } else {
            // Initialize to zeros on first step
            (
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![Complex::new(0.0, 0.0); self.weights_1.len()],
                vec![Complex::new(0.0, 0.0); self.weights_1.len()],
                vec![Complex::new(0.0, 0.0); self.weights_1.len()],
                vec![Complex::new(0.0, 0.0); self.weights_1.len()],
                vec![Complex::new(0.0, 0.0); self.weights_1.len()],
                vec![Complex::new(0.0, 0.0); self.weights_1.len()],
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
        calculate_adam_w_bias(
            &mut self.weights_1,
            &gradient.get_gradient_weights_vec_1(),
            &mut prev_m_weights_vec_1,
            &mut prev_v_weights_vec_1,
            &mut prev_v_weights_vec_hat_1,
            learning_rate,
            time_step,
        );
        calculate_adam_w_bias(
            &mut self.weights_2,
            &gradient.get_gradient_weights_vec_2(),
            &mut prev_m_weights_vec_2,
            &mut prev_v_weights_vec_2,
            &mut prev_v_weights_vec_hat_2,
            learning_rate,
            time_step,
        );

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);

        gradient.set_prev_m_weights_vec_1(prev_m_weights_vec_1);
        gradient.set_prev_v_weights_vec_1(prev_v_weights_vec_1);
        gradient.set_prev_v_weights_vec_hat_1(prev_v_weights_vec_hat_1);

        gradient.set_prev_m_weights_vec_2(prev_m_weights_vec_2);
        gradient.set_prev_v_weights_vec_2(prev_v_weights_vec_2);
        gradient.set_prev_v_weights_vec_hat_2(prev_v_weights_vec_hat_2);

        gradient.set_gradient_weights_vec_1(weight_gradeints_vec_1.clone());
        gradient.set_gradient_weights_vec_2(weight_gradients_vec_2.clone());
        gradient.set_gradient_bias(bias_gradients.clone());

        self.previous_gradient = Some(gradient.clone());
    }
}
