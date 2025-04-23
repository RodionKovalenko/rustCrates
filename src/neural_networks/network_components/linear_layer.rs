use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::{calculate_adam_w, calculate_adam_w_bias},
    matrix::{add_vector, check_nan_or_inf, clip_gradient_1d, clip_gradients, is_nan_or_inf, multiply_complex, transpose},
    weights_initializer::initialize_weights_complex,
};

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub bias: Vec<Complex<f64>>,
    #[serde(skip)]
    pub gradients: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub gradients_bias: Vec<Vec<Complex<f64>>>,
    pub learning_rate: f64,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub previous_gradient: Option<Gradient>,
    pub time_step: usize,
}

impl LinearLayer {
    pub fn new(learning_rate: f64, rows: usize, cols: usize) -> Self {
        let mut weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

        initialize_weights_complex(rows, cols, &mut weights);

        Self {
            weights,
            bias,
            learning_rate,
            gradients: vec![],
            gradients_bias: vec![],
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
        }
    }
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = input.get_input_batch();
        self.input_batch = Some(input_batch.clone());
        self.time_step = input.get_time_step();

        let output_batch = input_batch
            .par_iter() // Use a parallel iterator to process inputs in parallel
            .map(|input| {
                let mut output = multiply_complex(input, &self.weights);

                // Add the bias vector
                output = add_vector(&output, &self.bias);

                output // Return the processed output for this input
            })
            .collect();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);
        layer_output.set_input_gradient_batch(self.calculate_input_gradient_batch());

        layer_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let mut gradient_input_batch = previous_gradient_batch.clone();
        let mut gradient = Gradient::new_default();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.bias.len()]; input_batch.len()];

        // For each input sample in the batch
        for (batch_ind, (input_sample, previous_gradient)) in input_batch.iter().zip(previous_gradient_batch).enumerate() {
            // Multiply the transposed input sample with previous gradients (for weight gradients)
            weight_gradients[batch_ind] = multiply_complex(&transpose(input_sample), previous_gradient);

            // println!("\nprevious gradient: {:?}", &previous_gradient);
            // println!("\n weights linear: {:?}", &self.weights);

            //Accumulate gradients for biases
            for grad_row in previous_gradient.iter() {
                for (k, grad_val) in grad_row.iter().enumerate() {
                    bias_gradients[batch_ind][k] += grad_val.clone(); // Sum the gradients for biases
                }
            }

            gradient_input_batch[batch_ind] = multiply_complex(&previous_gradient, &transpose(&self.weights));
        }

        // let max = gradient_input_batch.iter().flat_map(|v| v.iter().flat_map(|w| w.iter())).max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Less));
        // let min = gradient_input_batch.iter().flat_map(|v| v.iter().flat_map(|w| w.iter())).min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Greater));

        // println!("max in backward linear gradient batch: {:?}", max);
        // println!("min in backward linear gradient batch: {:?}", min);

        gradient.set_gradient_input_batch(gradient_input_batch.clone());
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn calculate_input_gradient_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in dense layer");

        let mut input_gradient_batch = vec![vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        for batch_ind in 0..input_batch.len() {
            input_gradient_batch[batch_ind] = transpose(&self.weights);

        }
        
        input_gradient_batch
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());

        let input_batch = gradient.get_gradient_input_batch();
        let batch_size = input_batch.len() as f64;

        let threshold = 0.5;
        clip_gradients(&mut weight_gradients, threshold);
        clip_gradient_1d(&mut bias_gradients, threshold);

        check_nan_or_inf(&mut weight_gradients, "check weight gradients in linear layer");

        let mut prev_m_bias: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.bias.len()];
        let mut prev_v_bias: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.bias.len()];

        let mut prev_m_weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()];
        let mut prev_v_weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()];

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        if let Some(previous_gradient) = &mut self.previous_gradient {
            prev_m_bias = previous_gradient.get_prev_m_bias();
            prev_v_bias = previous_gradient.get_prev_v_bias();

            prev_m_weights = previous_gradient.get_prev_m_weights();
            prev_v_weights = previous_gradient.get_prev_v_weights();

            self.bias = calculate_adam_w_bias(&self.bias, &gradient.get_gradient_bias(), &mut prev_m_bias, &mut prev_v_bias, learning_rate, time_step);
            self.weights = calculate_adam_w(&self.weights, &gradient.get_gradient_weights(), &mut prev_m_weights, &mut prev_v_weights, learning_rate, time_step);
        } else {
            // Update weights and biases using gradient descent
            for (i, row) in self.weights.iter_mut().enumerate() {
                for (j, weight_value) in row.iter_mut().enumerate() {
                    if !is_nan_or_inf(&weight_gradients[i][j]) {
                        *weight_value -= self.learning_rate * (weight_gradients[i][j] / batch_size);
                    }
                }
            }

            for (i, value) in self.bias.iter_mut().enumerate() {
                if !is_nan_or_inf(&bias_gradients[i]) {
                    *value -= self.learning_rate * (bias_gradients[i] / batch_size);
                }
            }
        }

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        self.previous_gradient = Some(gradient.clone());
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
}
