use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_types::{transformer::transformer_network::EMA_SCALER, wavelet_discrete_layer::DiscreteWaveletLayer},
    utils::{
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        matrix::{add_matrix_2d_c, add_matrix_3d, add_vector, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, compute_global_norm, conjugate_transpose, multiply_complex, multiply_complex_with_f64, multiply_f64_complex, transpose},
        weights_initializer::initialize_weights_complex,
    },
};

use super::{
    gradient_struct::{Gradient, GradientBatch},
    layer_input_struct::LayerInput,
    layer_output_struct::LayerOutput,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub learning_rate: f64,
    pub bias: Vec<Complex<f64>>,
    pub smoothing: f64,
    pub ema: f64,
    pub discrete_wavelet_layer: Option<DiscreteWaveletLayer>,

    #[serde(skip)]
    pub gradients: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub gradients_bias: Vec<Vec<Complex<f64>>>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub previous_gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl LinearLayer {
    pub fn new(learning_rate: f64, rows: usize, cols: usize) -> Self {
        let mut weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let mut _dwt_layer = DiscreteWaveletLayer::new();
        _dwt_layer.is_linear_layer = true;

        initialize_weights_complex(rows, cols, &mut weights);

        Self {
            weights,
            bias,
            learning_rate,
            gradients: vec![],
            discrete_wavelet_layer: None,
            gradients_bias: vec![],
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
        }
    }
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let mut input_batch: Vec<Vec<Vec<Complex<f64>>>> = input.get_input_batch();

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        // println!("Input batch size in linear layer: {} {} {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());

        if self.discrete_wavelet_layer.is_some() {
            if let Some(dwt_layer) = self.discrete_wavelet_layer.as_mut() {
                let dwt_output = dwt_layer.forward(&input);
                input_batch = dwt_output.get_output_batch();
            }
        }

        self.input_batch = Some(input_batch.clone());

        // println!("Input batch size in linear layer after dwt: {} {} {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());

        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = input_batch
            .par_iter() // Use a parallel iterator to process inputs in parallel
            .map(|input| {
                let mut output = multiply_complex(input, &self.weights);

                // Add the bias vector
                output = add_vector(&output, &self.bias);
                output
            })
            .collect();

        // println!("Output batch size in linear layer after forward: {} {} {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());

        // Decompress Wavelet if the layer is present
        if self.discrete_wavelet_layer.is_some() {
            if let Some(dwt_layer) = self.discrete_wavelet_layer.as_mut() {
                let mut dwt_layer_input = LayerInput::new_default();
                dwt_layer_input.set_input_batch(output_batch.clone());
                let wavelet_inverse_output = dwt_layer.forward_inverse(&dwt_layer_input);
                output_batch = wavelet_inverse_output.get_output_batch();
            }
        }

        // println!("Output batch size in linear layer after dwt inverse:  {} {} {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let mut gradient = Gradient::new_default();

        let previous_gradient_batch = if !previous_gradient.get_gradient_input_batch().is_empty() {
            GradientBatch::Complex(previous_gradient.get_gradient_input_batch())
        } else {
            GradientBatch::Real(previous_gradient.get_gradient_input_batch_softmax())
        };

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.bias.len()]; input_batch.len()];
        let mut gradient_input_batch = vec![vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; input_batch[0].len()]; input_batch.len()];

        match previous_gradient_batch {
            GradientBatch::Complex(previous_gradient_input_batch) => {
                let mut previous_gradient_input_batch_clone = previous_gradient_input_batch.clone();
                // Apply DWT if the layer is present
                if self.discrete_wavelet_layer.is_some() {
                    if let Some(dwt_layer) = self.discrete_wavelet_layer.as_mut() {
                        gradient.set_gradient_input_batch(previous_gradient_input_batch.clone());

                        let gradient_inverse: Gradient = dwt_layer.backward_inverse(&gradient);
                        previous_gradient_input_batch_clone = gradient_inverse.get_gradient_input_batch();
                    }
                }

                for (batch_ind, (input_sample, previous_gradient)) in input_batch.iter().zip(previous_gradient_input_batch_clone).enumerate() {
                    weight_gradients[batch_ind] = multiply_complex(&conjugate_transpose(&input_sample), &previous_gradient);
                    //Accumulate gradients for biases
                    for grad_row in previous_gradient.iter() {
                        for (k, grad_val) in grad_row.iter().enumerate() {
                            bias_gradients[batch_ind][k] += grad_val;
                        }
                    }

                    gradient_input_batch[batch_ind] = multiply_complex(&previous_gradient, &conjugate_transpose(&self.weights));
                }

                if self.discrete_wavelet_layer.is_some() {
                    if let Some(dwt_layer) = self.discrete_wavelet_layer.as_mut() {
                        gradient.set_gradient_input_batch(gradient_input_batch.clone());

                        let dwt_gradient = dwt_layer.backward(&gradient);
                        gradient_input_batch = dwt_gradient.get_gradient_input_batch();
                    }
                }
            }
            GradientBatch::Real(previous_gradient_input_batch) => {
                // For each input sample in the batch
                for (batch_ind, (input_sample, previous_gradient)) in input_batch.iter().zip(previous_gradient_input_batch).enumerate() {
                    weight_gradients[batch_ind] = multiply_complex_with_f64(&transpose(input_sample), &previous_gradient);
                    //Accumulate gradients for biases
                    for grad_row in previous_gradient.iter() {
                        for (k, grad_val) in grad_row.iter().enumerate() {
                            bias_gradients[batch_ind][k] += grad_val;
                        }
                    }

                    gradient_input_batch[batch_ind] = multiply_f64_complex(&previous_gradient, &transpose(&self.weights));
                }
            }
        }

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            gradient_input_batch = add_matrix_3d(&gradient_input_batch, &previous_gradient.get_gradient_input_batch());
            weight_gradients = add_matrix_3d(&weight_gradients, &previous_gradient.get_gradient_weight_batch());
            bias_gradients = add_matrix_2d_c(&bias_gradients, &previous_gradient.get_gradient_bias_batch());
        }
        //  println!("batch size in linear layer: {}", self.batch_size);

        gradient.set_gradient_input_batch(gradient_input_batch.clone());
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
        let input_batch = gradient.get_gradient_input_batch();
        let mut batch_size = input_batch.len() as f64;
        let mut all_gradients = vec![weight_gradients];
        let global_norm = compute_global_norm(&all_gradients, &bias_gradients);
        self.ema = self.smoothing * self.ema + (1.0 - self.smoothing) * global_norm;
        let max_norm = self.ema * EMA_SCALER;
        clip_all_gradients_by_global_norm_2d(&mut all_gradients, &mut bias_gradients, global_norm, max_norm);
        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }
        let mut weight_gradients: Vec<Vec<Complex<f64>>> = all_gradients[0].clone();
        weight_gradients = average_matrix_by_scalar(&weight_gradients, batch_size);
        bias_gradients = average_vector_by_scalar(&bias_gradients, batch_size);
        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());
        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (previous_gradient.get_prev_m_bias(), previous_gradient.get_prev_v_bias(), previous_gradient.get_prev_m_weights(), previous_gradient.get_prev_v_weights())
        } else {
            // Initialize to zeros on first step
            (
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()],
                vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()],
            )
        };
        // prev_m_bias = average_gradient_polar_1d(&previous_gradient.get_prev_m_bias(), batch_size);
        // prev_v_bias = average_gradient_polar_1d(&previous_gradient.get_prev_v_bias(), batch_size);
        // prev_m_weights = average_gradient_polar(&previous_gradient.get_prev_m_weights(), batch_size);
        // prev_v_weights = average_gradient_polar(&previous_gradient.get_prev_v_weights(), batch_size);
        calculate_adam_w_bias(&mut self.bias, &gradient.get_gradient_bias(), &mut prev_m_bias, &mut prev_v_bias, learning_rate, time_step);

        calculate_adam_w(&mut self.weights, &gradient.get_gradient_weights(), &mut prev_m_weights, &mut prev_v_weights, learning_rate, time_step);
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
