use core::fmt::Debug;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::default_layer::LayerInterface,
    utils::{
        adam_w::calculate_adam_w_bias,
        matrix::{add_matrix, add_matrix_2d_c, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, normalize_bias},
    },
};

use crate::neural_networks::utils::dtype::{c, r, Real, C};

pub const EPSILON: f64 = 0.0000000000000000000000001;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer {
    pub gamma: Vec<C>,      // Learnable scaling parameter (for each feature)
    pub epsilon: Real,      // Small constant for numerical stability
    pub learning_rate: f64, // Learning rate for gamma updates
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl RMSNormLayer {
    // Initialize the RMSNorm layer with a given feature dimension (e.g., 16 for each token embedding)
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![c(1.0, 0.0); feature_dim], // Initialize gamma to 1.0 for all features
            epsilon: r(epsilon),
            smoothing: 0.9,
            ema: 0.0,
            learning_rate,
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    // RMSNorm function that works on a single token embedding (vector of Complex<Real>)
    pub fn rms_norm(&self, input: &Vec<C>) -> Vec<C> {
        if input.is_empty() {
            panic!("Input to RMSNorm cannot be empty");
        }
        let mut out = vec![c(0.0, 0.0); input.len()];
        rms_norm_row(input, &self.gamma, self.epsilon, &mut out);
        out
    }

    pub fn rms(&self, input: &Vec<C>) -> C {
        rms_of(input, self.epsilon)
    }

    // Forward pass for a batch of token embeddings (2D input)
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<C>>> = layer_input.get_input_batch();
        let input_before_transform_batch: Vec<Vec<Vec<C>>> = layer_input.get_input_batch_before();

        let mut output_batch: Vec<Vec<Vec<C>>> = Vec::new();
        let mut input_batch_added = input_batch.clone();

        for (batch_ind, input) in input_batch.iter().enumerate() {
            // println!("shape input in rms: {:?}, {:?}", input.len(), input[0].len());
            // println!("shape input before in rms: {:?}, {:?}", input_before_transform_batch[batch_ind].len(), input_before_transform_batch[batch_ind][0].len());
            let output = add_matrix(input, &input_before_transform_batch[batch_ind]);
            input_batch_added[batch_ind] = output.clone();

            output_batch.push(
                output
                    .iter()
                    .map(|vec| self.rms_norm(vec)) // Normalize each token embedding
                    .collect(),
            );
        }

        self.input_batch = Some(input_batch_added.clone());
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let previous_gradient_batch = previous_gradient.get_gradient_input_batch();
        let input_batch = self.input_batch.as_ref().expect("Input batch not found in RMSNorm layer");
        let mut gradient = Gradient::new_default();

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let dim_len = input_batch[0][0].len();

        let mut input_batch_gradients = vec![vec![vec![c(0.0, 0.0); dim_len]; seq_len]; batch_size];
        let mut gradient_gamma_batch = vec![vec![c(0.0, 0.0); dim_len]; batch_size];

        for b in 0..batch_size {
            for s in 0..seq_len {
                rms_norm_backward_row(&input_batch[b][s], &previous_gradient_batch[b][s], self.epsilon, &mut input_batch_gradients[b][s], &mut gradient_gamma_batch[b]);
            }
        }

        if self.gradient.is_some() {
            let stored_gradient = self.gradient.as_ref().expect("");
            gradient_gamma_batch = add_matrix_2d_c(&gradient_gamma_batch, &stored_gradient.get_gradient_gamma_batch());
        }

        gradient.set_gradient_input_batch(input_batch_gradients);
        gradient.set_gradient_gamma_batch(gradient_gamma_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.take().expect("No gradient found in rms norm layer");
        self.previous_gradient = Some(update_rms_norm_gamma(&mut self.gamma, gradient, self.previous_gradient.as_ref(), self.learning_rate, self.global_norm, self.max_norm));
    }
}

/// sqrt(mean(x * x) + epsilon) of one token embedding (complex, no conjugate).
pub fn rms_of(x: &[C], epsilon: Real) -> C {
    let mean_square = x.iter().map(|x| x * x).sum::<C>() / r(x.len() as f64);
    (mean_square + epsilon).sqrt()
}

/// RMSNorm of one row: out = x / rms(x) * gamma. Shared by `RMSNormLayer` and `RMSNormLayerRm`.
pub fn rms_norm_row(x: &[C], gamma: &[C], epsilon: Real, out: &mut [C]) {
    let rms = rms_of(x, epsilon);
    for ((o, &x), &g) in out.iter_mut().zip(x.iter()).zip(gamma.iter()) {
        *o = (x / rms) * g;
    }
}

/// Backward of `rms_norm_row` for one row: accumulates the input gradient into `input_gradient`
/// and x / rms(x) into `gamma_gradient`.
pub fn rms_norm_backward_row(x: &[C], previous_gradient: &[C], epsilon: Real, input_gradient: &mut [C], gamma_gradient: &mut [C]) {
    let dim_len = x.len();
    let rms = rms_of(x, epsilon);
    let rms_cubed = rms * rms * rms;
    let dim_r = r(dim_len as f64);

    for d_i in 0..dim_len {
        for d_j in 0..dim_len {
            let grad = if d_i == d_j {
                c(1.0, 0.0) / rms - (x[d_i] * x[d_j]) / (rms_cubed * dim_r)
            } else {
                -(x[d_i] * x[d_j]) / (rms_cubed * dim_r)
            };
            input_gradient[d_j] += grad.conj() * previous_gradient[d_i];
        }
        gamma_gradient[d_i] += x[d_i] / rms;
    }
}

/// AdamW update of gamma from the stored gradient; returns the gradient to keep as `previous_gradient`.
pub fn update_rms_norm_gamma(gamma: &mut Vec<C>, mut gradient: Gradient, previous_gradient: Option<&Gradient>, learning_rate: f64, global_norm: f64, max_norm: f64) -> Gradient {
    let mut gradient_gamma: Vec<C> = gradient.get_gradient_gamma();

    let total_valid_tokens = r(gradient.get_total_valid_tokens().max(1) as f64);
    gradient_gamma = average_vector_by_scalar(&gradient_gamma, total_valid_tokens);

    clip_all_gradients_by_global_norm_2d(&mut vec![], &mut gradient_gamma, global_norm, max_norm);
    normalize_bias(&mut gradient_gamma);

    let (mut prev_m_gamma, mut prev_v_gamma, mut prev_v_gamma_hat) = if let Some(previous_gradient) = previous_gradient {
        (previous_gradient.get_prev_m_gamma(), previous_gradient.get_prev_v_gamma(), previous_gradient.get_prev_v_gamma_hat())
    } else {
        // Initialize to zeros on first step
        (vec![c(0.0, 0.0); gradient_gamma.len()], vec![c(0.0, 0.0); gradient_gamma.len()], vec![c(0.0, 0.0); gradient_gamma.len()])
    };

    calculate_adam_w_bias(gamma, &gradient_gamma, &mut prev_m_gamma, &mut prev_v_gamma, &mut prev_v_gamma_hat, learning_rate, gradient.get_time_step());

    gradient.set_prev_m_gamma(prev_m_gamma);
    gradient.set_prev_v_gamma(prev_v_gamma);
    gradient.set_prev_v_gamma_hat(prev_v_gamma_hat);
    gradient.set_gradient_gamma(gradient_gamma);
    gradient
}

impl LayerInterface for RMSNormLayer {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        RMSNormLayer::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        RMSNormLayer::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        RMSNormLayer::update_parameters(self)
    }
}
