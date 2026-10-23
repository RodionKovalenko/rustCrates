use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::default_layer::LayerInterface,
    utils::{
        adam_w::calculate_adam_w_bias,
        dtype::{c, r, Real, C},
        matrix::{average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, normalize_bias},
        matrix_rm::RowMajorMatrix,
        matrix_rm::add_rm,
    },
};

/// RM counterpart of `RMSNormLayer`: RMSNorm(x + residual) * gamma per token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayerRm {
    pub gamma: Vec<C>,
    pub epsilon: Real,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl RMSNormLayerRm {
    pub fn new(feature_dim: usize, epsilon: f64, learning_rate: f64) -> Self {
        Self {
            gamma: vec![c(1.0, 0.0); feature_dim],
            epsilon: r(epsilon),
            smoothing: 0.9,
            ema: 0.0,
            learning_rate,
            input_batch_rm: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    /// RM counterpart of `RMSNormLayer::forward`: normalizes `input + input_before` (residual).
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm = layer_input.get_input_batch_rm();
        let residual_batch_rm = layer_input.get_input_batch_before_rm_ref().expect("RMSNormLayerRm::forward expects RM residual input (input_batch_before_rm)");
        assert_eq!(input_batch_rm.len(), residual_batch_rm.len(), "RMSNormLayerRm::forward batch size mismatch");

        let input_batch_added_rm: Vec<RowMajorMatrix<C>> = input_batch_rm.iter().zip(residual_batch_rm.iter()).map(|(x, residual)| add_rm(x, residual)).collect();

        let output_batch_rm: Vec<RowMajorMatrix<C>> = input_batch_added_rm
            .par_iter()
            .map(|added| {
                let mut out = RowMajorMatrix::create_empty(added.rows, added.cols);
                for row in 0..added.rows {
                    let range = added.row_range(row);
                    rms_norm_row(&added.data[range.clone()], &self.gamma, self.epsilon, &mut out.data[range]);
                }
                out
            })
            .collect();

        self.input_batch_rm = Some(input_batch_added_rm);
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    /// RM counterpart of `RMSNormLayer::backward`.
    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("Input batch RM not found in RMSNormLayerRm");
        assert_eq!(input_batch_rm.len(), previous_gradient_batch_rm.len(), "RMSNormLayerRm::backward batch size mismatch");

        let per_batch: Vec<(RowMajorMatrix<C>, Vec<C>)> = input_batch_rm
            .par_iter()
            .zip(previous_gradient_batch_rm.par_iter())
            .map(|(x, g)| {
                let mut input_gradient = RowMajorMatrix::create_empty(x.rows, x.cols);
                let mut gamma_gradient = vec![c(0.0, 0.0); x.cols];
                for row in 0..x.rows {
                    let range = x.row_range(row);
                    rms_norm_backward_row(&x.data[range.clone()], &g.data[range.clone()], self.epsilon, &mut input_gradient.data[range], &mut gamma_gradient);
                }
                (input_gradient, gamma_gradient)
            })
            .collect();

        let (input_gradient_batch_rm, mut gradient_gamma_batch): (Vec<RowMajorMatrix<C>>, Vec<Vec<C>>) = per_batch.into_iter().unzip();

        // Gamma gradients accumulate until update_parameters (like RMSNormLayer).
        if let Some(stored_gradient) = self.gradient.as_ref() {
            for (g, stored) in gradient_gamma_batch.iter_mut().zip(stored_gradient.get_gradient_gamma_batch().iter()) {
                for (a, &b) in g.iter_mut().zip(stored.iter()) {
                    *a += b;
                }
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);
        gradient.set_gradient_gamma_batch(gradient_gamma_batch);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        self.backward_rm(&previous_gradient.get_gradient_input_batch_rm_cow())
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.take().expect("No gradient found in RMSNormLayerRm");
        self.previous_gradient = Some(update_rms_norm_gamma(&mut self.gamma, gradient, self.previous_gradient.as_ref(), self.learning_rate, self.global_norm, self.max_norm));
    }
}

impl LayerInterface for RMSNormLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        RMSNormLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        RMSNormLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        RMSNormLayerRm::update_parameters(self)
    }
}

/// sqrt(mean(x * x) + epsilon) of one token embedding (complex, no conjugate).
fn rms_of(x: &[C], epsilon: Real) -> C {
    let mean_square = x.iter().map(|x| x * x).sum::<C>() / r(x.len() as f64);
    (mean_square + epsilon).sqrt()
}

/// RMSNorm of one row: out = x / rms(x) * gamma.
fn rms_norm_row(x: &[C], gamma: &[C], epsilon: Real, out: &mut [C]) {
    let rms = rms_of(x, epsilon);
    for ((o, &x), &g) in out.iter_mut().zip(x.iter()).zip(gamma.iter()) {
        *o = (x / rms) * g;
    }
}

/// Backward of `rms_norm_row` for one row: accumulates the input gradient into `input_gradient`
/// and x / rms(x) into `gamma_gradient`.
fn rms_norm_backward_row(x: &[C], previous_gradient: &[C], epsilon: Real, input_gradient: &mut [C], gamma_gradient: &mut [C]) {
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
fn update_rms_norm_gamma(gamma: &mut Vec<C>, mut gradient: Gradient, previous_gradient: Option<&Gradient>, learning_rate: f64, global_norm: f64, max_norm: f64) -> Gradient {
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
