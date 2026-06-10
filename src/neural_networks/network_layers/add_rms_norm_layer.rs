use core::fmt::Debug;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    utils::{
        adam_w::calculate_adam_w_bias,
        matrix::{add_matrix, add_matrix_2d_c, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, normalize_bias},
    },
};

use crate::neural_networks::utils::dtype::{c, r, Real, C};
use crate::neural_networks::utils::matrix::RowMajorMatrix;

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
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
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
            input_batch_rm: None,
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

        let rms = self.rms(input);

        // Normalize the input and apply the learned gamma scaling
        input.iter().zip(self.gamma.iter()).map(|(x, &g)| (*x / rms) * g).collect()
    }

    pub fn rms(&self, input: &Vec<C>) -> C {
        let mean_square = input
            .iter()
            .map(|x| {
                // println!("x {:?}, x * x {:?}", x, x * x);
                x * x
            })
            .sum::<C>()
            / r(input.len() as f64);
        (mean_square + self.epsilon).sqrt()
    }

    // Forward pass for a batch of token embeddings (2D input)
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm_ref = layer_input.get_input_batch_rm_ref();
        let input_batch_ref = layer_input.get_input_batch_ref();

        let use_rm = input_batch_rm_ref.is_some() && input_batch_ref.map_or(true, |v| v.is_empty());

        if use_rm {
            return self.forward_rm(layer_input);
        }

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
        self.input_batch_rm = None;
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn forward_rm(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm = layer_input.get_input_batch_rm_ref().expect("RMSNormLayer::forward_rm expects RM input").to_vec();
        let input_before_transform_batch_rm = layer_input
            .get_input_batch_before_rm_ref()
            .expect("RMSNormLayer::forward_rm expects RM residual input (input_batch_before_rm)")
            .to_vec();

        assert_eq!(input_batch_rm.len(), input_before_transform_batch_rm.len(), "RMSNormLayer::forward_rm batch size mismatch");
        let mut output_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(input_batch_rm.len());
        let mut input_batch_added_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(input_batch_rm.len());

        for (x, residual) in input_batch_rm.iter().zip(input_before_transform_batch_rm.iter()) {
            assert_eq!((x.rows, x.cols), (residual.rows, residual.cols), "RMSNormLayer::forward_rm shape mismatch");
            let mut added = RowMajorMatrix::from_data(x.rows, x.cols, vec![c(0.0, 0.0); x.rows * x.cols]);
            for i in 0..added.data.len() {
                added.data[i] = x.data[i] + residual.data[i];
            }

            let mut out = RowMajorMatrix::from_data(added.rows, added.cols, vec![c(0.0, 0.0); added.rows * added.cols]);
            for row in 0..added.rows {
                let rr = added.row_range(row);
                let row_slice = &added.data[rr.clone()];

                if row_slice.is_empty() {
                    continue;
                }

                let mean_square = row_slice.iter().map(|x| x * x).sum::<C>() / r(added.cols as f64);
                let rms = (mean_square + self.epsilon).sqrt();

                for c in 0..added.cols {
                    out.data[rr.start + c] = (row_slice[c] / rms) * self.gamma[c];
                }
            }

            input_batch_added_rm.push(added);
            output_batch_rm.push(out);
        }

        self.input_batch = None;
        self.input_batch_rm = Some(input_batch_added_rm);
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found in RMSNorm layer");
        let mut gradient = Gradient::new_default();

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let dim_len = input_batch[0][0].len();

        let mut input_batch_gradients = vec![vec![vec![c(0.0, 0.0); dim_len]; seq_len]; batch_size];
        let mut gradient_gamma_batch = vec![vec![c(0.0, 0.0); dim_len]; batch_size];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let rms = self.rms(&input_batch[b][s]);
                let rms_cubed = rms * rms * rms;
                let dim_r = r(dim_len as f64);

                for d_i in 0..dim_len {
                    for d_j in 0..dim_len {
                        let grad = if d_i == d_j {
                            c(1.0, 0.0) / rms - (input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (rms_cubed * dim_r)
                        } else {
                            -(input_batch[b][s][d_i] * input_batch[b][s][d_j]) / (rms_cubed * dim_r)
                        };
                        input_batch_gradients[b][s][d_j] += grad.conj() * previous_gradient_batch[b][s][d_i];
                    }

                    gradient_gamma_batch[b][d_i] += input_batch[b][s][d_i] / rms;
                }
            }
        }

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            gradient_gamma_batch = add_matrix_2d_c(&gradient_gamma_batch, &previous_gradient.get_gradient_gamma_batch());
        }

        gradient.set_gradient_input_batch(input_batch_gradients);
        gradient.set_gradient_gamma_batch(gradient_gamma_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("Input batch RM not found in RMSNorm layer (did you call forward_rm?)");

        let batch_size = input_batch_rm.len();
        if batch_size == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_gamma_batch(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let seq_len = input_batch_rm[0].rows;
        let dim_len = input_batch_rm[0].cols;

        // Pad/truncate gradients to match batch size.
        let mut prev_grads: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            if let Some(g) = previous_gradient_batch_rm.get(b) {
                prev_grads.push(g.clone());
            } else {
                prev_grads.push(RowMajorMatrix::from_data(seq_len, dim_len, vec![c(0.0, 0.0); seq_len * dim_len]));
            }
        }

        let mut input_batch_gradients_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
        let mut gradient_gamma_batch = vec![vec![c(0.0, 0.0); dim_len]; batch_size];

        for b in 0..batch_size {
            assert_eq!((input_batch_rm[b].rows, input_batch_rm[b].cols), (seq_len, dim_len));
            assert_eq!((prev_grads[b].rows, prev_grads[b].cols), (seq_len, dim_len));

            let mut grad_m = RowMajorMatrix::from_data(seq_len, dim_len, vec![c(0.0, 0.0); seq_len * dim_len]);

            for s in 0..seq_len {
                let row = input_batch_rm[b].row_range(s);
                let x = &input_batch_rm[b].data[row.clone()];
                let g = &prev_grads[b].data[row.clone()];

                let mean_square = x.iter().map(|v| v * v).sum::<C>() / r(dim_len as f64);
                let rms = (mean_square + self.epsilon).sqrt();
                let rms_cubed = rms * rms * rms;
                let dim_r = r(dim_len as f64);

                for d_i in 0..dim_len {
                    for d_j in 0..dim_len {
                        let grad = if d_i == d_j {
                            c(1.0, 0.0) / rms - (x[d_i] * x[d_j]) / (rms_cubed * dim_r)
                        } else {
                            -(x[d_i] * x[d_j]) / (rms_cubed * dim_r)
                        };
                        grad_m.data[row.start + d_j] += grad.conj() * g[d_i];
                    }
                    gradient_gamma_batch[b][d_i] += x[d_i] / rms;
                }
            }

            input_batch_gradients_rm.push(grad_m);
        }

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            gradient_gamma_batch = add_matrix_2d_c(&gradient_gamma_batch, &previous_gradient.get_gradient_gamma_batch());
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(input_batch_gradients_rm);
        gradient.set_gradient_gamma_batch(gradient_gamma_batch);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No gradient found in rms norm layer");
        let mut gradient_gamma: Vec<C> = gradient.get_gradient_gamma();

        let total_valid_tokens = r(gradient.get_total_valid_tokens().max(1) as f64);

        gradient_gamma = average_vector_by_scalar(&gradient_gamma, total_valid_tokens);

        clip_all_gradients_by_global_norm_2d(&mut vec![], &mut gradient_gamma, self.global_norm, self.max_norm);

        normalize_bias(&mut gradient_gamma);

        let learning_rate = self.learning_rate;

        let (mut prev_m_gamma, mut prev_v_gamma, mut prev_v_gamma_hat) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (previous_gradient.get_prev_m_gamma(), previous_gradient.get_prev_v_gamma(), previous_gradient.get_prev_v_gamma_hat())
        } else {
            // Initialize to zeros on first step
            (
                vec![c(0.0, 0.0); gradient_gamma.len()],
                vec![c(0.0, 0.0); gradient_gamma.len()],
                vec![c(0.0, 0.0); gradient_gamma.len()],
            )
        };

        calculate_adam_w_bias(
            &mut self.gamma,
            &gradient_gamma,
            &mut prev_m_gamma,
            &mut prev_v_gamma,
            &mut prev_v_gamma_hat,
            learning_rate,
            gradient.get_time_step(),
        );

        gradient.set_prev_m_gamma(prev_m_gamma);
        gradient.set_prev_v_gamma(prev_v_gamma);
        gradient.set_prev_v_gamma_hat(prev_v_gamma_hat);
        gradient.set_gradient_gamma(gradient_gamma.clone());
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}
