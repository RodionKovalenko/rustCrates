use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::{
    adam_w::calculate_adam_w,
    matrix::{average_matrix_by_scalar, RowMajorMatrix},
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
    pub input_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
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
            input_batch_rm: None,
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
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let input_batch_rm_ref: Option<&[RowMajorMatrix<Complex<f64>>]> = input.get_input_batch_rm_ref();
        let use_rm = input_batch_rm_ref.is_some_and(|rm| !rm.is_empty());

        // In rm_strict mode, never call get_input_batch() if RM activations exist (it would require RM->Vec conversion).
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = if use_rm { vec![] } else { input.get_input_batch() };

        // Store whichever representation was provided so backward can avoid reshaping.
        if !input_batch.is_empty() {
            self.input_batch = Some(input_batch.clone());
        } else {
            self.input_batch = None;
        }
        if let Some(rm) = input_batch_rm_ref {
            if !rm.is_empty() {
                self.input_batch_rm = Some(rm.to_vec());
            } else {
                self.input_batch_rm = None;
            }
        } else {
            self.input_batch_rm = None;
        }

        let in_f = self.weights_1.len();
        let out_f = self.weights_1[0].len();

        let output_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = if let Some(input_batch_rm) = input_batch_rm_ref {
            if !input_batch_rm.is_empty() {
                input_batch_rm
                    .par_iter()
                    .map(|input_rm| {
                        assert_eq!(input_rm.cols, in_f);
                        let time = input_rm.rows;
                        let mut out = RowMajorMatrix::from_data(time, out_f, vec![Complex::new(0.0, 0.0); time * out_f]);

                        for t in 0..time {
                            let in_row = input_rm.row_range(t);
                            for f in 0..out_f {
                                let mut sum_real = 0.0;
                                for k in 0..in_f {
                                    let x = input_rm.data[in_row.start + k];
                                    sum_real += x.re * self.weights_1[k][f].re + x.im * self.weights_2[k][f].re;
                                }
                                let out_i = out.idx(t, f);
                                out.data[out_i] = Complex::new(sum_real, 0.0);
                            }
                        }

                        out
                    })
                    .collect()
            } else {
                vec![]
            }
        } else {
            input_batch
                .par_iter()
                .map(|input_rows| {
                    let input_rm = RowMajorMatrix::from_rows(input_rows);
                    assert_eq!(input_rm.cols, in_f);
                    let time = input_rm.rows;
                    let mut out = RowMajorMatrix::from_data(time, out_f, vec![Complex::new(0.0, 0.0); time * out_f]);

                    for t in 0..time {
                        let in_row = input_rm.row_range(t);
                        for f in 0..out_f {
                            let mut sum_real = 0.0;
                            for k in 0..in_f {
                                let x = input_rm.data[in_row.start + k];
                                sum_real += x.re * self.weights_1[k][f].re + x.im * self.weights_2[k][f].re;
                            }
                            let out_i = out.idx(t, f);
                            out.data[out_i] = Complex::new(sum_real, 0.0);
                        }
                    }

                    out
                })
                .collect()
        };

        // Preserve legacy output when the legacy input representation is used.
        // If the caller provides only row-major inputs, we avoid allocating a nested Vec<Vec<...>> output.
        let need_legacy_output = !input_batch.is_empty();
        let mut layer_output = LayerOutput::new_default();
        if need_legacy_output {
            let output_batch: Vec<Vec<Vec<Complex<f64>>>> = output_batch_rm.iter().map(|m| m.to_rows()).collect();
            layer_output.set_output_batch(output_batch);
        }
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let input_batch_vec = self.input_batch.as_ref().filter(|b| !b.is_empty());
        let input_batch_rm = self.input_batch_rm.as_ref().filter(|b| !b.is_empty());
        if input_batch_vec.is_none() && input_batch_rm.is_none() {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_weight_2_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let previous_gradient_rm_ref = previous_gradient.get_gradient_input_batch_rm_ref().filter(|rm| !rm.is_empty());
        let previous_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient.get_gradient_input_batch();

        // Prefer non-empty row-major inputs; fall back to legacy Vec.
        let batch_len = if let Some(b) = input_batch_rm {
            b.len()
        } else {
            input_batch_vec.expect("vec batch").len()
        };

        let in_f = self.weights_1.len();
        let out_f = self.weights_1[0].len();

        let mut grad_w1 = vec![vec![vec![Complex::new(0.0, 0.0); out_f]; in_f]; batch_len];
        let mut grad_w2 = vec![vec![vec![Complex::new(0.0, 0.0); out_f]; in_f]; batch_len];
        let mut gradient_input_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = Vec::with_capacity(batch_len);

        if batch_len == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_weight_2_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        if previous_gradient_rm_ref.is_none() && previous_gradient_input_batch.is_empty() {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_weight_2_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        for batch_ind in 0..batch_len {
            let input_rm: RowMajorMatrix<Complex<f64>> = if let Some(rm_batch) = input_batch_rm {
                if let Some(rm) = rm_batch.get(batch_ind) {
                    rm.clone()
                } else if let Some(vec_batch) = input_batch_vec {
                    RowMajorMatrix::from_rows(&vec_batch[batch_ind])
                } else {
                    let mut gradient = Gradient::new_default();
                    gradient.set_gradient_input_batch_rm(vec![]);
                    gradient.set_gradient_weight_batch(vec![]);
                    gradient.set_gradient_weight_2_batch(vec![]);
                    gradient.set_total_valid_tokens(total_valid_tokens);
                    self.gradient = Some(gradient.clone());
                    return gradient;
                }
            } else {
                let input_sample = &input_batch_vec.expect("vec batch")[batch_ind];
                RowMajorMatrix::from_rows(input_sample)
            };

            let grad_rm: RowMajorMatrix<Complex<f64>> = if let Some(grads_rm) = previous_gradient_rm_ref {
                if let Some(rm) = grads_rm.get(batch_ind) {
                    rm.clone()
                } else if !previous_gradient_input_batch.is_empty() {
                    RowMajorMatrix::from_rows(&previous_gradient_input_batch[batch_ind])
                } else {
                    let mut gradient = Gradient::new_default();
                    gradient.set_gradient_input_batch_rm(vec![]);
                    gradient.set_gradient_weight_batch(vec![]);
                    gradient.set_gradient_weight_2_batch(vec![]);
                    gradient.set_total_valid_tokens(total_valid_tokens);
                    self.gradient = Some(gradient.clone());
                    return gradient;
                }
            } else {
                RowMajorMatrix::from_rows(&previous_gradient_input_batch[batch_ind])
            };

            assert_eq!(input_rm.cols, in_f);
            assert_eq!(grad_rm.cols, out_f);
            assert_eq!(input_rm.rows, grad_rm.rows);

            let time = input_rm.rows;
            let mut gx_rm = RowMajorMatrix::from_data(time, in_f, vec![Complex::new(0.0, 0.0); time * in_f]);

            for t in 0..time {
                let in_row = input_rm.row_range(t);
                let g_row = grad_rm.row_range(t);
                for f in 0..out_f {
                    let g = grad_rm.data[g_row.start + f].re;
                    for k in 0..in_f {
                        let x = input_rm.data[in_row.start + k];

                        // input gradients
                        let gi = gx_rm.idx(t, k);
                        let mut cur = gx_rm.data[gi];
                        cur.re += g * self.weights_1[k][f].re;
                        cur.im += g * self.weights_2[k][f].re;
                        gx_rm.data[gi] = cur;

                        // weight gradients
                        grad_w1[batch_ind][k][f].re += x.re * g;
                        grad_w2[batch_ind][k][f].re += x.im * g;
                    }
                }
            }

            gradient_input_batch_rm.push(gx_rm);
        }

        let mut gradient = Gradient::new_default();
        let legacy_mode = self.input_batch.is_some();
        if legacy_mode {
            let gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = gradient_input_batch_rm.iter().map(|m| m.to_rows()).collect();
            gradient.set_gradient_input_batch(gradient_input_batch);
        }
        gradient.set_gradient_input_batch_rm(gradient_input_batch_rm);
        gradient.set_gradient_weight_batch(grad_w1);
        gradient.set_gradient_weight_2_batch(grad_w2);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");

        let mut weight_gradients_1: Vec<Vec<Complex<f64>>> = gradient.get_gradient_weights();
        let mut weight_gradients_2: Vec<Vec<Complex<f64>>> = gradient.get_gradient_weights_2();

        // Get total valid tokens from gradient
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in complex_to_linear_layer");
        let total_valid_tokens = gradient.get_total_valid_tokens().max(1) as f64;

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
