use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    utils::{
        adam_w::calculate_adam_w,
        dtype::{r, Real, C, ZERO},
        matrix::{average_matrix_by_scalar, normalize_gradients, RowMajorMatrix},
        weights_initializer::initialize_weights_complex_only_real,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexToLinearLayerRm {
    pub weights_1: RowMajorMatrix<C>,
    pub weights_2: RowMajorMatrix<C>,
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
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl ComplexToLinearLayerRm {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut weights_1_vec: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let mut weights_2_vec: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];

        initialize_weights_complex_only_real(rows, cols, &mut weights_1_vec);
        initialize_weights_complex_only_real(rows, cols, &mut weights_2_vec);

        Self {
            weights_1: RowMajorMatrix::from_rows(&weights_1_vec),
            weights_2: RowMajorMatrix::from_rows(&weights_2_vec),
            learning_rate: r(learning_rate),
            gradients: vec![],
            gradients_bias: vec![],
            input_batch_rm: None,
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
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let rm_in = input.get_input_batch_rm_ref().filter(|b| !b.is_empty()).expect("ComplexToLinearLayerRm::forward expects RM input");

        if input.has_non_empty_input_batch() {
            panic!("ComplexToLinearLayerRm received Vec input; use ComplexToLinearLayer (Vec)");
        }

        let in_f = self.weights_1.rows;
        let out_f = self.weights_1.cols;

        let output_batch_rm: Vec<RowMajorMatrix<C>> = rm_in
            .par_iter()
            .map(|m| {
                assert_eq!(m.cols, in_f, "ComplexToLinearLayerRm: input cols must match weights rows");
                let rows = m.rows;
                let mut out = vec![C::new(ZERO, ZERO); rows * out_f];

                for r_i in 0..rows {
                    let row_start = r_i * in_f;
                    for f in 0..out_f {
                        let mut sum_real: Real = ZERO;
                        for k in 0..in_f {
                            let x = m.data[row_start + k];
                            let w1 = self.weights_1.data[k * out_f + f].re;
                            let w2 = self.weights_2.data[k * out_f + f].re;
                            sum_real += x.re * w1 + x.im * w2;
                        }
                        out[r_i * out_f + f] = C::new(sum_real, ZERO);
                    }
                }

                RowMajorMatrix::from_data(rows, out_f, out)
            })
            .collect();

        self.input_batch_rm = Some(rm_in.to_vec());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(vec![]);
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();

        let input_batch_rm = self.input_batch_rm.as_ref().filter(|b| !b.is_empty()).expect("ComplexToLinearLayerRm::backward missing input_batch_rm");

        let prev_rm = previous_gradient
            .get_gradient_input_batch_rm_ref()
            .filter(|b| !b.is_empty())
            .expect("ComplexToLinearLayerRm::backward expects RM gradient");

        if previous_gradient.get_gradient_input_batch_ref().is_some_and(|b| !b.is_empty()) {
            panic!("ComplexToLinearLayerRm received Vec gradient; use ComplexToLinearLayer (Vec)");
        }

        let batch_len = input_batch_rm.len();
        if batch_len == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_weight_2_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let in_f = self.weights_1.rows;
        let out_f = self.weights_1.cols;

        let mut grad_w1 = vec![vec![vec![C::new(ZERO, ZERO); out_f]; in_f]; batch_len];
        let mut grad_w2 = vec![vec![vec![C::new(ZERO, ZERO); out_f]; in_f]; batch_len];

        let mut grad_input_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_len);

        for b in 0..batch_len {
            let x = &input_batch_rm[b];
            let g = &prev_rm[b];

            assert_eq!(x.cols, in_f);
            assert_eq!(g.cols, out_f);
            assert_eq!(x.rows, g.rows);

            let rows = x.rows;
            let mut gx_data = vec![C::new(ZERO, ZERO); rows * in_f];

            for r_i in 0..rows {
                let x_row = r_i * in_f;
                let g_row = r_i * out_f;
                for f in 0..out_f {
                    let gg = g.data[g_row + f].re;
                    for k in 0..in_f {
                        let xk = x.data[x_row + k];
                        let w1 = self.weights_1.data[k * out_f + f].re;
                        let w2 = self.weights_2.data[k * out_f + f].re;

                        // input gradients
                        let idx = x_row + k;
                        gx_data[idx].re += gg * w1;
                        gx_data[idx].im += gg * w2;

                        // weight gradients
                        grad_w1[b][k][f].re += xk.re * gg;
                        grad_w2[b][k][f].re += xk.im * gg;
                    }
                }
            }

            grad_input_rm.push(RowMajorMatrix::from_data(rows, in_f, gx_data));
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(grad_input_rm);
        gradient.set_gradient_weight_batch(grad_w1);
        gradient.set_gradient_weight_2_batch(grad_w2);
        gradient.set_total_valid_tokens(total_valid_tokens);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in ComplexToLinearLayerRm");

        let mut weight_gradients_1: Vec<Vec<C>> = gradient.get_gradient_weights();
        let mut weight_gradients_2: Vec<Vec<C>> = gradient.get_gradient_weights_2();

        let total_valid_tokens: Real = r(gradient.get_total_valid_tokens().max(1) as f64);
        weight_gradients_1 = average_matrix_by_scalar(&weight_gradients_1, total_valid_tokens);
        weight_gradients_2 = average_matrix_by_scalar(&weight_gradients_2, total_valid_tokens);

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
                let rows = self.weights_1.rows;
                let cols = self.weights_1.cols;
                (
                    vec![vec![C::new(ZERO, ZERO); cols]; rows],
                    vec![vec![C::new(ZERO, ZERO); cols]; rows],
                    vec![vec![C::new(ZERO, ZERO); cols]; rows],
                    vec![vec![C::new(ZERO, ZERO); cols]; rows],
                    vec![vec![C::new(ZERO, ZERO); cols]; rows],
                    vec![vec![C::new(ZERO, ZERO); cols]; rows],
                )
            };

        let mut w1 = self.weights_1.to_rows();
        let mut w2 = self.weights_2.to_rows();

        calculate_adam_w(
            &mut w1,
            &weight_gradients_1,
            &mut prev_m_weights_1,
            &mut prev_v_weights_1,
            &mut prev_v_weights_hat_1,
            learning_rate as f64,
            time_step,
        );
        calculate_adam_w(
            &mut w2,
            &weight_gradients_2,
            &mut prev_m_weights_2,
            &mut prev_v_weights_2,
            &mut prev_v_weights_hat_2,
            learning_rate as f64,
            time_step,
        );

        self.weights_1 = RowMajorMatrix::from_rows(&w1);
        self.weights_2 = RowMajorMatrix::from_rows(&w2);

        gradient.set_prev_m_weights(prev_m_weights_1);
        gradient.set_prev_v_weights(prev_v_weights_1);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat_1);

        gradient.set_prev_m_weights_2(prev_m_weights_2);
        gradient.set_prev_v_weights_2(prev_v_weights_2);
        gradient.set_prev_v_weights_hat_2(prev_v_weights_hat_2);

        gradient.set_gradient_weights(weight_gradients_1);
        gradient.set_gradient_weights_2(weight_gradients_2);
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}
