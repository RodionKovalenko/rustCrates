use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::default_layer::LayerInterface,
    utils::{
        adam_w_rm::update_weights_adam_w_rm,
        dtype::{r, Real, C, ZERO},
        matrix_rm::RowMajorMatrix,
        matrix_rm::{average_matrix_by_scalar_rm, normalize_gradients_rm},
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

    /// RM counterpart of `ComplexToLinearLayer::forward`: `Re(x) * Re(W1) + Im(x) * Re(W2)` as a real output.
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let input_batch_rm = input.get_input_batch_rm();
        let (in_f, out_f) = (self.weights_1.rows, self.weights_1.cols);

        let output_batch_rm: Vec<RowMajorMatrix<C>> = input_batch_rm
            .par_iter()
            .map(|m| {
                assert_eq!(m.cols, in_f, "ComplexToLinearLayerRm: input cols must match weights rows");
                let mut out = RowMajorMatrix::create_empty(m.rows, out_f);
                for r_i in 0..m.rows {
                    let x_row = &m.data[m.row_range(r_i)];
                    for f in 0..out_f {
                        let mut sum_real: Real = ZERO;
                        for k in 0..in_f {
                            sum_real += x_row[k].re * self.weights_1.data[k * out_f + f].re + x_row[k].im * self.weights_2.data[k * out_f + f].re;
                        }
                        out.data[r_i * out_f + f] = C::new(sum_real, ZERO);
                    }
                }
                out
            })
            .collect();

        self.input_batch_rm = Some(input_batch_rm);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    /// RM counterpart of `ComplexToLinearLayer::backward`. Only the real part of the upstream gradient is used.
    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("ComplexToLinearLayerRm::backward missing input_batch_rm");
        let previous_gradient_rm = previous_gradient.get_gradient_input_batch_rm_cow();
        assert_eq!(input_batch_rm.len(), previous_gradient_rm.len(), "ComplexToLinearLayerRm::backward batch mismatch");

        let (in_f, out_f) = (self.weights_1.rows, self.weights_1.cols);

        let per_batch: Vec<(RowMajorMatrix<C>, RowMajorMatrix<C>, RowMajorMatrix<C>)> = input_batch_rm
            .par_iter()
            .zip(previous_gradient_rm.par_iter())
            .map(|(x, g)| {
                assert_eq!((x.cols, g.cols, x.rows), (in_f, out_f, g.rows));
                let mut grad_input = RowMajorMatrix::create_empty(x.rows, in_f);
                let mut grad_w1 = RowMajorMatrix::create_empty(in_f, out_f);
                let mut grad_w2 = RowMajorMatrix::create_empty(in_f, out_f);

                for r_i in 0..x.rows {
                    for f in 0..out_f {
                        let gg = g.data[r_i * out_f + f].re;
                        for k in 0..in_f {
                            let xk = x.data[r_i * in_f + k];
                            let w = k * out_f + f;
                            grad_input.data[r_i * in_f + k].re += gg * self.weights_1.data[w].re;
                            grad_input.data[r_i * in_f + k].im += gg * self.weights_2.data[w].re;
                            grad_w1.data[w].re += xk.re * gg;
                            grad_w2.data[w].re += xk.im * gg;
                        }
                    }
                }
                (grad_input, grad_w1, grad_w2)
            })
            .collect();

        let mut grad_input_rm = Vec::with_capacity(per_batch.len());
        let mut grad_w1 = Vec::with_capacity(per_batch.len());
        let mut grad_w2 = Vec::with_capacity(per_batch.len());
        for (gi, g1, g2) in per_batch {
            grad_input_rm.push(gi);
            grad_w1.push(g1);
            grad_w2.push(g2);
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(grad_input_rm);
        gradient.set_weight_gradient_batch_rm(WEIGHTS_1, grad_w1);
        gradient.set_weight_gradient_batch_rm(WEIGHTS_2, grad_w2);
        gradient.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());

        self.gradient = Some(gradient.clone());
        gradient
    }

    /// RM counterpart of `ComplexToLinearLayer::update_parameters`.
    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in ComplexToLinearLayerRm");

        let total_valid_tokens: Real = r(gradient.get_total_valid_tokens().max(1) as f64);
        let mut weight_gradients_1 = average_matrix_by_scalar_rm(&gradient.get_weight_gradient_rm(WEIGHTS_1).expect("missing weights_1 gradient"), total_valid_tokens);
        let mut weight_gradients_2 = average_matrix_by_scalar_rm(&gradient.get_weight_gradient_rm(WEIGHTS_2).expect("missing weights_2 gradient"), total_valid_tokens);

        normalize_gradients_rm(&mut weight_gradients_1);
        normalize_gradients_rm(&mut weight_gradients_2);

        let (learning_rate, time_step) = (self.learning_rate as f64, self.time_step);
        let previous_gradient = self.previous_gradient.as_ref();
        update_weights_adam_w_rm(WEIGHTS_1, &mut self.weights_1, &weight_gradients_1, previous_gradient, gradient, learning_rate, time_step);
        update_weights_adam_w_rm(WEIGHTS_2, &mut self.weights_2, &weight_gradients_2, previous_gradient, gradient, learning_rate, time_step);

        self.previous_gradient = self.gradient.take();
    }
}

const WEIGHTS_1: &str = Gradient::WEIGHTS;
const WEIGHTS_2: &str = "weights_2";

impl LayerInterface for ComplexToLinearLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        ComplexToLinearLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        ComplexToLinearLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        ComplexToLinearLayerRm::update_parameters(self)
    }
}
