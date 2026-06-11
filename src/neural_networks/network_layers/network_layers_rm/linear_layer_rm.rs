use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::default_layer::LayerInterface,
    utils::{
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        dtype::{r, C, ONE, ZERO},
        matrix::{
            add_vector_rm, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose_rm, multiply_complex_rm, normalize_bias, normalize_gradients,
            RowMajorMatrix,
        },
        weights_initializer::initialize_weights_complex,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayerRm {
    pub weights: RowMajorMatrix<C>,
    pub bias: Vec<C>,
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

impl LinearLayerRm {
    pub fn new(learning_rate: f64, rows: usize, cols: usize) -> Self {
        let mut weights_vec: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        initialize_weights_complex(rows, cols, &mut weights_vec);
        let weights = RowMajorMatrix::from_rows(&weights_vec);
        let bias: Vec<C> = vec![C::new(ONE, ZERO); cols];

        Self {
            weights,
            bias,
            learning_rate,
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
        if input.has_non_empty_input_batch() {
            panic!("LinearLayerRm received Vec input; use LinearLayer (Vec)");
        }

        let input_rm = input.get_input_batch_rm_ref().filter(|b| !b.is_empty()).expect("LinearLayerRm::forward expects RM input").to_vec();

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch_rm = Some(input_rm.clone());

        let mut out_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(input_rm.len());
        for m in input_rm.iter() {
            let mut out = multiply_complex_rm(m, &self.weights);
            add_vector_rm(&mut out, &self.bias);
            out_rm.push(out);
        }

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(vec![]);
        layer_output.set_output_batch_rm(out_rm);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();

        let input_batch_rm = self.input_batch_rm.as_ref().filter(|b| !b.is_empty()).expect("LinearLayerRm::backward missing input_batch_rm");

        let grads_rm = previous_gradient
            .get_gradient_input_batch_rm_ref()
            .filter(|b| !b.is_empty())
            .expect("LinearLayerRm::backward expects RM gradient");

        if previous_gradient.get_gradient_input_batch_ref().is_some_and(|b| !b.is_empty()) {
            panic!("LinearLayerRm received Vec gradient; use LinearLayer (Vec)");
        }

        let batch_len = input_batch_rm.len();
        if batch_len == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let weights_h_rm = conjugate_transpose_rm(&self.weights);

        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights.cols]; self.weights.rows]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];
        let mut input_grad_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_len);

        for b in 0..batch_len {
            let x_rm = &input_batch_rm[b];
            let g_rm = &grads_rm[b];

            let x_h = conjugate_transpose_rm(x_rm);
            let wgrad_rm = multiply_complex_rm(&x_h, g_rm);
            weight_gradients[b] = wgrad_rm.to_rows();

            // bias gradients: sum over rows
            for r_i in 0..g_rm.rows {
                let row = g_rm.row_range(r_i);
                for c in 0..g_rm.cols {
                    bias_gradients[b][c] += g_rm.data[row.start + c];
                }
            }

            let gx_rm = multiply_complex_rm(g_rm, &weights_h_rm);
            input_grad_rm.push(gx_rm);
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(input_grad_rm);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        gradient.set_total_valid_tokens(total_valid_tokens);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in LinearLayerRm");

        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());
        let total_valid_tokens = gradient.get_total_valid_tokens();

        weight_gradients = average_matrix_by_scalar(&weight_gradients, r(total_valid_tokens as f64));
        bias_gradients = average_vector_by_scalar(&bias_gradients, r(total_valid_tokens as f64));

        clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        normalize_gradients(&mut weight_gradients);
        normalize_bias(&mut bias_gradients);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights, mut prev_v_weights_hat, mut prev_v_bias_hat) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_bias(),
                previous_gradient.get_prev_v_bias(),
                previous_gradient.get_prev_m_weights(),
                previous_gradient.get_prev_v_weights(),
                previous_gradient.get_prev_v_weights_hat(),
                previous_gradient.get_prev_v_bias_hat(),
            )
        } else {
            (
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![vec![C::new(ZERO, ZERO); self.weights.cols]; self.weights.rows],
                vec![vec![C::new(ZERO, ZERO); self.weights.cols]; self.weights.rows],
                vec![vec![C::new(ZERO, ZERO); self.weights.cols]; self.weights.rows],
                vec![C::new(ZERO, ZERO); self.bias.len()],
            )
        };

        calculate_adam_w_bias(
            &mut self.bias,
            &bias_gradients,
            &mut prev_m_bias,
            &mut prev_v_bias,
            &mut prev_v_bias_hat,
            learning_rate,
            time_step,
        );

        let mut w = self.weights.to_rows();
        calculate_adam_w(
            &mut w,
            &weight_gradients,
            &mut prev_m_weights,
            &mut prev_v_weights,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
        );
        self.weights = RowMajorMatrix::from_rows(&w);

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);
        gradient.set_gradient_weights(weight_gradients);
        gradient.set_gradient_bias(bias_gradients);
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}

impl LayerInterface for LinearLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        LinearLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        LinearLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        LinearLayerRm::update_parameters(self)
    }
}
