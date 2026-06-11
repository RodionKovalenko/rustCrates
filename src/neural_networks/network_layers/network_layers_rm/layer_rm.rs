use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{default_layer::LayerInterface, layer::{ActivationType, LayerType}},
    utils::{
        activation::activate_output_complex,
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        dtype::{r, Real, C, ONE, ZERO},
        matrix::{
            add_vector_rm, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose_rm, multiply_complex_rm, normalize_bias, normalize_gradients,
            RowMajorMatrix,
        },
        weights_initializer::initialize_weights_complex,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRm {
    pub weights: RowMajorMatrix<C>,
    pub bias: Vec<C>,
    pub activation_type: ActivationType,
    pub layer_type: LayerType,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub inactivated_input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl LayerRm {
    pub fn new(rows: usize, cols: usize, learning_rate: &f64, activation: &ActivationType, layer_type: LayerType) -> Self {
        let mut weights_vec: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        initialize_weights_complex(rows, cols, &mut weights_vec);
        let weights = RowMajorMatrix::from_rows(&weights_vec);
        let bias: Vec<C> = vec![C::new(ONE, ZERO); cols];

        Self {
            weights,
            bias,
            activation_type: activation.clone(),
            layer_type,
            learning_rate: *learning_rate,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            previous_gradient: None,
            input_batch_rm: None,
            inactivated_input_batch_rm: None,
            output_batch_rm: None,
            gradient: None,
            padding_mask_batch: None,
            time_step: 0,
            batch_size: 0,
        }
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_rm = input.get_input_batch_rm_ref().expect("LayerRm::forward expects input_batch_rm").to_vec();

        let calculate_gradient = input.get_calculate_gradient();

        let activation_supported_rm = matches!(
            self.activation_type,
            ActivationType::LINEAR | ActivationType::TANH | ActivationType::RELU | ActivationType::LEAKYRELU | ActivationType::SIGMOID | ActivationType::GELU | ActivationType::SWiGLU
        );

        if !activation_supported_rm {
            if input.get_rm_strict() {
                panic!("RM strict mode violation: Dense RM layer activation {:?} lacks RM support", self.activation_type);
            }
        }

        let needs_raw_pre_activation_rm = matches!(self.activation_type, ActivationType::SWiGLU | ActivationType::GELU);

        let mut output_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(input_rm.len());
        let mut raw_output_batch_rm: Vec<RowMajorMatrix<C>> = if calculate_gradient && needs_raw_pre_activation_rm {
            Vec::with_capacity(input_rm.len())
        } else {
            Vec::new()
        };

        for m in input_rm.iter() {
            let mut out = multiply_complex_rm(m, &self.weights);
            add_vector_rm(&mut out, &self.bias);

            if calculate_gradient && needs_raw_pre_activation_rm {
                raw_output_batch_rm.push(out.clone());
            }

            let out_act = if activation_supported_rm {
                activate_output_complex_rm(out, &self.activation_type)
            } else {
                // Fallback: do activation in legacy (Vec) representation.
                let rows = out.to_rows();
                let activated = activate_output_complex(&rows, self.activation_type.clone());
                RowMajorMatrix::from_rows(&activated)
            };

            output_batch_rm.push(out_act);
        }

        self.batch_size = input.get_batch_size();
        self.time_step = input.get_time_step();
        self.input_batch_rm = if calculate_gradient { Some(input_rm) } else { None };
        self.inactivated_input_batch_rm = if calculate_gradient && needs_raw_pre_activation_rm { Some(raw_output_batch_rm) } else { None };
        self.output_batch_rm = if calculate_gradient { Some(output_batch_rm.clone()) } else { None };
        self.padding_mask_batch = Some(input.get_padding_mask_batch());

        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(output_batch_rm);
        output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("Input RM batch is missing in dense RM layer");

        let batch_len = input_batch_rm.len();
        if batch_len == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let (out_rows, out_cols) = if let Some(out_rm) = self.output_batch_rm.as_ref().and_then(|v| v.first()) {
            (out_rm.rows, out_rm.cols)
        } else {
            (input_batch_rm[0].rows, self.weights.cols)
        };

        let mut prev_grads: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_len);
        for b in 0..batch_len {
            if let Some(g) = previous_gradient_batch_rm.get(b) {
                prev_grads.push(g.clone());
            } else {
                prev_grads.push(RowMajorMatrix::from_data(out_rows, out_cols, vec![C::new(ZERO, ZERO); out_rows * out_cols]));
            }
        }

        let weights_h_rm = conjugate_transpose_rm(&self.weights);

        let mut gradient = Gradient::new_default();

        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights.cols]; self.weights.rows]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];
        let mut input_gradient_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_len);

        match &self.activation_type {
            ActivationType::SWiGLU => {
                let raw_output_batch_rm = self.inactivated_input_batch_rm.as_ref().expect("Raw output RM batch is missing in dense SWiGLU RM layer");

                let cols = self.weights.cols;
                assert!(cols % 2 == 0, "SWiGLU expects even column count");
                let half = cols / 2;

                let weights_1_rm = split_columns_rm(&self.weights, 0, half);
                let weights_2_rm = split_columns_rm(&self.weights, half, cols);
                let weights_1_h_rm = conjugate_transpose_rm(&weights_1_rm);
                let weights_2_h_rm = conjugate_transpose_rm(&weights_2_rm);

                for batch_ind in 0..batch_len {
                    let input_rm = &input_batch_rm[batch_ind];
                    let grad_y_rm = &prev_grads[batch_ind];

                    let raw_rm = &raw_output_batch_rm[batch_ind];
                    assert_eq!(raw_rm.cols, cols);
                    assert_eq!(grad_y_rm.cols, half, "SWiGLU gradient must match activated output cols");

                    let a_rm = split_columns_rm(raw_rm, 0, half);
                    let b_rm = split_columns_rm(raw_rm, half, cols);

                    let swish_b = swish_rm(&b_rm);
                    let grad_swish_b = swish_grad_rm(&b_rm);

                    let dl_da = hadamard_rm(grad_y_rm, &conjugate_rm(&swish_b));
                    let dl_db_1 = hadamard_rm(grad_y_rm, &conjugate_rm(&a_rm));
                    let dl_db = hadamard_rm(&dl_db_1, &conjugate_rm(&grad_swish_b));

                    let input_h_rm = conjugate_transpose_rm(input_rm);
                    let wgrad1_rm = multiply_complex_rm(&input_h_rm, &dl_da);
                    let wgrad2_rm = multiply_complex_rm(&input_h_rm, &dl_db);

                    accumulate_bias_from_rm(&mut bias_gradients[batch_ind][0..half], &dl_da);
                    accumulate_bias_from_rm(&mut bias_gradients[batch_ind][half..cols], &dl_db);

                    let gx_a = multiply_complex_rm(&dl_da, &weights_1_h_rm);
                    let gx_b = multiply_complex_rm(&dl_db, &weights_2_h_rm);
                    let gx = add_rm(&gx_a, &gx_b);
                    input_gradient_batch_rm.push(gx);

                    let mut rows = wgrad1_rm.to_rows();
                    let rows2 = wgrad2_rm.to_rows();
                    for (r_i, row2) in rows2.into_iter().enumerate() {
                        rows[r_i].extend_from_slice(&row2);
                    }
                    weight_gradients[batch_ind] = rows;
                }
            }
            _ => {
                let output_batch_rm = self.output_batch_rm.as_ref().expect("Output RM batch is missing in dense RM layer");

                let raw_output_batch_rm = if self.activation_type == ActivationType::GELU {
                    Some(self.inactivated_input_batch_rm.as_ref().expect("Raw output RM batch is missing in dense GELU RM layer"))
                } else {
                    None
                };

                for batch_ind in 0..batch_len {
                    let input_rm = &input_batch_rm[batch_ind];
                    let grad_y_rm = &prev_grads[batch_ind];
                    let out_act_rm = &output_batch_rm[batch_ind];

                    let d_act = if self.activation_type == ActivationType::GELU {
                        let raw_rm = &raw_output_batch_rm.unwrap()[batch_ind];
                        activation_derivative_from_raw_rm(raw_rm, &self.activation_type)
                    } else {
                        activation_derivative_rm(out_act_rm, &self.activation_type)
                    };

                    let dz = hadamard_rm(grad_y_rm, &conjugate_rm(&d_act));

                    let input_h_rm = conjugate_transpose_rm(input_rm);
                    let wgrad_rm = multiply_complex_rm(&input_h_rm, &dz);
                    weight_gradients[batch_ind] = wgrad_rm.to_rows();

                    accumulate_bias_from_rm(&mut bias_gradients[batch_ind], &dz);

                    let gx_rm = multiply_complex_rm(&dz, &weights_h_rm);
                    input_gradient_batch_rm.push(gx_rm);
                }
            }
        }

        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let prev_rm_ref = previous_gradient.get_gradient_input_batch_rm_ref().expect("LayerRm::backward expects RM gradients");
        self.backward_rm(prev_rm_ref)
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in dense RM layer");

        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());

        let total_valid_tokens: Real = r(gradient.get_total_valid_tokens().max(1) as f64);

        clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        weight_gradients = average_matrix_by_scalar(&weight_gradients, total_valid_tokens);
        bias_gradients = average_vector_by_scalar(&bias_gradients, total_valid_tokens);

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

        // Update weights by round-tripping through Vec<Vec<C>> to reuse the existing AdamW.
        let mut weights_vec = self.weights.to_rows();
        calculate_adam_w(
            &mut weights_vec,
            &weight_gradients,
            &mut prev_m_weights,
            &mut prev_v_weights,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
        );
        self.weights = RowMajorMatrix::from_rows(&weights_vec);

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);
        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}

fn activate_output_complex_rm(mut data: RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    match activation {
        ActivationType::LINEAR => data,
        ActivationType::TANH => {
            for v in data.data.iter_mut() {
                *v = tanh_complex(*v);
            }
            data
        }
        ActivationType::RELU => {
            for v in data.data.iter_mut() {
                *v = if v.re > ZERO { *v } else { C::new(ZERO, ZERO) };
            }
            data
        }
        ActivationType::LEAKYRELU => {
            for v in data.data.iter_mut() {
                *v = if v.re > ZERO { *v } else { *v * r(0.01) };
            }
            data
        }
        ActivationType::SIGMOID => {
            for v in data.data.iter_mut() {
                *v = sigmoid_complex(v);
            }
            data
        }
        ActivationType::GELU => {
            for v in data.data.iter_mut() {
                *v = C::new(gelu_real(v.re), v.im);
            }
            data
        }
        ActivationType::SWiGLU => {
            let cols = data.cols;
            assert!(cols % 2 == 0, "SWiGLU expects an even column count");
            let half = cols / 2;
            let mut out = RowMajorMatrix::from_data(data.rows, half, vec![C::new(ZERO, ZERO); data.rows * half]);
            for r_i in 0..data.rows {
                let row = data.row_range(r_i);
                let out_row = out.row_range(r_i);
                for c in 0..half {
                    let a = data.data[row.start + c];
                    let b = data.data[row.start + half + c];
                    out.data[out_row.start + c] = a * b * sigmoid_complex(&b);
                }
            }
            out
        }
        _ => data,
    }
}

fn activation_derivative_rm(activated: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(activated.rows, activated.cols, vec![C::new(ZERO, ZERO); activated.rows * activated.cols]);
    match activation {
        ActivationType::LINEAR => {
            for v in out.data.iter_mut() {
                *v = C::new(ONE, ZERO);
            }
        }
        ActivationType::TANH => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = C::new(ONE, ZERO) - (z * z);
            }
        }
        ActivationType::SIGMOID => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = z * (C::new(ONE, ZERO) - z);
            }
        }
        ActivationType::RELU => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = if z.re > ZERO { C::new(ONE, ZERO) } else { C::new(ZERO, ZERO) };
            }
        }
        ActivationType::LEAKYRELU => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = if z.re > ZERO { C::new(ONE, ZERO) } else { C::new(r(0.01), ZERO) };
            }
        }
        _ => {}
    }
    out
}

fn activation_derivative_from_raw_rm(raw: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(raw.rows, raw.cols, vec![C::new(ZERO, ZERO); raw.rows * raw.cols]);
    match activation {
        ActivationType::GELU => {
            for (dst, &z) in out.data.iter_mut().zip(raw.data.iter()) {
                *dst = C::new(gelu_derivative_real(z.re), ZERO);
            }
        }
        _ => {}
    }
    out
}

fn gelu_real(x: Real) -> Real {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let xf = x as f64;
    let x3 = xf * xf * xf;
    let inner = SQRT_2_OVER_PI * (xf + 0.044_715 * x3);
    r(0.5 * xf * (1.0 + inner.tanh()))
}

fn gelu_derivative_real(x: Real) -> Real {
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let xf = x as f64;
    let x2 = xf * xf;
    let inner = SQRT_2_OVER_PI * (xf + 0.044_715 * (xf * xf * xf));
    let t = inner.tanh();
    let sech2 = 1.0 - t * t;
    let inner_prime = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044_715 * x2);
    r(0.5 * (1.0 + t) + 0.5 * xf * sech2 * inner_prime)
}

fn conjugate_rm(m: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(m.rows, m.cols, vec![C::new(ZERO, ZERO); m.rows * m.cols]);
    for (dst, &v) in out.data.iter_mut().zip(m.data.iter()) {
        *dst = v.conj();
    }
    out
}

fn hadamard_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = RowMajorMatrix::from_data(a.rows, a.cols, vec![C::new(ZERO, ZERO); a.rows * a.cols]);
    for i in 0..out.data.len() {
        out.data[i] = a.data[i] * b.data[i];
    }
    out
}

fn add_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = RowMajorMatrix::from_data(a.rows, a.cols, vec![C::new(ZERO, ZERO); a.rows * a.cols]);
    for i in 0..out.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }
    out
}

fn split_columns_rm(matrix: &RowMajorMatrix<C>, start_col: usize, end_col: usize) -> RowMajorMatrix<C> {
    assert!(start_col <= end_col);
    assert!(end_col <= matrix.cols);
    let cols = end_col - start_col;
    let mut data = vec![C::new(ZERO, ZERO); matrix.rows * cols];
    for r_i in 0..matrix.rows {
        let src_row = matrix.row_range(r_i);
        let dst_row_start = r_i * cols;
        let src_start = src_row.start + start_col;
        let src_end = src_row.start + end_col;
        data[dst_row_start..dst_row_start + cols].copy_from_slice(&matrix.data[src_start..src_end]);
    }
    RowMajorMatrix::from_data(matrix.rows, cols, data)
}

fn swish_rm(b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(b.rows, b.cols, vec![C::new(ZERO, ZERO); b.rows * b.cols]);
    for i in 0..out.data.len() {
        let z = b.data[i];
        out.data[i] = z * sigmoid_complex(&z);
    }
    out
}

fn swish_grad_rm(b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(b.rows, b.cols, vec![C::new(ZERO, ZERO); b.rows * b.cols]);
    let one = C::new(ONE, ZERO);
    for i in 0..out.data.len() {
        let z = b.data[i];
        let sigma = sigmoid_complex(&z);
        out.data[i] = sigma + z * sigma * (one - sigma);
    }
    out
}

fn accumulate_bias_from_rm(dst: &mut [C], grad: &RowMajorMatrix<C>) {
    assert_eq!(dst.len(), grad.cols);
    for r_i in 0..grad.rows {
        let row = grad.row_range(r_i);
        for c in 0..grad.cols {
            dst[c] += grad.data[row.start + c];
        }
    }
}

fn sigmoid_complex(z: &C) -> C {
    let one = C::new(ONE, ZERO);
    one / (one + (-*z).exp())
}

fn tanh_complex(z: C) -> C {
    z.tanh()
}

impl LayerInterface for LayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        LayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        LayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        LayerRm::update_parameters(self)
    }
}
