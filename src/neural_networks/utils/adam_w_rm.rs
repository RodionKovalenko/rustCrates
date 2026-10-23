use crate::neural_networks::network_components::gradient_struct::Gradient;
use crate::neural_networks::utils::adam_w::{adam_w_step, calculate_adam_w_bias, get_current_learning_rate};
use crate::neural_networks::utils::dtype::{r, Real, C, ZERO};
use crate::neural_networks::utils::matrix_rm::RowMajorMatrix;
use serde::{Deserialize, Serialize};

/// AdamW (AMSGrad) moment state of one row-major weight matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWStateRm {
    pub m: RowMajorMatrix<C>,
    pub v: RowMajorMatrix<C>,
    pub v_hat: RowMajorMatrix<C>,
}

impl AdamWStateRm {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            m: RowMajorMatrix::create_empty(rows, cols),
            v: RowMajorMatrix::create_empty(rows, cols),
            v_hat: RowMajorMatrix::create_empty(rows, cols),
        }
    }

    fn has_shape(&self, rows: usize, cols: usize) -> bool {
        (self.m.rows, self.m.cols) == (rows, cols)
    }
}

// RM counterpart of `calculate_adam_w`: all matrices share the same row-major layout.
pub fn calculate_adam_w_rm(weights: &mut RowMajorMatrix<C>, weight_gradients: &RowMajorMatrix<C>, state: &mut AdamWStateRm, learning_rate: f64, t: usize) {
    if weight_gradients.data.is_empty() {
        return;
    }

    let shape = (weights.rows, weights.cols);
    if (weight_gradients.rows, weight_gradients.cols) != shape || !state.has_shape(shape.0, shape.1) {
        panic!("Shape mismatch in adam w rm weights: weights {:?}, gradients {:?}", shape, (weight_gradients.rows, weight_gradients.cols));
    }

    let t = t.max(1) as i32;
    let current_lr: Real = r(get_current_learning_rate(learning_rate, t as usize));

    for i in 0..weights.data.len() {
        adam_w_step(&mut weights.data[i], weight_gradients.data[i], &mut state.m.data[i], &mut state.v.data[i], &mut state.v_hat.data[i], current_lr, t);
    }
}

/// Runs AdamW on the RM weight matrix `name`: the moment state is read from `previous_gradient`
/// (zeros on the first step) and the updated state is stored in `gradient`.
pub fn update_weights_adam_w_rm(
    name: &str,
    weights: &mut RowMajorMatrix<C>,
    weight_gradients: &RowMajorMatrix<C>,
    previous_gradient: Option<&Gradient>,
    gradient: &mut Gradient,
    learning_rate: f64,
    time_step: usize,
) {
    let mut state = previous_gradient
        .and_then(|g| g.get_adam_w_state_rm(name))
        .filter(|s| s.has_shape(weights.rows, weights.cols))
        .cloned()
        .unwrap_or_else(|| AdamWStateRm::zeros(weights.rows, weights.cols));

    calculate_adam_w_rm(weights, weight_gradients, &mut state, learning_rate, time_step);

    gradient.set_adam_w_state_rm(name, state);
    gradient.set_weight_gradient_rm(name, weight_gradients.clone());
}

/// Runs AdamW on the bias vector: the moment state is read from `previous_gradient`
/// (zeros on the first step) and the updated state is stored in `gradient`.
pub fn update_bias_adam_w(bias: &mut Vec<C>, bias_gradients: &Vec<C>, previous_gradient: Option<&Gradient>, gradient: &mut Gradient, learning_rate: f64, time_step: usize) {
    let (mut prev_m_bias, mut prev_v_bias, mut prev_v_bias_hat) = match previous_gradient {
        Some(previous_gradient) => (previous_gradient.get_prev_m_bias(), previous_gradient.get_prev_v_bias(), previous_gradient.get_prev_v_bias_hat()),
        None => (vec![C::new(ZERO, ZERO); bias.len()], vec![C::new(ZERO, ZERO); bias.len()], vec![C::new(ZERO, ZERO); bias.len()]),
    };

    calculate_adam_w_bias(bias, bias_gradients, &mut prev_m_bias, &mut prev_v_bias, &mut prev_v_bias_hat, learning_rate, time_step);

    gradient.set_prev_m_bias(prev_m_bias);
    gradient.set_prev_v_bias(prev_v_bias);
    gradient.set_prev_v_bias_hat(prev_v_bias_hat);
    gradient.set_gradient_bias(bias_gradients.clone());
}
