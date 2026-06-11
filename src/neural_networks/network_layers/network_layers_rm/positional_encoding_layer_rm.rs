use crate::neural_networks::network_components::gradient_struct::Gradient;
use crate::neural_networks::network_components::layer_input_struct::LayerInput;
use crate::neural_networks::network_components::layer_output_struct::LayerOutput;
use crate::neural_networks::network_layers::default_layer::LayerInterface;
use crate::neural_networks::utils::dtype::{r, Real, C, ZERO};
use crate::neural_networks::utils::matrix::RowMajorMatrix;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_layers::positional_encoding_layer::{INITIAL_BASE, SCALING_FAKTOR};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionalEncodingLayerRm {
    pub embedding_dim: usize,
    pub base: f64,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
}

impl PositionalEncodingLayerRm {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            base: INITIAL_BASE,
            gradient: None,
            input_batch_rm: None,
        }
    }

    pub fn forward_inner(&mut self, layer_input: &LayerInput) -> Vec<RowMajorMatrix<C>> {
        let input_batch_rm = layer_input.get_input_batch_rm_ref().expect("PositionalEncodingLayerRm::forward expects RM input_batch_rm");

        let scaling_factor = SCALING_FAKTOR;
        let forward_only = layer_input.get_forward_only();

        if layer_input.get_calculate_gradient() {
            self.input_batch_rm = Some(input_batch_rm.to_vec());
        } else {
            self.input_batch_rm = None;
        }

        input_batch_rm
            .par_iter()
            .map(|m| {
                assert_eq!(m.cols, self.embedding_dim, "All token embeddings must match the specified dimension.");
                assert_eq!(self.embedding_dim % 2, 0, "Embedding dimension must be even for RoPE.");

                let mut out = RowMajorMatrix::from_data(m.rows, m.cols, vec![C::new(ZERO, ZERO); m.rows * m.cols]);

                for position in 0..m.rows {
                    let time_step = if forward_only && layer_input.get_time_step() > 0 { layer_input.get_time_step() } else { position };

                    let row = m.row_range(position);
                    let out_row = out.row_range(position);
                    let half_dim = self.embedding_dim / 2;

                    for i in 0..half_dim {
                        let even_idx = 2 * i;
                        let odd_idx = even_idx + 1;

                        let mut theta = time_step as f64 / ((self.base * scaling_factor).powf(2.0 * i as f64 / self.embedding_dim as f64));
                        theta = theta.clamp(-1.0, 1.0);

                        let (sin_theta_f64, cos_theta_f64) = theta.sin_cos();
                        let sin_theta: Real = r(sin_theta_f64);
                        let cos_theta: Real = r(cos_theta_f64);

                        let even = m.data[row.start + even_idx];
                        let odd = m.data[row.start + odd_idx];

                        out.data[out_row.start + even_idx] = C::new(even.re * cos_theta - odd.re * sin_theta, even.im * cos_theta - odd.im * sin_theta);
                        out.data[out_row.start + odd_idx] = C::new(even.re * sin_theta + odd.re * cos_theta, even.im * sin_theta + odd.im * cos_theta);
                    }
                }

                out
            })
            .collect()
    }

    pub fn backward_inner(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut gradient = Gradient::new_default();

        assert_eq!(self.embedding_dim % 2, 0, "Embedding dimension must be even for RoPE.");
        let half_dim = self.embedding_dim / 2;

        let input_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient_batch_rm
            .par_iter()
            .map(|g| {
                assert_eq!(g.cols, self.embedding_dim);
                let mut out = RowMajorMatrix::from_data(g.rows, g.cols, vec![C::new(ZERO, ZERO); g.rows * g.cols]);

                for position in 0..g.rows {
                    let row = g.row_range(position);
                    let out_row = out.row_range(position);

                    for i in 0..half_dim {
                        let even_idx = 2 * i;
                        let odd_idx = even_idx + 1;

                        let mut theta = position as f64 / ((self.base * SCALING_FAKTOR).powf(2.0 * i as f64 / self.embedding_dim as f64));
                        theta = theta.clamp(-1.0, 1.0);

                        let (sin_theta_f64, cos_theta_f64) = theta.sin_cos();
                        let sin_theta: Real = r(sin_theta_f64);
                        let cos_theta: Real = r(cos_theta_f64);

                        let grad_even = g.data[row.start + even_idx];
                        let grad_odd = g.data[row.start + odd_idx];

                        out.data[out_row.start + even_idx] = C::new(grad_even.re * cos_theta + grad_odd.re * sin_theta, grad_even.im * cos_theta + grad_odd.im * sin_theta);
                        out.data[out_row.start + odd_idx] = C::new(-grad_even.re * sin_theta + grad_odd.re * cos_theta, -grad_even.im * sin_theta + grad_odd.im * cos_theta);
                    }
                }

                out
            })
            .collect();

        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let enc_rm = PositionalEncodingLayerRm::forward_inner(self, layer_input);
        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(enc_rm);
        output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let gr_rm = previous_gradient
            .get_gradient_input_batch_rm_ref()
            .map(|r| r.to_vec())
            .unwrap_or_else(|| previous_gradient.get_gradient_input_batch_rm());
        PositionalEncodingLayerRm::backward_inner(self, &gr_rm)
    }

    pub fn update_parameters(&mut self) {}
}

impl LayerInterface for PositionalEncodingLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        PositionalEncodingLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        PositionalEncodingLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        PositionalEncodingLayerRm::update_parameters(self)
    }
}
