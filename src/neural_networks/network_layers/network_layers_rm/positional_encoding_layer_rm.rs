use crate::neural_networks::network_components::gradient_struct::Gradient;
use crate::neural_networks::network_components::layer_input_struct::LayerInput;
use crate::neural_networks::network_components::layer_output_struct::LayerOutput;
use crate::neural_networks::network_layers::default_layer::LayerInterface;
use crate::neural_networks::utils::dtype::C;
use crate::neural_networks::utils::matrix::RowMajorMatrix;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_layers::positional_encoding_layer::{rope_rotate, rope_rotate_backward, INITIAL_BASE, SCALING_FAKTOR};

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

    /// RM counterpart of `PositionalEncodingLayer::forward`.
    pub fn forward_inner(&mut self, layer_input: &LayerInput) -> Vec<RowMajorMatrix<C>> {
        let input_batch_rm = layer_input.get_input_batch_rm();

        let forward_only = layer_input.get_forward_only();
        let time_step = layer_input.get_time_step();
        let base = self.base * SCALING_FAKTOR;

        let output_batch_rm = input_batch_rm
            .par_iter()
            .map(|m| {
                assert_eq!(m.cols, self.embedding_dim, "All token embeddings must match the specified dimension.");
                let mut out = RowMajorMatrix::create_empty(m.rows, m.cols);
                for position in 0..m.rows {
                    let position_t = if forward_only && time_step > 0 { time_step } else { position };
                    let (row, out_row) = (m.row_range(position), out.row_range(position));
                    rope_rotate(&m.data[row], &mut out.data[out_row], position_t, base);
                }
                out
            })
            .collect();

        self.input_batch_rm = layer_input.get_calculate_gradient().then_some(input_batch_rm);
        output_batch_rm
    }

    /// RM counterpart of `PositionalEncodingLayer::backward`.
    pub fn backward_inner(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let base = self.base * SCALING_FAKTOR;

        let input_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient_batch_rm
            .par_iter()
            .map(|g| {
                assert_eq!(g.cols, self.embedding_dim);
                let mut out = RowMajorMatrix::create_empty(g.rows, g.cols);
                for position in 0..g.rows {
                    let (row, out_row) = (g.row_range(position), out.row_range(position));
                    rope_rotate_backward(&g.data[row], &mut out.data[out_row], position, base);
                }
                out
            })
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let enc_rm = PositionalEncodingLayerRm::forward_inner(self, layer_input);
        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(enc_rm);
        output.set_padding_mask_batch(layer_input.get_padding_mask_batch());
        output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        PositionalEncodingLayerRm::backward_inner(self, &previous_gradient.get_gradient_input_batch_rm_cow())
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
