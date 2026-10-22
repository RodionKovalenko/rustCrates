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
        let output_batch_rm = self.rotate_batch(&input_batch_rm, layer_input.get_forward_only(), layer_input.get_time_step());
        self.input_batch_rm = layer_input.get_calculate_gradient().then_some(input_batch_rm);
        output_batch_rm
    }

    /// RM counterpart of `PositionalEncodingLayer::backward`.
    pub fn backward_inner(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(self.rotate_batch_backward(previous_gradient_batch_rm));
        self.gradient = Some(gradient.clone());
        gradient
    }

    /// Applies RoPE to every row (RM counterpart of `apply_rope_to_sequence`). In forward-only mode with
    /// `time_step > 0` all rows use the angle of `time_step`, otherwise the angle of their row index.
    pub fn rotate_batch(&self, batch: &[RowMajorMatrix<C>], forward_only: bool, time_step: usize) -> Vec<RowMajorMatrix<C>> {
        self.rotate_batch_at(batch, |position| if forward_only && time_step > 0 { time_step } else { position })
    }

    /// Applies RoPE with explicit positions: row `i` is rotated by the angle of `position(i)`.
    pub fn rotate_batch_at(&self, batch: &[RowMajorMatrix<C>], position: impl Fn(usize) -> usize + Sync) -> Vec<RowMajorMatrix<C>> {
        let base = self.base * SCALING_FAKTOR;
        batch
            .par_iter()
            .map(|m| {
                assert_eq!(m.cols, self.embedding_dim, "All token embeddings must match the specified dimension.");
                let mut out = RowMajorMatrix::create_empty(m.rows, m.cols);
                for row in 0..m.rows {
                    let (src, dst) = (m.row_range(row), out.row_range(row));
                    rope_rotate(&m.data[src], &mut out.data[dst], position(row), base);
                }
                out
            })
            .collect()
    }

    /// Backward of `rotate_batch` (always by row index, like `PositionalEncodingLayer::backward_sequences`).
    pub fn rotate_batch_backward(&self, gradient_batch: &[RowMajorMatrix<C>]) -> Vec<RowMajorMatrix<C>> {
        let base = self.base * SCALING_FAKTOR;
        gradient_batch
            .par_iter()
            .map(|g| {
                assert_eq!(g.cols, self.embedding_dim);
                let mut out = RowMajorMatrix::create_empty(g.rows, g.cols);
                for row in 0..g.rows {
                    let (src, dst) = (g.row_range(row), out.row_range(row));
                    rope_rotate_backward(&g.data[src], &mut out.data[dst], row, base);
                }
                out
            })
            .collect()
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
