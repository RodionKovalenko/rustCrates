use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{default_layer::LayerInterface, eml_linear_layer::EmlLinearLayer},
    utils::{
        dtype::{Real, C, ZERO},
        matrix::RowMajorMatrix,
    },
};

/// RM counterpart of `EmlLinearLayer` (hierarchical output head with teacher forcing).
///
/// The parameters, the optimizer and the row-major forward live in `EmlLinearLayer`; this layer only
/// takes RM input and returns RM scores / input gradients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmlLinearLayerRm {
    pub layer: EmlLinearLayer,
}

impl EmlLinearLayerRm {
    pub fn new(learning_rate: f64, d_model: usize, vocab_size: usize) -> Self {
        Self { layer: EmlLinearLayer::new(learning_rate, d_model, vocab_size) }
    }

    /// Training: CE loss + stored RM gradient. Inference: RM top-k scores (rows shorter than k are padded
    /// with -inf / index 0) and their vocab indices.
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let mut output = self.layer.forward_rm(&input.get_input_batch_rm(), input);

        let scores = output.take_output_batch().unwrap_or_default();
        if !scores.is_empty() {
            let top_k = input.get_top_k_size().max(1);
            let mut indices = output.get_output_indices();
            let scores_rm: Vec<RowMajorMatrix<C>> = scores
                .iter()
                .zip(indices.iter_mut())
                .map(|(seq_scores, seq_indices)| {
                    let mut data = Vec::with_capacity(seq_scores.len() * top_k);
                    for (row, row_indices) in seq_scores.iter().zip(seq_indices.iter_mut()) {
                        data.extend(row.iter().copied().chain(std::iter::repeat(C::new(Real::NEG_INFINITY, ZERO))).take(top_k));
                        row_indices.resize(top_k, 0);
                    }
                    RowMajorMatrix::from_data(seq_scores.len(), top_k, data)
                })
                .collect();
            output.set_output_batch_rm(scores_rm);
            output.set_output_indices(indices);
        }
        output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        self.layer.backward(previous_gradient)
    }

    pub fn update_parameters(&mut self) {
        self.layer.update_parameters()
    }

    /// Gradient stored by the training forward pass.
    pub fn gradient(&self) -> Option<&Gradient> {
        self.layer.gradient.as_ref()
    }
}

impl LayerInterface for EmlLinearLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        EmlLinearLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        EmlLinearLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        EmlLinearLayerRm::update_parameters(self)
    }
}
