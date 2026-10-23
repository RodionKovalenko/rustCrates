use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{adaptive_linear_layer::AdaptiveLinearLayer, default_layer::LayerInterface},
};

/// RM counterpart of `AdaptiveLinearLayer` (hierarchical softmax output head).
///
/// The parameters, the optimizer and the row-major forward/backward live in `AdaptiveLinearLayer`;
/// this layer only takes RM input and returns RM logits / input gradients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLinearLayerRm {
    pub layer: AdaptiveLinearLayer,
}

impl AdaptiveLinearLayerRm {
    pub fn new(learning_rate: f64, hidden_size: usize, vocab_size: usize) -> Self {
        Self { layer: AdaptiveLinearLayer::new(learning_rate, hidden_size, vocab_size) }
    }

    /// Training forward also computes the CE loss and stores the gradient (see `AdaptiveLinearLayer::forward`).
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        self.layer.forward_rm(input.get_input_batch_rm(), input)
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        self.layer.backward_rm(&previous_gradient.get_gradient_input_batch_rm_cow(), previous_gradient.get_total_valid_tokens())
    }

    pub fn update_parameters(&mut self) {
        self.layer.update_parameters()
    }

    /// Gradient stored by the training forward pass.
    pub fn gradient(&self) -> Option<&Gradient> {
        self.layer.gradient.as_ref()
    }
}

impl LayerInterface for AdaptiveLinearLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        AdaptiveLinearLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        AdaptiveLinearLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        AdaptiveLinearLayerRm::update_parameters(self)
    }
}
