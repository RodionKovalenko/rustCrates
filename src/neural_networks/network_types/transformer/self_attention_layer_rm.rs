use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{default_layer::LayerInterface, layer::LayerType, network_layers_rm::norm_layer_rm::NormalNormLayerRm},
    utils::{
        dtype::{r, Real, C},
        matrix::RowMajorMatrix,
    },
};

use super::{
    attention_rm_common::{residual_attention_backward_rm, residual_attention_forward_rm},
    masked_attention_head_rm::MaskedAttentionHeadRm,
    transformer_updater::calculate_alpha,
};

/// RM counterpart of `SelfAttentionLayer`: norm -> causal attention heads -> concat -> beta * out + input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayerRm {
    pub attention_heads: Vec<MaskedAttentionHeadRm>,
    pub norm_layer: NormalNormLayerRm,
    pub alpha: Real,
    pub beta: Real,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
}

impl SelfAttentionLayerRm {
    pub fn new(num_heads: usize, rows: usize, cols: usize, learning_rate: f64) -> Self {
        let head_cols = cols / num_heads;
        let attention_heads = (0..num_heads).map(|_| MaskedAttentionHeadRm::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer, learning_rate)).collect();

        let alpha = calculate_alpha();
        let beta = r(1.0) / alpha;

        Self {
            attention_heads,
            norm_layer: NormalNormLayerRm::new(cols, 1e-12, learning_rate),
            alpha,
            beta,
            input_batch_rm: None,
            output_batch_rm: None,
            gradient: None,
            time_step: 0,
        }
    }

    /// RM counterpart of `SelfAttentionLayer::forward`.
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let output = residual_attention_forward_rm(&mut self.attention_heads, &mut self.norm_layer, layer_input, self.beta);

        self.input_batch_rm = Some(layer_input.get_input_batch_rm());
        self.output_batch_rm = Some(output.clone());
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output);
        layer_output.set_padding_mask_batch(layer_input.get_padding_mask_batch());
        layer_output
    }

    /// RM counterpart of `SelfAttentionLayer::backward`.
    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut out = Gradient::new_default();
        out.set_gradient_input_batch_rm(residual_attention_backward_rm(&mut self.attention_heads, &mut self.norm_layer, previous_gradient_batch_rm, self.beta));
        self.gradient = Some(out.clone());
        out
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        self.backward_rm(&previous_gradient.get_gradient_input_batch_rm_cow())
    }

    pub fn update_parameters(&mut self) {
        self.norm_layer.update_parameters();
        self.attention_heads.par_iter_mut().for_each(|head| head.update_parameters());
    }

    pub fn clear_cache(&mut self) {
        self.attention_heads.iter_mut().for_each(|head| head.clear_cache());
    }
}

impl LayerInterface for SelfAttentionLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        SelfAttentionLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        SelfAttentionLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        SelfAttentionLayerRm::update_parameters(self)
    }
}
