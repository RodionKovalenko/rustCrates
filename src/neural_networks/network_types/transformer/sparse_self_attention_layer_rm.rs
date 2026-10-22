use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{network_layers_rm::norm_layer_rm::NormalNormLayerRm},
    utils::{
        dtype::{r, C, Real, ZERO},
        matrix::RowMajorMatrix,
        matrix_rm::{add_assign_rm, concat_heads_rm, scale_rm_in_place, split_heads_rm, sum_heads_rm, tail_rows_rm},
    },
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    sparse_masked_attention_head_rm::SparseMaskedAttentionHeadRm,
    transformer_updater::calculate_alpha,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSelfAttentionLayerRm {
    pub attention_heads: Vec<SparseMaskedAttentionHeadRm>,
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

impl SparseSelfAttentionLayerRm {
    pub fn new(num_heads: usize, rows: usize, cols: usize, window_size: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<SparseMaskedAttentionHeadRm> = vec![];
        let head_cols = cols / num_heads;
        for _ in 0..num_heads {
            attention_heads.push(SparseMaskedAttentionHeadRm::new(rows, head_cols, window_size, learning_rate));
        }

        let epsilon: f64 = 1e-12;
        let norm_layer = NormalNormLayerRm::new(cols, epsilon, learning_rate);

        let alpha = calculate_alpha();
        let beta = r(1.0) / alpha;

        Self {
            attention_heads,
            norm_layer,
            alpha,
            beta,
            input_batch_rm: None,
            output_batch_rm: None,
            gradient: None,
            time_step: 0,
        }
    }

    /// RM counterpart of `SparseSelfAttentionLayer::forward`: norm -> heads -> concat -> beta * out + input.
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm = layer_input.get_input_batch_rm();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

        let mut local_input = layer_input.clone();
        local_input.clear_input_batch();
        local_input.set_input_batch_rm(input_batch_rm.clone());
        local_input.set_padding_mask_batch(padding_mask_batch.clone());

        let normed = self.norm_layer.forward(&local_input).take_output_batch_rm().expect("NormRm did not return RM output");
        local_input.set_input_batch_rm(normed);

        let head_outputs: Vec<Vec<RowMajorMatrix<C>>> = self
            .attention_heads
            .par_iter_mut()
            .map(|head| head.forward(&local_input).take_output_batch_rm().expect("SparseMaskedAttentionHeadRm did not return RM output"))
            .collect();

        // Residual: beta * out + input (in forward-only mode the output may only cover the last rows)
        let mut output = concat_heads_rm(&head_outputs);
        for (out, input) in output.iter_mut().zip(input_batch_rm.iter()) {
            scale_rm_in_place(out, C::new(self.beta, ZERO));
            if layer_input.get_forward_only() {
                add_assign_rm(out, &tail_rows_rm(input, out.rows));
            } else {
                add_assign_rm(out, input);
            }
        }

        self.input_batch_rm = Some(input_batch_rm);
        self.output_batch_rm = Some(output.clone());
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output);
        layer_output.set_padding_mask_batch(padding_mask_batch);
        layer_output
    }

    /// RM counterpart of `SparseSelfAttentionLayer::backward`.
    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut scaled_upstream: Vec<RowMajorMatrix<C>> = previous_gradient_batch_rm.to_vec();
        scaled_upstream.iter_mut().for_each(|m| scale_rm_in_place(m, C::new(self.beta, ZERO)));

        let head_gradients = split_heads_rm(&scaled_upstream, self.attention_heads.len());
        let head_input_gradients: Vec<Vec<RowMajorMatrix<C>>> = self
            .attention_heads
            .par_iter_mut()
            .zip(head_gradients.par_iter())
            .map(|(head, gradient)| head.backward_rm(gradient).get_gradient_input_batch_rm())
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(sum_heads_rm(head_input_gradients));
        let mut combined = self.norm_layer.backward(&gradient).get_gradient_input_batch_rm();

        // Residual: add unscaled upstream
        for (c, u) in combined.iter_mut().zip(previous_gradient_batch_rm.iter()) {
            add_assign_rm(c, u);
        }

        let mut out = Gradient::new_default();
        out.set_gradient_input_batch_rm(combined);
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
        self.attention_heads.par_iter_mut().for_each(|head| head.clear_cache());
    }
}
