//! Multi-head residual attention shared by the RM self-attention layers:
//! `out = beta * concat(heads(Norm(x))) + x`.

use rayon::prelude::*;

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::network_layers_rm::norm_layer_rm::NormalNormLayerRm,
    utils::{
        dtype::{Real, C, ZERO},
        matrix::RowMajorMatrix,
        matrix_rm::{add_assign_rm, concat_heads_rm, scale_rm_in_place, split_heads_rm, sum_heads_rm, tail_rows_rm},
    },
};

/// An attention head working on row-major batches.
pub trait AttentionHeadRm: Send {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput;
    fn backward_rm(&mut self, previous_gradient_batch: &[RowMajorMatrix<C>]) -> Gradient;
    fn update_parameters(&mut self);
}

/// Forward: Norm -> heads (in parallel) -> concat -> beta * out + input.
/// In forward-only mode the heads may only return the last rows; the residual is aligned to them.
pub fn residual_attention_forward_rm<H: AttentionHeadRm>(heads: &mut [H], norm_layer: &mut NormalNormLayerRm, layer_input: &LayerInput, beta: Real) -> Vec<RowMajorMatrix<C>> {
    let input_batch = layer_input.get_input_batch_rm();

    let mut local_input = layer_input.clone();
    local_input.clear_input_batch();
    local_input.set_input_batch_rm(input_batch.clone());

    let normed = norm_layer.forward(&local_input).take_output_batch_rm().expect("NormRm did not return RM output");
    local_input.set_input_batch_rm(normed);

    let head_outputs: Vec<Vec<RowMajorMatrix<C>>> = heads
        .par_iter_mut()
        .map(|head| head.forward(&local_input).take_output_batch_rm().expect("attention head did not return RM output"))
        .collect();

    let mut output = concat_heads_rm(&head_outputs);
    for (out, input) in output.iter_mut().zip(input_batch.iter()) {
        scale_rm_in_place(out, C::new(beta, ZERO));
        if layer_input.get_forward_only() {
            add_assign_rm(out, &tail_rows_rm(input, out.rows));
        } else {
            add_assign_rm(out, input);
        }
    }
    output
}

/// Backward of `residual_attention_forward_rm`: beta * upstream -> heads -> sum -> Norm, plus the residual.
pub fn residual_attention_backward_rm<H: AttentionHeadRm>(heads: &mut [H], norm_layer: &mut NormalNormLayerRm, previous_gradient_batch: &[RowMajorMatrix<C>], beta: Real) -> Vec<RowMajorMatrix<C>> {
    let mut scaled_upstream: Vec<RowMajorMatrix<C>> = previous_gradient_batch.to_vec();
    scaled_upstream.iter_mut().for_each(|m| scale_rm_in_place(m, C::new(beta, ZERO)));

    let head_gradients = split_heads_rm(&scaled_upstream, heads.len());
    let head_input_gradients: Vec<Vec<RowMajorMatrix<C>>> = heads.par_iter_mut().zip(head_gradients.par_iter()).map(|(head, gradient)| head.backward_rm(gradient).get_gradient_input_batch_rm()).collect();

    let mut gradient = Gradient::new_default();
    gradient.set_gradient_input_batch_rm(sum_heads_rm(head_input_gradients));
    let mut combined = norm_layer.backward(&gradient).get_gradient_input_batch_rm();

    // Residual: add unscaled upstream
    for (c, u) in combined.iter_mut().zip(previous_gradient_batch.iter()) {
        add_assign_rm(c, u);
    }
    combined
}
