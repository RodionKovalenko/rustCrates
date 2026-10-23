//! Shared helpers for the tests that compare a `*Rm` layer with its `Vec<Vec<C>>` counterpart.
#![cfg(test)]

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
    network_layers::{
        complex_to_linear_layer::ComplexToLinearLayer,
        default_layer::LayerInterface,
        layer::Layer,
        linear_layer::LinearLayer,
        network_layers_rm::{complex_to_linear_layer_rm::ComplexToLinearLayerRm, layer_rm::LayerRm, linear_layer_rm::LinearLayerRm, norm_layer_rm::NormalNormLayerRm},
        norm_layer::NormalNormLayer,
    },
    utils::{
        dtype::{Real, C},
        matrix_rm::RowMajorMatrix,
        random_arrays::{generate_random_complex_2d, generate_random_complex_3d},
    },
};

/// Relative tolerance for Vec vs RM comparisons. Both paths run the same arithmetic,
/// differences only come from a different summation order.
pub const TOLERANCE: f64 = 1e-10;

/// Builds an RM layer with exactly the parameters of `layer` (same serialized fields, same order).
/// The round trip must be byte-identical, so a field that differs between the two structs fails here.
pub fn copy_parameters<A: serde::Serialize, B: serde::Serialize + serde::de::DeserializeOwned>(layer: &A) -> B {
    let bytes = bincode::serialize(layer).expect("serialize layer");
    let layer_rm: B = bincode::deserialize(&bytes).expect("RM layer must have the same serialized parameters");
    assert_eq!(bincode::serialize(&layer_rm).expect("serialize RM layer"), bytes, "parameter layout differs between layer and RM layer");
    layer_rm
}

pub fn to_rm_batch(batch: &Vec<Vec<Vec<C>>>) -> Vec<RowMajorMatrix<C>> {
    batch.iter().map(|m| RowMajorMatrix::from_rows(m)).collect()
}

/// Two `LayerInput`s with the same content: one with a Vec batch, one with an RM batch.
pub fn layer_inputs(input_batch: &Vec<Vec<Vec<C>>>) -> (LayerInput, LayerInput) {
    let padding_mask_batch = vec![vec![1; input_batch[0].len()]; input_batch.len()];
    layer_inputs_with_mask(input_batch, &padding_mask_batch)
}

pub fn layer_inputs_with_mask(input_batch: &Vec<Vec<Vec<C>>>, padding_mask_batch: &Vec<Vec<u32>>) -> (LayerInput, LayerInput) {
    let mut input = LayerInput::new_default();
    input.set_input_batch(input_batch.clone());
    input.set_padding_mask_batch(padding_mask_batch.clone());
    input.set_batch_size(input_batch.len());

    let mut input_rm = LayerInput::new_default();
    input_rm.set_input_batch_rm(to_rm_batch(input_batch));
    input_rm.set_padding_mask_batch(padding_mask_batch.clone());
    input_rm.set_batch_size(input_batch.len());

    (input, input_rm)
}

/// Two upstream `Gradient`s with the same content: one Vec, one RM.
pub fn upstream_gradients(gradient_batch: &Vec<Vec<Vec<C>>>) -> (Gradient, Gradient) {
    let mut gradient = Gradient::new_default();
    gradient.set_gradient_input_batch(gradient_batch.clone());

    let mut gradient_rm = Gradient::new_default();
    gradient_rm.set_gradient_input_batch_rm(to_rm_batch(gradient_batch));

    (gradient, gradient_rm)
}

pub fn assert_close_1d(name: &str, expected: &[C], actual: &[C]) {
    assert_eq!(expected.len(), actual.len(), "{}: length mismatch", name);
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).norm();
        let scale = e.norm().max(a.norm()).max(1.0);
        assert!(diff / scale <= TOLERANCE, "{}: mismatch at {}: expected {:?}, actual {:?}", name, i, e, a);
    }
}

pub fn assert_close_2d(name: &str, expected: &Vec<Vec<C>>, actual: &RowMajorMatrix<C>) {
    assert_eq!(expected.len(), actual.rows, "{}: row count mismatch", name);
    if expected.is_empty() {
        return;
    }
    assert_eq!(expected[0].len(), actual.cols, "{}: col count mismatch", name);
    let flat: Vec<C> = expected.iter().flatten().copied().collect();
    assert_close_1d(name, &flat, &actual.data);
}

pub fn assert_close_3d(name: &str, expected: &Vec<Vec<Vec<C>>>, actual: &[RowMajorMatrix<C>]) {
    assert_eq!(expected.len(), actual.len(), "{}: batch size mismatch", name);
    for (batch_ind, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_close_2d(&format!("{} [batch {}]", name, batch_ind), e, a);
    }
}

pub fn assert_close_real_2d(name: &str, expected: &Vec<Vec<Real>>, actual: &Vec<Vec<Real>>) {
    assert_eq!(expected.len(), actual.len(), "{}: row count mismatch", name);
    for (row, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(e.len(), a.len(), "{}: col count mismatch in row {}", name, row);
        for (col, (e, a)) in e.iter().zip(a.iter()).enumerate() {
            let scale = e.abs().max(a.abs()).max(1.0);
            assert!((e - a).abs() / scale <= TOLERANCE as Real, "{}: mismatch at ({}, {}): expected {:?}, actual {:?}", name, row, col, e, a);
        }
    }
}

/// Both layers must have stored a gradient after backward (or neither of them).
pub fn stored_gradients<'a>(name: &str, gradient: Option<&'a Gradient>, gradient_rm: Option<&'a Gradient>) -> Option<(&'a Gradient, &'a Gradient)> {
    match (gradient, gradient_rm) {
        (Some(g), Some(g_rm)) => Some((g, g_rm)),
        (None, None) => None,
        _ => panic!("{}: only one of the layers stored a gradient", name),
    }
}

/// Weight and bias gradients stored by the backward pass of a dense layer.
pub fn assert_dense_gradients_close(name: &str, layer: &Layer, layer_rm: &LayerRm) {
    let (g, g_rm) = stored_gradients(name, layer.gradient.as_ref(), layer_rm.gradient.as_ref()).expect("dense layer must store its gradient");
    assert_close_2d(&format!("{} weight gradient", name), &g.get_gradient_weights(), &g_rm.get_gradient_weights_rm().expect("RM weight gradient"));
    assert_close_1d(&format!("{} bias gradient", name), &g.get_gradient_bias(), &g_rm.get_gradient_bias());
}

/// Weight and bias gradients stored by the backward pass of a linear layer.
pub fn assert_linear_gradients_close(name: &str, layer: &LinearLayer, layer_rm: &LinearLayerRm) {
    let (g, g_rm) = stored_gradients(name, layer.gradient.as_ref(), layer_rm.gradient.as_ref()).expect("linear layer must store its gradient");
    assert_close_2d(&format!("{} weight gradient", name), &g.get_gradient_weights(), &g_rm.get_gradient_weights_rm().expect("RM weight gradient"));
    assert_close_1d(&format!("{} bias gradient", name), &g.get_gradient_bias(), &g_rm.get_gradient_bias());
}

/// Gamma and beta gradients stored by the backward pass of a norm layer.
pub fn assert_norm_gradients_close(name: &str, layer: &NormalNormLayer, layer_rm: &NormalNormLayerRm) {
    let (g, g_rm) = stored_gradients(name, layer.gradient.as_ref(), layer_rm.gradient.as_ref()).expect("norm layer must store its gradient");
    assert_close_1d(&format!("{} gamma gradient", name), &g.get_gradient_gamma(), &g_rm.get_gradient_gamma());
    assert_close_1d(&format!("{} beta gradient", name), &g.get_gradient_beta(), &g_rm.get_gradient_beta());
}

/// Gradients of both weight matrices stored by the backward pass of a ComplexToLinear layer.
pub fn assert_ctl_gradients_close(name: &str, layer: &ComplexToLinearLayer, layer_rm: &ComplexToLinearLayerRm) {
    let (g, g_rm) = stored_gradients(name, layer.gradient.as_ref(), layer_rm.gradient.as_ref()).expect("CTL layer must store its gradient");
    assert_close_2d(&format!("{} weights_1 gradient", name), &g.get_gradient_weights(), &g_rm.get_weight_gradient_rm("weights").expect("RM weights gradient"));
    assert_close_2d(&format!("{} weights_2 gradient", name), &g.get_gradient_weights_2(), &g_rm.get_weight_gradient_rm("weights_2").expect("RM weights_2 gradient"));
}

/// Compares the input gradients of a Vec layer and an RM layer.
pub fn assert_input_gradients_close(name: &str, gradient: &Gradient, gradient_rm: &Gradient) {
    let gradient_rm_batch = gradient_rm.get_gradient_input_batch_rm_ref().expect("RM layer must return RM input gradients");
    assert_close_3d(&format!("{} input gradient", name), &gradient.get_gradient_input_batch(), gradient_rm_batch);
}

/// Runs `steps` training steps (forward, backward, update) on a Vec layer and its RM counterpart with
/// identical random inputs (`shape` = batch, seq_len, input_dim) and upstream gradients and checks that outputs and input gradients match.
/// `compare_gradients` checks the parameter gradients after each backward (it gets both layers and the
/// returned gradients), `compare_parameters` the parameters after each update.
pub fn compare_training_steps<V: LayerInterface, R: LayerInterface>(
    name: &str,
    layer: &mut V,
    layer_rm: &mut R,
    shape: (usize, usize, usize),
    padding_mask_batch: Option<Vec<Vec<u32>>>,
    steps: usize,
    mut compare_gradients: impl FnMut(&V, &R, &Gradient, &Gradient, &str),
    mut compare_parameters: impl FnMut(&V, &R, &str),
) {
    let (batch_size, seq_len, input_dim) = shape;
    let padding_mask_batch = padding_mask_batch.unwrap_or_else(|| vec![vec![1; seq_len]; batch_size]);

    for step in 1..=steps {
        let step_name = format!("{} step {}", name, step);
        let input_batch = generate_random_complex_3d(batch_size, seq_len, input_dim);
        let (mut input, mut input_rm) = layer_inputs_with_mask(&input_batch, &padding_mask_batch);
        input.set_time_step(step);
        input_rm.set_time_step(step);

        let output = layer.forward(&input).get_output_batch();
        let output_rm = layer_rm.forward(&input_rm);
        assert_close_3d(&format!("{} output", step_name), &output, output_rm.get_output_batch_rm_ref().expect("RM layer must return RM output"));

        // Upstream gradient with the shape of the layer output.
        let upstream_batch: Vec<Vec<Vec<C>>> = output.iter().map(|m| generate_random_complex_2d(m.len(), m[0].len())).collect();

        let (upstream, upstream_rm) = upstream_gradients(&upstream_batch);
        let gradient = layer.backward(&upstream);
        let gradient_rm = layer_rm.backward(&upstream_rm);
        assert_input_gradients_close(&step_name, &gradient, &gradient_rm);
        compare_gradients(layer, layer_rm, &gradient, &gradient_rm, &step_name);

        layer.update_parameters();
        layer_rm.update_parameters();
        compare_parameters(layer, layer_rm, &step_name);
    }
}
