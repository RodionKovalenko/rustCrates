//! The candle block against the RM layers: same parameters, same input -> same output and same input gradient.

use candle_core::{DType, Device, Tensor, Var};

use super::{
    block::{AttentionHead, AttentionLayer, BlockContext, ComplexNorm, FeedForwardLayer},
    complex::{CTensor, CVar},
};
use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
    network_layers::{
        layer::LayerEnum,
        network_layers_rm::{feedforward_layer_rm::FeedForwardLayerRm, norm_layer_rm::NormalNormLayerRm},
    },
    network_types::transformer::sparse_self_attention_layer_rm::SparseSelfAttentionLayerRm,
    utils::{
        dtype::{Real, C},
        matrix_rm::RowMajorMatrix,
        random_arrays::generate_random_complex_3d,
    },
};

/// f64 build: candle and RM agree to rounding; f32 build: to f32 precision.
#[cfg(feature = "dtype-f64")]
const TOLERANCE: f64 = 1e-9;
#[cfg(not(feature = "dtype-f64"))]
const TOLERANCE: f64 = 2e-3;

fn pairs(values: &[C]) -> Vec<(f64, f64)> {
    values.iter().map(|c| (c.re as f64, c.im as f64)).collect()
}

fn cvar(values: &[C], shape: &[usize]) -> CVar {
    CVar::from_pairs(&pairs(values), shape, DType::F64, &Device::Cpu).unwrap()
}

fn real_var(m: &RowMajorMatrix<C>) -> Var {
    let re: Vec<f64> = m.data.iter().map(|c| c.re as f64).collect();
    Var::from_tensor(&Tensor::from_vec(re, (m.rows, m.cols), &Device::Cpu).unwrap()).unwrap()
}

fn norm_from_rm(norm: &NormalNormLayerRm) -> ComplexNorm {
    let d = norm.gamma.len();
    ComplexNorm { gamma: cvar(&norm.gamma, &[d]), beta: cvar(&norm.beta, &[d]), epsilon: norm.epsilon }
}

fn attention_from_rm(layer: &SparseSelfAttentionLayerRm) -> AttentionLayer {
    let heads = layer
        .attention_heads
        .iter()
        .map(|h| {
            let shape = [h.weights_q_rm.rows, h.weights_q_rm.cols];
            let (ctl_q, ctl_k) = (h.ctl_q.as_ref().unwrap(), h.ctl_k.as_ref().unwrap());
            AttentionHead {
                w_q: cvar(&h.weights_q_rm.data, &shape),
                w_k: cvar(&h.weights_k_rm.data, &shape),
                w_v: cvar(&h.weights_v_rm.data, &shape),
                ctl_q_1: real_var(&ctl_q.weights_1),
                ctl_q_2: real_var(&ctl_q.weights_2),
                ctl_k_1: real_var(&ctl_k.weights_1),
                ctl_k_2: real_var(&ctl_k.weights_2),
            }
        })
        .collect();
    AttentionLayer { norm: norm_from_rm(&layer.norm_layer), heads, beta: layer.beta as f64 }
}

fn ffn_from_rm(layer: &FeedForwardLayerRm) -> FeedForwardLayer {
    let Some(LayerEnum::NormRm(norm)) = &layer.norm_layer else { panic!("FeedForwardLayerRm without NormRm") };
    let (LayerEnum::DenseRm(d1), LayerEnum::DenseRm(d2)) = (&layer.layers[0], &layer.layers[1]) else { panic!("FeedForwardLayerRm without two DenseRm layers") };
    FeedForwardLayer {
        norm: norm_from_rm(norm),
        w_1: cvar(&d1.weights.data, &[d1.weights.rows, d1.weights.cols]),
        b_1: cvar(&d1.bias, &[d1.bias.len()]),
        w_2: cvar(&d2.weights.data, &[d2.weights.rows, d2.weights.cols]),
        b_2: cvar(&d2.bias, &[d2.bias.len()]),
        beta: layer.beta as f64,
    }
}

fn to_ctensor(batch: &[Vec<Vec<C>>]) -> CTensor {
    let (b, t, d) = (batch.len(), batch[0].len(), batch[0][0].len());
    let flat: Vec<C> = batch.iter().flatten().flatten().copied().collect();
    CTensor::from_pairs(&pairs(&flat), &[b, t, d], DType::F64, &Device::Cpu).unwrap()
}

fn assert_close(name: &str, expected: &[(f64, f64)], actual: &[(f64, f64)]) {
    assert_eq!(expected.len(), actual.len(), "{name}: length");
    let mut worst = 0.0f64;
    for (e, a) in expected.iter().zip(actual) {
        let diff = ((e.0 - a.0).powi(2) + (e.1 - a.1).powi(2)).sqrt();
        let scale = (e.0.hypot(e.1)).max(a.0.hypot(a.1)).max(1.0);
        worst = worst.max(diff / scale);
    }
    println!("{name}: max relative difference {worst:.2e}");
    assert!(worst <= TOLERANCE, "{name}: max relative difference {worst:.2e} > {TOLERANCE:.0e}");
}

fn rm_pairs(batch: &[RowMajorMatrix<C>]) -> Vec<(f64, f64)> {
    batch.iter().flat_map(|m| pairs(&m.data)).collect()
}

/// Input gradient of L = sum(Re(G) Re(Y) + Im(G) Im(Y)) as dL/dRe(x) + i dL/dIm(x), the convention of the RM
/// backward pass for an upstream gradient G.
fn candle_input_gradient(forward: impl Fn(&CTensor) -> CTensor, input: &[Vec<Vec<C>>], upstream: &[Vec<Vec<C>>]) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let x = to_ctensor(input);
    let (x_re, x_im) = (Var::from_tensor(&x.re).unwrap(), Var::from_tensor(&x.im).unwrap());
    let y = forward(&CTensor::new(x_re.as_tensor().clone(), x_im.as_tensor().clone()));
    let g = to_ctensor(upstream);
    let loss = (y.re.mul(&g.re).unwrap().sum_all().unwrap() + y.im.mul(&g.im).unwrap().sum_all().unwrap()).unwrap();
    let grads = loss.backward().unwrap();
    let gx = CTensor::new(grads.get(x_re.as_tensor()).unwrap().clone(), grads.get(x_im.as_tensor()).unwrap().clone());
    (y.to_pairs().unwrap(), gx.to_pairs().unwrap())
}

fn rm_input(input: &[Vec<Vec<C>>]) -> LayerInput {
    let mut layer_input = LayerInput::new_default();
    layer_input.set_input_batch_rm(input.iter().map(|m| RowMajorMatrix::from_rows(m)).collect());
    layer_input.set_padding_mask_batch(vec![vec![1; input[0].len()]; input.len()]);
    layer_input.set_batch_size(input.len());
    layer_input
}

fn rm_upstream(upstream: &[Vec<Vec<C>>]) -> Gradient {
    let mut gradient = Gradient::new_default();
    gradient.set_gradient_input_batch_rm(upstream.iter().map(|m| RowMajorMatrix::from_rows(m)).collect());
    gradient.set_total_valid_tokens(upstream.iter().map(|m| m.len()).sum());
    gradient
}

#[test]
fn feed_forward_layer_matches_rm() {
    let (batch_size, seq_len, dim, hidden) = (2, 6, 8, 12);
    let mut layer_rm = FeedForwardLayerRm::new(dim, hidden, 0.01);
    let layer = ffn_from_rm(&layer_rm);
    let input = generate_random_complex_3d(batch_size, seq_len, dim);
    let upstream = generate_random_complex_3d(batch_size, seq_len, dim);

    let output_rm = layer_rm.forward(&rm_input(&input)).take_output_batch_rm().unwrap();
    let input_gradient_rm = layer_rm.backward(&rm_upstream(&upstream)).get_gradient_input_batch_rm();

    let (output, input_gradient) = candle_input_gradient(|x| layer.forward(x).unwrap(), &input, &upstream);
    assert_close("FFN output", &rm_pairs(&output_rm), &output);
    assert_close("FFN input gradient", &rm_pairs(&input_gradient_rm), &input_gradient);
}

#[test]
fn attention_layer_matches_rm() {
    let (batch_size, seq_len, dim, heads, window) = (2, 9, 8, 2, 2);
    let mut layer_rm = SparseSelfAttentionLayerRm::new(heads, dim, dim, window, 0.01);
    let layer = attention_from_rm(&layer_rm);
    let input = generate_random_complex_3d(batch_size, seq_len, dim);
    let upstream = generate_random_complex_3d(batch_size, seq_len, dim);

    let output_rm = layer_rm.forward(&rm_input(&input)).take_output_batch_rm().unwrap();
    let input_gradient_rm = layer_rm.backward(&rm_upstream(&upstream)).get_gradient_input_batch_rm();

    let ctx = BlockContext::new(seq_len, dim / heads, window, DType::F64, &Device::Cpu).unwrap();
    let (output, input_gradient) = candle_input_gradient(|x| layer.forward(x, &ctx).unwrap(), &input, &upstream);
    assert_close("attention output", &rm_pairs(&output_rm), &output);
    // The RM attention backward is hand-written; this reports how far it is from the exact (autograd) gradient.
    let worst = {
        let rm = rm_pairs(&input_gradient_rm);
        rm.iter().zip(&input_gradient).map(|(e, a)| ((e.0 - a.0).powi(2) + (e.1 - a.1).powi(2)).sqrt() / e.0.hypot(e.1).max(a.0.hypot(a.1)).max(1.0)).fold(0.0f64, f64::max)
    };
    println!("attention input gradient: RM vs exact autograd, max relative difference {worst:.2e}");
    // Measured 1e-6 in f64: the hand-written RM attention backward is a close approximation of the exact gradient.
    assert!(worst <= 1e-4_f64.max(TOLERANCE), "attention input gradient differs from RM by {worst:.2e}");
}

#[test]
fn complex_ops_match_num_complex() {
    let z: Vec<C> = generate_random_complex_3d(1, 4, 6).concat().concat();
    let z = z.into_iter().map(|c| c * (3.0 as Real)).collect::<Vec<C>>();
    let t = CTensor::from_pairs(&pairs(&z), &[z.len()], DType::F64, &Device::Cpu).unwrap();
    let sqrt: Vec<C> = z.iter().map(|c| c.sqrt()).collect();
    assert_close("sqrt", &pairs(&sqrt), &t.sqrt().unwrap().to_pairs().unwrap());
    let sigmoid: Vec<C> = z.iter().map(crate::neural_networks::utils::activation::sigmoid_complex).collect();
    assert_close("sigmoid", &pairs(&sigmoid), &t.sigmoid().unwrap().to_pairs().unwrap());
}
