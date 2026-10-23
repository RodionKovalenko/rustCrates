use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        layer::LayerType,
        network_layers_rm::{
            complex_to_linear_layer_rm::ComplexToLinearLayerRm, linear_layer_rm::LinearLayerRm, norm_layer_rm::NormalNormLayerRm, positional_encoding_layer_rm::PositionalEncodingLayerRm,
        },
    },
    utils::{
        activation::softmax_row_complex,
        adam_w_rm::update_weights_adam_w_rm,
        dtype::{r, Real, C, ONE, ZERO},
        matrix_rm::RowMajorMatrix,
        matrix_rm::{
            add_rm, append_rows_rm, average_matrix_by_scalar_rm, clip_all_gradients_by_global_norm_rm, conjugate_rm, conjugate_transpose_rm, linear_forward_batch_rm, multiply_complex_rm,
            normalize_gradients_rm, scale_rm_in_place, transpose_rm,
        },
        weights_initializer::initialize_weights_complex,
    },
};

use super::attention_rm_common::AttentionHeadRm;

const W_Q: &str = "w_q";
const W_K: &str = "w_k";
const W_V: &str = "w_v";

/// RM counterpart of `MaskedAttentionHead` (causal full attention):
///
/// Q = CTL(X Wq), K = CTL(Norm(RoPE(X Wk))), V = Linear(Norm(X Wv)), O = softmax(mask(Q K^T / sqrt(d_k))) V
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedAttentionHeadRm {
    pub weights_q: RowMajorMatrix<C>,
    pub weights_k: RowMajorMatrix<C>,
    pub weights_v: RowMajorMatrix<C>,

    pub layer_type: LayerType,
    pub learning_rate: f64,
    pub smoothing: Real,
    pub ema: Real,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub ctl_q: Option<ComplexToLinearLayerRm>,
    pub ctl_k: Option<ComplexToLinearLayerRm>,

    pub positional_encoding_layer: PositionalEncodingLayerRm,
    pub norm_layer_v: NormalNormLayerRm,
    pub norm_layer_k: NormalNormLayerRm,
    pub linear_v: LinearLayerRm,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub attention_weights_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub k_cache_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub v_cache_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub total_valid_tokens: usize,
    #[serde(skip)]
    pub q_ctl_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub k_ctl_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub v_norm_rm: Option<Vec<RowMajorMatrix<C>>>,
}

impl MaskedAttentionHeadRm {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let init = |rows, cols| {
            let mut w: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
            initialize_weights_complex(rows, cols, &mut w);
            RowMajorMatrix::from_rows(&w)
        };

        Self {
            weights_q: init(rows, cols),
            weights_k: init(rows, cols),
            weights_v: init(rows, cols),
            layer_type: LayerType::InputLayer,
            learning_rate,
            smoothing: r(0.99),
            ema: ZERO,
            global_norm: 0.0,
            max_norm: 0.0,
            previous_gradient: None,
            ctl_q: Some(ComplexToLinearLayerRm::new(cols, cols, learning_rate)),
            ctl_k: Some(ComplexToLinearLayerRm::new(cols, cols, learning_rate)),
            positional_encoding_layer: PositionalEncodingLayerRm::new(cols),
            norm_layer_v: NormalNormLayerRm::new(cols, 1e-8, learning_rate),
            norm_layer_k: NormalNormLayerRm::new(cols, 1e-8, learning_rate),
            linear_v: LinearLayerRm::new(learning_rate, cols, cols),
            gradient: None,
            time_step: 0,
            input_batch_rm: None,
            attention_weights_batch_rm: None,
            padding_mask_batch: None,
            k_cache_rm: None,
            v_cache_rm: None,
            batch_size: 0,
            total_valid_tokens: 1,
            q_ctl_rm: None,
            k_ctl_rm: None,
            v_norm_rm: None,
        }
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, learning_rate: f64) -> Self {
        let mut layer = Self::new(rows, cols, learning_rate);
        layer.layer_type = layer_type;
        layer
    }

    pub fn clear_cache(&mut self) {
        self.k_cache_rm = None;
        self.v_cache_rm = None;
    }

    /// `layer_input` with an RM batch replaced (keeps time step, padding mask, flags).
    fn with_batch(layer_input: &LayerInput, batch: Vec<RowMajorMatrix<C>>) -> LayerInput {
        let mut li = layer_input.clone();
        li.clear_input_batch();
        li.set_input_batch_rm(batch);
        li
    }

    /// K = CTL(Norm(RoPE(K)))
    fn transform_k(&mut self, k_batch: &[RowMajorMatrix<C>], layer_input: &LayerInput) -> Vec<RowMajorMatrix<C>> {
        let k_pos = self.positional_encoding_layer.rotate_batch(k_batch, layer_input.get_forward_only(), layer_input.get_time_step());
        let k_norm = self.norm_layer_k.forward(&Self::with_batch(layer_input, k_pos)).take_output_batch_rm().expect("NormRm must return RM output");
        match self.ctl_k.as_mut() {
            Some(ctl_k) => ctl_k.forward(&Self::with_batch(layer_input, k_norm)).take_output_batch_rm().expect("ComplexToLinearLayerRm must return RM output"),
            None => k_norm,
        }
    }

    /// V = Linear(Norm(V))
    fn transform_v(&mut self, v_batch: &[RowMajorMatrix<C>], layer_input: &LayerInput) -> Vec<RowMajorMatrix<C>> {
        let v_norm = self.norm_layer_v.forward(&Self::with_batch(layer_input, v_batch.to_vec())).take_output_batch_rm().expect("NormRm must return RM output");
        self.linear_v.forward(&Self::with_batch(layer_input, v_norm)).take_output_batch_rm().expect("LinearLayerRm must return RM output")
    }

    /// RM counterpart of `MaskedAttentionHead::forward` (including its K/V cache behavior in inference).
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch = layer_input.get_input_batch_rm();
        let padding_mask_batch = layer_input.get_padding_mask_batch();
        let calculate_k_v_cache = layer_input.get_calculate_k_v_cache();

        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();
        self.total_valid_tokens = layer_input.get_total_valid_tokens();

        let q_batch = linear_forward_batch_rm(&input_batch, &self.weights_q, None);

        // K/V for all tokens, or (with an existing cache) only for the last token of the last sequence.
        let (k_new_batch, v_new_batch) = if self.k_cache_rm.is_none() || !calculate_k_v_cache {
            (linear_forward_batch_rm(&input_batch, &self.weights_k, None), linear_forward_batch_rm(&input_batch, &self.weights_v, None))
        } else {
            match input_batch.last().filter(|m| m.rows > 0) {
                Some(last) => {
                    let last_token = RowMajorMatrix::from_data(1, last.cols, last.data[last.row_range(last.rows - 1)].to_vec());
                    (vec![multiply_complex_rm(&last_token, &self.weights_k)], vec![multiply_complex_rm(&last_token, &self.weights_v)])
                }
                None => (vec![], vec![]),
            }
        };

        let (k_batch, v_batch) = if calculate_k_v_cache {
            match (self.k_cache_rm.as_mut(), self.v_cache_rm.as_mut()) {
                (Some(k_cache), Some(v_cache)) => {
                    for (cache, new) in k_cache.iter_mut().zip(k_new_batch.iter()) {
                        append_rows_rm(cache, new);
                    }
                    for (cache, new) in v_cache.iter_mut().zip(v_new_batch.iter()) {
                        append_rows_rm(cache, new);
                    }
                }
                _ => {
                    self.k_cache_rm = Some(k_new_batch);
                    self.v_cache_rm = Some(v_new_batch);
                }
            }
            (self.k_cache_rm.clone().unwrap(), self.v_cache_rm.clone().unwrap())
        } else {
            (k_new_batch, v_new_batch)
        };

        let q_ctl_batch = match self.ctl_q.as_mut() {
            Some(ctl_q) => {
                let mut li = LayerInput::new_default();
                li.set_input_batch_rm(q_batch);
                ctl_q.forward(&li).take_output_batch_rm().expect("ComplexToLinearLayerRm must return RM output")
            }
            None => q_batch,
        };
        let k_ctl_batch = self.transform_k(&k_batch, layer_input);
        let v_batch = self.transform_v(&v_batch, layer_input);

        let attention_weights_batch: Vec<RowMajorMatrix<C>> = (0..q_ctl_batch.len())
            .into_par_iter()
            .map(|b| attention_weights(&q_ctl_batch[b], &k_ctl_batch[b], &padding_mask_batch[b]))
            .collect();
        let output_batch: Vec<RowMajorMatrix<C>> = attention_weights_batch.par_iter().zip(v_batch.par_iter()).map(|(a, v)| multiply_complex_rm(a, v)).collect();

        self.input_batch_rm = Some(input_batch);
        self.q_ctl_rm = Some(q_ctl_batch);
        self.k_ctl_rm = Some(k_ctl_batch);
        self.v_norm_rm = Some(v_batch);
        self.attention_weights_batch_rm = Some(attention_weights_batch);

        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(output_batch);
        output
    }

    /// RM counterpart of `MaskedAttentionHead::backward`.
    pub fn backward_rm(&mut self, previous_gradient_batch: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch = self.input_batch_rm.clone().expect("Input batch is missing in MaskedAttentionHeadRm");
        let padding_mask_batch = self.padding_mask_batch.clone().expect("Padding mask batch is missing in MaskedAttentionHeadRm");
        let attention_weights_batch = self.attention_weights_batch_rm.clone().expect("Attention weights batch is missing in MaskedAttentionHeadRm");
        let q_ctl_batch = self.q_ctl_rm.clone().expect("Q CTL is missing in MaskedAttentionHeadRm");
        let k_ctl_batch = self.k_ctl_rm.clone().expect("K CTL is missing in MaskedAttentionHeadRm");
        let v_batch = self.v_norm_rm.clone().expect("V norm output is missing in MaskedAttentionHeadRm");

        // V path: dL/dV = A^H dL/dO, back through Linear and Norm
        let dl_dv: Vec<RowMajorMatrix<C>> = attention_weights_batch.iter().zip(previous_gradient_batch.iter()).map(|(a, g)| multiply_complex_rm(&conjugate_transpose_rm(a), g)).collect();
        let mut gradient_v = Gradient::new_default();
        gradient_v.set_gradient_input_batch_rm(dl_dv);
        let gradient_v_linear = self.linear_v.backward(&gradient_v);
        let dl_dv_input = self.norm_layer_v.backward(&gradient_v_linear).get_gradient_input_batch_rm();

        // Scores path: dL/dz -> dL/dQ_ctl, dL/dK_ctl
        let per_batch: Vec<(RowMajorMatrix<C>, RowMajorMatrix<C>)> = (0..previous_gradient_batch.len())
            .into_par_iter()
            .map(|b| {
                let dl_dz = softmax_attention_backward_rm(&attention_weights_batch[b], &previous_gradient_batch[b], &v_batch[b], &padding_mask_batch[b]);
                let scaling_factor: Real = ONE / (r(1e-8) + (k_ctl_batch[b].cols as Real).sqrt());
                let mut k_scaled = conjugate_rm(&k_ctl_batch[b]);
                scale_rm_in_place(&mut k_scaled, C::new(scaling_factor, ZERO));
                let mut q_scaled = q_ctl_batch[b].clone();
                scale_rm_in_place(&mut q_scaled, C::new(scaling_factor, ZERO));
                (multiply_complex_rm(&dl_dz, &k_scaled), multiply_complex_rm(&conjugate_transpose_rm(&dl_dz), &q_scaled))
            })
            .collect();
        let (dl_dq_ctl, dl_dk_ctl): (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>) = per_batch.into_iter().unzip();

        let dl_dq = match self.ctl_q.as_mut() {
            Some(ctl_q) => {
                let mut g = Gradient::new_default();
                g.set_gradient_input_batch_rm(dl_dq_ctl);
                ctl_q.backward(&g).get_gradient_input_batch_rm()
            }
            None => dl_dq_ctl,
        };

        // K path: CTL -> Norm -> RoPE
        let mut gradient_k = Gradient::new_default();
        gradient_k.set_gradient_input_batch_rm(match self.ctl_k.as_mut() {
            Some(ctl_k) => {
                let mut g = Gradient::new_default();
                g.set_gradient_input_batch_rm(dl_dk_ctl);
                ctl_k.backward(&g).get_gradient_input_batch_rm()
            }
            None => dl_dk_ctl,
        });
        let dl_dk_norm = self.norm_layer_k.backward(&gradient_k).get_gradient_input_batch_rm();
        let dl_dk = self.positional_encoding_layer.rotate_batch_backward(&dl_dk_norm);

        let (wq_h, wk_h, wv_h) = (conjugate_transpose_rm(&self.weights_q), conjugate_transpose_rm(&self.weights_k), conjugate_transpose_rm(&self.weights_v));
        let per_batch: Vec<(RowMajorMatrix<C>, RowMajorMatrix<C>, RowMajorMatrix<C>, RowMajorMatrix<C>)> = (0..input_batch.len())
            .into_par_iter()
            .map(|b| {
                let x_h = conjugate_transpose_rm(&input_batch[b]);
                let input_gradient = add_rm(&add_rm(&multiply_complex_rm(&dl_dq[b], &wq_h), &multiply_complex_rm(&dl_dk[b], &wk_h)), &multiply_complex_rm(&dl_dv_input[b], &wv_h));
                (multiply_complex_rm(&x_h, &dl_dq[b]), multiply_complex_rm(&x_h, &dl_dk[b]), multiply_complex_rm(&x_h, &dl_dv_input[b]), input_gradient)
            })
            .collect();

        let mut gradient = Gradient::new_default();
        let mut gradients: [Vec<RowMajorMatrix<C>>; 4] = Default::default();
        for (gq, gk, gv, gi) in per_batch {
            gradients[0].push(gq);
            gradients[1].push(gk);
            gradients[2].push(gv);
            gradients[3].push(gi);
        }
        let [gq, gk, gv, gi] = gradients;
        gradient.set_weight_gradient_batch_rm(W_Q, gq);
        gradient.set_weight_gradient_batch_rm(W_K, gk);
        gradient.set_weight_gradient_batch_rm(W_V, gv);
        gradient.set_gradient_input_batch_rm(gi);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        self.backward_rm(&previous_gradient.get_gradient_input_batch_rm_cow())
    }

    /// RM counterpart of `MaskedAttentionHead::update_parameters`.
    pub fn update_parameters(&mut self) {
        let mut gradient = self.gradient.take().expect("Gradient is missing in MaskedAttentionHeadRm");
        // Like MaskedAttentionHead: no max(1) guard here.
        let total_valid_tokens: Real = r(self.total_valid_tokens as f64);

        let mut weight_gradients: Vec<RowMajorMatrix<C>> = [W_Q, W_V, W_K]
            .iter()
            .map(|name| average_matrix_by_scalar_rm(&gradient.get_weight_gradient_rm(name).expect("missing attention weight gradient"), total_valid_tokens))
            .collect();
        for g in weight_gradients.iter_mut() {
            clip_all_gradients_by_global_norm_rm(g, &mut vec![], self.global_norm, self.max_norm);
        }
        for g in weight_gradients.iter_mut() {
            normalize_gradients_rm(g);
        }

        if let Some(ctl_q) = self.ctl_q.as_mut() {
            ctl_q.update_parameters();
        }
        if let Some(ctl_k) = self.ctl_k.as_mut() {
            ctl_k.update_parameters();
        }
        self.norm_layer_v.update_parameters();
        self.norm_layer_k.update_parameters();
        self.linear_v.update_parameters();

        let (learning_rate, time_step) = (self.learning_rate, self.time_step);
        let previous_gradient = self.previous_gradient.as_ref();
        update_weights_adam_w_rm(W_Q, &mut self.weights_q, &weight_gradients[0], previous_gradient, &mut gradient, learning_rate, time_step);
        update_weights_adam_w_rm(W_K, &mut self.weights_k, &weight_gradients[2], previous_gradient, &mut gradient, learning_rate, time_step);
        update_weights_adam_w_rm(W_V, &mut self.weights_v, &weight_gradients[1], previous_gradient, &mut gradient, learning_rate, time_step);

        self.previous_gradient = Some(gradient);
    }
}

impl AttentionHeadRm for MaskedAttentionHeadRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        MaskedAttentionHeadRm::forward(self, layer_input)
    }
    fn backward_rm(&mut self, previous_gradient_batch: &[RowMajorMatrix<C>]) -> Gradient {
        MaskedAttentionHeadRm::backward_rm(self, previous_gradient_batch)
    }
    fn update_parameters(&mut self) {
        MaskedAttentionHeadRm::update_parameters(self)
    }
}

/// Causal softmax(Q K^T / sqrt(d_k)) with padded query rows set to zero.
fn attention_weights(q: &RowMajorMatrix<C>, k: &RowMajorMatrix<C>, padding_mask: &[u32]) -> RowMajorMatrix<C> {
    let mut scores = multiply_complex_rm(q, &transpose_rm(k));
    let scaling_factor: Real = ONE / (r(1e-8) + (k.cols as Real).sqrt());
    for row in 0..scores.rows {
        let range = scores.row_range(row);
        for (col, score) in scores.data[range].iter_mut().enumerate() {
            // causal mask (cyclic like MaskedAttentionHead's mask when K is longer than Q)
            *score = if col % scores.rows > row % scores.rows { C::new(Real::NEG_INFINITY, Real::NEG_INFINITY) } else { *score * scaling_factor };
        }
    }

    let mut weights = RowMajorMatrix::create_empty(scores.rows, scores.cols);
    for row in 0..scores.rows {
        assert!(row < padding_mask.len(), "Row mask is smaller than the index: {}, {}", padding_mask.len(), row);
        if padding_mask[row] == 0 {
            continue;
        }
        let range = scores.row_range(row);
        weights.data[range.clone()].copy_from_slice(&softmax_row_complex(&scores.data[range].to_vec()));
    }
    weights
}

/// RM counterpart of `MaskedAttentionHead::softmax_attention_backward_full`: dL/dz for one sample.
fn softmax_attention_backward_rm(softmax_vals: &RowMajorMatrix<C>, dl_do: &RowMajorMatrix<C>, v: &RowMajorMatrix<C>, padding_mask: &[u32]) -> RowMajorMatrix<C> {
    let (q_len, k_len) = (dl_do.rows, softmax_vals.cols);
    let mut dl_dz = RowMajorMatrix::create_empty(q_len, k_len);
    if q_len == 0 || k_len == 0 {
        return dl_dz;
    }

    let k_compute = k_len.min(v.rows);
    let is_padded = |p: usize| padding_mask.get(p).copied().unwrap_or(1) == 0;

    for i in 0..q_len {
        if is_padded(i) {
            continue;
        }
        let g_row = &dl_do.data[dl_do.row_range(i)];

        // u[p] = sum_j dL/dO[i][j] * conj(V[p][j])
        let mut u = vec![C::new(ZERO, ZERO); k_len];
        for p in 0..k_compute {
            if is_padded(p) {
                continue;
            }
            let v_row = &v.data[v.row_range(p)];
            let feat_len = g_row.len().min(v_row.len());
            u[p] = (0..feat_len).fold(C::new(ZERO, ZERO), |acc, j| acc + g_row[j] * v_row[j].conj());
        }

        let s_row = &softmax_vals.data[softmax_vals.row_range(i)];
        let dot = (0..k_compute).filter(|&p| !is_padded(p)).fold(C::new(ZERO, ZERO), |acc, p| acc + u[p] * C::new(s_row[p].re, ZERO));

        let dz_row = dl_dz.row_range(i);
        for p in 0..k_compute {
            if !is_padded(p) {
                dl_dz.data[dz_row.start + p] = (u[p] - dot) * C::new(s_row[p].re, ZERO);
            }
        }
    }

    dl_dz
}
