use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        layer::LayerType,
        network_layers_rm::{complex_to_linear_layer_rm::ComplexToLinearLayerRm, positional_encoding_layer_rm::PositionalEncodingLayerRm},
    },
    utils::{
        activation::softmax_row_complex,
        adam_w_rm::update_weights_adam_w_rm,
        dtype::{r, Real, C, ZERO},
        matrix_rm::RowMajorMatrix,
        matrix_rm::{
            append_rows_rm, average_matrix_by_scalar_rm, clip_all_gradients_by_global_norm_rm, linear_forward_batch_rm, multiply_complex_rm, normalize_gradients_rm,
        },
        weights_initializer::initialize_weights_complex,
    },
};

use super::attention_rm_common::{qkv_projection_backward_rm, AttentionHeadRm};

const W_Q: &str = "w_q";
const W_K: &str = "w_k";
const W_V: &str = "w_v";

/// RM counterpart of `SparseMaskedAttentionHead`: causal attention over a local window of `window_size`
/// tokens on each side, with RoPE on Q/K and ComplexToLinear layers on Q/K.
///
/// Attention weights are stored with a fixed width of `2 * window_size + 1` columns; column `c` of
/// row `i` belongs to key `start(i) + c`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMaskedAttentionHeadRm {
    pub weights_q_rm: RowMajorMatrix<C>,
    pub weights_k_rm: RowMajorMatrix<C>,
    pub weights_v_rm: RowMajorMatrix<C>,

    pub layer_type: LayerType,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub window_size: usize,

    pub ctl_q: Option<ComplexToLinearLayerRm>,
    pub ctl_k: Option<ComplexToLinearLayerRm>,

    pub positional_encoding_layer: PositionalEncodingLayerRm,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub attention_weights_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub k_cache_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub v_cache_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub k_ctl_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub q_ctl_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub total_valid_tokens: usize,
}

impl SparseMaskedAttentionHeadRm {
    pub fn new(rows: usize, cols: usize, window_size: usize, learning_rate: f64) -> Self {
        let init = |rows, cols| {
            let mut w: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
            initialize_weights_complex(rows, cols, &mut w);
            RowMajorMatrix::from_rows(&w)
        };

        Self {
            weights_q_rm: init(rows, cols),
            weights_k_rm: init(rows, cols),
            weights_v_rm: init(rows, cols),
            layer_type: LayerType::InputLayer,
            learning_rate,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            previous_gradient: None,
            window_size,
            ctl_q: Some(ComplexToLinearLayerRm::new(cols, cols, learning_rate)),
            ctl_k: Some(ComplexToLinearLayerRm::new(cols, cols, learning_rate)),
            positional_encoding_layer: PositionalEncodingLayerRm::new(cols),
            gradient: None,
            time_step: 0,
            input_batch_rm: None,
            attention_weights_batch_rm: None,
            output_batch_rm: None,
            padding_mask_batch: None,
            k_cache_rm: None,
            v_cache_rm: None,
            k_ctl_rm: None,
            q_ctl_rm: None,
            batch_size: 0,
            total_valid_tokens: 1,
        }
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, window_size: usize, learning_rate: f64) -> Self {
        let mut layer = Self::new(rows, cols, window_size, learning_rate);
        layer.layer_type = layer_type;
        layer
    }

    pub fn clear_cache(&mut self) {
        self.k_cache_rm = None;
        self.v_cache_rm = None;
        self.k_ctl_rm = None;
        self.q_ctl_rm = None;
        self.attention_weights_batch_rm = None;
        self.input_batch_rm = None;
        self.output_batch_rm = None;
        self.padding_mask_batch = None;
    }

    fn window_width(&self) -> usize {
        2 * self.window_size + 1
    }

    /// Q/K/V projections for a batch; Q and K are rotated by RoPE with `position(row)`.
    fn project_qkv(&self, input_batch: &[RowMajorMatrix<C>], position: impl Fn(usize) -> usize + Sync) -> (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>) {
        let q = self.positional_encoding_layer.rotate_batch_at(&linear_forward_batch_rm(input_batch, &self.weights_q_rm, None), &position);
        let k = self.positional_encoding_layer.rotate_batch_at(&linear_forward_batch_rm(input_batch, &self.weights_k_rm, None), &position);
        let v = linear_forward_batch_rm(input_batch, &self.weights_v_rm, None);
        (q, k, v)
    }

    /// Applies the optional ComplexToLinear layer (real-valued output).
    fn apply_ctl(ctl: Option<&mut ComplexToLinearLayerRm>, batch: Vec<RowMajorMatrix<C>>) -> Vec<RowMajorMatrix<C>> {
        match ctl {
            Some(ctl) => {
                let mut li = LayerInput::new_default();
                li.set_input_batch_rm(batch);
                ctl.forward(&li).take_output_batch_rm().expect("ComplexToLinearLayerRm must return RM output")
            }
            None => batch,
        }
    }

    /// RM counterpart of `SparseMaskedAttentionHead::forward`.
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        if layer_input.get_forward_only() && layer_input.get_calculate_k_v_cache() {
            return self.forward_cached(layer_input);
        }

        let input_batch = layer_input.get_input_batch_rm();
        let padding_mask_batch = layer_input.get_padding_mask_batch();
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();
        self.total_valid_tokens = layer_input.get_total_valid_tokens();

        let (forward_only, time_step) = (layer_input.get_forward_only(), layer_input.get_time_step());
        let (q_batch, k_batch, v_batch) = self.project_qkv(&input_batch, |position| if forward_only && time_step > 0 { time_step } else { position });

        let q_ctl_batch = Self::apply_ctl(self.ctl_q.as_mut(), q_batch);
        let k_ctl_batch = Self::apply_ctl(self.ctl_k.as_mut(), k_batch);

        let attention_weights_batch: Vec<RowMajorMatrix<C>> = (0..input_batch.len())
            .into_par_iter()
            .map(|b| self.attention_weights_rm(&q_ctl_batch[b], &k_ctl_batch[b], &padding_mask_batch[b]))
            .collect();

        let output_batch: Vec<RowMajorMatrix<C>> = attention_weights_batch.par_iter().zip(v_batch.par_iter()).map(|(a, v)| self.multiply_sparse_rm(a, v)).collect();

        self.input_batch_rm = Some(input_batch);
        self.q_ctl_rm = Some(q_ctl_batch);
        self.k_ctl_rm = Some(k_ctl_batch);
        self.attention_weights_batch_rm = Some(attention_weights_batch);
        self.output_batch_rm = Some(output_batch.clone());

        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(output_batch);
        output
    }

    /// Causal local-window softmax(Q K^T / sqrt(d_k)) in fixed-window layout (padded query rows are zero).
    fn attention_weights_rm(&self, q: &RowMajorMatrix<C>, k: &RowMajorMatrix<C>, padding_mask: &[u32]) -> RowMajorMatrix<C> {
        let seq_len = q.rows;
        let window_width = self.window_width();
        let d_k_sqrt: Real = (q.cols as Real).sqrt() + r(1e-12);
        let mut weights = RowMajorMatrix::create_empty(seq_len, window_width);

        for i in 0..seq_len {
            if padding_mask[i] == 0 {
                continue;
            }
            let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
            let q_row = &q.data[q.row_range(i)];

            let scores: Vec<C> = (start_ind..end_ind)
                .map(|j| {
                    if j > i {
                        C::new(Real::NEG_INFINITY, Real::NEG_INFINITY)
                    } else {
                        q_row.iter().zip(k.data[k.row_range(j)].iter()).fold(C::new(ZERO, ZERO), |acc, (&q, &k)| acc + q * k) / d_k_sqrt
                    }
                })
                .collect();

            let row = weights.row_range(i);
            weights.data[row.start..row.start + scores.len()].copy_from_slice(&softmax_row_complex(&scores));
        }

        weights
    }

    /// `A V` with the fixed-window attention weights (real-valued, like `SparseMaskedAttentionHead::multiply_sparse`).
    fn multiply_sparse_rm(&self, attention_weights: &RowMajorMatrix<C>, v: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
        let seq_len = attention_weights.rows;
        let mut output = RowMajorMatrix::create_empty(seq_len, v.cols);

        for i in 0..seq_len {
            let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
            let a_row = &attention_weights.data[attention_weights.row_range(i)];
            let out_row = output.row_range(i);
            for col in 0..v.cols {
                for f in 0..end_ind - start_ind {
                    output.data[out_row.start + col] += (a_row[f].re * v.data[v.idx(start_ind + f, col)]).re;
                }
            }
        }

        output
    }

    /// Incremental-decoding forward with a K/V cache (inference only), RM counterpart of
    /// `SparseMaskedAttentionHead::forward_cached`. The input holds only the new suffix tokens.
    pub fn forward_cached(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch = layer_input.get_input_batch_rm();
        let batch_size = input_batch.len();
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();

        let cache_len = self.k_cache_rm.as_ref().and_then(|c| c.first()).map(|m| m.rows).unwrap_or(0);

        // Q/K are rotated by their absolute position in the full sequence.
        let (q_batch, k_batch, v_batch) = self.project_qkv(&input_batch, |i| cache_len + i);

        // CTL layers are position-wise, so applying them to the suffix alone is exact.
        let q_ctl_batch = Self::apply_ctl(self.ctl_q.as_mut(), q_batch);
        let k_ctl_batch = Self::apply_ctl(self.ctl_k.as_mut(), k_batch);

        let k_cache = self.k_cache_rm.get_or_insert_with(|| k_ctl_batch.iter().map(|k| RowMajorMatrix::create_empty(0, k.cols)).collect());
        let v_cache = self.v_cache_rm.get_or_insert_with(|| v_batch.iter().map(|v| RowMajorMatrix::create_empty(0, v.cols)).collect());
        for b in 0..batch_size {
            append_rows_rm(&mut k_cache[b], &k_ctl_batch[b]);
            append_rows_rm(&mut v_cache[b], &v_batch[b]);
        }

        let d_k_sqrt: Real = (q_ctl_batch[0].cols as Real).sqrt() + r(1e-12);

        let output_batch: Vec<RowMajorMatrix<C>> = (0..batch_size)
            .map(|b| {
                let (q, k, v) = (&q_ctl_batch[b], &k_cache[b], &v_cache[b]);
                let (suffix_len, seq_total) = (q.rows, k.rows);
                let mut out = RowMajorMatrix::create_empty(suffix_len, v.cols);

                for i in 0..suffix_len {
                    // Absolute position of this query in the full (cached) sequence.
                    let p = seq_total - suffix_len + i;
                    let (start_ind, end_ind) = calculate_start_end_indices(p, self.window_size, seq_total);
                    let q_row = &q.data[q.row_range(i)];

                    let scores: Vec<C> = (start_ind..end_ind)
                        .map(|j| {
                            if j > p {
                                C::new(Real::NEG_INFINITY, Real::NEG_INFINITY)
                            } else {
                                q_row.iter().zip(k.data[k.row_range(j)].iter()).fold(C::new(ZERO, ZERO), |acc, (&q, &k)| acc + q * k) / d_k_sqrt
                            }
                        })
                        .collect();
                    let attention_weights = softmax_row_complex(&scores);

                    let out_row = out.row_range(i);
                    for (f, attention_weight) in attention_weights.iter().enumerate() {
                        let v_row = &v.data[v.row_range(start_ind + f)];
                        for col in 0..v.cols {
                            out.data[out_row.start + col] += (attention_weight.re * v_row[col]).re;
                        }
                    }
                }
                out
            })
            .collect();

        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(output_batch);
        output
    }

    /// Gradients w.r.t. V and the pre-softmax scores (dL/dz) for one sample.
    fn attention_backward_rm(&self, grad_output: &RowMajorMatrix<C>, attention_weights: &RowMajorMatrix<C>, v: &RowMajorMatrix<C>, padding_mask: &[u32]) -> (RowMajorMatrix<C>, RowMajorMatrix<C>) {
        let seq_len = grad_output.rows;
        let dim = grad_output.cols;
        let mut dl_dv = RowMajorMatrix::create_empty(seq_len, dim);
        let mut dl_dz = RowMajorMatrix::create_empty(seq_len, self.window_width());

        // dL/dV[q] = sum over queries p attending q of A[p][q] * dL/dO[p] (real parts)
        for col in 0..dim {
            for q in 0..seq_len {
                for p in 0..seq_len {
                    let (start_ind, end_ind) = calculate_start_end_indices(p, self.window_size, seq_len);
                    if q >= start_ind && q < end_ind {
                        let a = attention_weights.data[attention_weights.idx(p, q - start_ind)].re;
                        let idx = dl_dv.idx(q, col);
                        dl_dv.data[idx] += (grad_output.data[grad_output.idx(p, col)] * a).re;
                    }
                }
            }
        }

        // Softmax backward (real parts), see `SparseMaskedAttentionHead::softmax_backward_sparse_compressed`
        for i in 0..seq_len {
            if padding_mask[i] == 0 {
                continue;
            }
            let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
            let g_row = &grad_output.data[grad_output.row_range(i)];
            let a_row = &attention_weights.data[attention_weights.row_range(i)];

            let u: Vec<Real> = (start_ind..end_ind)
                .map(|col| g_row.iter().zip(v.data[v.row_range(col)].iter()).fold(r(0.0), |acc, (g, v)| acc + g.re * v.re))
                .collect();
            let dot: Real = u.iter().enumerate().fold(r(0.0), |acc, (p, &u_p)| acc + a_row[p].re * u_p);

            let dz_row = dl_dz.row_range(i);
            for (p, &u_p) in u.iter().enumerate() {
                dl_dz.data[dz_row.start + p] = C::new(a_row[p].re * (u_p - dot), r(0.0));
            }
        }

        (dl_dv, dl_dz)
    }

    /// Gradients w.r.t. the (CTL) queries and keys from dL/dz for one sample.
    fn scores_backward_rm(&self, dl_dz: &RowMajorMatrix<C>, q: &RowMajorMatrix<C>, k: &RowMajorMatrix<C>) -> (RowMajorMatrix<C>, RowMajorMatrix<C>) {
        let seq_len = q.rows;
        let scaling_factor: Real = r(1.0) / (r(1e-8) + (k.cols as Real).sqrt());
        let mut dl_dq = RowMajorMatrix::create_empty(seq_len, q.cols);
        let mut dl_dk = RowMajorMatrix::create_empty(seq_len, k.cols);

        for col in 0..q.cols {
            for i in 0..seq_len {
                for p in 0..seq_len {
                    let (start_p, end_p) = calculate_start_end_indices(p, self.window_size, seq_len);
                    if i < start_p || i >= end_p {
                        continue;
                    }
                    // p is in the window of i as well (symmetric windows)
                    let (start_i, _) = calculate_start_end_indices(i, self.window_size, seq_len);

                    // dL/dq[i] += dz[i][p] * k[p] / sqrt(d_k)
                    let dz_ip = dl_dz.data[dl_dz.idx(i, p - start_i)];
                    let idx = dl_dq.idx(i, col);
                    dl_dq.data[idx] += k.data[k.idx(p, col)] * scaling_factor * dz_ip.conj();

                    // dL/dk[i] += dz[p][i] * q[p] / sqrt(d_k)
                    let dz_pi = dl_dz.data[dl_dz.idx(p, i - start_p)];
                    let idx = dl_dk.idx(i, col);
                    dl_dk.data[idx] += q.data[q.idx(p, col)] * scaling_factor * dz_pi.conj();
                }
            }
        }

        (dl_dq, dl_dk)
    }

    /// Backward through the optional ComplexToLinear layer.
    fn ctl_backward(ctl: Option<&mut ComplexToLinearLayerRm>, gradient_batch: Vec<RowMajorMatrix<C>>) -> Vec<RowMajorMatrix<C>> {
        match ctl {
            Some(ctl) => {
                let mut gradient = Gradient::new_default();
                gradient.set_gradient_input_batch_rm(gradient_batch);
                ctl.backward(&gradient).get_gradient_input_batch_rm()
            }
            None => gradient_batch,
        }
    }

    /// RM counterpart of `SparseMaskedAttentionHead::backward`.
    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch = self.input_batch_rm.as_ref().expect("Input batch RM is missing in SparseMaskedAttentionHeadRm");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Padding mask batch is missing in SparseMaskedAttentionHeadRm");
        let attention_weights_batch = self.attention_weights_batch_rm.as_ref().expect("Attention weights RM batch is missing in SparseMaskedAttentionHeadRm");
        let q_ctl_batch = self.q_ctl_rm.as_ref().expect("Q CTL RM is missing");
        let k_ctl_batch = self.k_ctl_rm.as_ref().expect("K CTL RM is missing");

        let batch_size = previous_gradient_batch_rm.len();
        assert_eq!(input_batch.len(), batch_size);

        let per_batch: Vec<(RowMajorMatrix<C>, RowMajorMatrix<C>, RowMajorMatrix<C>)> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let v = multiply_complex_rm(&input_batch[b], &self.weights_v_rm);
                let (dl_dv, dl_dz) = self.attention_backward_rm(&previous_gradient_batch_rm[b], &attention_weights_batch[b], &v, &padding_mask_batch[b]);
                let (dl_dq_ctl, dl_dk_ctl) = self.scores_backward_rm(&dl_dz, &q_ctl_batch[b], &k_ctl_batch[b]);
                (dl_dv, dl_dq_ctl, dl_dk_ctl)
            })
            .collect();

        let mut dl_dv_batch = Vec::with_capacity(batch_size);
        let mut dl_dq_ctl_batch = Vec::with_capacity(batch_size);
        let mut dl_dk_ctl_batch = Vec::with_capacity(batch_size);
        for (dv, dq, dk) in per_batch {
            dl_dv_batch.push(dv);
            dl_dq_ctl_batch.push(dq);
            dl_dk_ctl_batch.push(dk);
        }

        // Back through CTL and RoPE
        let dl_dq_batch = self.positional_encoding_layer.rotate_batch_backward(&Self::ctl_backward(self.ctl_q.as_mut(), dl_dq_ctl_batch));
        let dl_dk_batch = self.positional_encoding_layer.rotate_batch_backward(&Self::ctl_backward(self.ctl_k.as_mut(), dl_dk_ctl_batch));

        let [gradient_q_batch, gradient_k_batch, gradient_v_batch, gradient_input_batch] =
            qkv_projection_backward_rm(input_batch, [&dl_dq_batch, &dl_dk_batch, &dl_dv_batch], [&self.weights_q_rm, &self.weights_k_rm, &self.weights_v_rm]);

        let mut gradient = Gradient::new_default();
        gradient.set_weight_gradient_batch_rm(W_Q, gradient_q_batch);
        gradient.set_weight_gradient_batch_rm(W_K, gradient_k_batch);
        gradient.set_weight_gradient_batch_rm(W_V, gradient_v_batch);
        gradient.set_gradient_input_batch_rm(gradient_input_batch);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        self.backward_rm(&previous_gradient.get_gradient_input_batch_rm_cow())
    }

    /// RM counterpart of `SparseMaskedAttentionHead::update_parameters`.
    pub fn update_parameters(&mut self) {
        let mut gradient = self.gradient.take().expect("Gradient is missing in SparseMaskedAttentionHeadRm");
        let total_valid_tokens: Real = r(self.total_valid_tokens.max(1) as f64);

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

        let (learning_rate, time_step) = (self.learning_rate, self.time_step);
        let previous_gradient = self.previous_gradient.as_ref();
        update_weights_adam_w_rm(W_Q, &mut self.weights_q_rm, &weight_gradients[0], previous_gradient, &mut gradient, learning_rate, time_step);
        update_weights_adam_w_rm(W_K, &mut self.weights_k_rm, &weight_gradients[2], previous_gradient, &mut gradient, learning_rate, time_step);
        update_weights_adam_w_rm(W_V, &mut self.weights_v_rm, &weight_gradients[1], previous_gradient, &mut gradient, learning_rate, time_step);

        self.previous_gradient = Some(gradient);
    }
}

impl AttentionHeadRm for SparseMaskedAttentionHeadRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        SparseMaskedAttentionHeadRm::forward(self, layer_input)
    }
    fn backward_rm(&mut self, previous_gradient_batch: &[RowMajorMatrix<C>]) -> Gradient {
        SparseMaskedAttentionHeadRm::backward_rm(self, previous_gradient_batch)
    }
    fn update_parameters(&mut self) {
        SparseMaskedAttentionHeadRm::update_parameters(self)
    }
}

/// Attention window [start, end) of `token_index` (same window as `SparseMaskedAttentionHead`).
fn calculate_start_end_indices(token_index: usize, window_size: usize, seq_len: usize) -> (usize, usize) {
    let start_ind = if token_index >= window_size { token_index - window_size } else { 0 };
    let end_ind = if token_index + window_size + 1 <= seq_len { token_index + window_size + 1 } else { seq_len };

    (start_ind, end_ind)
}
