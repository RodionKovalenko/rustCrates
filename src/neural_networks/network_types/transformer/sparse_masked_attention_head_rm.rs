use num::{Complex, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    utils::{
        adam_w::calculate_adam_w,
        dtype::{r, C, Real, ZERO},
        matrix::{
            add_matrix, average_matrix_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose_rm, multiply_complex_rm, RowMajorMatrix,
        },
        weights_initializer::initialize_weights_complex,
    },
};

use super::sparse_masked_attention_head::calculate_start_end_indices;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexToLinearLayerRm {
    pub weights_1_rm: RowMajorMatrix<C>,
    pub weights_2_rm: RowMajorMatrix<C>,
    pub learning_rate: Real,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl ComplexToLinearLayerRm {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut w1: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let mut w2: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        // Reuse existing initializer (temporary Vec is not stored).
        initialize_weights_complex(rows, cols, &mut w1);
        initialize_weights_complex(rows, cols, &mut w2);

        Self {
            weights_1_rm: RowMajorMatrix::from_rows(&w1),
            weights_2_rm: RowMajorMatrix::from_rows(&w2),
            learning_rate: r(learning_rate),
            input_batch_rm: None,
            gradient: None,
            time_step: 0,
            batch_size: 0,
        }
    }

    pub fn forward(&mut self, input: &LayerInput, input_batch_rm: &[RowMajorMatrix<C>]) -> Vec<RowMajorMatrix<C>> {
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch_rm = Some(input_batch_rm.to_vec());

        let in_f = self.weights_1_rm.rows;
        let out_f = self.weights_1_rm.cols;
        assert_eq!(self.weights_2_rm.rows, in_f);
        assert_eq!(self.weights_2_rm.cols, out_f);

        input_batch_rm
            .par_iter()
            .map(|input_rm| {
                assert_eq!(input_rm.cols, in_f);
                let time = input_rm.rows;
                let mut out = RowMajorMatrix::from_data(time, out_f, vec![C::new(ZERO, ZERO); time * out_f]);

                for t in 0..time {
                    let in_row = input_rm.row_range(t);
                    for f in 0..out_f {
                        let mut sum_real: Real = ZERO;
                        for k in 0..in_f {
                            let x = input_rm.data[in_row.start + k];
                            // Use only real parts of weights (consistent with existing ComplexToLinearLayer).
                            sum_real += x.re * self.weights_1_rm.data[self.weights_1_rm.idx(k, f)].re
                                + x.im * self.weights_2_rm.data[self.weights_2_rm.idx(k, f)].re;
                        }
                        let out_i = out.idx(t, f);
                        out.data[out_i] = C::new(sum_real, ZERO);
                    }
                }

                out
            })
            .collect()
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let total_valid_tokens = 1usize;
        let input_batch_rm = self
            .input_batch_rm
            .as_ref()
            .expect("ComplexToLinearLayerRm missing input_batch_rm");
        let batch_len = input_batch_rm.len();
        let in_f = self.weights_1_rm.rows;
        let out_f = self.weights_1_rm.cols;

        let mut grad_w1 = vec![vec![vec![C::new(ZERO, ZERO); out_f]; in_f]; batch_len];
        let mut grad_w2 = vec![vec![vec![C::new(ZERO, ZERO); out_f]; in_f]; batch_len];
        let mut gradient_input_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_len);

        for b in 0..batch_len {
            let input_rm = &input_batch_rm[b];
            let grad_rm = &previous_gradient_batch_rm[b];
            assert_eq!(input_rm.cols, in_f);
            assert_eq!(grad_rm.cols, out_f);
            assert_eq!(input_rm.rows, grad_rm.rows);

            let time = input_rm.rows;
            let mut gx_rm = RowMajorMatrix::from_data(time, in_f, vec![C::new(ZERO, ZERO); time * in_f]);

            for t in 0..time {
                let in_row = input_rm.row_range(t);
                let g_row = grad_rm.row_range(t);
                for f in 0..out_f {
                    let g = grad_rm.data[g_row.start + f].re;
                    for k in 0..in_f {
                        // grads for weights (stored as complex with imag=0)
                        grad_w1[b][k][f].re += input_rm.data[in_row.start + k].re * g;
                        grad_w2[b][k][f].re += input_rm.data[in_row.start + k].im * g;

                        // grad for input
                        let gx_i = gx_rm.idx(t, k);
                        gx_rm.data[gx_i].re += g * self.weights_1_rm.data[self.weights_1_rm.idx(k, f)].re;
                        gx_rm.data[gx_i].im += g * self.weights_2_rm.data[self.weights_2_rm.idx(k, f)].re;
                    }
                }
            }

            gradient_input_batch_rm.push(gx_rm);
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(gradient_input_batch_rm);
        gradient.set_gradient_weight_batch(grad_w1);
        gradient.set_gradient_weight_2_batch(grad_w2);
        gradient.set_total_valid_tokens(total_valid_tokens);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.as_mut().expect("ComplexToLinearLayerRm missing gradient");
        let (mut grad_w1, mut grad_w2) = (gradient.get_gradient_weights(), gradient.get_gradient_weights_2());

        let total_valid_tokens = r(gradient.get_total_valid_tokens().max(1) as f64);
        grad_w1 = average_matrix_by_scalar(&grad_w1, total_valid_tokens);
        grad_w2 = average_matrix_by_scalar(&grad_w2, total_valid_tokens);

        // Update via existing adam util working on Vec weights.
        let mut w1 = self.weights_1_rm.to_rows();
        let mut w2 = self.weights_2_rm.to_rows();
        let time_step = self.time_step;
        let learning_rate = self.learning_rate as f64;

        let mut prev_m_w1 = vec![vec![C::new(ZERO, ZERO); w1[0].len()]; w1.len()];
        let mut prev_v_w1 = vec![vec![C::new(ZERO, ZERO); w1[0].len()]; w1.len()];
        let mut prev_v_w1_hat = vec![vec![C::new(ZERO, ZERO); w1[0].len()]; w1.len()];
        let mut prev_m_w2 = vec![vec![C::new(ZERO, ZERO); w2[0].len()]; w2.len()];
        let mut prev_v_w2 = vec![vec![C::new(ZERO, ZERO); w2[0].len()]; w2.len()];
        let mut prev_v_w2_hat = vec![vec![C::new(ZERO, ZERO); w2[0].len()]; w2.len()];

        calculate_adam_w(&mut w1, &grad_w1, &mut prev_m_w1, &mut prev_v_w1, &mut prev_v_w1_hat, learning_rate, time_step);
        calculate_adam_w(&mut w2, &grad_w2, &mut prev_m_w2, &mut prev_v_w2, &mut prev_v_w2_hat, learning_rate, time_step);

        self.weights_1_rm = RowMajorMatrix::from_rows(&w1);
        self.weights_2_rm = RowMajorMatrix::from_rows(&w2);
        self.gradient = None;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMaskedAttentionHeadRm {
    pub weights_q_rm: RowMajorMatrix<C>,
    pub weights_k_rm: RowMajorMatrix<C>,
    pub weights_v_rm: RowMajorMatrix<C>,

    pub learning_rate: f64,

    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub window_size: usize,

    pub ctl_q: ComplexToLinearLayerRm,
    pub ctl_k: ComplexToLinearLayerRm,

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
        let mut wq: Vec<Vec<C>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut wk: Vec<Vec<C>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut wv: Vec<Vec<C>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        initialize_weights_complex(rows, cols, &mut wq);
        initialize_weights_complex(rows, cols, &mut wk);
        initialize_weights_complex(rows, cols, &mut wv);

        Self {
            weights_q_rm: RowMajorMatrix::from_rows(&wq),
            weights_k_rm: RowMajorMatrix::from_rows(&wk),
            weights_v_rm: RowMajorMatrix::from_rows(&wv),
            learning_rate,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            previous_gradient: None,
            window_size,
            ctl_q: ComplexToLinearLayerRm::new(cols, cols, learning_rate),
            ctl_k: ComplexToLinearLayerRm::new(cols, cols, learning_rate),
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

    fn tail_rows_rm(matrix: &RowMajorMatrix<C>, max_rows: usize) -> RowMajorMatrix<C> {
        if matrix.rows <= max_rows {
            return matrix.clone();
        }
        let start_row = matrix.rows - max_rows;
        let start = start_row * matrix.cols;
        RowMajorMatrix::from_data(max_rows, matrix.cols, matrix.data[start..].to_vec())
    }

    fn last_row_rm(matrix: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
        assert!(matrix.rows > 0);
        let start = (matrix.rows - 1) * matrix.cols;
        RowMajorMatrix::from_data(1, matrix.cols, matrix.data[start..start + matrix.cols].to_vec())
    }

    pub fn clear_cache(&mut self) {
        self.k_cache_rm = None;
        self.v_cache_rm = None;
        self.k_ctl_rm = None;
        self.q_ctl_rm = None;
        self.attention_weights_batch_rm = None;
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm = layer_input
            .get_input_batch_rm_ref()
            .expect("SparseMaskedAttentionHeadRm requires RM input");
        if input_batch_rm.is_empty() {
            let mut out = LayerOutput::new_default();
            out.set_output_batch_rm(vec![]);
            return out;
        }

        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();
        self.total_valid_tokens = layer_input.get_total_valid_tokens();

        let cache_limit = 2 * self.window_size;

        // Q for entire sequence
        let q_batch_rm: Vec<_> = input_batch_rm
            .par_iter()
            .map(|input| multiply_complex_rm(input, &self.weights_q_rm))
            .collect();

        // Trim Q/mask to align with KV cache
        let mut q_batch_trimmed_rm: Vec<RowMajorMatrix<C>> = if layer_input.get_calculate_k_v_cache() {
            q_batch_rm.iter().map(|q| Self::tail_rows_rm(q, cache_limit)).collect()
        } else {
            q_batch_rm.clone()
        };

        let padding_mask_batch_trimmed: Vec<Vec<u32>> = if layer_input.get_calculate_k_v_cache() {
            padding_mask_batch
                .iter()
                .map(|mask_seq| {
                    let seq_len = mask_seq.len();
                    if seq_len > cache_limit {
                        mask_seq[seq_len - cache_limit..].to_vec()
                    } else {
                        mask_seq.clone()
                    }
                })
                .collect()
        } else {
            padding_mask_batch.clone()
        };

        // Compute/update K/V cache
        let (k_new_batch_rm, v_new_batch_rm): (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>) = if self.k_cache_rm.is_none() || !layer_input.get_calculate_k_v_cache() {
            let k_new: Vec<_> = input_batch_rm.par_iter().map(|input| multiply_complex_rm(input, &self.weights_k_rm)).collect();
            let v_new: Vec<_> = input_batch_rm.par_iter().map(|input| multiply_complex_rm(input, &self.weights_v_rm)).collect();
            (k_new, v_new)
        } else {
            let k_new: Vec<_> = input_batch_rm
                .par_iter()
                .map(|input| multiply_complex_rm(&Self::last_row_rm(input), &self.weights_k_rm))
                .collect();
            let v_new: Vec<_> = input_batch_rm
                .par_iter()
                .map(|input| multiply_complex_rm(&Self::last_row_rm(input), &self.weights_v_rm))
                .collect();
            (k_new, v_new)
        };

        let (k_cache_rm, v_cache_rm): (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>) = if layer_input.get_calculate_k_v_cache() {
            if self.k_cache_rm.is_none() {
                self.k_cache_rm = Some(k_new_batch_rm.iter().map(|k| Self::tail_rows_rm(k, cache_limit)).collect());
                self.v_cache_rm = Some(v_new_batch_rm.iter().map(|v| Self::tail_rows_rm(v, cache_limit)).collect());
            } else {
                let k_cache = self.k_cache_rm.as_mut().unwrap();
                let v_cache = self.v_cache_rm.as_mut().unwrap();

                for b in 0..k_cache.len() {
                    let mut new_k = k_cache[b].data.clone();
                    new_k.extend_from_slice(&k_new_batch_rm[b].data);
                    let mut new_v = v_cache[b].data.clone();
                    new_v.extend_from_slice(&v_new_batch_rm[b].data);

                    let total_rows = new_k.len() / k_cache[b].cols;
                    let keep_rows = total_rows.min(cache_limit);
                    let start_row = total_rows - keep_rows;

                    k_cache[b] = RowMajorMatrix::from_data(keep_rows, k_cache[b].cols, new_k[start_row * k_cache[b].cols..].to_vec());
                    v_cache[b] = RowMajorMatrix::from_data(keep_rows, v_cache[b].cols, new_v[start_row * v_cache[b].cols..].to_vec());
                }
            }

            (self.k_cache_rm.as_ref().unwrap().clone(), self.v_cache_rm.as_ref().unwrap().clone())
        } else {
            (k_new_batch_rm, v_new_batch_rm)
        };

        // CTL Q/K
        q_batch_trimmed_rm = self.ctl_q.forward(layer_input, &q_batch_trimmed_rm);
        self.q_ctl_rm = Some(q_batch_trimmed_rm.clone());

        let mut k_cache_ctl_rm = k_cache_rm.clone();
        self.k_ctl_rm = Some(k_cache_ctl_rm.clone());
        k_cache_ctl_rm = self.ctl_k.forward(layer_input, &k_cache_ctl_rm);
        self.k_ctl_rm = Some(k_cache_ctl_rm.clone());

        let batch_output_rm = self.calculated_sparse_masked_attention_rm(&q_batch_trimmed_rm, &k_cache_ctl_rm, &v_cache_rm, &padding_mask_batch_trimmed);
        self.output_batch_rm = Some(batch_output_rm.clone());
        self.input_batch_rm = Some(input_batch_rm.to_vec());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(batch_output_rm);
        layer_output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch_rm = self
            .input_batch_rm
            .as_ref()
            .expect("Input batch RM is missing in SparseMaskedAttentionHeadRm");
        let padding_mask_batch = self
            .padding_mask_batch
            .as_ref()
            .expect("Padding mask batch is missing in SparseMaskedAttentionHeadRm");
        let attention_weights_batch_rm = self
            .attention_weights_batch_rm
            .as_ref()
            .expect("Attention weights RM batch is missing in SparseMaskedAttentionHeadRm");

        let batch_size = previous_gradient_batch_rm.len();
        assert_eq!(input_batch_rm.len(), batch_size);
        assert_eq!(attention_weights_batch_rm.len(), batch_size);
        assert_eq!(padding_mask_batch.len(), batch_size);

        let mut gradient_input_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);

        let (rows, cols) = (self.weights_q_rm.rows, self.weights_q_rm.cols);
        let mut gradient_q_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(0.0, 0.0); cols]; rows]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(0.0, 0.0); cols]; rows]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(0.0, 0.0); cols]; rows]; batch_size];

        let q_ctl_rm = self.q_ctl_rm.as_ref().expect("Q CTL RM is missing");
        let k_ctl_rm = self.k_ctl_rm.as_ref().expect("K CTL RM is missing");

        let mut dl_dq_ctl_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
        let mut dl_dk_ctl_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
        let mut dl_dv_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let prev_grad_rm = &previous_gradient_batch_rm[b];
            let seq_len = prev_grad_rm.rows;
            let embed_dim = prev_grad_rm.cols;

            let attn_probs_rm = &attention_weights_batch_rm[b];
            assert_eq!(attn_probs_rm.rows, seq_len);
            let window_width = attn_probs_rm.cols;

            let v_used_rm: RowMajorMatrix<C> = if let Some(v_cache) = self.v_cache_rm.as_ref() {
                v_cache[b].clone()
            } else {
                multiply_complex_rm(&input_batch_rm[b], &self.weights_v_rm)
            };
            assert_eq!(v_used_rm.rows, seq_len);
            assert_eq!(v_used_rm.cols, embed_dim);

            let mut dl_dv = RowMajorMatrix::from_data(seq_len, embed_dim, vec![Complex::zero(); seq_len * embed_dim]);
            let mut dl_da = RowMajorMatrix::from_data(seq_len, window_width, vec![Complex::zero(); seq_len * window_width]);

            for i in 0..seq_len {
                if padding_mask_batch[b][i] == 0 {
                    continue;
                }
                let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
                let len = (end_ind - start_ind).min(window_width);

                let do_row = prev_grad_rm.row_range(i);

                for local_pos in 0..len {
                    let j = start_ind + local_pos;
                    let w = attn_probs_rm.data[attn_probs_rm.idx(i, local_pos)].re;

                    let dv_row = dl_dv.row_range(j);
                    for f in 0..embed_dim {
                        dl_dv.data[dv_row.start + f].re += w * prev_grad_rm.data[do_row.start + f].re;
                    }

                    let v_row = v_used_rm.row_range(j);
                    let mut dot: Real = 0.0;
                    for f in 0..embed_dim {
                        dot += prev_grad_rm.data[do_row.start + f].re * v_used_rm.data[v_row.start + f].re;
                    }
                    let dl_da_i = dl_da.idx(i, local_pos);
                    dl_da.data[dl_da_i] = Complex::new(dot, 0.0);
                }
            }

            // Softmax backward (fixed-window)
            let mut dl_dz = RowMajorMatrix::from_data(seq_len, window_width, vec![Complex::zero(); seq_len * window_width]);
            for i in 0..seq_len {
                if padding_mask_batch[b][i] == 0 {
                    continue;
                }

                let mut sum: Real = 0.0;
                for local_pos in 0..window_width {
                    sum += attn_probs_rm.data[attn_probs_rm.idx(i, local_pos)].re * dl_da.data[dl_da.idx(i, local_pos)].re;
                }

                for local_pos in 0..window_width {
                    let a = attn_probs_rm.data[attn_probs_rm.idx(i, local_pos)].re;
                    let g = dl_da.data[dl_da.idx(i, local_pos)].re;
                    let dl_dz_i = dl_dz.idx(i, local_pos);
                    dl_dz.data[dl_dz_i] = Complex::new(a * (g - sum), 0.0);
                }
            }

            // dQ and dK (fixed-window)
            let q_used_rm = &q_ctl_rm[b];
            let k_used_rm = &k_ctl_rm[b];

            let mut dl_dq_ctl = RowMajorMatrix::from_data(seq_len, embed_dim, vec![Complex::zero(); seq_len * embed_dim]);
            let mut dl_dk_ctl = RowMajorMatrix::from_data(seq_len, embed_dim, vec![Complex::zero(); seq_len * embed_dim]);

            let inv_scale: Real = r(1.0) / ((embed_dim as Real).sqrt() + r(1e-12));

            for i in 0..seq_len {
                let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
                let len = (end_ind - start_ind).min(window_width);

                for local_pos in 0..len {
                    let j = start_ind + local_pos;
                    if j > i {
                        continue;
                    }
                    let g = dl_dz.data[dl_dz.idx(i, local_pos)].re * inv_scale;
                    if g == 0.0 {
                        continue;
                    }

                    for f in 0..embed_dim {
                        let dq_i = dl_dq_ctl.idx(i, f);
                        dl_dq_ctl.data[dq_i].re += g * k_used_rm.data[k_used_rm.idx(j, f)].re;

                        let dk_i = dl_dk_ctl.idx(j, f);
                        dl_dk_ctl.data[dk_i].re += g * q_used_rm.data[q_used_rm.idx(i, f)].re;
                    }
                }
            }

            dl_dq_ctl_batch_rm.push(dl_dq_ctl);
            dl_dk_ctl_batch_rm.push(dl_dk_ctl);
            dl_dv_batch_rm.push(dl_dv);
        }

        // Backprop through CTL layers
        let gq = self.ctl_q.backward_rm(&dl_dq_ctl_batch_rm);
        let gradient_q_ctl_input_rm = gq.get_gradient_input_batch_rm();
        let gk = self.ctl_k.backward_rm(&dl_dk_ctl_batch_rm);
        let gradient_k_ctl_input_rm = gk.get_gradient_input_batch_rm();

        // Project gradients back to inputs and compute weight gradients
        for b in 0..batch_size {
            let input_rm = &input_batch_rm[b];
            let dq_rm = &gradient_q_ctl_input_rm[b];
            let dk_rm = &gradient_k_ctl_input_rm[b];
            let dv_rm = &dl_dv_batch_rm[b];

            // Weight grads: X^H * dY
            let x_h = conjugate_transpose_rm(input_rm);
            let grad_wq_rm = multiply_complex_rm(&x_h, dq_rm);
            let grad_wk_rm = multiply_complex_rm(&x_h, dk_rm);
            let grad_wv_rm = multiply_complex_rm(&x_h, dv_rm);

            gradient_q_batch[b] = grad_wq_rm.to_rows();
            gradient_k_batch[b] = grad_wk_rm.to_rows();
            gradient_v_batch[b] = grad_wv_rm.to_rows();

            // Input grad: dQ*Wq^H + dK*Wk^H + dV*Wv^H
            let wq_h = conjugate_transpose_rm(&self.weights_q_rm);
            let wk_h = conjugate_transpose_rm(&self.weights_k_rm);
            let wv_h = conjugate_transpose_rm(&self.weights_v_rm);

            let gx_q = multiply_complex_rm(dq_rm, &wq_h);
            let gx_k = multiply_complex_rm(dk_rm, &wk_h);
            let gx_v = multiply_complex_rm(dv_rm, &wv_h);

            let gx = add_matrix(&add_matrix(&gx_q.to_rows(), &gx_k.to_rows()), &gx_v.to_rows());
            gradient_input_batch_rm.push(RowMajorMatrix::from_rows(&gx));
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_weights_q_batch(gradient_q_batch);
        gradient.set_gradient_weights_k_batch(gradient_k_batch);
        gradient.set_gradient_weights_v_batch(gradient_v_batch);
        gradient.set_gradient_input_batch_rm(gradient_input_batch_rm);
        gradient.set_total_valid_tokens(self.total_valid_tokens.max(1));

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in SparseMaskedAttentionHeadRm");
        let (mut grad_w_q, mut grad_w_v, mut grad_w_k) = (gradient.get_gradient_weights_q(), gradient.get_gradient_weights_v(), gradient.get_gradient_weights_k());

        let total_valid_tokens = r(self.total_valid_tokens.max(1) as f64);
        grad_w_q = average_matrix_by_scalar(&grad_w_q, total_valid_tokens);
        grad_w_v = average_matrix_by_scalar(&grad_w_v, total_valid_tokens);
        grad_w_k = average_matrix_by_scalar(&grad_w_k, total_valid_tokens);

        clip_all_gradients_by_global_norm_2d(&mut grad_w_q, &mut vec![], self.global_norm, self.max_norm);
        clip_all_gradients_by_global_norm_2d(&mut grad_w_v, &mut vec![], self.global_norm, self.max_norm);
        clip_all_gradients_by_global_norm_2d(&mut grad_w_k, &mut vec![], self.global_norm, self.max_norm);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        let mut wq = self.weights_q_rm.to_rows();
        let mut wk = self.weights_k_rm.to_rows();
        let mut wv = self.weights_v_rm.to_rows();

        let (
            mut prev_m_weights_q,
            mut prev_v_weights_q,
            mut prev_m_weights_k,
            mut prev_v_weights_k,
            mut prev_m_weights_v,
            mut prev_v_weights_v,
            mut prev_v_weights_q_hat,
            mut prev_v_weights_k_hat,
            mut prev_v_weights_v_hat,
        ) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_weigths_q(),
                previous_gradient.get_prev_v_weigths_q(),
                previous_gradient.get_prev_m_weigths_k(),
                previous_gradient.get_prev_v_weights_k(),
                previous_gradient.get_prev_m_weigths_v(),
                previous_gradient.get_prev_v_weights_v(),
                previous_gradient.get_prev_v_weights_q_hat(),
                previous_gradient.get_prev_v_weights_k_hat(),
                previous_gradient.get_prev_v_weights_v_hat(),
            )
        } else {
            (
                vec![vec![Complex::new(0.0, 0.0); wq[0].len()]; wq.len()],
                vec![vec![Complex::new(0.0, 0.0); wq[0].len()]; wq.len()],
                vec![vec![Complex::new(0.0, 0.0); wk[0].len()]; wk.len()],
                vec![vec![Complex::new(0.0, 0.0); wk[0].len()]; wk.len()],
                vec![vec![Complex::new(0.0, 0.0); wv[0].len()]; wv.len()],
                vec![vec![Complex::new(0.0, 0.0); wv[0].len()]; wv.len()],
                vec![vec![Complex::new(0.0, 0.0); wq[0].len()]; wq.len()],
                vec![vec![Complex::new(0.0, 0.0); wk[0].len()]; wk.len()],
                vec![vec![Complex::new(0.0, 0.0); wv[0].len()]; wv.len()],
            )
        };

        calculate_adam_w(&mut wq, &grad_w_q, &mut prev_m_weights_q, &mut prev_v_weights_q, &mut prev_v_weights_q_hat, learning_rate, time_step);
        calculate_adam_w(&mut wk, &grad_w_k, &mut prev_m_weights_k, &mut prev_v_weights_k, &mut prev_v_weights_k_hat, learning_rate, time_step);
        calculate_adam_w(&mut wv, &grad_w_v, &mut prev_m_weights_v, &mut prev_v_weights_v, &mut prev_v_weights_v_hat, learning_rate, time_step);

        self.weights_q_rm = RowMajorMatrix::from_rows(&wq);
        self.weights_k_rm = RowMajorMatrix::from_rows(&wk);
        self.weights_v_rm = RowMajorMatrix::from_rows(&wv);

        gradient.set_prev_m_weights_q(prev_m_weights_q);
        gradient.set_prev_v_weights_q(prev_v_weights_q);
        gradient.set_prev_v_weights_q_hat(prev_v_weights_q_hat);
        gradient.set_prev_m_weights_k(prev_m_weights_k);
        gradient.set_prev_v_weights_k(prev_v_weights_k);
        gradient.set_prev_v_weights_k_hat(prev_v_weights_k_hat);
        gradient.set_prev_m_weights_v(prev_m_weights_v);
        gradient.set_prev_v_weights_v(prev_v_weights_v);
        gradient.set_prev_v_weights_v_hat(prev_v_weights_v_hat);

        self.previous_gradient = Some(gradient.clone());
        self.gradient = None;

        self.ctl_q.update_parameters();
        self.ctl_k.update_parameters();
    }

    fn calculated_sparse_masked_attention_rm(
        &mut self,
        q_batch: &[RowMajorMatrix<C>],
        k_batch: &[RowMajorMatrix<C>],
        v_batch: &[RowMajorMatrix<C>],
        padding_mask_batch: &[Vec<u32>],
    ) -> Vec<RowMajorMatrix<C>> {
        assert_eq!(q_batch.len(), k_batch.len());
        assert_eq!(q_batch.len(), v_batch.len());
        assert_eq!(q_batch.len(), padding_mask_batch.len());

        let window_width = 2 * self.window_size + 1;

        let attn_probs_batch_rm: Vec<RowMajorMatrix<C>> = q_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, q_rm)| {
                let logits = self.calculate_local_attention_logits_rm_fixed_window(q_rm, &k_batch[batch_ind], window_width, true);
                self.softmax_fixed_window_rm(&logits, &padding_mask_batch[batch_ind])
            })
            .collect();

        let batch_output_rm: Vec<RowMajorMatrix<C>> = attn_probs_batch_rm
            .iter()
            .enumerate()
            .map(|(batch_ind, probs)| self.multiply_sparse_fixed_window_rm(probs, &v_batch[batch_ind]))
            .collect();

        self.attention_weights_batch_rm = Some(attn_probs_batch_rm);
        self.output_batch_rm = Some(batch_output_rm.clone());
        batch_output_rm
    }

    fn calculate_local_attention_logits_rm_fixed_window(
        &self,
        q: &RowMajorMatrix<C>,
        k: &RowMajorMatrix<C>,
        window_width: usize,
        scale_by_dk: bool,
    ) -> RowMajorMatrix<C> {
        assert_eq!(q.cols, k.cols);
        assert_eq!(q.rows, k.rows);

        let seq_len = q.rows;
        let d_k_sqrt: Real = (q.cols as Real).sqrt() + r(1e-12);
        let inv_scale: Real = if scale_by_dk { r(1.0) / d_k_sqrt } else { r(1.0) };

        let neg_inf = Complex::new(Real::NEG_INFINITY, Real::NEG_INFINITY);
        let mut logits = RowMajorMatrix::from_data(seq_len, window_width, vec![neg_inf; seq_len * window_width]);

        for i in 0..seq_len {
            let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
            let len = (end_ind - start_ind).min(window_width);

            let q_row = q.row_range(i);
            for local_pos in 0..len {
                let j = start_ind + local_pos;
                if j > i {
                    continue;
                }
                let k_row = k.row_range(j);
                let mut sum = Complex::zero();
                for f in 0..q.cols {
                    sum += q.data[q_row.start + f] * k.data[k_row.start + f];
                }
                sum *= inv_scale;
                logits.data[i * window_width + local_pos] = sum;
            }
        }

        logits
    }

    fn softmax_fixed_window_rm(&self, logits: &RowMajorMatrix<C>, padding_mask: &Vec<u32>) -> RowMajorMatrix<C> {
        assert_eq!(logits.rows, padding_mask.len());
        let seq_len = logits.rows;
        let window_width = logits.cols;

        let mut out = RowMajorMatrix::from_data(seq_len, window_width, vec![Complex::zero(); seq_len * window_width]);

        for i in 0..seq_len {
            if padding_mask[i] == 0 {
                continue;
            }

            let row = logits.row_range(i);
            let mut max_re = Real::NEG_INFINITY;
            for local_pos in 0..window_width {
                let v = logits.data[row.start + local_pos].re;
                if v.is_finite() {
                    max_re = max_re.max(v);
                }
            }

            if !max_re.is_finite() {
                continue;
            }

            let mut sum: Real = r(0.0);
            for local_pos in 0..window_width {
                let v = logits.data[row.start + local_pos].re;
                if v.is_finite() {
                    sum += (v - max_re).exp();
                }
            }

            if sum <= r(0.0) {
                continue;
            }

            for local_pos in 0..window_width {
                let v = logits.data[row.start + local_pos].re;
                if v.is_finite() {
                    let p = (v - max_re).exp() / sum;
                    out.data[i * window_width + local_pos] = Complex::new(p, r(0.0));
                }
            }
        }

        out
    }

    fn multiply_sparse_fixed_window_rm(&self, attention_probs: &RowMajorMatrix<C>, v: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
        let seq_len = attention_probs.rows;
        let window_width = attention_probs.cols;
        let embedding_dim = v.cols;
        assert_eq!(v.rows, seq_len);

        let mut out = RowMajorMatrix::from_data(seq_len, embedding_dim, vec![Complex::zero(); seq_len * embedding_dim]);

        for i in 0..seq_len {
            let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
            let len = (end_ind - start_ind).min(window_width);

            for k_col in 0..embedding_dim {
                let mut acc: Real = r(0.0);
                for local_pos in 0..len {
                    let j = start_ind + local_pos;
                    let w = attention_probs.data[attention_probs.idx(i, local_pos)].re;
                    acc += w * v.data[v.idx(j, k_col)].re;
                }
                out.data[i * embedding_dim + k_col] = Complex::new(acc, r(0.0));
            }
        }

        out
    }
}
