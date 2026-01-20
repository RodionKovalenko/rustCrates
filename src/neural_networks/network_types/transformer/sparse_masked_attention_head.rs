use std::{
    f64,
};

use num::{Complex, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_layers::complex_to_linear_layer::ComplexToLinearLayer;
use crate::neural_networks::network_layers::layer::LayerType;
use crate::neural_networks::utils::dtype::{r, C, Real};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    utils::{
        activation::softmax_complex_padding_complex,
        adam_w::calculate_adam_w,
        low_rank_approx::transpose,
        matrix::{
            add_matrix, average_matrix_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose, conjugate_transpose_rm, multiply_complex, multiply_complex_rm,
            RowMajorMatrix,
        },
        weights_initializer::initialize_weights_complex,
    },
};
use std::fmt::Debug;

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMaskedAttentionHead {
    pub weights_q: Vec<Vec<C>>,
    pub weights_k: Vec<Vec<C>>,
    pub weights_v: Vec<Vec<C>>,

    pub bias_q: Vec<C>,
    pub bias_k: Vec<C>,
    pub bias_v: Vec<C>,

    pub layer_type: LayerType,
    pub learning_rate: f64,

    pub m1: Vec<Vec<C>>,
    pub v1: Vec<Vec<C>>,

    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub window_size: usize,

    pub ctl_q: Option<ComplexToLinearLayer>,
    pub ctl_k: Option<ComplexToLinearLayer>,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub attention_weights_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub attention_weights_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub attention_weights_batch_raw: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub k_cache: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub v_cache: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub k_cache_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub v_cache_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub k_ctl: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub q_ctl: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub k_ctl_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub q_ctl_rm: Option<Vec<RowMajorMatrix<C>>>,

    pub weights_q_rm: Option<RowMajorMatrix<C>>,
    pub weights_k_rm: Option<RowMajorMatrix<C>>,
    pub weights_v_rm: Option<RowMajorMatrix<C>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub total_valid_tokens: usize,
}

impl SparseMaskedAttentionHead {
    pub fn new(rows: usize, cols: usize, window_size: usize, learning_rate: f64) -> Self {
        let mut weights_q: Vec<Vec<C>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_k: Vec<Vec<C>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_v: Vec<Vec<C>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

        let ctl_q = ComplexToLinearLayer::new(cols, cols, learning_rate);
        let ctl_k = ComplexToLinearLayer::new(cols, cols, learning_rate);

        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);

        let bias_q: Vec<C> = vec![Complex::new(1.0, 0.0); cols];
        let bias_k: Vec<C> = vec![Complex::new(1.0, 0.0); cols];
        let bias_v: Vec<C> = vec![Complex::new(1.0, 0.0); cols];

        SparseMaskedAttentionHead {
            weights_q,
            weights_k,
            weights_v,

            bias_q,
            bias_k,
            bias_v,
            layer_type: LayerType::InputLayer,
            learning_rate: learning_rate,

            smoothing: 0.99,
            ema: 0.0,
            window_size: window_size,

            global_norm: 0.0,
            max_norm: 0.0,

            ctl_q: Some(ctl_q),
            ctl_k: Some(ctl_k),

            gradient: None,
            previous_gradient: None,
            input_batch: None,
            input_batch_rm: None,
            output_batch: None,
            output_batch_rm: None,
            padding_mask_batch: None,
            attention_weights_batch: None,
            attention_weights_batch_rm: None,
            attention_weights_batch_raw: None,
            k_cache: None,
            v_cache: None,
            k_cache_rm: None,
            v_cache_rm: None,
            k_ctl: None,
            q_ctl: None,
            k_ctl_rm: None,
            q_ctl_rm: None,
            weights_q_rm: None,
            weights_k_rm: None,
            weights_v_rm: None,
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            time_step: 0,
            batch_size: 0,
            total_valid_tokens: 1,
        }
    }

    fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }

    pub fn clear_cache(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
        self.k_ctl = None;
        self.q_ctl = None;

        self.k_cache_rm = None;
        self.v_cache_rm = None;
        self.k_ctl_rm = None;
        self.q_ctl_rm = None;

        self.attention_weights_batch_rm = None;
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, window_size: usize, learning_rate: f64) -> SparseMaskedAttentionHead {
        let mut attention_layer: SparseMaskedAttentionHead = SparseMaskedAttentionHead::new(rows, cols, window_size, learning_rate);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl SparseMaskedAttentionHead {
    fn ensure_weights_cache_rm(&mut self) {
        let rebuild_q = self.weights_q_rm.is_none()
            || self
                .weights_q_rm
                .as_ref()
                .is_some_and(|w| w.rows != self.weights_q.len() || w.cols != self.weights_q[0].len());
        if rebuild_q {
            self.weights_q_rm = Some(RowMajorMatrix::from_rows(&self.weights_q));
        }
        let rebuild_k = self.weights_k_rm.is_none()
            || self
                .weights_k_rm
                .as_ref()
                .is_some_and(|w| w.rows != self.weights_k.len() || w.cols != self.weights_k[0].len());
        if rebuild_k {
            self.weights_k_rm = Some(RowMajorMatrix::from_rows(&self.weights_k));
        }
        let rebuild_v = self.weights_v_rm.is_none()
            || self
                .weights_v_rm
                .as_ref()
                .is_some_and(|w| w.rows != self.weights_v.len() || w.cols != self.weights_v[0].len());
        if rebuild_v {
            self.weights_v_rm = Some(RowMajorMatrix::from_rows(&self.weights_v));
        }
    }

    pub fn prepare_for_save(&mut self) {
        self.ensure_weights_cache_rm();
    }

    fn tail_rows_rm(matrix: &RowMajorMatrix<C>, max_rows: usize) -> RowMajorMatrix<C> {
        if matrix.rows <= max_rows {
            return matrix.clone();
        }

        let start_row = matrix.rows - max_rows;
        let start = start_row * matrix.cols;
        let data = matrix.data[start..].to_vec();
        RowMajorMatrix::from_data(max_rows, matrix.cols, data)
    }

    fn last_row_rm(matrix: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
        assert!(matrix.rows > 0);
        let start = (matrix.rows - 1) * matrix.cols;
        let data = matrix.data[start..start + matrix.cols].to_vec();
        RowMajorMatrix::from_data(1, matrix.cols, data)
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm_ref = layer_input.get_input_batch_rm_ref();
        let input_batch_ref = layer_input.get_input_batch_ref();
        let use_rm = input_batch_ref.is_none() && input_batch_rm_ref.is_some();

        if use_rm {
            let input_batch_rm = input_batch_rm_ref.unwrap();
            self.ensure_weights_cache_rm();
            let weights_q_rm = self.weights_q_rm.as_ref().unwrap();
            let weights_k_rm = self.weights_k_rm.as_ref().unwrap();
            let weights_v_rm = self.weights_v_rm.as_ref().unwrap();

            let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
            // Keep the original mask around; softmax uses row-index masking.
            self.padding_mask_batch = Some(padding_mask_batch.clone());
            self.time_step = layer_input.get_time_step();
            self.batch_size = layer_input.get_batch_size();
            self.total_valid_tokens = layer_input.get_total_valid_tokens();

            self.input_batch = None;
            self.output_batch = None;
            self.k_cache = None;
            self.v_cache = None;
            self.k_ctl = None;
            self.q_ctl = None;

            self.attention_weights_batch = None;
            self.attention_weights_batch_rm = None;

            self.input_batch_rm = Some(input_batch_rm.to_vec());

            let cache_limit = 2 * self.window_size;

            // Step 1: Compute Q for the entire sequence
            let q_batch_rm: Vec<RowMajorMatrix<C>> = input_batch_rm.par_iter().map(|input| multiply_complex_rm(input, weights_q_rm)).collect();

            // Step 1b: Trim Q batch for KV-cache inference
            let mut q_batch_trimmed_rm: Vec<RowMajorMatrix<C>> = if layer_input.get_calculate_k_v_cache() {
                q_batch_rm.iter().map(|q| Self::tail_rows_rm(q, cache_limit)).collect()
            } else {
                q_batch_rm.clone()
            };

            // Cache pre-CTL Q (will be overwritten if CTL is enabled)
            self.q_ctl_rm = Some(q_batch_trimmed_rm.clone());

            // Trim padding mask to match trimmed Q tokens
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

            // Steps 2 & 3: Compute/update K/V cache
            let (k_new_batch_rm, v_new_batch_rm): (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>) =
                if self.k_cache_rm.is_none() || !layer_input.get_calculate_k_v_cache() {
                    let k_new: Vec<_> = input_batch_rm.par_iter().map(|input| multiply_complex_rm(input, weights_k_rm)).collect();
                    let v_new: Vec<_> = input_batch_rm.par_iter().map(|input| multiply_complex_rm(input, weights_v_rm)).collect();
                    (k_new, v_new)
                } else {
                    // In KV-cache inference mode, only compute projections for the newest token.
                    let k_new: Vec<_> = input_batch_rm
                        .par_iter()
                        .map(|input| {
                            let last = Self::last_row_rm(input);
                            multiply_complex_rm(&last, weights_k_rm)
                        })
                        .collect();
                    let v_new: Vec<_> = input_batch_rm
                        .par_iter()
                        .map(|input| {
                            let last = Self::last_row_rm(input);
                            multiply_complex_rm(&last, weights_v_rm)
                        })
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

            // CTL Q/K (RM)
            if let Some(ctl_q) = &mut self.ctl_q {
                let mut local_in = layer_input.clone();
                local_in.clear_input_batch();
                local_in.set_input_batch_rm(q_batch_trimmed_rm.clone());
                let ctl_q_output = ctl_q.forward(&local_in);
                if let Some(out_rm) = ctl_q_output.get_output_batch_rm_ref() {
                    q_batch_trimmed_rm = out_rm.to_vec();
                    self.q_ctl_rm = Some(q_batch_trimmed_rm.clone());
                } else {
                    panic!("ComplexToLinearLayer did not return RM output");
                }
            }

            let mut k_cache_ctl_rm = k_cache_rm.clone();

            // Cache pre-CTL K (will be overwritten if CTL is enabled)
            self.k_ctl_rm = Some(k_cache_ctl_rm.clone());
            if let Some(ctl_k) = &mut self.ctl_k {
                let mut local_in = layer_input.clone();
                local_in.clear_input_batch();
                local_in.set_input_batch_rm(k_cache_ctl_rm.clone());
                let ctl_k_output = ctl_k.forward(&local_in);
                if let Some(out_rm) = ctl_k_output.get_output_batch_rm_ref() {
                    k_cache_ctl_rm = out_rm.to_vec();
                    self.k_ctl_rm = Some(k_cache_ctl_rm.clone());
                } else {
                    panic!("ComplexToLinearLayer did not return RM output");
                }
            }

            let batch_output_rm = self.calculated_sparse_masked_attention_rm(&q_batch_trimmed_rm, &k_cache_ctl_rm, &v_cache_rm, &padding_mask_batch_trimmed);
            self.output_batch_rm = Some(batch_output_rm.clone());

            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch_rm(batch_output_rm);
            return layer_output;
        }

        let input_batch: Vec<Vec<Vec<C>>> = layer_input.get_input_batch();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();

        self.input_batch = Some(input_batch.clone());
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();
        self.total_valid_tokens = layer_input.get_total_valid_tokens();
        let cache_limit = 2 * self.window_size;

        // Step 1: Compute Q for the entire sequence (all tokens up to current step)
        let q_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_q)).collect();

        // Step 1b: Trim Q batch to last cache_limit tokens for inference to align with KV cache
        let mut q_batch_trimmed: Vec<Vec<Vec<C>>> = if layer_input.get_calculate_k_v_cache() {
            q_batch
                .iter()
                .map(|q_seq| {
                    let seq_len = q_seq.len();
                    if seq_len > cache_limit {
                        q_seq[seq_len - cache_limit..].to_vec()
                    } else {
                        q_seq.clone()
                    }
                })
                .collect()
        } else {
            q_batch.clone()
        };

        // Similarly trim padding mask to align with trimmed Q tokens
        let padding_mask_batch: Vec<Vec<u32>> = if layer_input.get_calculate_k_v_cache() {
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

        // Steps 2 & 3: Compute K and V for new tokens and update KV cache (your existing logic)
        let (k_new_batch, v_new_batch): (Vec<_>, Vec<_>) = if self.k_cache.is_none() || !layer_input.get_calculate_k_v_cache() {
            let k_new_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_k)).collect();
            let v_new_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_v)).collect();
            (k_new_batch, v_new_batch)
        } else {
            let k_new_batch: Vec<_> = input_batch.last().map_or(vec![], |input| {
                input.last().map_or(vec![], |last_token: &Vec<C>| vec![multiply_complex(&vec![last_token.clone()], &self.weights_k)])
            });

            let v_new_batch: Vec<_> = input_batch.last().map_or(vec![], |input| {
                input.last().map_or(vec![], |last_token| vec![multiply_complex(&vec![last_token.clone()], &self.weights_v)])
            });

            (k_new_batch, v_new_batch)
        };

        let (mut k_cache, v_cache): (Vec<Vec<Vec<C>>>, Vec<Vec<Vec<C>>>) = if layer_input.get_calculate_k_v_cache() {
            if self.k_cache.is_none() {
                self.k_cache = Some(
                    k_new_batch
                        .iter()
                        .map(|k_batch| k_batch.iter().rev().take(cache_limit).cloned().collect::<Vec<_>>().into_iter().rev().collect())
                        .collect(),
                );

                self.v_cache = Some(
                    v_new_batch
                        .iter()
                        .map(|v_batch| v_batch.iter().rev().take(cache_limit).cloned().collect::<Vec<_>>().into_iter().rev().collect())
                        .collect(),
                );
            } else {
                let k_cache = self.k_cache.as_mut().unwrap();
                let v_cache = self.v_cache.as_mut().unwrap();

                for ((cache_k, new_k), (cache_v, new_v)) in k_cache.iter_mut().zip(&k_new_batch).zip(v_cache.iter_mut().zip(&v_new_batch)) {
                    cache_k.extend_from_slice(new_k);
                    cache_v.extend_from_slice(new_v);

                    if cache_k.len() > cache_limit {
                        let start = cache_k.len() - cache_limit;
                        *cache_k = cache_k[start..].to_vec();
                        *cache_v = cache_v[start..].to_vec();
                    }
                }
            }
            (self.k_cache.as_ref().unwrap().clone(), self.v_cache.as_ref().unwrap().clone())
        } else {
            (k_new_batch, v_new_batch)
        };

        if let Some(ctl_q) = &mut self.ctl_q {
            let mut layer_input = layer_input.clone();
            layer_input.set_input_batch(q_batch_trimmed.clone());
            let ctl_q_output = ctl_q.forward(&layer_input);
            q_batch_trimmed = ctl_q_output.get_output_batch();

            self.q_ctl = Some(q_batch_trimmed.clone());
        }

        if let Some(ctl_k) = &mut self.ctl_k {
            let mut layer_input = layer_input.clone();
            layer_input.set_input_batch(k_cache.clone());
            let ctl_k_output = ctl_k.forward(&layer_input);
            k_cache = ctl_k_output.get_output_batch();

            self.k_ctl = Some(k_cache.clone());
        }

        // Step 4: Use trimmed Q batch and trimmed padding mask for attention computation
        let batch_output: Vec<_> = self.calculated_sparse_masked_attention(q_batch_trimmed, k_cache, v_cache, padding_mask_batch);
        self.output_batch = Some(batch_output.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output);

        layer_output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        self.ensure_weights_cache_rm();
        let weights_q_rm = self.weights_q_rm.as_ref().unwrap();
        let weights_k_rm = self.weights_k_rm.as_ref().unwrap();
        let weights_v_rm = self.weights_v_rm.as_ref().unwrap();

        let input_batch_rm = self
            .input_batch_rm
            .as_ref()
            .expect("Input batch RM is missing in sparse attention head");
        let padding_mask_batch = self
            .padding_mask_batch
            .as_ref()
            .expect("Padding mask batch is missing in sparse attention head");
        let attention_weights_batch_rm = self
            .attention_weights_batch_rm
            .as_ref()
            .expect("Attention weights RM batch is missing in sparse attention head");

        let batch_size = previous_gradient_batch_rm.len();
        assert_eq!(input_batch_rm.len(), batch_size);
        assert_eq!(attention_weights_batch_rm.len(), batch_size);
        assert_eq!(padding_mask_batch.len(), batch_size);

        let mut gradient_input_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
        let mut gradient_q_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_v[0].len()]; self.weights_v.len()]; batch_size];

        let q_ctl_rm = self
            .q_ctl_rm
            .as_ref()
            .expect("Q CTL RM is missing in sparse attention head");
        let k_ctl_rm = self
            .k_ctl_rm
            .as_ref()
            .expect("K CTL RM is missing in sparse attention head");

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

            // V used in forward. Prefer cache if present.
            let v_used_rm: RowMajorMatrix<C> = if let Some(v_cache) = self.v_cache_rm.as_ref() {
                v_cache[b].clone()
            } else {
                multiply_complex_rm(&input_batch_rm[b], weights_v_rm)
            };
            assert_eq!(v_used_rm.rows, seq_len);
            assert_eq!(v_used_rm.cols, embed_dim);

            // dV and dA (sparse, stored as fixed-width RM: seq_len x window_width)
            let mut dl_dv = RowMajorMatrix::from_data(seq_len, embed_dim, vec![Complex::zero(); seq_len * embed_dim]);
            let mut dl_da = RowMajorMatrix::from_data(seq_len, window_width, vec![Complex::zero(); seq_len * window_width]);

            for i in 0..seq_len {
                if padding_mask_batch[b][i] == 0 {
                    continue;
                }
                let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
                let len = end_ind - start_ind;
                let len = len.min(window_width);

                let do_row = prev_grad_rm.row_range(i);

                for local_pos in 0..len {
                    let j = start_ind + local_pos;
                    let w = attn_probs_rm.data[attn_probs_rm.idx(i, local_pos)].re;

                    // dV[j] += w * dO[i]
                    let dv_row = dl_dv.row_range(j);
                    for f in 0..embed_dim {
                        dl_dv.data[dv_row.start + f].re += w * prev_grad_rm.data[do_row.start + f].re;
                    }

                    // dA[i, local_pos] = dO[i] · V[j]
                    let v_row = v_used_rm.row_range(j);
                    let mut dot: Real = 0.0;
                    for f in 0..embed_dim {
                        dot += prev_grad_rm.data[do_row.start + f].re * v_used_rm.data[v_row.start + f].re;
                    }
                    let a_i = dl_da.idx(i, local_pos);
                    dl_da.data[a_i] = Complex::new(dot, 0.0);
                }
            }

            // Softmax backward: dZ = A * (dA - <A,dA>)
            let mut dl_dz = RowMajorMatrix::from_data(seq_len, window_width, vec![Complex::zero(); seq_len * window_width]);
            for i in 0..seq_len {
                if padding_mask_batch[b][i] == 0 {
                    continue;
                }
                let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
                let len = (end_ind - start_ind).min(window_width);

                let mut dot: Real = 0.0;
                for local_pos in 0..len {
                    let a = attn_probs_rm.data[attn_probs_rm.idx(i, local_pos)].re;
                    let u = dl_da.data[dl_da.idx(i, local_pos)].re;
                    dot += a * u;
                }

                for local_pos in 0..len {
                    let a = attn_probs_rm.data[attn_probs_rm.idx(i, local_pos)].re;
                    let u = dl_da.data[dl_da.idx(i, local_pos)].re;
                    let dz_i = i * window_width + local_pos;
                    dl_dz.data[dz_i] = Complex::new(a * (u - dot), 0.0);
                }
            }

            // dQ_ctl and dK_ctl from sparse scores: scores = (Q · K^T) / sqrt(dk)
            let d_k = k_ctl_rm[b].cols as Real;
            let scale: Real = r(1.0) / (d_k.sqrt() + r(1e-12));
            let mut dl_dq_ctl = RowMajorMatrix::from_data(seq_len, q_ctl_rm[b].cols, vec![Complex::zero(); seq_len * q_ctl_rm[b].cols]);
            let mut dl_dk_ctl = RowMajorMatrix::from_data(seq_len, k_ctl_rm[b].cols, vec![Complex::zero(); seq_len * k_ctl_rm[b].cols]);

            for i in 0..seq_len {
                if padding_mask_batch[b][i] == 0 {
                    continue;
                }
                let (start_ind, end_ind) = calculate_start_end_indices(i, self.window_size, seq_len);
                let len = (end_ind - start_ind).min(window_width);

                let q_row = q_ctl_rm[b].row_range(i);

                for local_pos in 0..len {
                    let j = start_ind + local_pos;
                    let dz = dl_dz.data[dl_dz.idx(i, local_pos)];
                    let k_row = k_ctl_rm[b].row_range(j);

                    // dQ[i] += dz * K[j] * scale
                    let dq_row = dl_dq_ctl.row_range(i);
                    for f in 0..q_ctl_rm[b].cols {
                        dl_dq_ctl.data[dq_row.start + f] += dz * (k_ctl_rm[b].data[k_row.start + f] * scale);
                    }

                    // dK[j] += dz * Q[i] * scale
                    let dk_row = dl_dk_ctl.row_range(j);
                    for f in 0..k_ctl_rm[b].cols {
                        dl_dk_ctl.data[dk_row.start + f] += dz * (q_ctl_rm[b].data[q_row.start + f] * scale);
                    }
                }
            }

            // dWv = X^H * dV
            let x_h = conjugate_transpose_rm(&input_batch_rm[b]);
            let d_wv_rm = multiply_complex_rm(&x_h, &dl_dv);
            gradient_v_batch[b] = d_wv_rm.to_rows();

            dl_dq_ctl_batch_rm.push(dl_dq_ctl);
            dl_dk_ctl_batch_rm.push(dl_dk_ctl);
            dl_dv_batch_rm.push(dl_dv);

            let _ = embed_dim;
        }

        // Backprop through CTL layers (RM)
        let mut dl_dq_batch_rm: Vec<RowMajorMatrix<C>> = dl_dq_ctl_batch_rm.clone();
        if self.ctl_q.is_some() {
            let mut g = Gradient::new_default();
            g.set_gradient_input_batch_rm(dl_dq_ctl_batch_rm);
            let ctl_q = self.ctl_q.as_mut().expect("CTL Q is missing");
            let ctl_q_gradient = ctl_q.backward(&g);
            dl_dq_batch_rm = ctl_q_gradient.get_gradient_input_batch_rm();
        }

        let mut dl_dk_batch_rm: Vec<RowMajorMatrix<C>> = dl_dk_ctl_batch_rm.clone();
        if self.ctl_k.is_some() {
            let mut g = Gradient::new_default();
            g.set_gradient_input_batch_rm(dl_dk_ctl_batch_rm);
            let ctl_k = self.ctl_k.as_mut().expect("CTL K is missing");
            let ctl_k_gradient = ctl_k.backward(&g);
            dl_dk_batch_rm = ctl_k_gradient.get_gradient_input_batch_rm();
        }

        // Parameter grads and input grads
        for b in 0..batch_size {
            let dl_dq = &dl_dq_batch_rm[b];
            let dl_dk = &dl_dk_batch_rm[b];

            let x_h = conjugate_transpose_rm(&input_batch_rm[b]);

            let d_wq_rm = multiply_complex_rm(&x_h, dl_dq);
            gradient_q_batch[b] = d_wq_rm.to_rows();

            let d_wk_rm = multiply_complex_rm(&x_h, dl_dk);
            gradient_k_batch[b] = d_wk_rm.to_rows();

            let wq_h = conjugate_transpose_rm(weights_q_rm);
            let wk_h = conjugate_transpose_rm(weights_k_rm);
            let wv_h = conjugate_transpose_rm(weights_v_rm);

            let dl_dqx = multiply_complex_rm(dl_dq, &wq_h);
            let dl_dkx = multiply_complex_rm(dl_dk, &wk_h);

            let dl_dvx = multiply_complex_rm(&dl_dv_batch_rm[b], &wv_h);

            let mut grad_in = dl_dqx;
            for i in 0..grad_in.data.len() {
                grad_in.data[i] += dl_dkx.data[i];
                grad_in.data[i] += dl_dvx.data[i];
            }
            gradient_input_batch_rm.push(grad_in);
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_weights_v_batch(gradient_v_batch);
        gradient.set_gradient_weights_q_batch(gradient_q_batch);
        gradient.set_gradient_weights_k_batch(gradient_k_batch);
        gradient.set_gradient_input_batch_rm(gradient_input_batch_rm);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn calculated_sparse_masked_attention(
        &mut self,
        q_batch: Vec<Vec<Vec<C>>>,
        k_batch: Vec<Vec<Vec<C>>>,
        v_batch: Vec<Vec<Vec<C>>>,
        padding_mask_batch: Vec<Vec<u32>>,
    ) -> Vec<Vec<Vec<C>>> {
        let k_windows: Vec<Vec<Vec<Vec<C>>>> = calculate_window_tokens_batch(&k_batch, self.window_size);

        let attention_weights_inactivated_compressed: Vec<Vec<Vec<C>>> = q_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, q)| {
                let mut sparse_attention_weights_inactivated = self.calculate_local_attention(q, &k_windows[batch_ind], true);

                // println!("Attention weights before mask: {:?}", sparse_attention_weights_inactivated);
                self.apply_sparse_causal_mask(&mut sparse_attention_weights_inactivated);
                sparse_attention_weights_inactivated
            })
            .collect();

        let attention_weights_activated_compressed: Vec<Vec<Vec<C>>> = attention_weights_inactivated_compressed
            .par_iter()
            .enumerate()
            .map(|(batch_ind, sparse_attention_weights_inactivated)| {
                let softmax_output: Vec<Vec<C>> = softmax_complex_padding_complex(sparse_attention_weights_inactivated, &padding_mask_batch[batch_ind]);
                softmax_output
            })
            .collect();

        let batch_output_compr: Vec<Vec<Vec<C>>> = attention_weights_activated_compressed
            .iter()
            .enumerate()
            .map(|(batch_ind, attention_weights)| self.multiply_sparse(attention_weights, &v_batch[batch_ind]))
            .collect();

        self.attention_weights_batch = Some(attention_weights_activated_compressed);
        self.output_batch = Some(batch_output_compr.clone());

        batch_output_compr
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
                    // causal mask
                    continue;
                }
                let k_row = k.row_range(j);
                let mut sum = Complex::zero();
                for f in 0..q.cols {
                    sum += q.data[q_row.start + f] * k.data[k_row.start + f];
                }
                sum *= inv_scale;
                let logits_i = i * window_width + local_pos;
                logits.data[logits_i] = sum;
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
                // All masked: keep zeros
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
                    let out_i = i * window_width + local_pos;
                    out.data[out_i] = Complex::new(p, r(0.0));
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
                let out_i = i * embedding_dim + k_col;
                out.data[out_i] = Complex::new(acc, r(0.0));
            }
        }

        out
    }

    pub fn multiply_sparse(&self, attention_sparse_weights: &Vec<Vec<C>>, v: &Vec<Vec<C>>) -> Vec<Vec<C>> {
        let seq_len = attention_sparse_weights.len();
        let embedding_dim = v[0].len();

        let mut output = vec![vec![Complex::zero(); embedding_dim]; seq_len];

        for q in 0..seq_len {
            let (start_ind, _) = calculate_start_end_indices(q, self.window_size, seq_len);

            for k in 0..v[q].len() {
                for f in 0..attention_sparse_weights[q].len() {
                    output[q][k] += (attention_sparse_weights[q][f].re * v[start_ind + f][k]).re;
                }
            }
        }

        output
    }

    pub fn multiply_sparse_backward(&self, v: &Vec<Vec<C>>, attention_sparse_weights: &Vec<Vec<C>>, use_conjugate: bool) -> Vec<Vec<C>> {
        let n_rows = v.len();
        let n_cols = attention_sparse_weights.len();

        let mut output = vec![vec![Complex::zero(); n_cols]; n_rows];

        for k in 0..n_rows {
            for q in 0..n_cols {
                for p in 0..n_cols {
                    let (start_ind, end_ind) = calculate_start_end_indices(p, self.window_size, n_cols);

                    if q >= start_ind && q < end_ind {
                        let position = (start_ind..end_ind).position(|x| x == q).unwrap();

                        if use_conjugate {
                            output[k][q] += v[k][p] * attention_sparse_weights[p][position].conj();
                        } else {
                            output[k][q] += (v[k][p] * attention_sparse_weights[p][position].re).re;
                        }
                    }
                }
            }
        }

        output
    }

    pub fn transpose_sparse(&self, sparse_matrix: &Vec<Vec<C>>, sparse_matrix_ind: &Vec<Vec<usize>>) -> Vec<Vec<C>> {
        let n_rows = sparse_matrix.len();

        let mut output: Vec<Vec<C>> = Vec::new();

        for i in 0..n_rows {
            output.push(vec![Complex::zero(); sparse_matrix[i].len()]);
        }

        for i in 0..n_rows {
            for j in 0..sparse_matrix[i].len() {
                let position_j = sparse_matrix_ind[i][j];

                if let Some((sparse_i, sparse_j)) = calculate_sparse_i_j(position_j, i, self.window_size, n_rows) {
                    output[sparse_i][sparse_j] = sparse_matrix[i][j];
                }
            }
        }

        output
    }

    pub fn calculate_local_attention(&self, q: &Vec<Vec<C>>, k_windows: &Vec<Vec<Vec<C>>>, scale_by_dk: bool) -> Vec<Vec<C>> {
        let mut q_v_k_dot: Vec<Vec<C>> = Vec::new();
        let mut _k_token_start = 0;
        let mut _k_token_end = 0;
        let d_k_sqrt: Real = (q[0].len() as Real).sqrt() + r(1e-12);

        for (seq_ind, q_vec) in q.iter().enumerate() {
            let mut q_v_k_dot_token: Vec<C> = Vec::new();
            // e.g. 2x4 or 3x4
            let k_window_tokens: Vec<Vec<C>> = k_windows[seq_ind].clone();

            // println!("attention_vec dim: {}", attention_vec.len());
            // println!("window_tokens dim: {}, {}", window_tokens.len(), window_tokens[0].len());

            for k_token in k_window_tokens.iter() {
                let mut q_k_v_sum = Complex::zero();

                // no autoregressive mask
                for (feature_indx, feature) in q_vec.iter().enumerate() {
                    q_k_v_sum += *feature * k_token[feature_indx];
                }

                if scale_by_dk {
                    q_k_v_sum = q_k_v_sum / d_k_sqrt;
                }

                q_v_k_dot_token.push(q_k_v_sum);
            }

            q_v_k_dot.push(q_v_k_dot_token);
        }

        q_v_k_dot
    }

    pub fn apply_unified_mask(&self, scores: &mut Vec<Vec<C>>, padding_mask: &Vec<u32>) {
        let seq_len = scores.len();

        for token_ind in 0..seq_len {
            let (start_ind, _) = calculate_start_end_indices(token_ind, self.window_size, seq_len);
            let row = &mut scores[token_ind];

            for (local_pos, score) in row.iter_mut().enumerate() {
                let global_k = start_ind + local_pos;
                if global_k >= padding_mask.len() || padding_mask[global_k] == 0 || global_k > token_ind {
                    *score = Complex::new(Real::NEG_INFINITY, Real::NEG_INFINITY);
                }
            }
        }
    }

    pub fn apply_sparse_causal_mask(&self, q: &mut Vec<Vec<C>>) {
        // row i can only attend to rows <= i
        /*
            [1 2 0 0 0 0] => bekomes [1 0 0 0 0 0]
            [1 2 3 0 0 0]            [1 2 0 0 0 0]
            [0 1 2 3 0 0]            [0 1 2 0 0 0]
            [0 0 1 2 3 0]            [0 0 1 2 0 0]
            [0 0 0 1 2 3]            [0 0 0 1 2 0]
            [0 0 0 0 1 2]            [0 0 0 0 1 2]
        */
        /*
           apply it for sparse matrix
           [1, 2]
           [1, 2, 3]
           [1, 2, 3]
           [1, 2, 3]
           [1, 2, 3]
           [1, 2]

           becomes:

           [1, 0]
           [1, 2, 0]
           [1, 2, 0]
           [1, 2, 0]
           [1, 2, 0]
           [1, 2]

           token indices:
           0 => 0..2
           1 => 0..3
           2 => 1..4
           3 => 2..5
           4 => 3..6
           5 => 4..6
        */

        let seq_len = q.len();
        let mut range: Vec<usize>;

        for (token_ind, q_vec) in q.iter_mut().enumerate() {
            let (start_ind, end_ind) = calculate_start_end_indices(token_ind, self.window_size, seq_len);

            if token_ind > start_ind || token_ind < end_ind {
                range = (start_ind..end_ind).collect();

                for i in start_ind..end_ind {
                    if let Some(pos_compr) = range.iter().position(|&x| x == i) {
                        if i > token_ind {
                            q_vec[pos_compr] = Complex::new(Real::NEG_INFINITY, Real::NEG_INFINITY);
                        }
                    }
                }
            }
        }
    }

    pub fn restore_sparse_matrix(&self, input: &Vec<Vec<C>>, window_size: usize, seq_len: usize) -> Vec<Vec<C>> {
        let mut restored_matrix: Vec<Vec<C>> = Vec::new();

        for (token_index, row) in input.iter().enumerate() {
            restored_matrix.push(restore_sparse_row(&row, token_index, window_size, seq_len));
        }

        restored_matrix
    }

    pub fn restore_sparse_matrix_zeroes(&self, input: &Vec<Vec<C>>, window_size: usize, seq_len: usize) -> Vec<Vec<C>> {
        let mut restored_matrix: Vec<Vec<C>> = Vec::new();

        for (token_index, row) in input.iter().enumerate() {
            restored_matrix.push(restore_sparse_row_zeroes(&row, token_index, window_size, seq_len));
        }

        restored_matrix
    }

    pub fn restore_sparse_matrix_f64(&self, input: &Vec<Vec<f64>>, window_size: usize, seq_len: usize) -> Vec<Vec<f64>> {
        let mut restored_matrix: Vec<Vec<f64>> = Vec::new();

        for (token_index, row) in input.iter().enumerate() {
            restored_matrix.push(restore_sparse_row_f64(&row, token_index, window_size, seq_len));
        }

        restored_matrix
    }

    pub fn build_original_indices(&self, input: &Vec<Vec<C>>) -> Vec<Vec<usize>> {
        let mut indices_matrix: Vec<Vec<usize>> = Vec::new();

        for token_index in 0..input.len() {
            indices_matrix.push(self.build_sparse_row_indices(token_index, self.window_size, input.len()));
        }

        indices_matrix
    }

    pub fn build_sparse_row_indices(&self, token_index: usize, window_size: usize, seq_len: usize) -> Vec<usize> {
        let (start_ind, end_ind) = calculate_start_end_indices(token_index, window_size, seq_len);
        let mut indices: Vec<usize> = Vec::new();

        for i in start_ind..end_ind {
            indices.push(i);
        }

        indices
    }

    pub fn softmax_attention_backward_full(
        &self,
        softmax_vals: &Vec<Vec<Real>>,
        softmax_idx: &Vec<Vec<usize>>,
        dl_do: &Vec<Vec<C>>,
        do_ds: &Vec<Vec<C>>,
        padding_mask: &Vec<u32>,
    ) -> Vec<Vec<C>> {
        let n = dl_do.len();
        let d = dl_do[0].len();

        let mut dl_dz = vec![vec![Complex::new(r(0.0), r(0.0)); n]; n];

        // Pre-transpose V
        let mut v_t = vec![vec![Complex::new(r(0.0), r(0.0)); n]; d];
        for i in 0..n {
            for j in 0..d {
                v_t[j][i] = do_ds[i][j];
            }
        }

        for i in 0..n {
            if padding_mask[i] == 0 {
                continue;
            }

            let cols = &softmax_idx[i]; // indices in this row that exist
            let values = &softmax_vals[i]; // sparse softmax row

            // Compute dot = Σ s[k] * u_k only over sparse entries
            let mut dot: Real = r(0.0);

            for (p, &col) in cols.iter().enumerate() {
                let mut u_k: Real = r(0.0);
                for j in 0..d {
                    u_k += dl_do[i][j].re * v_t[j][col].re;
                }
                dot += values[p] * u_k;
            }

            // Now compute dl/dz for sparse positions
            for (p, &col) in cols.iter().enumerate() {
                let mut u_j: Real = r(0.0);
                for j in 0..d {
                    u_j += dl_do[i][j].re * v_t[j][col].re;
                }

                let s = values[p];
                dl_dz[i][col] = Complex::new(s * (u_j - dot), r(0.0));
            }
        }

        dl_dz
    }

    pub fn softmax_backward_sparse_compressed(
        &self,
        softmax_vals: &Vec<Vec<C>>, // sparse softmax values per row
        softmax_idx: &Vec<Vec<usize>>,         // original column indices
        dl_do: &Vec<Vec<C>>,        // ∂L/∂o
        do_ds: &Vec<Vec<C>>,        // ∂o/∂s
        padding_mask: &Vec<u32>,
    ) -> (Vec<Vec<C>>, Vec<Vec<usize>>) // sparse ∂L/∂z
    {
        let n = softmax_vals.len();
        let d = dl_do[0].len();

        let mut dl_dz_vals: Vec<Vec<C>> = Vec::with_capacity(n);
        let mut dl_dz_idx: Vec<Vec<usize>> = Vec::with_capacity(n);

        // Pre-transpose do_ds for easier indexing
        let mut v_t = vec![vec![Complex::new(r(0.0), r(0.0)); n]; d];
        for i in 0..n {
            for j in 0..d {
                v_t[j][i] = do_ds[i][j];
            }
        }

        for i in 0..n {
            if padding_mask[i] == 0 {
                dl_dz_vals.push(vec![Complex::new(r(0.0), r(0.0)); softmax_vals[i].len()]);
                dl_dz_idx.push(vec![0; softmax_vals[i].len()]);
                continue;
            }

            let cols = &softmax_idx[i];
            let values = &softmax_vals[i];

            // Compute u[col] = dl_do[i] ⋅ do_ds[:, col] for sparse columns
            let mut u_vals: Vec<Real> = Vec::with_capacity(cols.len());
            for &col in cols.iter() {
                let mut u: Real = r(0.0);
                for j in 0..d {
                    u += dl_do[i][j].re * v_t[j][col].re;
                }
                u_vals.push(u);
            }

            // Compute dot = Σ s[k] * u_k
            let mut dot: Real = r(0.0);
            for (p, _) in cols.iter().enumerate() {
                dot += values[p].re * u_vals[p];
            }

            // Compute final dl/dz only for sparse entries
            let mut grad_vals = Vec::with_capacity(cols.len());
            for (p, _) in cols.iter().enumerate() {
                grad_vals.push(Complex::new(values[p].re * (u_vals[p] - dot), r(0.0)));
            }

            dl_dz_vals.push(grad_vals);
            dl_dz_idx.push(cols.clone());
        }

        (dl_dz_vals, dl_dz_idx)
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        if self.input_batch.is_none() {
            if self.input_batch_rm.is_some() {
                let previous_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient_batch
                    .iter()
                    .map(|rows| RowMajorMatrix::from_rows(rows))
                    .collect();

                let mut gradient = self.backward_rm(&previous_gradient_batch_rm);
                let legacy_gx: Vec<Vec<Vec<C>>> = gradient.get_gradient_input_batch_rm().iter().map(|m| m.to_rows()).collect();
                gradient.set_gradient_input_batch(legacy_gx);
                return gradient;
            }

            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        // Input shape e.g. [2][5] and out shape of weights [5][4] => we get final output [2][4]
        let output_batch = self.output_batch.as_ref().expect("Output batch is missing in attention head layer");
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in lattention head inear layer");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Padding mask batch is missing in attention head ");

        // dimensions [seq_len][seq_len] -> A
        let attention_weights_batch: &Vec<Vec<Vec<C>>> = self.attention_weights_batch.as_ref().expect("Attention weights batch is missing in attention head");
        let batch_size = output_batch.len();

        let mut dl_da_batch: Vec<Vec<Vec<C>>> = Vec::new();
        let mut softmax_sparse_indices_batch: Vec<Vec<Vec<usize>>> = Vec::new();
        let mut grad_wv_batch: Vec<Vec<Vec<C>>> = Vec::new();

        // Initialize gradients for each parameter (weights and biases)
        let mut gradient_input_batch = vec![vec![vec![Complex::new(r(0.0), r(0.0)); previous_gradient_batch[0][0].len()]; previous_gradient_batch[0].len()]; input_batch.len()];
        let mut gradient_q_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(r(0.0), r(0.0)); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(r(0.0), r(0.0)); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(r(0.0), r(0.0)); self.weights_v[0].len()]; self.weights_v.len()]; batch_size];

        for (batch_ind, previous_gradient) in previous_gradient_batch.iter().enumerate() {
            let v: Vec<Vec<C>> = multiply_complex(&input_batch[batch_ind], &self.weights_v);

            let grad_wv: Vec<Vec<C>> = self.multiply_sparse_backward(&transpose(&previous_gradient), &attention_weights_batch[batch_ind], false);
            gradient_v_batch[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &transpose(&grad_wv));

            let sparse_attention_idxs = self.build_original_indices(&attention_weights_batch[batch_ind]);
            let (dl_da, _dl_da_inds) = self.softmax_backward_sparse_compressed(&attention_weights_batch[batch_ind], &sparse_attention_idxs, &previous_gradient, &v, &padding_mask_batch[batch_ind]);

            dl_da_batch.push(dl_da);
            softmax_sparse_indices_batch.push(sparse_attention_idxs);
            grad_wv_batch.push(grad_wv);
        }

        let mut dl_dq_ctl_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(r(0.0), r(0.0)); dl_da_batch[0][0].len()]; dl_da_batch[0].len()]; batch_size];
        let mut dl_dk_ctl_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![Complex::new(r(0.0), r(0.0)); dl_da_batch[0][0].len()]; dl_da_batch[0].len()]; batch_size];

        for batch_ind in 0..previous_gradient_batch.len() {
            let q_ctl: Vec<Vec<C>> = self.q_ctl.as_ref().expect("Q CTL is missing in attention head layer")[batch_ind].clone();
            let k_ctl: Vec<Vec<C>> = self.k_ctl.as_ref().expect("K CTL is missing in attention head layer")[batch_ind].clone();

            let d_k = k_ctl[0].len() as Real;
            // Compute gradient of k_scaled w.r.t. k
            let k_scaled: Vec<Vec<C>> = scale_attention_scores(&transpose(&k_ctl), d_k);
            let q_scaled: Vec<Vec<C>> = scale_attention_scores(&q_ctl, d_k);

            let dl_da_transposed: Vec<Vec<C>> = self.transpose_sparse(&dl_da_batch[batch_ind], &softmax_sparse_indices_batch[batch_ind]);
            dl_dq_ctl_batch[batch_ind] = transpose(&self.multiply_sparse_backward(&k_scaled, &dl_da_transposed, true));

            // Gradient Wk
            let dl_dk: Vec<Vec<C>> = self.multiply_sparse_backward(&transpose(&q_scaled), &dl_da_batch[batch_ind], true);
            dl_dk_ctl_batch[batch_ind] = transpose(&dl_dk);
        }

        let mut gradient_kcl_q_batch = Vec::new();
        let mut gradient_k_ctl_batch = Vec::new();

        if self.ctl_q.is_some() {
            let mut gradient_softmax = Gradient::new_default();
            gradient_softmax.set_gradient_input_batch(dl_dq_ctl_batch.clone());

            let ctl_q: &mut ComplexToLinearLayer = self.ctl_q.as_mut().expect("CTL Q is missing in attention head layer");
            let ctl_q_gradient: Gradient = ctl_q.backward(&gradient_softmax);
            gradient_kcl_q_batch = ctl_q_gradient.get_gradient_input_batch();
        }

        if self.ctl_k.is_some() {
            let mut gradient_softmax = Gradient::new_default();
            gradient_softmax.set_gradient_input_batch(dl_dk_ctl_batch.clone());

            let ctl_k = self.ctl_k.as_mut().expect("CTL K is missing in attention head layer");
            let ctl_k_gradient = ctl_k.backward(&gradient_softmax);
            gradient_k_ctl_batch = ctl_k_gradient.get_gradient_input_batch();
        }

        for batch_ind in 0..previous_gradient_batch.len() {
            // Gradient Wq
            let dl_dq: &Vec<Vec<C>> = &gradient_kcl_q_batch[batch_ind];
            gradient_q_batch[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &dl_dq);

            // Gradient Wk
            let dl_dk: &Vec<Vec<C>> = &gradient_k_ctl_batch[batch_ind];
            gradient_k_batch[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &dl_dk);

            // Gradient input
            let dl_dqx = multiply_complex(dl_dq, &conjugate_transpose(&self.weights_q));
            let dl_dkx = multiply_complex(dl_dk, &conjugate_transpose(&self.weights_k));
            let dl_dvx = multiply_complex(&conjugate_transpose(&grad_wv_batch[batch_ind]), &conjugate_transpose(&self.weights_v));

            gradient_input_batch[batch_ind] = add_matrix(&dl_dqx, &dl_dkx);
            gradient_input_batch[batch_ind] = add_matrix(&gradient_input_batch[batch_ind], &dl_dvx);
        }

        // if self.gradient.is_some() {
        //     let previous_gradient = self.gradient.as_ref().expect("");
        //     gradient_v_batch = add_matrix_3d(&gradient_v_batch, &previous_gradient.get_gradient_weights_v_batch());
        //     gradient_q_batch = add_matrix_3d(&gradient_q_batch, &previous_gradient.get_gradient_weights_q_batch());
        //     gradient_k_batch = add_matrix_3d(&gradient_k_batch, &previous_gradient.get_gradient_weights_k_batch());
        // }

        // Compute the gradients for the parameters and store them
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_weights_v_batch(gradient_v_batch);
        gradient.set_gradient_weights_q_batch(gradient_q_batch);
        gradient.set_gradient_weights_k_batch(gradient_k_batch);
        gradient.set_gradient_input_batch(gradient_input_batch);

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in attention head layer");
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
                vec![vec![Complex::new(0.0, 0.0); grad_w_q[0].len()]; grad_w_q.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_q[0].len()]; grad_w_q.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_k[0].len()]; grad_w_k.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_k[0].len()]; grad_w_k.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
            )
        };

        calculate_adam_w(
            &mut self.weights_q,
            &grad_w_q,
            &mut prev_m_weights_q,
            &mut prev_v_weights_q,
            &mut prev_v_weights_q_hat,
            learning_rate,
            time_step,
        );
        calculate_adam_w(
            &mut self.weights_k,
            &grad_w_k,
            &mut prev_m_weights_k,
            &mut prev_v_weights_k,
            &mut prev_v_weights_k_hat,
            learning_rate,
            time_step,
        );
        calculate_adam_w(
            &mut self.weights_v,
            &grad_w_v,
            &mut prev_m_weights_v,
            &mut prev_v_weights_v,
            &mut prev_v_weights_v_hat,
            learning_rate,
            time_step,
        );

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

        // weights changed; invalidate RM caches
        self.weights_q_rm = None;
        self.weights_k_rm = None;
        self.weights_v_rm = None;

        self.gradient = None;
    }
}

pub fn scale_attention_scores(attention_scores: &Vec<Vec<C>>, d_k: Real) -> Vec<Vec<C>> {
    let scaling_factor: Real = r(1.0) / (r(1e-8) + d_k.sqrt());
    let mut scaled_scores = attention_scores.clone();

    // Scale each attention score
    for row in 0..scaled_scores.len() {
        for col in 0..scaled_scores[row].len() {
            scaled_scores[row][col] = scaled_scores[row][col] * scaling_factor;
        }
    }

    scaled_scores
}

pub fn scale_attention_scores_f64(attention_scores: &Vec<Vec<f64>>, d_k: f64) -> Vec<Vec<f64>> {
    let scaling_factor = 1.0 / (1e-8 + d_k.sqrt());
    let mut scaled_scores = attention_scores.clone();

    // Scale each attention score
    for row in 0..scaled_scores.len() {
        for col in 0..scaled_scores[row].len() {
            scaled_scores[row][col] = scaled_scores[row][col] * scaling_factor;
        }
    }

    scaled_scores
}

pub fn create_causal_mask(rows: usize) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0; rows]; rows]; // Initialize with zeros

    for i in 0..rows {
        for j in 0..=i {
            mask[i][j] = 1; // Allow attention to current and previous tokens
        }
    }

    mask
}

pub fn calculate_window_tokens_batch(input: &Vec<Vec<Vec<C>>>, window_size: usize) -> Vec<Vec<Vec<Vec<C>>>> {
    input.par_iter().map(|sequence| calculate_window_tokens(sequence, window_size)).collect()
}

pub fn calculate_window_tokens(input: &Vec<Vec<C>>, window_size: usize) -> Vec<Vec<Vec<C>>> {
    let seq_len = input.len();
    let mut windowed_tokens: Vec<Vec<Vec<C>>> = Vec::new();

    for i in 0..seq_len {
        let (start_ind, end_ind) = calculate_start_end_indices(i, window_size, seq_len);

        let mut window: Vec<Vec<C>> = Vec::new();
        for j in start_ind..end_ind {
            window.push(input[j].clone());
        }
        windowed_tokens.push(window);
    }

    windowed_tokens
}

pub fn calculate_start_end_indices(token_index: usize, window_size: usize, seq_len: usize) -> (usize, usize) {
    let start_ind = if token_index >= window_size { token_index - window_size } else { 0 };
    let end_ind = if token_index + window_size + 1 <= seq_len { token_index + window_size + 1 } else { seq_len };

    (start_ind, end_ind)
}

pub fn calculate_sparse_i_j(original_i: usize, original_j: usize, window_size: usize, seq_len: usize) -> Option<(usize, usize)> {
    let (start_ind, end_ind) = calculate_start_end_indices(original_i, window_size, seq_len);

    if original_j >= start_ind && original_j < end_ind {
        let sparse_j = original_j - start_ind;
        Some((original_i, sparse_j))
    } else {
        None
    }
}

pub fn restore_sparse_row(sparse_row: &Vec<C>, token_index: usize, window_size: usize, seq_len: usize) -> Vec<C> {
    let (start_ind, end_ind) = calculate_start_end_indices(token_index, window_size, seq_len);
    let mut row = vec![Complex::new(Real::NEG_INFINITY, Real::NEG_INFINITY); seq_len];
    let mut index_in_row = 0;

    for i in 0..seq_len {
        if i >= start_ind && i < end_ind {
            row[i] = sparse_row[index_in_row];
            index_in_row += 1;
        }
    }
    row
}

pub fn restore_sparse_row_zeroes(sparse_row: &Vec<C>, token_index: usize, window_size: usize, seq_len: usize) -> Vec<C> {
    let (start_ind, end_ind) = calculate_start_end_indices(token_index, window_size, seq_len);
    let mut row = vec![Complex::new(r(0.0), r(0.0)); seq_len];
    let mut index_in_row = 0;

    for i in 0..seq_len {
        if i >= start_ind && i < end_ind {
            row[i] = sparse_row[index_in_row];
            index_in_row += 1;
        }
    }
    row
}

pub fn restore_sparse_row_f64(sparse_row: &Vec<f64>, token_index: usize, window_size: usize, seq_len: usize) -> Vec<f64> {
    let (start_ind, end_ind) = calculate_start_end_indices(token_index, window_size, seq_len);
    let mut row = vec![0.0; seq_len];
    let mut index_in_row = 0;

    for i in 0..seq_len {
        if i >= start_ind && i < end_ind {
            row[i] = sparse_row[index_in_row];
            index_in_row += 1;
        }
    }
    row
}
