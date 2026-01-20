use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput}, network_layers::layer::LayerType, utils::{
        activation::softmax_complex_padding_real,
        adam_w::calculate_adam_w,
        derivative::{backpropagate_softmax_masked_real, softmax_derivative_complex_jacobian},
        dtype::{C, ONE, Real, ZERO, r},
        matrix::{
            RowMajorMatrix, add_matrix, append_rows_rm, average_matrix_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose, conjugate_transpose_rm, multiply_complex, multiply_complex_rm, multiply_complex_with_f64, multiply_complex_with_f64_rm, multiply_f64_complex, multiply_f64_complex_rm, transpose, transpose_rm, transpose_rm_f64
        },
        weights_initializer::initialize_weights_complex,
    }
};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedAttentionHead {
    pub weights_q: Vec<Vec<C>>,
    pub weights_k: Vec<Vec<C>>,
    pub weights_v: Vec<Vec<C>>,

    pub bias_pos: Vec<Vec<C>>,

    pub bias_q: Vec<C>,
    pub bias_k: Vec<C>,
    pub bias_v: Vec<C>,

    pub layer_type: LayerType,
    pub learning_rate: f64,

    pub m1: Vec<Vec<C>>,
    pub v1: Vec<Vec<C>>,

    pub smoothing: Real,
    pub ema: Real,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub attention_weights_batch: Option<Vec<Vec<Vec<Real>>>>,
    #[serde(skip)]
    pub attention_weights_batch_raw: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub attention_weights_batch_rm: Option<Vec<RowMajorMatrix<Real>>>,
    #[serde(skip)]
    pub attention_weights_batch_raw_rm: Option<Vec<RowMajorMatrix<C>>>,
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
    pub weights_q_rm: Option<RowMajorMatrix<C>>,
    pub weights_k_rm: Option<RowMajorMatrix<C>>,
    pub weights_v_rm: Option<RowMajorMatrix<C>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub total_valid_tokens: usize,
}

impl MaskedAttentionHead {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut weights_q: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let mut weights_k: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let mut weights_v: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];

        let mut bias_pos: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); 5]; 5];

        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);

        initialize_weights_complex(5, 5, &mut bias_pos);

        let bias_q: Vec<C> = vec![C::new(ONE, ZERO); cols];
        let bias_k: Vec<C> = vec![C::new(ONE, ZERO); cols];
        let bias_v: Vec<C> = vec![C::new(ONE, ZERO); cols];

        MaskedAttentionHead {
            weights_q,
            weights_k,
            weights_v,

            bias_pos,
            bias_q,
            bias_k,
            bias_v,
            layer_type: LayerType::InputLayer,
            learning_rate: learning_rate,

            smoothing: r(0.99),
            ema: ZERO,

            global_norm: 0.0,
            max_norm: 0.0,

            gradient: None,
            previous_gradient: None,
            input_batch: None,
            input_batch_rm: None,
            output_batch: None,
            output_batch_rm: None,
            padding_mask_batch: None,
            attention_weights_batch: None,
            attention_weights_batch_raw: None,
            attention_weights_batch_rm: None,
            attention_weights_batch_raw_rm: None,
            k_cache: None,
            v_cache: None,
            k_cache_rm: None,
            v_cache_rm: None,
            weights_q_rm: None,
            weights_k_rm: None,
            weights_v_rm: None,
            m1: vec![vec![C::new(ZERO, ZERO); cols]; rows],
            v1: vec![vec![C::new(ZERO, ZERO); cols]; rows],
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
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, learning_rate: f64) -> MaskedAttentionHead {
        let mut attention_layer: MaskedAttentionHead = MaskedAttentionHead::new(rows, cols, learning_rate);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl MaskedAttentionHead {
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

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm_ref = layer_input.get_input_batch_rm_ref();
        let input_batch_ref = layer_input.get_input_batch_ref();
        let use_rm = input_batch_ref.is_none() && input_batch_rm_ref.is_some();

        if use_rm {
            let input_batch_rm = input_batch_rm_ref.unwrap();
            self.ensure_weights_cache_rm();

            let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
            self.input_batch = None;
            self.input_batch_rm = Some(input_batch_rm.to_vec());
            self.padding_mask_batch = Some(padding_mask_batch.clone());
            self.time_step = layer_input.get_time_step();
            self.batch_size = layer_input.get_batch_size();
            self.total_valid_tokens = layer_input.get_total_valid_tokens();

            let wq = self.weights_q_rm.as_ref().unwrap();
            let wk = self.weights_k_rm.as_ref().unwrap();
            let wv = self.weights_v_rm.as_ref().unwrap();

            let q_batch_rm: Vec<_> = input_batch_rm.par_iter().map(|x| multiply_complex_rm(x, wq)).collect();

            let (k_new_batch_rm, v_new_batch_rm): (Vec<_>, Vec<_>) = if self.k_cache_rm.is_none() || !layer_input.get_calculate_k_v_cache() {
                (
                    input_batch_rm.par_iter().map(|x| multiply_complex_rm(x, wk)).collect(),
                    input_batch_rm.par_iter().map(|x| multiply_complex_rm(x, wv)).collect(),
                )
            } else {
                let mut ks = Vec::with_capacity(input_batch_rm.len());
                let mut vs = Vec::with_capacity(input_batch_rm.len());
                for x in input_batch_rm.iter() {
                    let last_row_start = (x.rows - 1) * x.cols;
                    let last_row_end = last_row_start + x.cols;
                    let one_row = RowMajorMatrix::from_data(1, x.cols, x.data[last_row_start..last_row_end].to_vec());
                    ks.push(multiply_complex_rm(&one_row, wk));
                    vs.push(multiply_complex_rm(&one_row, wv));
                }
                (ks, vs)
            };

            let (k_cache_rm, v_cache_rm): (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>) = if layer_input.get_calculate_k_v_cache() {
                if self.k_cache_rm.is_none() {
                    self.k_cache_rm = Some(k_new_batch_rm.clone());
                    self.v_cache_rm = Some(v_new_batch_rm.clone());
                } else {
                    let k_cache = self.k_cache_rm.as_mut().unwrap();
                    let v_cache = self.v_cache_rm.as_mut().unwrap();
                    for (cache_k, new_k) in k_cache.iter_mut().zip(&k_new_batch_rm) {
                        append_rows_rm(cache_k, new_k);
                    }
                    for (cache_v, new_v) in v_cache.iter_mut().zip(&v_new_batch_rm) {
                        append_rows_rm(cache_v, new_v);
                    }
                }
                (self.k_cache_rm.as_ref().unwrap().clone(), self.v_cache_rm.as_ref().unwrap().clone())
            } else {
                (k_new_batch_rm, v_new_batch_rm)
            };

            let attention_weights_batch_inactivated_rm: Vec<_> = q_batch_rm
                .par_iter()
                .enumerate()
                .map(|(batch_ind, q)| {
                    let mask = create_causal_mask(q.rows);
                    let attn_scores = multiply_complex_rm(q, &transpose_rm(&k_cache_rm[batch_ind]));
                    let mut scaled_scores = scale_attention_scores_rm(&attn_scores, r(k_cache_rm[batch_ind].cols as f64));
                    apply_attention_mask_inplace_rm(&mut scaled_scores, &mask);
                    scaled_scores
                })
                .collect();

            let attention_weights_activated_rm: Vec<_> = attention_weights_batch_inactivated_rm
                .par_iter()
                .zip(padding_mask_batch.par_iter())
                .map(|(scores, padding_mask)| softmax_complex_padding_real_rm(scores, padding_mask))
                .collect();

            let batch_output_rm: Vec<_> = attention_weights_activated_rm
                .par_iter()
                .enumerate()
                .map(|(batch_ind, attention_weights)| multiply_f64_complex_rm(attention_weights, &v_cache_rm[batch_ind]))
                .collect();

            self.attention_weights_batch = None;
            self.attention_weights_batch_raw = None;
            self.output_batch = None;

            self.attention_weights_batch_rm = Some(attention_weights_activated_rm);
            self.attention_weights_batch_raw_rm = Some(attention_weights_batch_inactivated_rm);
            self.output_batch_rm = Some(batch_output_rm.clone());

            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch_rm(batch_output_rm);
            return layer_output;
        }

        let input_batch: Vec<Vec<Vec<C>>> = layer_input.get_input_batch();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();

        self.input_batch = Some(input_batch.clone());
        self.input_batch_rm = None;
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();
        self.total_valid_tokens = layer_input.get_total_valid_tokens();

        // Step 1: Compute Q for the entire sequence (all tokens up to current step)

        let q_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_q)).collect();

        // Step 2: Check if it's the first input or an extended input
        let (k_new_batch, v_new_batch): (Vec<_>, Vec<_>) = if self.k_cache.is_none() || !layer_input.get_calculate_k_v_cache() {
            // First input (compute for all tokens)
            let k_new_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_k)).collect();
            let v_new_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_v)).collect();
            (k_new_batch, v_new_batch)
        } else {
            let k_new_batch: Vec<_> = input_batch.last().map_or(vec![], |input| {
                input.last().map_or(vec![], |last_token: &Vec<C>| {
                    // Only compute K for the last token
                    vec![multiply_complex(&vec![last_token.clone()], &self.weights_k)]
                })
            });

            let v_new_batch: Vec<_> = input_batch.last().map_or(vec![], |input| {
                input.last().map_or(vec![], |last_token| {
                    // Only compute V for the last token
                    vec![multiply_complex(&vec![last_token.clone()], &self.weights_v)]
                })
            });

            (k_new_batch, v_new_batch)
        };

        // Step 3: Update the K/V cache (only if inference mode)
        let (k_cache, v_cache): (Vec<Vec<Vec<C>>>, Vec<Vec<Vec<C>>>) = if layer_input.get_calculate_k_v_cache() {
            // Update K and V cache with new token's values if inference mode
            if self.k_cache.is_none() {
                self.k_cache = Some(k_new_batch.clone());
                self.v_cache = Some(v_new_batch.clone());
            } else {
                let k_cache = self.k_cache.as_mut().unwrap();
                let v_cache = self.v_cache.as_mut().unwrap();
                for (cache_k, new_k) in k_cache.iter_mut().zip(&k_new_batch) {
                    cache_k.extend_from_slice(&new_k); // Only extend with new K values
                }
                for (cache_v, new_v) in v_cache.iter_mut().zip(&v_new_batch) {
                    cache_v.extend_from_slice(new_v); // Only extend with new V values
                }
            }
            (self.k_cache.as_ref().unwrap().clone(), self.v_cache.as_ref().unwrap().clone())
        } else {
            // If it's training mode, we don't use the cache
            (k_new_batch, v_new_batch)
        };

        // println!("k_cache: {} {} {}", k_cache.len(), k_cache[0].len(), k_cache[0][0].len());
        // println!("v_cache: {} {} {}", v_cache.len(), v_cache[0].len(), v_cache[0][0].len());
        // println!("q_batch: {} {} {}", q_batch.len(), q_batch[0].len(), q_batch[0][0].len());

        // Step 4: Compute attention weights in parallel using the entire Q batch and cached K/V
        let attention_weights_batch_inactivated: Vec<Vec<Vec<C>>> = q_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, q)| {
                let mask = create_causal_mask(q.len());
                let attn_scores = multiply_complex(q, &transpose(&k_cache[batch_ind])); // Use the full K cache
            let mut scaled_scores = scale_attention_scores(&attn_scores, r(k_cache[batch_ind][0].len() as f64));

                apply_attention_mask_inplace(&mut scaled_scores, &mask);
                scaled_scores
            })
            .collect();

        let attention_weights_activated: Vec<_> = attention_weights_batch_inactivated
            .par_iter()
            .zip(padding_mask_batch)
            .map(|(scaled_scores_positioned, padding_mask)| softmax_complex_padding_real(scaled_scores_positioned, &padding_mask))
            .collect();

        // Step 5: Compute final output using cached V values
        let batch_output: Vec<_> = attention_weights_activated
            .par_iter()
            .enumerate()
            .map(|(batch_ind, attention_weights)| multiply_f64_complex(attention_weights, &v_cache[batch_ind]))
            .collect();
        // Step 6: Store intermediate results
        self.attention_weights_batch = Some(attention_weights_activated);
        self.attention_weights_batch_raw = Some(attention_weights_batch_inactivated);
        self.output_batch = Some(batch_output.clone());
        self.attention_weights_batch_rm = None;
        self.attention_weights_batch_raw_rm = None;
        self.output_batch_rm = None;

        // println!("output_batch in attention head: {} {} {}", batch_output.len(), batch_output[0].len(), batch_output[0][0].len());
        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output);

        layer_output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        self.ensure_weights_cache_rm();

        let input_batch_rm = self.input_batch_rm.as_ref().expect("Input batch RM is missing in attention head");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Padding mask batch is missing in attention head");
        let attention_weights_batch_rm = self.attention_weights_batch_rm.as_ref().expect("Attention weights RM missing in attention head");
        let batch_size = input_batch_rm.len();

        let wq = self.weights_q_rm.as_ref().unwrap();
        let wk = self.weights_k_rm.as_ref().unwrap();
        let wv = self.weights_v_rm.as_ref().unwrap();

        let mut gradient_input_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
        let mut gradient_q_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights_v[0].len()]; self.weights_v.len()]; batch_size];

        for (batch_ind, previous_gradient) in previous_gradient_batch_rm.iter().enumerate() {
            let x = &input_batch_rm[batch_ind];
            let s = &attention_weights_batch_rm[batch_ind];

            let q = multiply_complex_rm(x, wq);
            let k = multiply_complex_rm(x, wk);
            let v = multiply_complex_rm(x, wv);

            // grad_wv = G^T * S
            let grad_wv = multiply_complex_with_f64_rm(&transpose_rm(previous_gradient), s);
            let dl_dwv = multiply_complex_rm(&conjugate_transpose_rm(x), &transpose_rm(&grad_wv));
            gradient_v_batch[batch_ind] = dl_dwv.to_rows();

            let d_k = r(k.cols as f64);
            let k_scaled = scale_attention_scores_rm(&transpose_rm(&k), d_k);
            let q_scaled = scale_attention_scores_rm(&q, d_k);

            // dl_ds = G * V^H
            let dl_ds = multiply_complex_rm(previous_gradient, &conjugate_transpose_rm(&v));

            // Masked softmax backward in RM without explicit Jacobian:
            // For each row i: dL/dA[i,j] = S[i,j] * (dL/dS[i,j] - sum_k dL/dS[i,k] * S[i,k])
            // Here we only use real part of dL/dS (matches legacy backpropagate_softmax_masked_real behavior).
            let mut dl_da_rm = RowMajorMatrix::from_data(s.rows, s.cols, vec![ZERO; s.rows * s.cols]);
            for i in 0..s.rows {
                if padding_mask_batch[batch_ind][i] == 0 {
                    continue;
                }

                let s_row = s.row_range(i);
                let ds_row = dl_ds.row_range(i);
                let mut dot: Real = ZERO;
                for k in 0..s.cols {
                    dot += dl_ds.data[ds_row.start + k].re * s.data[s_row.start + k];
                }

                for j in 0..s.cols {
                    let sij = s.data[s_row.start + j];
                    let dsij = dl_ds.data[ds_row.start + j].re;
                    let idx = dl_da_rm.idx(i, j);
                    dl_da_rm.data[idx] = sij * (dsij - dot);
                }
            }

            let dl_dq = multiply_complex_with_f64_rm(&k_scaled, &transpose_rm_f64(&dl_da_rm));
            let dl_dwq = multiply_complex_rm(&conjugate_transpose_rm(x), &conjugate_transpose_rm(&dl_dq));
            gradient_q_batch[batch_ind] = dl_dwq.to_rows();

            let dl_dk = multiply_complex_with_f64_rm(&conjugate_transpose_rm(&q_scaled), &dl_da_rm);
            let dl_dwk = multiply_complex_rm(&conjugate_transpose_rm(x), &transpose_rm(&dl_dk));
            gradient_k_batch[batch_ind] = dl_dwk.to_rows();

            let dl_dqx = multiply_complex_rm(&conjugate_transpose_rm(&dl_dq), &conjugate_transpose_rm(wq));
            let dl_dkx = multiply_complex_rm(&transpose_rm(&dl_dk), &conjugate_transpose_rm(wk));
            let dl_dvx = multiply_complex_rm(&transpose_rm(&grad_wv), &conjugate_transpose_rm(wv));

            let mut gx = dl_dqx;
            for i in 0..gx.data.len() {
                gx.data[i] += dl_dkx.data[i];
                gx.data[i] += dl_dvx.data[i];
            }
            gradient_input_batch_rm.push(gx);
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_weights_v_batch(gradient_v_batch);
        gradient.set_gradient_weights_q_batch(gradient_q_batch);
        gradient.set_gradient_weights_k_batch(gradient_k_batch);
        gradient.set_gradient_input_batch_rm(gradient_input_batch_rm);
        gradient.set_total_valid_tokens(self.total_valid_tokens);

        self.gradient = Some(gradient.clone());
        gradient
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
        let attention_weights_batch: &Vec<Vec<Vec<Real>>> = self.attention_weights_batch.as_ref().expect("Attention weights batch is missing in attention head");
        let _attention_weights_batch_raw: &Vec<Vec<Vec<C>>> = self.attention_weights_batch_raw.as_ref().expect("Attention weights batch is missing in attention head");

        let batch_size = output_batch.len();

        // Initialize gradients for each parameter (weights and biases)
        let mut gradient_input_batch = vec![vec![vec![C::new(ZERO, ZERO); previous_gradient_batch[0][0].len()]; previous_gradient_batch[0].len()]; input_batch.len()];
        let mut gradient_q_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights_v[0].len()]; self.weights_v.len()]; batch_size];

        for (batch_ind, previous_gradient) in previous_gradient_batch.iter().enumerate() {
            let q: Vec<Vec<C>> = multiply_complex(&input_batch[batch_ind], &self.weights_q);
            let k: Vec<Vec<C>> = multiply_complex(&input_batch[batch_ind], &self.weights_k);
            let v: Vec<Vec<C>> = multiply_complex(&input_batch[batch_ind], &self.weights_v);

            /*
               A = Q*KT/sqtr(dk)
               S = sigma(A)
               O = S * V

               dl/ds = Gt * VT
               dl/da = Gt * VT * ds/da = dl/ds * ds/da
               dl/dq = dl/ds * ds/da * da/dq = dl/da  * da/dq
               dl/dwq = XT * dl/ds * ds/da * da/dq = XT * dl/dq

               //Wq
               => dl/dwq = XT * (Gt * VT * grad(A) * Kt/sqtr(dk))
               dl/dwq = dl/ds * ds/da * da/dq * dq/dWq

               //Wv
               => dl/dWv = dl/do * do/dv * dv/dwv

               //Wk
               dl/dWk = dl/ds * ds/da * da/dk * dk/dWk

            */

            // Compute gradient of Wv
            // => dl/dWv = dl/do * do/dv * dv/dwv
            // 2, 4 * 2,2 = 4,2 * 2,2 = 4,2
            let grad_wv = multiply_complex_with_f64(&transpose(&previous_gradient), &attention_weights_batch[batch_ind]);
            // 2,5 * 2, 4 = 5, 2 * 2, 4 = 5, 4
            gradient_v_batch[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &transpose(&grad_wv));

            let d_k = r(k[0].len() as f64);
            // Compute gradient of k_scaled w.r.t. k
            let k_scaled: Vec<Vec<C>> = scale_attention_scores(&transpose(&k), d_k);
            let q_scaled: Vec<Vec<C>> = scale_attention_scores(&q, d_k);

            // Compute activation derivative softmax
            let softmax_derivative: Vec<Vec<Vec<Real>>> = softmax_derivative_complex_jacobian(&attention_weights_batch[batch_ind]);

            // println!("softmax_derivative dim: {}, {}, {}", &softmax_derivative.len(), &softmax_derivative[0].len(),  &softmax_derivative[0][0].len());

            // Gradient Wq
            //    => dl/dwq = XT * (Gt * VT * grad(A) * Kt/sqtr(dk))
            //    dl/dwq = dl/ds * ds/da * da/dq * dq/dWq
            // dl/dWq = (((dl/dO * dO/dS) * dS/dA) * dA/dQ) * dQ/dWq
            // 2, 4 * 2, 4 = 2,2
            let dl_ds: Vec<Vec<C>> = multiply_complex(&previous_gradient, &conjugate_transpose(&v));
            // 2,2 * 2,2  = 2,2
            let dl_da: Vec<Vec<Real>> = backpropagate_softmax_masked_real(&softmax_derivative, &dl_ds, &padding_mask_batch[batch_ind]);
            // println!("dl_da dim: {}, {}", &dl_da.len(), &dl_da[0].len(),);
            // 2,2 * 4, 2 = 2 * 2 * 2, 4 = 2,4
            let dl_dq: Vec<Vec<C>> = multiply_complex_with_f64(&k_scaled, &transpose(&dl_da));

            // println!("dl_dq dim: {}, {}", &dl_dq.len(), &dl_dq[0].len(),);
            // 2,5 * 4,2 = 5,2 * 2, 4 = 5, 4
            let dl_dwq: Vec<Vec<C>> = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &conjugate_transpose(&dl_dq));
            // println!("dl_dwq dim: {}, {}", &dl_dwq.len(), &dl_dwq[0].len());
            gradient_q_batch[batch_ind] = dl_dwq;

            // Gradient Wk
            //dl/dWk = (((dl/dO * dO/dS) * dS/dA) * dA/dKT) * dKT/dWk
            // 2,2 * 2,4 = 2,2 * 2,4 = 2,4
            let dl_dk: Vec<Vec<C>> = multiply_complex_with_f64(&conjugate_transpose(&q_scaled), &dl_da);

            // println!("dl_dk dim: {}, {}", &dl_dk.len(), &dl_dk[0].len());
            // 2,5 * 2,4 = 5,2 * 2, 4 = 5,4
            let dl_dwk: Vec<Vec<C>> = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &transpose(&dl_dk));
            // println!("dl_dwk dim: {}, {}", &dl_dwk.len(), &dl_dwk[0].len());
            gradient_k_batch[batch_ind] = dl_dwk;

            // println!("\n weights_q dim: {}, {}", &self.weights_q.len(), &self.weights_q[0].len());
            // println!("\n dl_dq dim: {}, {}", &dl_dq.len(), &dl_dq[0].len());
            // println!("\n dl_dk dim: {}, {}", &dl_dk.len(), &dl_dk[0].len());
            // println!("\n grad_wv dim: {}, {}", &grad_wv.len(), &grad_wv[0].len());

            // 4,2 * 5, 4 = 2, 4 * 4, 5 = 2,5
            // 4,2 * 5, 4 = 2, 4 * 4, 5 = 2,5
            let dl_dqx = multiply_complex(&conjugate_transpose(&dl_dq), &conjugate_transpose(&self.weights_q));
            // 4,2 * 5, 4 = 2, 4 * 4, 5 = 2,5
            let dl_dkx = multiply_complex(&transpose(&dl_dk), &conjugate_transpose(&self.weights_k));
            // 4,2 * 5, 4 = 2, 4 * 4, 5 = 2,5
            let dl_dvx = multiply_complex(&transpose(&grad_wv), &conjugate_transpose(&self.weights_v));

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

        let total_valid_tokens: Real = r(self.total_valid_tokens as f64);

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
                vec![vec![C::new(ZERO, ZERO); grad_w_q[0].len()]; grad_w_q.len()],
                vec![vec![C::new(ZERO, ZERO); grad_w_q[0].len()]; grad_w_q.len()],
                vec![vec![C::new(ZERO, ZERO); grad_w_k[0].len()]; grad_w_k.len()],
                vec![vec![C::new(ZERO, ZERO); grad_w_k[0].len()]; grad_w_k.len()],
                vec![vec![C::new(ZERO, ZERO); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![C::new(ZERO, ZERO); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![C::new(ZERO, ZERO); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![C::new(ZERO, ZERO); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![C::new(ZERO, ZERO); grad_w_v[0].len()]; grad_w_v.len()],
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

        gradient.set_prev_m_weights_k(prev_m_weights_k);
        gradient.set_prev_v_weights_k(prev_v_weights_k);

        gradient.set_prev_m_weights_v(prev_m_weights_v);
        gradient.set_prev_v_weights_v(prev_v_weights_v);

        gradient.set_prev_v_weights_q_hat(prev_v_weights_q_hat);
        gradient.set_prev_v_weights_k_hat(prev_v_weights_k_hat);
        gradient.set_prev_v_weights_v_hat(prev_v_weights_v_hat);

        self.previous_gradient = Some(gradient.clone());

        // weights changed; invalidate RM caches
        self.weights_q_rm = None;
        self.weights_k_rm = None;
        self.weights_v_rm = None;

        self.gradient = None;
    }
}

fn scale_attention_scores_rm(attention_scores: &RowMajorMatrix<C>, d_k: Real) -> RowMajorMatrix<C> {
    let scaling_factor: Real = ONE / (r(1e-8) + d_k.sqrt());
    let mut out = attention_scores.clone();
    for v in out.data.iter_mut() {
        *v *= scaling_factor;
    }
    out
}

fn apply_attention_mask_inplace_rm(attention_scores: &mut RowMajorMatrix<C>, mask: &Vec<Vec<u8>>) {
    let large_negative = C::new(r(-1e12), r(-1e12));
    for r in 0..attention_scores.rows {
        for c in 0..attention_scores.cols {
            if mask[r % mask.len()][c % mask[0].len()] == 0 {
                attention_scores.data[r * attention_scores.cols + c] = large_negative;
            }
        }
    }
}

fn softmax_complex_padding_real_rm(input: &RowMajorMatrix<C>, padding_mask: &Vec<u32>) -> RowMajorMatrix<Real> {
    assert_eq!(padding_mask.len(), input.rows);
    let mut data = vec![ZERO; input.rows * input.cols];
    for r in 0..input.rows {
        if padding_mask[r] == 0 {
            continue;
        }
        let row_start = r * input.cols;
        let row = &input.data[row_start..row_start + input.cols];
        let max_re = row.iter().map(|z| z.re).fold(Real::NEG_INFINITY, Real::max);
        let mut sum: Real = ZERO;
        for c in 0..input.cols {
            let e = (row[c].re - max_re).exp();
            data[row_start + c] = e;
            sum += e;
        }
        if sum != ZERO {
            let inv = ONE / sum;
            for c in 0..input.cols {
                data[row_start + c] *= inv;
            }
        }
    }
    RowMajorMatrix::from_data(input.rows, input.cols, data)
}

pub fn scale_attention_scores(attention_scores: &Vec<Vec<C>>, d_k: Real) -> Vec<Vec<C>> {
    let scaling_factor: Real = ONE / (r(1e-8) + d_k.sqrt());
    let mut scaled_scores = attention_scores.clone();

    // Scale each attention score
    for row in 0..scaled_scores.len() {
        for col in 0..scaled_scores[row].len() {
            scaled_scores[row][col] = scaled_scores[row][col] * scaling_factor;
        }
    }

    scaled_scores
}

pub fn scale_attention_scores_f64(attention_scores: &Vec<Vec<Real>>, d_k: Real) -> Vec<Vec<Real>> {
    let scaling_factor: Real = ONE / (r(1e-8) + d_k.sqrt());
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

fn apply_attention_mask_inplace(attention_scores: &mut Vec<Vec<C>>, mask: &Vec<Vec<u8>>) {
    let large_negative = C::new(r(-1e12), r(-1e12));

    for row in 0..attention_scores.len() {
        for col in 0..attention_scores[row].len() {
            if mask[row % mask.len()][col % mask[0].len()] == 0 {
                attention_scores[row][col] = large_negative; // Apply the mask
            }
        }
    }
}
