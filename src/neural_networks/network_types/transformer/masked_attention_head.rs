use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        adaptive_pooling::adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer, complex_to_linear_layer::ComplexToLinearLayer, layer::LayerType, norm_layer::NormalNormLayer,
        positional_encoding_layer::PositionalEncodingLayer,
    },
    utils::{
        activation::softmax_complex_padding_complex,
        adam_w::calculate_adam_w,
        dtype::{r, Real, C, ONE, ZERO},
        matrix::{add_matrix, average_matrix_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose, multiply_complex, transpose},
        weights_initializer::initialize_weights_complex,
    },
};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedAttentionHead {
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

    pub smoothing: Real,
    pub ema: Real,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub ctl_q: Option<ComplexToLinearLayer>,
    pub ctl_k: Option<ComplexToLinearLayer>,

    pub positional_encoding_layer: PositionalEncodingLayer,
    pub norm_layer_v: NormalNormLayer,
    pub norm_layer_k: NormalNormLayer,

    pub pool_k_layer: AdaptiveAvgPool1dLayer,
    pub pool_v_layer: AdaptiveAvgPool1dLayer,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub attention_weights_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub attention_weights_batch_raw: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub k_cache: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub v_cache: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub batch_size: usize,
    #[serde(skip)]
    pub total_valid_tokens: usize,

    #[serde(skip)]
    pub k_ctl: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub q_ctl: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub v_norm: Option<Vec<Vec<Vec<C>>>>,
}

impl MaskedAttentionHead {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut weights_q: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let mut weights_k: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let mut weights_v: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];

        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);

        let ctl_q = ComplexToLinearLayer::new(cols, cols, learning_rate);
        let ctl_k = ComplexToLinearLayer::new(cols, cols, learning_rate);

        let bias_q: Vec<C> = vec![C::new(ONE, ZERO); cols];
        let bias_k: Vec<C> = vec![C::new(ONE, ZERO); cols];
        let bias_v: Vec<C> = vec![C::new(ONE, ZERO); cols];

        let positional_encoding_layer = PositionalEncodingLayer::new(cols);
        let norm_layer_q = NormalNormLayer::new(cols, 1e-8, learning_rate);
        let norm_layer_k = NormalNormLayer::new(cols, 1e-8, learning_rate);
        let pool_k_layer = AdaptiveAvgPool1dLayer::new(64);
        let pool_v_layer = AdaptiveAvgPool1dLayer::new(64);

        MaskedAttentionHead {
            weights_q,
            weights_k,
            weights_v,

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
            ctl_k: Some(ctl_k),
            ctl_q: Some(ctl_q),
            positional_encoding_layer,
            norm_layer_v: norm_layer_q,
            norm_layer_k,

            pool_k_layer,
            pool_v_layer,

            previous_gradient: None,
            input_batch: None,
            output_batch: None,
            padding_mask_batch: None,
            attention_weights_batch: None,
            attention_weights_batch_raw: None,
            k_cache: None,
            v_cache: None,
            k_ctl: None,
            q_ctl: None,
            v_norm: None,
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
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<C>>> = layer_input.get_input_batch();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();

        self.input_batch = Some(input_batch.clone());
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

        /*
           e.g. X [258][16] -> [64][16]

           Q = X * Wq
           Qctl = CTL(Qnorm) (258x16)

           K = X * Wk
           Kpos = Rope(K)
           Kpool = Pool(Kpos) (64x16)
           Knorm = Norm(Kpool) (64x16)
           Kctl = CTL(Knorm) (64x16)

           V = X * Wv
           Vpool = Pool(V)
           Vnorm = Norm(Vpool) (64x16)

           // Attention
           A_raw = Qct * Kctl^T / sqrt(d_k) (258x16 * 16x64) = 258x64
           A_masked = Mask(A_raw)
           S = softmax(A) (258x64)
           O = S * V -> 258x64 * 64*16 = 258x16
        */

        //Optional CTL layers
        let q_ctl_batch: Vec<Vec<Vec<C>>> = if let Some(ctl_q) = self.ctl_q.as_mut() {
            let mut li = LayerInput::new_default();
            li.set_input_batch(q_batch.clone());
            ctl_q.forward(&li).get_output_batch()
        } else {
            q_batch
        };

        let k_ctl_batch = self.transform_k(&k_cache, layer_input);
        let v_cache = self.transform_v(&v_cache, layer_input);

        self.q_ctl = Some(q_ctl_batch.clone());
        self.k_ctl = Some(k_ctl_batch.clone());
        self.v_norm = Some(v_cache.clone());

        // Step 4: Compute attention weights in parallel using the entire Q batch and cached K/V
        let attention_weights_batch_inactivated: Vec<Vec<Vec<C>>> = q_ctl_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, q)| {
                let mut attn_scores = multiply_complex(q, &transpose(&k_ctl_batch[batch_ind])); // Use the full K cache
                let mask = create_causal_mask(q.len());

                apply_attention_mask_inplace(&mut attn_scores, &mask, r(k_ctl_batch[batch_ind][0].len() as f64));
                attn_scores
            })
            .collect();

        let attention_weights_activated: Vec<_> = attention_weights_batch_inactivated
            .par_iter()
            .enumerate()
            .map(|(batch_ind, sparse_attention_weights_inactivated)| {
                let softmax_output: Vec<Vec<C>> = softmax_complex_padding_complex(sparse_attention_weights_inactivated, &padding_mask_batch[batch_ind]);
                softmax_output
            })
            .collect();

        // Step 5: Compute final output using cached V values
        let batch_output: Vec<_> = attention_weights_activated
            .par_iter()
            .enumerate()
            .map(|(batch_ind, attention_weights)| multiply_complex(attention_weights, &v_cache[batch_ind]))
            .collect();
        // Step 6: Store intermediate results
        self.attention_weights_batch = Some(attention_weights_activated);
        self.attention_weights_batch_raw = Some(attention_weights_batch_inactivated);
        self.output_batch = Some(batch_output.clone());

        // println!("output_batch in attention head: {} {} {}", batch_output.len(), batch_output[0].len(), batch_output[0][0].len());
        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        // Input shape e.g. [2][5] and out shape of weights [5][4] => we get final output [2][4]
        let output_batch = self.output_batch.as_ref().expect("Output batch is missing in attention head layer");
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in lattention head inear layer");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Padding mask batch is missing in attention head ");

        // dimensions [seq_len][seq_len] -> A
        let attention_weights_batch: &Vec<Vec<Vec<C>>> = self.attention_weights_batch.as_ref().expect("Attention weights batch is missing in attention head");
        let _attention_weights_batch_raw: &Vec<Vec<Vec<C>>> = self.attention_weights_batch_raw.as_ref().expect("Attention weights batch is missing in attention head");

        let batch_size = output_batch.len();

        // Initialize gradients for each parameter (weights and biases)
        let mut gradient_input_batch = vec![vec![vec![C::new(ZERO, ZERO); previous_gradient_batch[0][0].len()]; previous_gradient_batch[0].len()]; input_batch.len()];
        let mut gradient_q_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<C>>> = Vec::new();
        let mut dl_da_batch: Vec<Vec<Vec<C>>> = Vec::new();
        let mut grad_wv_batch_pool: Vec<Vec<Vec<C>>> = Vec::new();

        for (batch_ind, previous_gradient) in previous_gradient_batch.iter().enumerate() {
            // dl_do * dl_dv_norm
            let grad_dl_dv_pool = multiply_complex(&conjugate_transpose(&attention_weights_batch[batch_ind]), &previous_gradient);
            grad_wv_batch_pool.push(grad_dl_dv_pool);
        }

        /*
           V = X * Wv
           Vpool = Pool(V)
           Vnorm = Norm(Vpool) (64x16)
           backward = V
        */

        let mut gradient_v = Gradient::new_default();
        
        gradient_v.set_gradient_input_batch(grad_wv_batch_pool.clone());
        // let gradient_v_norm = self.norm_layer_v.backward(&gradient_v);
        let gradient_v_pool = self.pool_v_layer.backward(&gradient_v);
        let v_pool_gradient_batch = gradient_v_pool.get_gradient_input_batch();

        for (batch_ind, previous_gradient) in previous_gradient_batch.iter().enumerate() {
            let v_norm: Vec<Vec<C>> = self.v_norm.as_ref().expect("V norm output is missing in attention head layer")[batch_ind].clone();

            // do_dv * do_dwv
            let grad_dl_dwv = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &v_pool_gradient_batch[batch_ind]);
            let dl_da: Vec<Vec<C>> = self.softmax_attention_backward_full(&attention_weights_batch[batch_ind], &previous_gradient, &v_norm, &padding_mask_batch[batch_ind]);

            gradient_v_batch.push(grad_dl_dwv);
            dl_da_batch.push(dl_da);
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

            dl_dq_ctl_batch[batch_ind] = multiply_complex(&dl_da_batch[batch_ind], &conjugate_transpose(&k_scaled));
            dl_dk_ctl_batch[batch_ind] = multiply_complex(&conjugate_transpose(&dl_da_batch[batch_ind]), &q_scaled);
        }

        let mut gradient_ctl_q_batch: Vec<Vec<Vec<C>>> = Vec::new();
        let mut gradient_ctl_k_batch: Vec<Vec<Vec<C>>> = Vec::new();

        if self.ctl_q.is_some() {
            let mut gradient_softmax = Gradient::new_default();
            gradient_softmax.set_gradient_input_batch(dl_dq_ctl_batch.clone());

            let ctl_q: &mut ComplexToLinearLayer = self.ctl_q.as_mut().expect("CTL Q is missing in attention head layer");
            let ctl_q_gradient: Gradient = ctl_q.backward(&gradient_softmax);
            gradient_ctl_q_batch = ctl_q_gradient.get_gradient_input_batch();
        }

        if self.ctl_k.is_some() {
            let mut gradient_softmax = Gradient::new_default();
            gradient_softmax.set_gradient_input_batch(dl_dk_ctl_batch.clone());

            let ctl_k = self.ctl_k.as_mut().expect("CTL K is missing in attention head layer");
            let ctl_k_gradient = ctl_k.backward(&gradient_softmax);
            gradient_ctl_k_batch = ctl_k_gradient.get_gradient_input_batch();
        }

        // K -> Kpos -> Kpool -> Knorm -> Kctl
        // Kctl -> Knorm -> Kpool -> Kpos -> K
        let mut gradient_k = Gradient::new_default();
        gradient_k.set_gradient_input_batch(gradient_ctl_k_batch.clone());
        let gradient_k_norm = self.norm_layer_k.backward(&gradient_k);
        let gradient_k_pool = self.pool_k_layer.backward(&gradient_k_norm);
        gradient_ctl_k_batch = self.positional_encoding_layer.backward_sequences(&gradient_k_pool.get_gradient_input_batch());

        for batch_ind in 0..previous_gradient_batch.len() {
            // Gradient Wq
            let dl_dq: &Vec<Vec<C>> = &gradient_ctl_q_batch[batch_ind];
            gradient_q_batch[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &dl_dq);

            // Gradient Wk
            let dl_dk: &Vec<Vec<C>> = &gradient_ctl_k_batch[batch_ind];
            gradient_k_batch[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &dl_dk);

            // Gradient input
            let dl_dqx = multiply_complex(dl_dq, &conjugate_transpose(&self.weights_q));
            let dl_dkx = multiply_complex(dl_dk, &conjugate_transpose(&self.weights_k));
            let dl_dvx = multiply_complex(&v_pool_gradient_batch[batch_ind], &conjugate_transpose(&self.weights_v));

            gradient_input_batch[batch_ind] = add_matrix(&dl_dqx, &dl_dkx);
            gradient_input_batch[batch_ind] = add_matrix(&gradient_input_batch[batch_ind], &dl_dvx);
        }

        // Compute the gradients for the parameters and store them
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_weights_v_batch(gradient_v_batch);
        gradient.set_gradient_weights_q_batch(gradient_q_batch);
        gradient.set_gradient_weights_k_batch(gradient_k_batch);
        gradient.set_gradient_input_batch(gradient_input_batch);

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn transform_k(&mut self, k_cache: &Vec<Vec<Vec<C>>>, layer_input: &LayerInput) -> Vec<Vec<Vec<C>>> {
        /*
             K = X * Wk
             Kpos = Rope(K)
             Kpool = Pool(Kpos) (64x16)
             Knorm = Norm(Kpool) (64x16)
             Kctl = CTL(Knorm) (64x16)
        */
        let k_pos: Vec<Vec<Vec<C>>> = self.positional_encoding_layer.forward_sequences(k_cache, layer_input);
        let mut layer_input = layer_input.clone();
        layer_input.set_input_batch(k_pos);

        let k_pool = self.pool_k_layer.forward(&layer_input);
        layer_input.set_input_batch(k_pool.get_output_batch());

        let k_norm = self.norm_layer_k.forward(&layer_input);
        layer_input.set_input_batch(k_norm.get_output_batch());

        let k_ctl_batch: Vec<Vec<Vec<C>>> = if let Some(ctl_k) = self.ctl_k.as_mut() {
            ctl_k.forward(&layer_input).get_output_batch()
        } else {
            k_norm.get_output_batch()
            //k_pool.get_output_batch()
        };

        k_ctl_batch
    }

    pub fn transform_v(&mut self, v_cache: &Vec<Vec<Vec<C>>>, layer_input: &LayerInput) -> Vec<Vec<Vec<C>>> {
        /*
           V = X * Wv
           Vpool = Pool(V)
           Vnorm = Norm(Vpool) (64x16)
        */
        let mut layer_input = layer_input.clone();
        layer_input.set_input_batch(v_cache.clone());

        let v_pool = self.pool_v_layer.forward(&layer_input);
        layer_input.set_input_batch(v_pool.get_output_batch());

        // let v_norm = self.norm_layer_v.forward(&layer_input);
        // v_norm.get_output_batch()

        v_pool.get_output_batch()
    }

    pub fn softmax_attention_backward_full(&self, softmax_vals: &Vec<Vec<C>>, dl_do: &Vec<Vec<C>>, do_ds: &Vec<Vec<C>>, padding_mask: &Vec<u32>) -> Vec<Vec<C>> {
        let q_len = dl_do.len();
        if q_len == 0 {
            return vec![];
        }

        let d = dl_do[0].len();
        let k_len = softmax_vals.get(0).map(|row| row.len()).unwrap_or(0);
        if k_len == 0 {
            return vec![];
        }

        // Output matches softmax shape: [q_len][k_len]
        let mut dl_dz: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); k_len]; q_len];

        let v_len = do_ds.len();
        let k_compute = usize::min(k_len, v_len);

        for i in 0..q_len {
            if padding_mask.get(i).copied().unwrap_or(1) == 0 {
                continue;
            }

            // u[p] = dL/dS_{i,p} = Σ_j dL/dO_{i,j} * conj(V_{p,j})
            let mut u: Vec<C> = vec![C::new(ZERO, ZERO); k_len];
            for p in 0..k_compute {
                if padding_mask.get(p).copied().unwrap_or(1) == 0 {
                    continue;
                }

                let mut acc = C::new(ZERO, ZERO);
                let feat_len = usize::min(d, do_ds[p].len());
                for j in 0..feat_len {
                    acc += dl_do[i][j] * do_ds[p][j].conj();
                }
                u[p] = acc; // Original line
            }

            // dot = Σ_p s[p] * u[p]
            let mut dot: C = C::new(ZERO, ZERO);
            for p in 0..k_compute {
                if padding_mask.get(p).copied().unwrap_or(1) == 0 {
                    continue;
                }
                dot += u[p] * C::new(softmax_vals[i][p].re, ZERO); // Original line
            }

            // dL/dz[p] = s[p] * (u[p] - dot)
            for p in 0..k_compute {
                if padding_mask.get(p).copied().unwrap_or(1) == 0 {
                    dl_dz[i][p] = C::new(ZERO, ZERO);
                    continue;
                }
                let s = softmax_vals[i][p].re;
                dl_dz[i][p] = (u[p] - dot) * C::new(s, ZERO);
            }
        }

        dl_dz
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

        self.gradient = None;
    }
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

fn apply_attention_mask_inplace(attention_scores: &mut Vec<Vec<C>>, mask: &Vec<Vec<u8>>, d_k: Real) {
    let scaling_factor: Real = ONE / (r(1e-8) + d_k.sqrt());

    let large_negative = Complex::new(Real::NEG_INFINITY, Real::NEG_INFINITY);

    for row in 0..attention_scores.len() {
        for col in 0..attention_scores[row].len() {
            if mask[row % mask.len()][col % mask[0].len()] == 0 {
                attention_scores[row][col] = large_negative; // Apply the mask
            } else {
                attention_scores[row][col] = attention_scores[row][col] * scaling_factor;
                // Scale the unmasked scores
            }
        }
    }
}
