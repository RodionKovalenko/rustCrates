use std::f64;

use num::{Complex, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::{
        network_components::{gradient_struct::Gradient, layer::LayerType, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
        utils::{
            activation::softmax_complex_padding_real,
            adam_w::calculate_adam_w,
            derivative::{backpropagate_softmax_masked_real, softmax_derivative_complex_jacobian, test_gradient_batch_error_f64},
            matrix::{
                add_matrix, add_matrix_3d, average_matrix_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose, multiply_complex, multiply_complex_with_f64, multiply_f64_complex,
                transpose,
            },
            weights_initializer::initialize_weights_complex,
        },
    },
    utils::data_converter::convert_to_c_f64_2d,
};
use std::fmt::Debug;

use super::transformer_network::MAX_CONTEXT_WINDOW_SIZE;

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMaskedAttentionHead {
    pub weights_q: Vec<Vec<Complex<f64>>>,
    pub weights_k: Vec<Vec<Complex<f64>>>,
    pub weights_v: Vec<Vec<Complex<f64>>>,

    pub bias_pos: Vec<Vec<Complex<f64>>>,

    pub bias_q: Vec<Complex<f64>>,
    pub bias_k: Vec<Complex<f64>>,
    pub bias_v: Vec<Complex<f64>>,

    pub layer_type: LayerType,
    pub learning_rate: f64,

    pub m1: Vec<Vec<Complex<f64>>>,
    pub v1: Vec<Vec<Complex<f64>>>,

    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    pub window_size: usize,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub attention_weights_batch: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(skip)]
    pub attention_weights_batch_raw: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub k_cache: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub v_cache: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub batch_size: usize,
}

impl SparseMaskedAttentionHead {
    pub fn new(rows: usize, cols: usize, window_size: usize, learning_rate: f64) -> Self {
        let mut weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

        let mut bias_pos: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); MAX_CONTEXT_WINDOW_SIZE * 5]; MAX_CONTEXT_WINDOW_SIZE * 5];

        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);

        initialize_weights_complex(MAX_CONTEXT_WINDOW_SIZE * 5, MAX_CONTEXT_WINDOW_SIZE * 5, &mut bias_pos);

        let bias_q: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let bias_k: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let bias_v: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

        SparseMaskedAttentionHead {
            weights_q,
            weights_k,
            weights_v,

            bias_pos,
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

            gradient: None,
            previous_gradient: None,
            input_batch: None,
            output_batch: None,
            padding_mask_batch: None,
            attention_weights_batch: None,
            attention_weights_batch_raw: None,
            k_cache: None,
            v_cache: None,
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            time_step: 0,
            batch_size: 0,
        }
    }

    fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, window_size: usize, learning_rate: f64) -> SparseMaskedAttentionHead {
        let mut attention_layer: SparseMaskedAttentionHead = SparseMaskedAttentionHead::new(rows, cols, window_size, learning_rate);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl SparseMaskedAttentionHead {
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();

        self.input_batch = Some(input_batch.clone());
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();

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
                input.last().map_or(vec![], |last_token: &Vec<Complex<f64>>| {
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
        let (k_cache, v_cache): (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<Vec<Complex<f64>>>>) = if layer_input.get_calculate_k_v_cache() {
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

        // Step 4: Compute attention weights in parallel using the entire Q batch and cached K/V
        let batch_output: Vec<_> = self.calculated_sparse_masked_attention(q_batch, k_cache, v_cache, padding_mask_batch);
        self.output_batch = Some(batch_output.clone());

        // println!("output_batch in attention head: {} {} {}", batch_output.len(), batch_output[0].len(), batch_output[0][0].len());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output);

        layer_output
    }

    pub fn calculated_sparse_masked_attention(
        &mut self,
        q_batch: Vec<Vec<Vec<Complex<f64>>>>,
        k_batch: Vec<Vec<Vec<Complex<f64>>>>,
        v_batch: Vec<Vec<Vec<Complex<f64>>>>,
        padding_mask_batch: Vec<Vec<u32>>,
    ) -> Vec<Vec<Vec<Complex<f64>>>> {
        let k_windows: Vec<Vec<Vec<Vec<Complex<f64>>>>> = calculate_window_tokens_batch(&k_batch, self.window_size);

        let attention_weights_batch_inactivated_compressed: Vec<Vec<Vec<Complex<f64>>>> = q_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, q)| {
                // println!("q_batch dim: {}, {}", q.len(), q[0].len());
                let mut sparse_attention_weights_inactivated = self.calculate_local_attention(q, &k_windows[batch_ind], true);
                self.apply_sparse_causal_mask(&mut sparse_attention_weights_inactivated);
                sparse_attention_weights_inactivated
            })
            .collect();

        // let attention_weights_restored = attention_weights_batch_inactivated_compressed
        //     .iter()
        //     .enumerate()
        //     .map(|(batch_ind, attention_weights)| self.restore_sparse_matrix(&attention_weights, self.window_size, q_batch[batch_ind].len()))
        //     .collect::<Vec<_>>();

        let attention_weights_activated_compressed: Vec<_> = attention_weights_batch_inactivated_compressed
            .par_iter()
            .zip(padding_mask_batch.clone())
            .map(|(scaled_scores_positioned, padding_mask)| softmax_complex_padding_real(scaled_scores_positioned, &padding_mask))
            .collect();

        let batch_output_compr: Vec<_> = attention_weights_activated_compressed
            .iter()
            .enumerate()
            .map(|(batch_ind, attention_weights)| {
                // println!("attention_weights dim: {}, {}", attention_weights.len(), attention_weights[0].len());
                self.multiply_sparse(attention_weights, &v_batch[batch_ind])
                //multiply_f64_complex(attention_weights, &v_batch[batch_ind])
            })
            .collect();

        self.attention_weights_batch = Some(attention_weights_activated_compressed);
        self.output_batch = Some(batch_output_compr.clone());

        batch_output_compr
    }

    pub fn multiply_sparse(&self, attention_sparse_weights: &Vec<Vec<f64>>, v: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let seq_len = attention_sparse_weights.len();
        let embedding_dim = v[0].len();

        let mut output = vec![vec![Complex::zero(); embedding_dim]; seq_len];

        for q in 0..seq_len {
            let (start_ind, _) = calculate_start_end_indices(q, self.window_size, seq_len);

            for k in 0..v[q].len() {
                for f in 0..attention_sparse_weights[q].len() {
                    output[q][k] += attention_sparse_weights[q][f] * v[start_ind + f][k];
                }
            }
        }

        output
    }

    pub fn multiply_sparse_backward(&self, v: &Vec<Vec<Complex<f64>>>, attention_sparse_weights: &Vec<Vec<f64>>) -> Vec<Vec<Complex<f64>>> {
        let n_rows = v.len();
        let n_cols = attention_sparse_weights.len();

        let mut output = vec![vec![Complex::zero(); n_cols]; n_rows];

        let mut range: Vec<usize>;
        let mut position: usize;

        for k in 0..n_rows {
            for q in 0..n_cols {
                for p in 0..n_cols {
                    let (start_ind, end_ind) = calculate_start_end_indices(p, self.window_size, n_cols);

                    // e.g. p = 3, start_ind = 2, end_ind = 5
                    if q >= start_ind && q < end_ind {
                        // now we should find at what position the q index is located with regard to the start_ind and end_ind
                        // the range is start_ind..end_ind, e.g. 2..5 = [2,3,4]

                        range = (start_ind..end_ind).collect();
                        // find position in the range
                        position = range.iter().position(|&x| x == q).unwrap();

                        output[k][q] += v[k][p] * attention_sparse_weights[p][position];
                    }
                }
            }
        }

        output
    }

    pub fn calculate_local_attention<T>(&self, q: &Vec<Vec<T>>, k_windows: &Vec<Vec<Vec<Complex<f64>>>>, scale_by_dk: bool) -> Vec<Vec<Complex<f64>>>
    where
        T: Copy + Into<Complex<f64>>,
    {
        let mut q_v_k_dot: Vec<Vec<Complex<f64>>> = Vec::new();
        let mut _k_token_start = 0;
        let mut _k_token_end = 0;
        let d_k_sqrt: f64 = (q[0].len() as f64).sqrt() + 1e-12;

        for (seq_ind, q_vec) in q.iter().enumerate() {
            let mut q_v_k_dot_token: Vec<Complex<f64>> = Vec::new();
            // e.g. 2x4 or 3x4
            let k_window_tokens: Vec<Vec<Complex<f64>>> = k_windows[seq_ind].clone();

            // println!("attention_vec dim: {}", attention_vec.len());
            // println!("window_tokens dim: {}, {}", window_tokens.len(), window_tokens[0].len());

            for k_token in k_window_tokens.iter() {
                let mut q_k_v_sum = Complex::zero();

                // no autoregressive mask
                for (feature_indx, feature) in q_vec.iter().enumerate() {
                    q_k_v_sum += (*feature).into() * k_token[feature_indx];
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

    pub fn apply_sparse_causal_mask(&self, q: &mut Vec<Vec<Complex<f64>>>) {
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
                            q_vec[pos_compr] = Complex::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
                        }
                    }
                }
            }
        }
    }

    pub fn restore_sparse_matrix(&self, input: &Vec<Vec<Complex<f64>>>, window_size: usize, seq_len: usize) -> Vec<Vec<Complex<f64>>> {
        let mut restored_matrix: Vec<Vec<Complex<f64>>> = Vec::new();

        for (token_index, row) in input.iter().enumerate() {
            restored_matrix.push(restore_sparse_row(&row, token_index, window_size, seq_len));
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

    fn sparse_softmax_derivative_complex(&self, data: &Vec<f64>, token_ind: usize, seq_len: usize) -> Vec<Vec<f64>> {
        // let mut jacobian: Vec<Vec<f64>> = vec![vec![0.0; seq_len]; seq_len];
        let mut jacobian: Vec<Vec<f64>> = vec![vec![0.0; data.len()]; data.len()];

        let (start_ind, end_ind) = calculate_start_end_indices(token_ind, self.window_size, seq_len);

        let range: Vec<usize> = (start_ind..end_ind).collect::<Vec<usize>>();

        // Loop through each pair of indices (i, j)
        for i in range.iter() {
            let position_i = range.iter().position(|&x| x == *i).unwrap();
            for j in range.iter() {
                let position_j = range.iter().position(|&x| x == *j).unwrap();
                if i == j {
                    // Diagonal elements: s_i * (1 - s_i)
                    jacobian[position_i][position_j] = data[position_i] * (1.0 - data[position_i]);
                    // jacobian[*i][*j] = data[position_i] * (1.0 - data[position_i]);
                } else {
                    // Off-diagonal elements: -s_i * s_j
                    jacobian[position_i][position_j] = -data[position_i] * data[position_j];
                    //jacobian[*i][*j] = -data[position_i] * data[position_j];
                }
            }
        }

        jacobian
    }

    pub fn sparse_softmax_derivative_complex_jacobian(&self, softmax_values: &Vec<Vec<f64>>) -> Vec<Vec<Vec<f64>>> {
        let seq_len = softmax_values.len();

        // 3D tensor to hold Jacobian matrices for each row
        let mut derivative: Vec<Vec<Vec<f64>>> = Vec::new();

        for i in 0..seq_len {
            derivative.push(self.sparse_softmax_derivative_complex(&softmax_values[i], i, seq_len));
        }

        derivative
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        // Input shape e.g. [2][5] and out shape of weights [5][4] => we get final output [2][4]
        let output_batch = self.output_batch.as_ref().expect("Output batch is missing in attention head layer");
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in lattention head inear layer");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Padding mask batch is missing in attention head ");

        // dimensions [seq_len][seq_len] -> A
        let attention_weights_batch: &Vec<Vec<Vec<f64>>> = self.attention_weights_batch.as_ref().expect("Attention weights batch is missing in attention head");

        let batch_size = output_batch.len();

        // Initialize gradients for each parameter (weights and biases)
        let mut gradient_input_batch = vec![vec![vec![Complex::new(0.0, 0.0); previous_gradient_batch[0][0].len()]; previous_gradient_batch[0].len()]; input_batch.len()];
        let mut gradient_q_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_v[0].len()]; self.weights_v.len()]; batch_size];

        let mut gradient_bias_pos_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_v.len()]; self.weights_v.len()]; batch_size];

        for (batch_ind, previous_gradient) in previous_gradient_batch.iter().enumerate() {
            let q: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_q);
            let k: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_k);
            let v: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_v);

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
            // 2, 4 * 2,2 = 4,2 * 2,2 = 4,2#

            //let grad_wv = multiply_complex_with_f64(&transpose(&previous_gradient), &attention_weights_batch[batch_ind]);
            let grad_wv: Vec<Vec<Complex<f64>>> = self.multiply_sparse_backward(&transpose(&previous_gradient), &attention_weights_batch[batch_ind]);
            // 2,5 * 2, 4 = 5, 2 * 2, 4 = 5, 4
            gradient_v_batch[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &transpose(&grad_wv));

            let d_k = k[0].len() as f64;
            // Compute gradient of k_scaled w.r.t. k
            let k_scaled: Vec<Vec<Complex<f64>>> = scale_attention_scores(&transpose(&k), d_k);
            let q_scaled: Vec<Vec<Complex<f64>>> = scale_attention_scores(&q, d_k);

            // Compute activation derivative softmax
            // 16x(2/3)
            let attention_weights_restored = self.restore_sparse_matrix_f64(&attention_weights_batch[batch_ind], self.window_size, input_batch[batch_ind].len());
            let softmax_derivative: Vec<Vec<Vec<f64>>> = softmax_derivative_complex_jacobian(&attention_weights_restored);

            // 16 x(2x2/3x3)
            let sparse_softmax_derivative: Vec<Vec<Vec<f64>>> = self.sparse_softmax_derivative_complex_jacobian(&attention_weights_batch[batch_ind]);

            // println!("softmax_derivative dim: {}, {}, {}", &softmax_derivative.len(), &softmax_derivative[0].len(),  &softmax_derivative[0][0].len());

            // Gradient Wq
            //    => dl/dwq = XT * (Gt * VT * grad(A) * Kt/sqtr(dk))
            //    dl/dwq = dl/ds * ds/da * da/dq * dq/dWq
            // dl/dWq = (((dl/dO * dO/dS) * dS/dA) * dA/dQ) * dQ/dWq
            // 16x4 x 4x16 = 16x16
            let dl_ds: Vec<Vec<Complex<f64>>> = multiply_complex(&previous_gradient, &conjugate_transpose(&v));
            // 16x16 x 16x16 = 16x16
            let dl_da: Vec<Vec<f64>> = backpropagate_softmax_masked_real(&softmax_derivative, &dl_ds, &padding_mask_batch[batch_ind]);
            // println!("dl_da dim: {}, {}", &dl_da.len(), &dl_da[0].len(),);
            // 2,2 * 4, 2 = 2 * 2 * 2, 4 = 2,4
            let dl_dq: Vec<Vec<Complex<f64>>> = multiply_complex_with_f64(&k_scaled, &transpose(&dl_da));

            // println!("dl_dq dim: {}, {}", &dl_dq.len(), &dl_dq[0].len(),);
            // 2,5 * 4,2 = 5,2 * 2, 4 = 5, 4
            let dl_dwq: Vec<Vec<Complex<f64>>> = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &conjugate_transpose(&dl_dq));
            // println!("dl_dwq dim: {}, {}", &dl_dwq.len(), &dl_dwq[0].len());
            gradient_q_batch[batch_ind] = dl_dwq;

            // Gradient Wk
            //dl/dWk = (((dl/dO * dO/dS) * dS/dA) * dA/dKT) * dKT/dWk
            // 2,2 * 2,4 = 2,2 * 2,4 = 2,4
            let dl_dk: Vec<Vec<Complex<f64>>> = multiply_complex_with_f64(&conjugate_transpose(&q_scaled), &dl_da);

            // println!("dl_dk dim: {}, {}", &dl_dk.len(), &dl_dk[0].len());
            // 2,5 * 2,4 = 5,2 * 2, 4 = 5,4
            let dl_dwk: Vec<Vec<Complex<f64>>> = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &transpose(&dl_dk));
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

            gradient_bias_pos_batch[batch_ind] = convert_to_c_f64_2d(&dl_da);
            gradient_input_batch[batch_ind] = add_matrix(&dl_dqx, &dl_dkx);
            // println!("dl_dqx dim: {}, {}", &dl_dqx.len(), &dl_dqx[0].len());
            // println!("dl_dkx dim: {}, {}", &dl_dkx.len(), &dl_dkx[0].len());
            gradient_input_batch[batch_ind] = add_matrix(&gradient_input_batch[batch_ind], &dl_dvx);
        }

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            // gradient_input_batch = add_matrix_3d(&gradient_input_batch, &previous_gradient.get_gradient_input_batch());
            gradient_bias_pos_batch = add_matrix_3d(&gradient_bias_pos_batch, &&previous_gradient.get_gradient_bias_pos_batch());
            gradient_v_batch = add_matrix_3d(&gradient_v_batch, &&previous_gradient.get_gradient_weights_v_batch());
            gradient_q_batch = add_matrix_3d(&gradient_q_batch, &&previous_gradient.get_gradient_weights_q_batch());
            gradient_k_batch = add_matrix_3d(&gradient_k_batch, &&previous_gradient.get_gradient_weights_k_batch());
        }

        // Compute the gradients for the parameters and store them
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_weights_v_batch(gradient_v_batch);
        gradient.set_gradient_weights_q_batch(gradient_q_batch);
        gradient.set_gradient_weights_k_batch(gradient_k_batch);
        gradient.set_gradient_input_batch(gradient_input_batch);
        gradient.set_gradient_bias_pos_batch(gradient_bias_pos_batch);

        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in attention head layer");
        let (mut grad_w_q, mut grad_w_v, mut grad_w_k) = (gradient.get_gradient_weights_q(), gradient.get_gradient_weights_v(), gradient.get_gradient_weights_k());

        let mut grad_bias_pos = gradient.get_gradient_bias_pos();
        let input_batch = gradient.get_gradient_input_batch();
        let mut batch_size = input_batch.len() as f64;

        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }

        grad_w_q = average_matrix_by_scalar(&grad_w_q, batch_size);
        grad_w_v = average_matrix_by_scalar(&grad_w_v, batch_size);
        grad_w_k = average_matrix_by_scalar(&grad_w_k, batch_size);
        grad_bias_pos = average_matrix_by_scalar(&grad_bias_pos, batch_size);

        clip_all_gradients_by_global_norm_2d(&mut grad_w_q, &mut vec![], self.global_norm, self.max_norm);
        clip_all_gradients_by_global_norm_2d(&mut grad_w_v, &mut vec![], self.global_norm, self.max_norm);
        clip_all_gradients_by_global_norm_2d(&mut grad_w_k, &mut vec![], self.global_norm, self.max_norm);
        clip_all_gradients_by_global_norm_2d(&mut grad_bias_pos, &mut vec![], self.global_norm, self.max_norm);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        let (
            mut prev_m_weights_q,
            mut prev_v_weights_q,
            mut prev_m_weights_k,
            mut prev_v_weights_k,
            mut prev_m_weights_v,
            mut prev_v_weights_v,
            mut prev_m_bias_pos,
            mut prev_v_bias_pos,
            mut prev_v_weights_q_hat,
            mut prev_v_weights_k_hat,
            mut prev_v_weights_v_hat,
            mut prev_v_weights_p_hat,
        ) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_weigths_q(),
                previous_gradient.get_prev_v_weigths_q(),
                previous_gradient.get_prev_m_weigths_k(),
                previous_gradient.get_prev_v_weights_k(),
                previous_gradient.get_prev_m_weigths_v(),
                previous_gradient.get_prev_v_weights_v(),
                previous_gradient.get_prev_m_bias_pos(),
                previous_gradient.get_prev_v_bias_pos(),
                // vec![vec![Complex::new(0.0, 0.0); self.bias_pos[0].len()]; self.bias_pos.len()],
                // vec![vec![Complex::new(0.0, 0.0); self.bias_pos[0].len()]; self.bias_pos.len()],
                previous_gradient.get_prev_v_weights_q_hat(),
                previous_gradient.get_prev_v_weights_k_hat(),
                previous_gradient.get_prev_v_weights_v_hat(),
                previous_gradient.get_prev_v_weights_p_hat(),
            )
        } else {
            (
                vec![vec![Complex::new(0.0, 0.0); grad_w_q[0].len()]; grad_w_q.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_q[0].len()]; grad_w_q.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_k[0].len()]; grad_w_k.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_k[0].len()]; grad_w_k.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); self.bias_pos[0].len()]; self.bias_pos.len()],
                vec![vec![Complex::new(0.0, 0.0); self.bias_pos[0].len()]; self.bias_pos.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()],
                vec![vec![Complex::new(0.0, 0.0); self.bias_pos[0].len()]; self.bias_pos.len()],
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

        let seq_len = grad_bias_pos.len();
        let mut bias_pos_slice: Vec<Vec<Complex<f64>>> = self.bias_pos[0..seq_len].iter().map(|row| row[0..seq_len].to_vec()).collect();
        calculate_adam_w(
            &mut bias_pos_slice,
            &grad_bias_pos,
            &mut prev_m_bias_pos,
            &mut prev_v_bias_pos,
            &mut prev_v_weights_p_hat,
            learning_rate,
            time_step,
        );

        for i in 0..seq_len {
            for j in 0..seq_len {
                self.bias_pos[i][j] = bias_pos_slice[i][j];
            }
        }

        gradient.set_prev_m_weights_q(prev_m_weights_q);
        gradient.set_prev_v_weights_q(prev_v_weights_q);
        gradient.set_prev_m_weights_k(prev_m_weights_k);
        gradient.set_prev_v_weights_k(prev_v_weights_k);
        gradient.set_prev_m_weights_v(prev_m_weights_v);
        gradient.set_prev_v_weights_v(prev_v_weights_v);
        gradient.set_prev_m_bias_pos(prev_m_bias_pos);
        gradient.set_prev_v_bias_pos(prev_v_bias_pos);

        gradient.set_prev_v_weights_q_hat(prev_v_weights_q_hat);
        gradient.set_prev_v_weights_k_hat(prev_v_weights_k_hat);
        gradient.set_prev_v_weights_v_hat(prev_v_weights_v_hat);
        gradient.set_prev_v_weights_p_hat(prev_v_weights_p_hat);

        self.previous_gradient = Some(gradient.clone());
    }
}

pub fn scale_attention_scores(attention_scores: &Vec<Vec<Complex<f64>>>, d_k: f64) -> Vec<Vec<Complex<f64>>> {
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

pub fn calculate_window_tokens_batch<T: Clone + Send + Sync>(input: &Vec<Vec<Vec<T>>>, window_size: usize) -> Vec<Vec<Vec<Vec<Complex<f64>>>>>
where
    T: Copy + Into<Complex<f64>> + From<f64>,
{
    input.par_iter().map(|sequence| calculate_window_tokens(sequence, window_size)).collect()
}

pub fn calculate_window_tokens<T: Clone>(input: &Vec<Vec<T>>, window_size: usize) -> Vec<Vec<Vec<Complex<f64>>>>
where
    T: Copy + Into<Complex<f64>> + From<f64>,
{
    let seq_len = input.len();
    let mut windowed_tokens: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

    for i in 0..seq_len {
        let (start_ind, end_ind) = calculate_start_end_indices(i, window_size, seq_len);

        let mut window: Vec<Vec<Complex<f64>>> = Vec::new();
        for j in start_ind..end_ind {
            window.push(input[j].clone().into_iter().map(|x| x.into()).collect());
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

pub fn restore_sparse_row(sparse_row: &Vec<Complex<f64>>, token_index: usize, window_size: usize, seq_len: usize) -> Vec<Complex<f64>> {
    let (start_ind, end_ind) = calculate_start_end_indices(token_index, window_size, seq_len);
    let mut row = vec![Complex::new(f64::NEG_INFINITY, f64::NEG_INFINITY); seq_len];
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
