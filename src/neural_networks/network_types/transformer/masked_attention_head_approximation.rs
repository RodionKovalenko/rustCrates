use num::Complex;
use rand_distr::{Distribution, Normal, Uniform};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer::LayerType, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    utils::{
        adam_w::calculate_adam_w,
        matrix::{add_matrix, clip_gradients, is_nan_or_inf, multiply_complex},
        matrix_approximation::phi_stable,
        weights_initializer::initialize_weights_complex,
    },
};

use super::transformer_network::MAX_CONTEXT_WINDOW_SIZE;

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedAttentionHeadApproximation {
    pub weights_q: Vec<Vec<Complex<f64>>>,
    pub weights_k: Vec<Vec<Complex<f64>>>,
    pub weights_v: Vec<Vec<Complex<f64>>>,

    pub bias_pos: Vec<Vec<Complex<f64>>>,

    pub bias_q: Vec<Complex<f64>>,
    pub bias_k: Vec<Complex<f64>>,
    pub bias_v: Vec<Complex<f64>>,

    pub layer_type: LayerType,
    pub learning_rate: f64,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub previous_gradient: Option<Gradient>,
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

    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,

    pub m1: Vec<Vec<Complex<f64>>>,
    pub v1: Vec<Vec<Complex<f64>>>,
}

impl MaskedAttentionHeadApproximation {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

        let mut bias_pos: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); MAX_CONTEXT_WINDOW_SIZE]; MAX_CONTEXT_WINDOW_SIZE];

        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);

        initialize_weights_complex(MAX_CONTEXT_WINDOW_SIZE, MAX_CONTEXT_WINDOW_SIZE, &mut bias_pos);

        let bias_q: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let bias_k: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let bias_v: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

        // Initialize random normal distribution for weights
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).expect("no normal distribution found");
        let num_features = cols;
        let d_k = cols;

        // Random matrix w: shape (num_features, d_k)
        let w: Vec<Vec<f64>> = (0..num_features).map(|_| (0..d_k).map(|_| normal.sample(&mut rng)).collect()).collect();

        // Random biases b: shape (num_features), uniform from 0 to 2pi
        let uniform = Uniform::new(0.0, 2.0 * std::f64::consts::PI).expect("no uniform distribution");
        let b: Vec<f64> = (0..num_features).map(|_| uniform.sample(&mut rng)).collect();

        MaskedAttentionHeadApproximation {
            weights_q,
            weights_k,
            weights_v,

            bias_pos,
            bias_q,
            bias_k,
            bias_v,
            layer_type: LayerType::InputLayer,
            learning_rate: learning_rate,

            w,
            b,

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
        }
    }

    fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, learning_rate: f64) -> MaskedAttentionHeadApproximation {
        let mut attention_layer: MaskedAttentionHeadApproximation = MaskedAttentionHeadApproximation::new(rows, cols, learning_rate);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl MaskedAttentionHeadApproximation {
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
        let seq_len = input_batch[0].len();
        let batch_size = input_batch.len();
        let d_k = self.weights_k[0].len();

        self.input_batch = Some(input_batch.clone());
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();

        // Step 1: Compute Q, K, V for all tokens in the batch
        let q_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_q)).collect();

        let k_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_k)).collect();

        let v_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_v)).collect();

        // Step 2: Add positional bias to Q and K before phi mapping
        // Assuming bias_pos has shape [seq_len][d_k]
        let q_pos_batch: Vec<_> = q_batch.iter().map(|q_seq| add_matrix::<Complex<f64>>(q_seq, &self.bias_pos)).collect();

        let k_pos_batch: Vec<_> = k_batch.iter().map(|k_seq| add_matrix::<Complex<f64>>(k_seq, &self.bias_pos)).collect();

        // Step 3: Scale Q and K by sqrt(d_k)
        let scale = 1.0 / (d_k as f64).sqrt();

        // Step 4: Compute phi(Q) and phi(K) using stable kernel mapping
        let phi_q_batch: Vec<Vec<Vec<f64>>> = q_pos_batch
            .iter()
            .map(|q_seq| {
                q_seq
                    .iter()
                    .map(|q_token| {
                        let scaled_q: Vec<Complex<f64>> = q_token.iter().map(|c| *c * scale).collect();
                        phi_stable(&scaled_q, &self.w, &self.b) // self.w, self.b are random features params
                    })
                    .collect()
            })
            .collect();

        let phi_k_batch: Vec<Vec<Vec<f64>>> = k_pos_batch
            .iter()
            .map(|k_seq| {
                k_seq
                    .iter()
                    .map(|k_token| {
                        let scaled_k: Vec<Complex<f64>> = k_token.iter().map(|c| *c * scale).collect();
                        phi_stable(&scaled_k, &self.w, &self.b)
                    })
                    .collect()
            })
            .collect();

        // Step 5: Initialize prefix sums for phi_k and phi_k * v
        let mut prefix_phi_k: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; self.w.len()]; seq_len]; batch_size];
        let mut prefix_phi_kv: Vec<Vec<Vec<Vec<Complex<f64>>>>> = vec![vec![vec![vec![Complex::new(0.0, 0.0); v_batch[0][0].len()]; self.w.len()]; seq_len]; batch_size];

        for b in 0..batch_size {
            for t in 0..seq_len {
                // Accumulate prefix sums for phi_k
                if t == 0 {
                    prefix_phi_k[b][t] = phi_k_batch[b][t].clone();
                } else {
                    prefix_phi_k[b][t] = prefix_phi_k[b][t - 1].iter().zip(&phi_k_batch[b][t]).map(|(prev, curr)| prev + curr).collect();
                }

                let d_v = v_batch[b][0].len();

                if t == 0 {
                    // Initialize prefix_phi_kv at t=0 as outer product phi_k[t] * v[t]
                    prefix_phi_kv[b][t] = (0..self.w.len())
                        .map(|r_idx| {
                            let phi_k_val = phi_k_batch[b][t][r_idx];
                            (0..d_v).map(|dv_idx| v_batch[b][t][dv_idx] * phi_k_val).collect::<Vec<Complex<f64>>>()
                        })
                        .collect::<Vec<Vec<Complex<f64>>>>();
                } else {
                    // Accumulate prefix sums for phi_kv
                    prefix_phi_kv[b][t] = prefix_phi_kv[b][t - 1]
                        .iter()
                        .enumerate()
                        .map(|(r_idx, prev_vec)| prev_vec.iter().zip(&v_batch[b][t]).map(|(&prev_val, &v_val)| prev_val + v_val * phi_k_batch[b][t][r_idx]).collect::<Vec<Complex<f64>>>())
                        .collect::<Vec<Vec<Complex<f64>>>>();
                }
            }
        }

        // Step 6: Compute final outputs using prefix sums and phi_q
        // output[t] = (phi_q[t]^T * prefix_phi_kv[t]) / (phi_q[t]^T * prefix_phi_k[t])
        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); v_batch[0][0].len()]; seq_len]; batch_size];

        for b in 0..batch_size {
            for t in 0..seq_len {
                let d_v = v_batch[b][0].len();
                let mut numerator = vec![Complex::new(0.0, 0.0); d_v];

                for r_idx in 0..self.w.len() {
                    for dv_idx in 0..d_v {
                        numerator[dv_idx] += prefix_phi_kv[b][t][r_idx][dv_idx] * phi_q_batch[b][t][r_idx];
                    }
                }

                let denominator: f64 = prefix_phi_k[b][t].iter().zip(&phi_q_batch[b][t]).map(|(a, b)| a * b).sum::<f64>().max(1e-8); // avoid division by zero

                for dv_idx in 0..d_v {
                    output_batch[b][t][dv_idx] = numerator[dv_idx] / denominator;
                }
            }
        }

        // Step 7: Apply padding mask to zero out padded tokens
        for b in 0..batch_size {
            for t in 0..seq_len {
                if padding_mask_batch[b][t] == 0 {
                    output_batch[b][t].iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
                }
            }
        }

        // Step 8: Store intermediate results for debugging or further use
        self.output_batch = Some(output_batch.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }
    pub fn backward(&mut self, _grad_output_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        // let batch_size = grad_output_batch.len();
        // let seq_len = grad_output_batch[0].len();
        // let r = self.phi_q_batch.as_ref().unwrap()[0][0].len(); // feature dim
        // let d_v = grad_output_batch[0][0].len(); // output dim

        // // Retrieve stored forward intermediates
        // let phi_q_batch = self.phi_q_batch.as_ref().expect("phi_q missing");
        // let phi_k_batch = self.phi_k_batch.as_ref().expect("phi_k missing");
        // let phi_kv_prefix_sums = self.phi_kv_prefix_sums.as_ref().expect("phi_kv prefix sums missing");
        // let phi_k_prefix_sums = self.phi_k_prefix_sums.as_ref().expect("phi_k prefix sums missing");
        // let output_batch = self.output_batch.as_ref().expect("output batch missing");
        // let input_batch = self.input_batch.as_ref().expect("input batch missing");

        // // Initialize gradients w.r.t inputs and weights
        // let mut grad_phi_q = vec![vec![vec![Complex::new(0.0, 0.0); r]; seq_len]; batch_size];
        // let mut grad_phi_k = vec![vec![vec![Complex::new(0.0, 0.0); r]; seq_len]; batch_size];
        // let mut grad_v = vec![vec![vec![Complex::new(0.0, 0.0); d_v]; seq_len]; batch_size];

        // // Gradients w.r.t weight matrices
        // let mut grad_w_q = vec![vec![Complex::new(0.0, 0.0); self.weights_q[0].len()]; self.weights_q.len()];
        // let mut grad_w_k = vec![vec![Complex::new(0.0, 0.0); self.weights_k[0].len()]; self.weights_k.len()];
        // let mut grad_w_v = vec![vec![Complex::new(0.0, 0.0); self.weights_v[0].len()]; self.weights_v.len()];

        // // Gradients w.r.t input batch
        // let mut grad_input_batch = vec![vec![vec![Complex::new(0.0, 0.0); input_batch[0][0].len()]; seq_len]; batch_size];

        // for b in 0..batch_size {
        //     // To accumulate gradients through prefix sums for phi_k and phi_kv
        //     let mut grad_phi_k_prefix = vec![vec![Complex::new(0.0, 0.0); r]; seq_len];
        //     let mut grad_phi_kv_prefix = vec![vec![vec![Complex::new(0.0, 0.0); d_v]; r]; seq_len];

        //     for t in 0..seq_len {
        //         let grad_output_t = &grad_output_batch[b][t]; // dL/d output_t, shape [d_v]
        //         let phi_q_t = &phi_q_batch[b][t]; // shape [r]
        //         let denom = dot_product_complex(phi_q_t, &phi_k_prefix_sums[b][t]); // scalar Complex<f64>
        //         let numerator = &phi_kv_prefix_sums[b][t]; // shape [r][d_v]
        //         let output_t = &output_batch[b][t]; // [d_v]

        //         // 1. Gradient w.r.t phi(Q_t):
        //         // dL/dphiQ_t = (grad_output_t^T * numerator) / denom - output_t * (grad_output_t^T * denom_gradient) / denom^2
        //         // but denom_gradient = phi_k_prefix_sums[b][t]
        //         // Simplified to:
        //         // grad_phi_q[t] = (numerator * grad_output_t) / denom - (phi_k_prefix_sums * (grad_output_t dot output_t)) / denom
        //         // We compute elementwise below:

        //         let mut grad_phi_q_t = vec![Complex::new(0.0, 0.0); r];
        //         for i in 0..r {
        //             let mut sum_num = Complex::new(0.0, 0.0);
        //             for dv_i in 0..d_v {
        //                 sum_num += numerator[i][dv_i] * grad_output_t[dv_i];
        //             }
        //             let output_grad_dot = dot_product_complex(grad_output_t, output_t);
        //             grad_phi_q_t[i] = sum_num / denom - phi_k_prefix_sums[b][t][i] * output_grad_dot / (denom * denom);
        //         }
        //         grad_phi_q[b][t] = grad_phi_q_t;

        //         // 2. Gradient w.r.t prefix sums phi_kv and phi_k:
        //         // dL/d prefix_phi_kv[t] += phi_q_t * grad_output_t / denom
        //         // dL/d prefix_phi_k[t] += -phi_q_t * output_t * (grad_output_t) / denom
        //         // We accumulate these to later backprop through prefix sums to individual timesteps

        //         for i in 0..r {
        //             for dv_i in 0..d_v {
        //                 grad_phi_kv_prefix[t][i][dv_i] += phi_q_t[i] * grad_output_t[dv_i] / denom;
        //             }
        //         }
        //         for i in 0..r {
        //             let output_grad_dot = dot_product_complex(grad_output_t, output_t);
        //             grad_phi_k_prefix[t][i] -= phi_q_t[i] * output_grad_dot / (denom * denom);
        //         }
        //     }

        //     // 3. Backprop through prefix sums to get grad_phi_k and grad_v:
        //     // prefix sums: prefix_phi_k[t] = sum_{j=0}^t phi_k[j]
        //     // So grad_phi_k[j] = sum_{t>=j} grad_phi_k_prefix[t]
        //     // similarly for phi_kv

        //     for j in 0..seq_len {
        //         for t in j..seq_len {
        //             for i in 0..r {
        //                 grad_phi_k[b][j][i] += grad_phi_k_prefix[t][i];
        //                 for dv_i in 0..d_v {
        //                     grad_v[b][j][dv_i] += grad_phi_kv_prefix[t][i][dv_i];
        //                 }
        //             }
        //         }
        //     }

        //     // 4. Backprop through phi (feature map) to get grad w.r.t Q_t, K_t:
        //     // For example, if phi(x) = exp(x) or your kernel feature,
        //     // grad_Q_t = grad_phi_q_t * dphi/dQ_t
        //     // grad_K_t = grad_phi_k_t * dphi/dK_t
        //     // Assuming phi = exp with pre-stored norm factors etc. (adjust as needed)

        //     for t in 0..seq_len {
        //         // Placeholder: compute grad w.r.t Q_t and K_t from grad_phi_q and grad_phi_k
        //         // You need to implement dphi/dx here
        //         // For example, if phi(x) = exp(some transformation), then dphi/dx = phi(x) * derivative of inside

        //         let grad_q_t = compute_grad_phi_input(&grad_phi_q[b][t], &self.phi_q_cache[b][t]);
        //         let grad_k_t = compute_grad_phi_input(&grad_phi_k[b][t], &self.phi_k_cache[b][t]);

        //         // 5. Backprop through linear layers to get grad w.r.t weights and input
        //         // Q = input * W_q
        //         // grad_W_q += input^T * grad_q_t
        //         let input_t = &input_batch[b][t];
        //         let grad_wq_t = multiply_complex(&conjugate_transpose(input_t), &grad_q_t);
        //         add_to_matrix(&mut grad_w_q, &grad_wq_t);

        //         // K = input * W_k
        //         let grad_wk_t = multiply_complex(&conjugate_transpose(input_t), &grad_k_t);
        //         add_to_matrix(&mut grad_w_k, &grad_wk_t);

        //         // V = input * W_v
        //         let grad_v_t = &grad_v[b][t]; // gradient w.r.t V_t from step 3
        //         let grad_wv_t = multiply_complex(&conjugate_transpose(input_t), grad_v_t);
        //         add_to_matrix(&mut grad_w_v, &grad_wv_t);

        //         // Also compute gradient w.r.t input through all three projections:
        //         let grad_input_q = multiply_complex(&grad_q_t, &conjugate_transpose(&self.weights_q));
        //         let grad_input_k = multiply_complex(&grad_k_t, &conjugate_transpose(&self.weights_k));
        //         let grad_input_v = multiply_complex(&grad_v_t, &conjugate_transpose(&self.weights_v));

        //         grad_input_batch[b][t] = add_matrix(&add_matrix(&grad_input_q, &grad_input_k), &grad_input_v);
        //     }
        // }

        // Compose the gradient struct and save in self.gradient
        let gradient = Gradient::new_default();
        // gradient.set_gradient_weights_q_batch(vec![grad_w_q]);
        // gradient.set_gradient_weights_k_batch(vec![grad_w_k]);
        // gradient.set_gradient_weights_v_batch(vec![grad_w_v]);
        // gradient.set_gradient_input_batch(grad_input_batch);

        // self.gradient = Some(gradient.clone());
        gradient
    }
    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in attention head layer");
        let (mut grad_w_q, mut grad_w_v, mut grad_w_k) = (gradient.get_gradient_weights_q(), gradient.get_gradient_weights_v(), gradient.get_gradient_weights_k());

        let input_batch = gradient.get_gradient_input_batch();
        let mut grad_bias_pos = gradient.get_gradient_bias_pos();
        let batch_size = input_batch.len() as f64;

        let threshold = 1.0;
        clip_gradients(&mut grad_w_q, threshold);
        clip_gradients(&mut grad_w_v, threshold);
        clip_gradients(&mut grad_w_k, threshold);
        clip_gradients(&mut grad_bias_pos, threshold);

        let mut prev_m_weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_q[0].len()]; grad_w_q.len()];
        let mut prev_v_weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_q[0].len()]; grad_w_q.len()];

        let mut prev_m_weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_k[0].len()]; grad_w_k.len()];
        let mut prev_v_weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_k[0].len()]; grad_w_k.len()];

        let mut prev_m_weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()];
        let mut prev_v_weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()];

        let mut prev_m_bias_pos: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_bias_pos[0].len()]; grad_bias_pos.len()];
        let mut prev_v_bias_pos: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_bias_pos[0].len()]; grad_bias_pos.len()];

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        if let Some(previous_gradient) = &mut self.previous_gradient {
            prev_m_weights_q = previous_gradient.get_prev_m_weigths_q();
            prev_v_weights_q = previous_gradient.get_prev_v_weigths_q();

            prev_m_weights_k = previous_gradient.get_prev_m_weigths_k();
            prev_v_weights_k = previous_gradient.get_prev_v_weigths_k();

            prev_m_weights_v = previous_gradient.get_prev_m_weigths_v();
            prev_v_weights_v = previous_gradient.get_prev_v_weigths_v();

            self.weights_q = calculate_adam_w(&self.weights_q, &grad_w_q, &mut prev_m_weights_q, &mut prev_v_weights_q, learning_rate, time_step);
            self.weights_k = calculate_adam_w(&self.weights_k, &grad_w_k, &mut prev_m_weights_k, &mut prev_v_weights_k, learning_rate, time_step);
            self.weights_v = calculate_adam_w(&self.weights_v, &grad_w_v, &mut prev_m_weights_v, &mut prev_v_weights_v, learning_rate, time_step);

            let seq_len = grad_bias_pos.len();
            let bias_pos_slice: Vec<Vec<Complex<f64>>> = self.bias_pos[0..seq_len].iter().map(|row| row[0..seq_len].to_vec()).collect();
            let updated_slice = calculate_adam_w(&bias_pos_slice, &grad_bias_pos, &mut prev_m_bias_pos, &mut prev_v_bias_pos, learning_rate, time_step);

            for i in 0..seq_len {
                for j in 0..seq_len {
                    self.bias_pos[i][j] = updated_slice[i][j];
                }
            }
        } else {
            // Update weights q
            for (i, row) in self.weights_q.iter_mut().enumerate() {
                for (j, weight_value) in row.iter_mut().enumerate() {
                    if !is_nan_or_inf(&grad_w_q[i][j]) {
                        *weight_value -= self.learning_rate * (grad_w_q[i][j] / batch_size);
                    }
                }
            }

            // Update weights v
            for (i, row) in self.weights_v.iter_mut().enumerate() {
                for (j, weight_value) in row.iter_mut().enumerate() {
                    if !is_nan_or_inf(&grad_w_v[i][j]) {
                        *weight_value -= self.learning_rate * (grad_w_v[i][j] / batch_size);
                    }
                }
            }

            // Update weights k
            for (i, row) in self.weights_k.iter_mut().enumerate() {
                for (j, weight_value) in row.iter_mut().enumerate() {
                    if !is_nan_or_inf(&grad_w_k[i][j]) {
                        *weight_value -= self.learning_rate * (grad_w_k[i][j] / batch_size);
                    }
                }
            }

            // Update bias pos
            for (i, row) in self.bias_pos.iter_mut().enumerate() {
                for (j, bias_pos_v) in row.iter_mut().enumerate() {
                    if i < grad_bias_pos.len() && j < grad_bias_pos[i].len() && !is_nan_or_inf(&grad_bias_pos[i][j]) {
                        *bias_pos_v -= self.learning_rate * (grad_bias_pos[i][j] / batch_size);
                    }
                }
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

        self.previous_gradient = Some(gradient.clone());
    }
}
