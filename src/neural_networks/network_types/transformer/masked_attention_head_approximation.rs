use num::Complex;
use num_complex::ComplexFloat;
use rand_distr::{Distribution, Normal, Uniform};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer::LayerType, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_types::transformer::transformer_network::EMA_SCALER,
    utils::{
        adam_w::calculate_adam_w,
        matrix::{add_matrix, average_matrix_by_scalar, clip_all_gradients_by_global_norm_2d, compute_global_norm, get_reduced_matrix, multiply_complex},
        matrix_approximation::phi,
        weights_initializer::initialize_weights_complex,
    },
};

use super::transformer_network::MAX_CONTEXT_WINDOW_SIZE;

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

    pub smoothing: f64,
    pub ema: f64,

    pub batch_size: usize,

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

    pub w: Vec<Vec<Complex<f64>>>,
    pub b: Vec<Complex<f64>>,

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

        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).expect("no normal distribution found");
        let num_features = cols;
        let d_k = cols;

        let w: Vec<Vec<Complex<f64>>> = (0..num_features).map(|_| (0..d_k).map(|_| Complex::new(normal.sample(&mut rng), 0.0)).collect()).collect();

        let uniform = Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
        let b: Vec<Complex<f64>> = (0..num_features).map(|_| Complex::new(uniform.sample(&mut rng), 0.0)).collect();

        Self {
            weights_q,
            weights_k,
            weights_v,
            bias_pos,
            bias_q,
            bias_k,
            bias_v,
            layer_type: LayerType::InputLayer,
            learning_rate,
            w,
            b,
            smoothing: 0.99,
            ema: 0.0,
            batch_size: 0,
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

    pub fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, learning_rate: f64) -> Self {
        let mut attention_layer = Self::new(rows, cols, learning_rate);
        attention_layer.set_layer_type(layer_type);
        attention_layer
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch = layer_input.get_input_batch();
        let padding_mask_batch = layer_input.get_padding_mask_batch();
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let d_k = self.weights_k[0].len();

        self.input_batch = Some(input_batch.clone());
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();

        let q_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_q)).collect();
        let k_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_k)).collect();
        let v_batch: Vec<_> = input_batch.par_iter().map(|input| multiply_complex(input, &self.weights_v)).collect();

        let q_pos_batch: Vec<_> = q_batch.iter().map(|q_seq| add_matrix::<Complex<f64>>(&q_seq, &get_reduced_matrix(&self.bias_pos, q_seq.len(), q_seq[0].len()))).collect();
        let k_pos_batch: Vec<_> = k_batch.iter().map(|k_seq| add_matrix::<Complex<f64>>(&k_seq, &get_reduced_matrix(&self.bias_pos, k_seq.len(), k_seq[0].len()))).collect();

        let scale = 1.0 / (d_k as f64).sqrt();

        let phi_q_batch: Vec<Vec<Vec<Complex<f64>>>> = q_pos_batch
            .iter()
            .map(|q_seq| {
                q_seq
                    .iter()
                    .map(|q_token| {
                        let scaled_q: Vec<Complex<f64>> = q_token.iter().map(|c| *c * scale).collect();
                        self.w.iter().map(|omega_i| phi(&scaled_q, omega_i)).collect()
                    })
                    .collect()
            })
            .collect();

        let phi_k_batch: Vec<Vec<Vec<Complex<f64>>>> = k_pos_batch
            .iter()
            .map(|k_seq| {
                k_seq
                    .iter()
                    .map(|k_token| {
                        let scaled_k: Vec<Complex<f64>> = k_token.iter().map(|c| *c * scale).collect();
                        self.w.iter().map(|omega_i| phi(&scaled_k, omega_i)).collect()
                    })
                    .collect()
            })
            .collect();

        let mut prefix_phi_k: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.w.len()]; seq_len]; batch_size];
        let mut prefix_phi_kv: Vec<Vec<Vec<Vec<Complex<f64>>>>> = vec![vec![vec![vec![Complex::new(0.0, 0.0); v_batch[0][0].len()]; self.w.len()]; seq_len]; batch_size];

        for b in 0..batch_size {
            for t in 0..seq_len {
                if t == 0 {
                    prefix_phi_k[b][t] = phi_k_batch[b][t].clone();

                    prefix_phi_kv[b][t] = (0..self.w.len())
                        .map(|r_idx| {
                            let phi_k_val = phi_k_batch[b][t][r_idx];
                            (0..v_batch[b][t].len()).map(|dv_idx| v_batch[b][t][dv_idx] * phi_k_val).collect()
                        })
                        .collect();
                } else {
                    prefix_phi_k[b][t] = prefix_phi_k[b][t - 1].iter().zip(&phi_k_batch[b][t]).map(|(prev, curr)| prev + curr).collect();

                    prefix_phi_kv[b][t] = prefix_phi_kv[b][t - 1]
                        .iter()
                        .enumerate()
                        .map(|(r_idx, prev_vec)| prev_vec.iter().zip(&v_batch[b][t]).map(|(&prev_val, &v_val)| prev_val + v_val * phi_k_batch[b][t][r_idx]).collect())
                        .collect();
                }
            }
        }

        let mut output_batch = vec![vec![vec![Complex::new(0.0, 0.0); v_batch[0][0].len()]; seq_len]; batch_size];

        for b in 0..batch_size {
            for t in 0..seq_len {
                let d_v = v_batch[b][0].len();
                let mut numerator = vec![Complex::new(0.0, 0.0); d_v];

                for r_idx in 0..self.w.len() {
                    for dv_idx in 0..d_v {
                        numerator[dv_idx] += prefix_phi_kv[b][t][r_idx][dv_idx] * phi_q_batch[b][t][r_idx];
                    }
                }

                let mut denominator: Complex<f64> = prefix_phi_k[b][t].iter().zip(&phi_q_batch[b][t]).map(|(a, b)| a * b).sum::<Complex<f64>>();
                // Clamp real part to minimum value to avoid division by zero
                if denominator.abs() < 1e-8 {
                    denominator += Complex::new(1e-8, 0.0);
                }

                for dv_idx in 0..d_v {
                    output_batch[b][t][dv_idx] = numerator[dv_idx] / denominator;
                }
            }
        }

        for b in 0..batch_size {
            for t in 0..seq_len {
                if padding_mask_batch[b][t] == 0 {
                    output_batch[b][t].iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
                }
            }
        }

        self.output_batch = Some(output_batch.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, grad_output_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("No input");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("No mask");
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();
        let d_model = input_batch[0][0].len();
        let d_k = self.weights_k[0].len();
        let scale = 1.0 / (d_k as f64).sqrt();

        // Retain correct shapes
        let mut grad_in = vec![vec![vec![Complex::new(0.0, 0.0); d_model]; seq_len]; batch_size];
        let mut grad_wq_b: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut grad_wk_b: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut grad_wv_b: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_v[0].len()]; self.weights_v.len()]; batch_size];
        let mut grad_bias_pos: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); seq_len]; seq_len]; batch_size];

        let q_batch: Vec<_> = input_batch.iter().map(|inp| multiply_complex(inp, &self.weights_q)).collect();
        let k_batch: Vec<_> = input_batch.iter().map(|inp| multiply_complex(inp, &self.weights_k)).collect();
        let v_batch: Vec<_> = input_batch.iter().map(|inp| multiply_complex(inp, &self.weights_v)).collect();

        let q_pos: Vec<_> = q_batch.iter().map(|seq| add_matrix(seq, &get_reduced_matrix(&self.bias_pos, seq.len(), seq[0].len()))).collect();
        let k_pos: Vec<_> = k_batch.iter().map(|seq| add_matrix(seq, &get_reduced_matrix(&self.bias_pos, seq.len(), seq[0].len()))).collect();

        let q_scaled: Vec<Vec<Vec<Complex<f64>>>> = q_pos.iter().map(|seq| seq.iter().map(|tok| tok.iter().map(|c| *c * scale).collect()).collect()).collect();
        let k_scaled: Vec<Vec<Vec<Complex<f64>>>> = k_pos.iter().map(|seq| seq.iter().map(|tok| tok.iter().map(|c| *c * scale).collect()).collect()).collect();

        let (phi_q, dphi_q) = self.compute_phi_and_derivative(&q_scaled);
        let (phi_k, dphi_k) = self.compute_phi_and_derivative(&k_scaled);

        let (prefix_phi_k, prefix_phi_kv) = self.recompute_prefix_sums(&phi_k, &v_batch);

        for b in 0..batch_size {
            for t in 0..seq_len {
                if padding_mask_batch[b][t] == 0 {
                    continue; // correctly skip masked
                }
                let grad_out = &grad_output_batch[b][t];

                // Same as forward numerator/denominator
                let mut numerator = vec![Complex::new(0.0, 0.0); grad_out.len()];
                for r in 0..self.w.len() {
                    for dv in 0..grad_out.len() {
                        numerator[dv] += prefix_phi_kv[b][t][r][dv] * phi_q[b][t][r];
                    }
                }
                let denominator: Complex<f64> = prefix_phi_k[b][t].iter().zip(&phi_q[b][t]).map(|(a, b)| *a * *b).sum();
                let safe_denominator = if denominator.norm() < 1e-8 { Complex::new(1e-8, 0.0) } else { denominator };

                // Numerator and denominator gradients
                let grad_num: Vec<Complex<f64>> = grad_out.iter().map(|&g| g / safe_denominator).collect();
                let grad_den: Complex<f64> = grad_out.iter().zip(&numerator).map(|(&g, &n)| -g * n / (safe_denominator * safe_denominator)).sum();

                // Through phi_q
                let mut grad_phi_q_num = vec![Complex::new(0.0, 0.0); self.w.len()];
                let mut grad_pref_kv_t = vec![vec![Complex::new(0.0, 0.0); grad_out.len()]; self.w.len()];
                for (dv, &gn) in grad_num.iter().enumerate() {
                    for r in 0..self.w.len() {
                        grad_phi_q_num[r] += gn * prefix_phi_kv[b][t][r][dv];
                        grad_pref_kv_t[r][dv] += gn * phi_q[b][t][r];
                    }
                }
                let mut grad_phi_q_den = vec![Complex::new(0.0, 0.0); self.w.len()];
                let mut grad_pref_k_t = vec![Complex::new(0.0, 0.0); self.w.len()];
                for r in 0..self.w.len() {
                    grad_phi_q_den[r] = grad_den * prefix_phi_k[b][t][r];
                    grad_pref_k_t[r] = grad_den * phi_q[b][t][r];
                }
                let grad_phi_q: Vec<Complex<f64>> = grad_phi_q_num.iter().zip(&grad_phi_q_den).map(|(a, b)| *a + *b).collect();

                // Backprop through phi_q -> q_scaled -> q_pos
                let grad_q_scaled = self.backward_through_phi(&grad_phi_q, &q_scaled[b][t], &dphi_q[b][t]);
                let grad_q_pos: Vec<Complex<f64>> = grad_q_scaled.iter().map(|&g| g * scale).collect();

                // Accumulate weights and input grad for Q path (keep shapes)
                for i in 0..d_model {
                    for j in 0..d_k.min(grad_q_pos.len()) {
                        grad_wq_b[b][i][j] += (input_batch[b][t][i] * grad_q_pos[j]).conj();
                        grad_in[b][t][i] += (self.weights_q[i][j] * grad_q_pos[j]).conj();
                    }
                }
                // Accumulate bias_pos gradient: only the relevant slice for this token position t
                for pos in 0..seq_len.min(grad_q_pos.len()) {
                    grad_bias_pos[b][t][pos] += grad_q_pos[pos].conj(); // only for q_path, t-th row
                }

                // Recurse K/V
                self.backward_through_prefix_sums_batch_fixed(b, t, &grad_pref_k_t, &grad_pref_kv_t, &phi_k, &v_batch, &dphi_k, &k_scaled, input_batch, &mut grad_wk_b, &mut grad_wv_b, &mut grad_bias_pos, &mut grad_in, scale);
            }
        }
        let mut grad = Gradient::new_default();
        grad.set_gradient_weights_q_batch(grad_wq_b);
        grad.set_gradient_weights_k_batch(grad_wk_b);
        grad.set_gradient_weights_v_batch(grad_wv_b);
        grad.set_gradient_bias_pos_batch(grad_bias_pos);
        grad.set_gradient_input_batch(grad_in);

        self.gradient = Some(grad.clone());
        grad
    }

    fn backward_through_prefix_sums_batch_fixed(
        &self,
        batch_idx: usize,
        current_t: usize,
        grad_prefix_phi_k_t: &[Complex<f64>],
        grad_prefix_phi_kv_t: &[Vec<Complex<f64>>],
        phi_k_batch: &[Vec<Vec<Complex<f64>>>],
        v_batch: &[Vec<Vec<Complex<f64>>>],
        dphi_k_batch: &[Vec<Vec<Vec<Complex<f64>>>>],
        k_scaled_batch: &[Vec<Vec<Complex<f64>>>],
        input_batch: &[Vec<Vec<Complex<f64>>>],
        grad_weights_k_batch: &mut Vec<Vec<Vec<Complex<f64>>>>,
        grad_weights_v_batch: &mut Vec<Vec<Vec<Complex<f64>>>>,
        grad_bias_pos: &mut Vec<Vec<Vec<Complex<f64>>>>,
        grad_input_batch: &mut Vec<Vec<Vec<Complex<f64>>>>,
        scale: f64,
    ) {
        let num_features = self.w.len();
        let d_model = input_batch[batch_idx][0].len();
        let d_k = k_scaled_batch[batch_idx][0].len();
        let d_v = self.weights_v[0].len();
        let d_v_output = grad_prefix_phi_kv_t[0].len();

        let mut accumulated_grad_prefix_phi_k = vec![Complex::new(0.0, 0.0); num_features];
        let mut accumulated_grad_prefix_phi_kv = vec![vec![Complex::new(0.0, 0.0); d_v_output]; num_features];

        for t in (0..=current_t).rev() {
            let mut grad_prefix_phi_k_current = accumulated_grad_prefix_phi_k.clone();
            let mut grad_prefix_phi_kv_current = accumulated_grad_prefix_phi_kv.clone();

            if t == current_t {
                for r_idx in 0..num_features {
                    grad_prefix_phi_k_current[r_idx] += grad_prefix_phi_k_t[r_idx];
                    for dv_idx in 0..d_v_output {
                        grad_prefix_phi_kv_current[r_idx][dv_idx] += grad_prefix_phi_kv_t[r_idx][dv_idx];
                    }
                }
            }

            let mut grad_phi_k_local = vec![Complex::new(0.0, 0.0); num_features];
            let mut grad_v_local = vec![Complex::new(0.0, 0.0); d_v];

            for r_idx in 0..num_features {
                let phi_k_val = phi_k_batch[batch_idx][t][r_idx];
                for dv_idx in 0..d_v_output.min(d_v) {
                    if dv_idx < v_batch[batch_idx][t].len() {
                        grad_phi_k_local[r_idx] += grad_prefix_phi_kv_current[r_idx][dv_idx] * v_batch[batch_idx][t][dv_idx];
                        grad_v_local[dv_idx] += grad_prefix_phi_kv_current[r_idx][dv_idx] * phi_k_val;
                    }
                }
            }

            let total_grad_phi_k: Vec<Complex<f64>> = grad_prefix_phi_k_current.iter().zip(grad_phi_k_local.iter()).map(|(a, b)| *a + *b).collect();
            let grad_k_scaled_t = self.backward_through_phi(&total_grad_phi_k, &k_scaled_batch[batch_idx][t], &dphi_k_batch[batch_idx][t]);
            let grad_k_pos_t: Vec<Complex<f64>> = grad_k_scaled_t.iter().map(|&g| g * scale).collect();

            // Gradient accumulation for weights_k and grad_input (correct as before)
            for i in 0..d_model {
                for j in 0..d_k.min(grad_k_pos_t.len()) {
                    grad_weights_k_batch[batch_idx][i][j] += (input_batch[batch_idx][t][i] * grad_k_pos_t[j]).conj();
                    grad_input_batch[batch_idx][t][i] += (self.weights_k[i][j] * grad_k_pos_t[j]).conj();
                }
                for j in 0..d_v.min(grad_v_local.len()) {
                    grad_weights_v_batch[batch_idx][i][j] += (input_batch[batch_idx][t][i] * grad_v_local[j]).conj();
                    grad_input_batch[batch_idx][t][i] += (self.weights_v[i][j] * grad_v_local[j]).conj();
                }
            }

            // Corrected: accumulate grad_bias_pos per sequence positions, NOT over input feature i
            for pos_j in 0..d_k.min(grad_k_pos_t.len()) {
                grad_bias_pos[batch_idx][t][pos_j] += grad_k_pos_t[pos_j].conj();
            }

            if t > 0 {
                accumulated_grad_prefix_phi_k = grad_prefix_phi_k_current;
                accumulated_grad_prefix_phi_kv = grad_prefix_phi_kv_current;
            }
        }
    }

    // Helper method to compute phi function and its derivative
    fn compute_phi_and_derivative(&self, scaled_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<Vec<Vec<Complex<f64>>>>>) {
        let batch_size = scaled_batch.len();
        let seq_len = scaled_batch[0].len();
        let num_features = self.w.len();
        let d_k = scaled_batch[0][0].len();

        let mut phi_batch = vec![vec![vec![Complex::new(0.0, 0.0); num_features]; seq_len]; batch_size];
        let mut dphi_batch = vec![vec![vec![vec![Complex::new(0.0, 0.0); d_k]; num_features]; seq_len]; batch_size];

        for b in 0..batch_size {
            for t in 0..seq_len {
                let x = &scaled_batch[b][t];
                for (r_idx, omega_i) in self.w.iter().enumerate() {
                    // Compute phi
                    phi_batch[b][t][r_idx] = phi(x, omega_i);
                    let phi_val = phi_batch[b][t][r_idx];

                    // Compute derivative: dφ/dx_d = φ(x) * (conj(ω_d) - conj(x_d))
                    for d_idx in 0..d_k {
                        dphi_batch[b][t][r_idx][d_idx] = phi_val * (omega_i[d_idx].conj() - x[d_idx].conj());
                    }
                }
            }
        }

        (phi_batch, dphi_batch)
    }

    // Helper method to backward through phi transformation
    fn backward_through_phi(&self, grad_phi: &[Complex<f64>], _scaled_input: &[Complex<f64>], dphi_dx: &[Vec<Complex<f64>>]) -> Vec<Complex<f64>> {
        let d_model = _scaled_input.len();
        let mut grad_scaled = vec![Complex::new(0.0, 0.0); d_model];

        for (r_idx, &grad_phi_r) in grad_phi.iter().enumerate() {
            if r_idx < dphi_dx.len() {
                for d_idx in 0..d_model {
                    if d_idx < dphi_dx[r_idx].len() {
                        grad_scaled[d_idx] += grad_phi_r * dphi_dx[r_idx][d_idx];
                    }
                }
            }
        }

        grad_scaled
    }

    // Helper method to recompute prefix sums
    fn recompute_prefix_sums(&self, phi_k_batch: &[Vec<Vec<Complex<f64>>>], v_batch: &[Vec<Vec<Complex<f64>>>]) -> (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<Vec<Vec<Complex<f64>>>>>) {
        let batch_size = phi_k_batch.len();
        let seq_len = phi_k_batch[0].len();
        let num_features = phi_k_batch[0][0].len();
        let d_v = v_batch[0][0].len();

        let mut prefix_phi_k = vec![vec![vec![Complex::new(0.0, 0.0); num_features]; seq_len]; batch_size];
        let mut prefix_phi_kv = vec![vec![vec![vec![Complex::new(0.0, 0.0); d_v]; num_features]; seq_len]; batch_size];

        for b in 0..batch_size {
            for t in 0..seq_len {
                if t == 0 {
                    prefix_phi_k[b][t] = phi_k_batch[b][t].clone();

                    for r_idx in 0..num_features {
                        for dv_idx in 0..d_v {
                            prefix_phi_kv[b][t][r_idx][dv_idx] = v_batch[b][t][dv_idx] * phi_k_batch[b][t][r_idx];
                        }
                    }
                } else {
                    for r_idx in 0..num_features {
                        prefix_phi_k[b][t][r_idx] = prefix_phi_k[b][t - 1][r_idx] + phi_k_batch[b][t][r_idx];

                        for dv_idx in 0..d_v {
                            prefix_phi_kv[b][t][r_idx][dv_idx] = prefix_phi_kv[b][t - 1][r_idx][dv_idx] + v_batch[b][t][dv_idx] * phi_k_batch[b][t][r_idx];
                        }
                    }
                }
            }
        }

        (prefix_phi_k, prefix_phi_kv)
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in attention head layer");
        let (grad_w_q, grad_w_v, grad_w_k) = (gradient.get_gradient_weights_q(), gradient.get_gradient_weights_v(), gradient.get_gradient_weights_k());

        let input_batch = gradient.get_gradient_input_batch();
        let grad_bias_pos = gradient.get_gradient_bias_pos();
        let mut batch_size = input_batch.len() as f64;

        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }

        let mut all_gradients = vec![grad_w_q, grad_w_v, grad_w_k, grad_bias_pos];
        let global_norm = compute_global_norm(&all_gradients, &vec![]);
        self.ema = self.smoothing * self.ema + (1.0 - self.smoothing) * global_norm;
        let max_norm = self.ema * EMA_SCALER;
        clip_all_gradients_by_global_norm_2d(&mut all_gradients, &mut vec![], global_norm, max_norm);

        let mut grad_w_q = all_gradients[0].clone();
        let mut grad_w_v = all_gradients[1].clone();
        let mut grad_w_k = all_gradients[2].clone();
        let mut grad_bias_pos = all_gradients[3].clone();

        grad_w_q = average_matrix_by_scalar(&grad_w_q, batch_size);
        grad_w_v = average_matrix_by_scalar(&grad_w_v, batch_size);
        grad_w_k = average_matrix_by_scalar(&grad_w_k, batch_size);
        grad_bias_pos = average_matrix_by_scalar(&grad_bias_pos, batch_size);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        let (mut prev_m_weights_q, mut prev_v_weights_q, mut prev_m_weights_k, mut prev_v_weights_k, mut prev_m_weights_v, mut prev_v_weights_v, mut prev_m_bias_pos, mut prev_v_bias_pos) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_weigths_q(),
                previous_gradient.get_prev_v_weigths_q(),
                previous_gradient.get_prev_m_weigths_k(),
                previous_gradient.get_prev_v_weigths_k(),
                previous_gradient.get_prev_m_weigths_v(),
                previous_gradient.get_prev_v_weigths_v(),
                vec![vec![Complex::new(0.0, 0.0); self.bias_pos[0].len()]; self.bias_pos.len()],
                vec![vec![Complex::new(0.0, 0.0); self.bias_pos[0].len()]; self.bias_pos.len()],
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
            )
        };

        calculate_adam_w(&mut self.weights_q, &grad_w_q, &mut prev_m_weights_q, &mut prev_v_weights_q, learning_rate, time_step);
        calculate_adam_w(&mut self.weights_k, &grad_w_k, &mut prev_m_weights_k, &mut prev_v_weights_k, learning_rate, time_step);
        calculate_adam_w(&mut self.weights_v, &grad_w_v, &mut prev_m_weights_v, &mut prev_v_weights_v, learning_rate, time_step);

        let seq_len = grad_bias_pos.len();
        let mut bias_pos_slice: Vec<Vec<Complex<f64>>> = self.bias_pos[0..seq_len].iter().map(|row| row[0..seq_len].to_vec()).collect();
        calculate_adam_w(&mut bias_pos_slice, &grad_bias_pos, &mut prev_m_bias_pos, &mut prev_v_bias_pos, learning_rate, time_step);

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

        self.previous_gradient = Some(gradient.clone());
    }
}
