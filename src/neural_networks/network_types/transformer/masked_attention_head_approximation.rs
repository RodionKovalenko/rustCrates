use num::Complex;
use num_complex::ComplexFloat;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer::LayerType, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    utils::{
        adam_w::calculate_adam_w,
        matrix::{average_matrix_by_scalar, clip_all_gradients_by_global_norm_2d},
        weights_initializer::initialize_weights_complex,
    },
};

// Precomputed gradient caches for optimization
#[derive(Clone)]
struct PrecomputedGradients {
    numerator: Vec<Vec<Vec<Complex<f64>>>>,      // [batch][time][dv]
    denominator: Vec<Vec<Complex<f64>>>,         // [batch][time]
    grad_numerator: Vec<Vec<Vec<Complex<f64>>>>, // [batch][time][dv]
    grad_denominator: Vec<Vec<Complex<f64>>>,    // [batch][time]
    grad_phi_q: Vec<Vec<Vec<Complex<f64>>>>,     // [batch][time][r]
}

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
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch: Vec<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub padding_mask: Vec<Vec<u32>>,
    #[serde(skip)]
    pub q_scaled: Vec<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub k_scaled: Vec<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub v_batch: Vec<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub phi_q: Vec<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub phi_k: Vec<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub prefix_phi_k: Vec<Vec<Vec<Complex<f64>>>>,
    #[serde(skip)]
    pub prefix_phi_kv: Vec<Vec<Vec<Vec<Complex<f64>>>>>,
    pub w: Vec<Vec<Complex<f64>>>,
    pub b: Vec<Complex<f64>>,
    pub m1: Vec<Vec<Complex<f64>>>,
    pub v1: Vec<Vec<Complex<f64>>>,
}

impl MaskedAttentionHeadApproximation {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut rng = rand::rng();
        let mut weights_q = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_k = weights_q.clone();
        let mut weights_v = weights_q.clone();
        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);
        let mut bias_pos = vec![vec![Complex::new(0.0, 0.0); 5]; 5];
        initialize_weights_complex(5, 5, &mut bias_pos);
        let bias_q = vec![Complex::new(1.0, 0.0); cols];
        let bias_k = bias_q.clone();
        let bias_v = bias_q.clone();
        let w = (0..cols).map(|_| (0..cols).map(|_| Complex::new(rng.random(), 0.0)).collect()).collect();
        let b = (0..cols).map(|_| Complex::new(rng.random(), 0.0)).collect();

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
            smoothing: 0.99,
            ema: 0.0,
            batch_size: 0,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            input_batch: vec![],
            output_batch: None,
            padding_mask: vec![],
            q_scaled: vec![],
            k_scaled: vec![],
            v_batch: vec![],
            phi_q: vec![],
            phi_k: vec![],
            prefix_phi_k: vec![],
            prefix_phi_kv: vec![],
            w,
            b,
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, learning_rate: f64) -> MaskedAttentionHeadApproximation {
        let mut attention_layer: MaskedAttentionHeadApproximation = MaskedAttentionHeadApproximation::new(rows, cols, learning_rate);
        attention_layer.layer_type = layer_type;

        attention_layer
    }

    fn phi(&self, x: &[Complex<f64>], omega: &[Complex<f64>]) -> Complex<f64> {
        let dot = omega.iter().zip(x).map(|(w, x)| w.conj() * x).sum::<Complex<f64>>();
        let norm_sq = x.iter().map(|c| c.norm_sqr()).sum::<f64>();
        (dot - Complex::new(norm_sq / 2.0, 0.0)).exp()
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input = layer_input.get_input_batch();
        let pad = layer_input.get_padding_mask_batch();
        let bsz = input.len();
        let seq = input[0].len();
        let d_model = input[0][0].len();
        let d_k = self.weights_k[0].len();
        self.input_batch = input.clone();
        self.padding_mask = pad.clone();
        self.batch_size = bsz;
        self.time_step = layer_input.get_time_step();
        let mut q = vec![vec![vec![Complex::new(0.0, 0.0); d_k]; seq]; bsz];
        let mut k = q.clone();
        let mut v = q.clone();
        for b in 0..bsz {
            for t in 0..seq {
                for i in 0..d_model {
                    for j in 0..d_k {
                        q[b][t][j] += input[b][t][i] * self.weights_q[i][j];
                        k[b][t][j] += input[b][t][i] * self.weights_k[i][j];
                        v[b][t][j] += input[b][t][i] * self.weights_v[i][j];
                    }
                }
            }
        }
        self.q_scaled = q.clone();
        self.k_scaled = k.clone();
        self.v_batch = v.clone();
        let scale = 1.0 / (d_k as f64).sqrt();
        for b in 0..bsz {
            for t in 0..seq {
                for j in 0..d_k {
                    self.q_scaled[b][t][j] = (q[b][t][j] + self.bias_pos[t][j]) * scale;
                    self.k_scaled[b][t][j] = (k[b][t][j] + self.bias_pos[t][j]) * scale;
                }
            }
        }
        self.phi_q = vec![vec![vec![Complex::new(0.0, 0.0); self.w.len()]; seq]; bsz];
        self.phi_k = self.phi_q.clone();
        for b in 0..bsz {
            for t in 0..seq {
                for r in 0..self.w.len() {
                    self.phi_q[b][t][r] = self.phi(&self.q_scaled[b][t], &self.w[r]);
                    self.phi_k[b][t][r] = self.phi(&self.k_scaled[b][t], &self.w[r]);
                }
            }
        }
        self.prefix_phi_k = self.phi_k.clone();
        self.prefix_phi_kv = vec![vec![vec![vec![Complex::new(0.0, 0.0); d_k]; self.w.len()]; seq]; bsz];
        for b in 0..bsz {
            for t in 0..seq {
                if t == 0 {
                    for r in 0..self.w.len() {
                        for dv in 0..d_k {
                            self.prefix_phi_kv[b][t][r][dv] = v[b][t][dv] * self.phi_k[b][t][r];
                        }
                    }
                } else {
                    for r in 0..self.w.len() {
                        self.prefix_phi_k[b][t][r] = self.prefix_phi_k[b][t - 1][r] + self.phi_k[b][t][r];
                        for dv in 0..d_k {
                            self.prefix_phi_kv[b][t][r][dv] = self.prefix_phi_kv[b][t - 1][r][dv] + v[b][t][dv] * self.phi_k[b][t][r];
                        }
                    }
                }
            }
        }
        let mut out = vec![vec![vec![Complex::new(0.0, 0.0); d_k]; seq]; bsz];
        for b in 0..bsz {
            for t in 0..seq {
                if pad[b][t] == 0 {
                    continue;
                }
                let mut num = vec![Complex::new(0.0, 0.0); d_k];
                let mut den = Complex::new(0.0, 0.0);
                for r in 0..self.w.len() {
                    for dv in 0..d_k {
                        num[dv] += self.prefix_phi_kv[b][t][r][dv] * self.phi_q[b][t][r];
                    }
                    den += self.prefix_phi_k[b][t][r] * self.phi_q[b][t][r];
                }
                if den.abs() < 1e-8 {
                    den += Complex::new(1e-8, 0.0)
                }
                for dv in 0..d_k {
                    out[b][t][dv] = num[dv] / den;
                }
            }
        }
        self.output_batch = Some(out.clone());
        let mut lo = LayerOutput::new_default();
        lo.set_output_batch(out);
        lo
    }

    /// Precompute numerator, denominator, and local gradient components for all (b,t)
    fn precompute_gradients(&self, batch_size: usize, seq_len: usize, grad_output: &Vec<Vec<Vec<Complex<f64>>>>) -> PrecomputedGradients {
        let d_v = self.weights_v[0].len();
        let mut pg = PrecomputedGradients {
            numerator: vec![vec![vec![Complex::new(0.0, 0.0); d_v]; seq_len]; batch_size],
            denominator: vec![vec![Complex::new(0.0, 0.0); seq_len]; batch_size],
            grad_numerator: vec![vec![vec![Complex::new(0.0, 0.0); d_v]; seq_len]; batch_size],
            grad_denominator: vec![vec![Complex::new(0.0, 0.0); seq_len]; batch_size],
            grad_phi_q: vec![vec![vec![Complex::new(0.0, 0.0); self.w.len()]; seq_len]; batch_size],
        };

        for b in 0..batch_size {
            for t in 0..seq_len {
                if self.padding_mask[b][t] == 0 {
                    continue;
                }
                // Compute numerator & denominator once
                for r in 0..self.w.len() {
                    let phi_q_rt = self.phi_q[b][t][r];
                    for dv in 0..d_v {
                        pg.numerator[b][t][dv] += self.prefix_phi_kv[b][t][r][dv] * phi_q_rt;
                    }
                    pg.denominator[b][t] += self.prefix_phi_k[b][t][r] * phi_q_rt;
                }
                if pg.denominator[b][t].abs() < 1e-8 {
                    pg.denominator[b][t] += Complex::new(1e-8, 0.0);
                }
                // Compute grad_numerator & grad_denominator
                for dv in 0..d_v {
                    let go = grad_output[b][t][dv];
                    pg.grad_numerator[b][t][dv] = go / pg.denominator[b][t];
                    pg.grad_denominator[b][t] += -go * pg.numerator[b][t][dv] / (pg.denominator[b][t] * pg.denominator[b][t]);
                }
                // Compute grad_phi_q
                for r in 0..self.w.len() {
                    let mut sum = Complex::new(0.0, 0.0);
                    for dv in 0..d_v {
                        sum += pg.grad_numerator[b][t][dv] * self.prefix_phi_kv[b][t][r][dv];
                    }
                    pg.grad_phi_q[b][t][r] = sum + pg.grad_denominator[b][t] * self.prefix_phi_k[b][t][r];
                }
            }
        }
        pg
    }

    pub fn backward(&mut self, grad_output: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let bsz = self.input_batch.len();
        let seq = self.input_batch[0].len();
        let d_model = self.input_batch[0][0].len();
        let d_k = self.weights_k[0].len();
        let scale = 1.0 / (d_k as f64).sqrt();

        // Precompute shared gradient terms
        let pg = self.precompute_gradients(bsz, seq, grad_output);

        // Global accumulators
        let mut grad_in = vec![vec![vec![Complex::new(0.0, 0.0); d_model]; seq]; bsz];
        let mut grad_wq = vec![vec![vec![Complex::new(0.0, 0.0); d_k]; d_model]; bsz];
        let mut grad_wk = grad_wq.clone();
        let mut grad_wv = grad_wq.clone();

        // Per-batch scratch buffers in parallel
        let scratch: Vec<_> = (0..bsz)
            .into_par_iter()
            .map(|b| {
                let mut local_wq = vec![vec![Complex::new(0.0, 0.0); d_k]; d_model];
                let mut local_wk = vec![vec![Complex::new(0.0, 0.0); d_k]; d_model];
                let mut local_wv = vec![vec![Complex::new(0.0, 0.0); d_k]; d_model];
                let mut local_in = vec![vec![Complex::new(0.0, 0.0); d_model]; seq];

                // Q path
                for t in 0..seq {
                    if self.padding_mask[b][t] == 0 {
                        continue;
                    }
                    for r in 0..self.w.len() {
                        let phi = self.phi_q[b][t][r];
                        for i in 0..d_k {
                            let dphi = phi * (self.w[r][i].conj() - self.q_scaled[b][t][i]);
                            let gq = pg.grad_phi_q[b][t][r] * dphi * scale;
                            for j in 0..d_model {
                                local_wq[j][i] += (gq * self.input_batch[b][t][j]).conj();
                                local_in[t][j] += (gq * self.weights_q[j][i]).conj();
                            }
                        }
                    }
                    // V path
                    for tau in 0..=t {
                        for dv in 0..d_k {
                            let mut gv = Complex::new(0.0, 0.0);
                            for r in 0..self.w.len() {
                                gv += pg.grad_numerator[b][t][dv] * self.phi_q[b][t][r] * self.phi_k[b][tau][r];
                            }
                            for j in 0..d_model {
                                local_wv[j][dv] += (gv * self.input_batch[b][tau][j]).conj();
                                local_in[tau][j] += (gv * self.weights_v[j][dv]).conj();
                            }
                        }
                    }
                }

                // K path
                for tau in 0..seq {
                    for r in 0..self.w.len() {
                        let phi = self.phi_k[b][tau][r];
                        for i in 0..d_k {
                            let dphi = phi * (self.w[r][i].conj() - self.k_scaled[b][tau][i]);
                            let mut gk = Complex::new(0.0, 0.0);
                            for t in tau..seq {
                                if self.padding_mask[b][t] == 0 {
                                    continue;
                                }
                                let mut gp = Complex::new(0.0, 0.0);
                                for dv in 0..d_k {
                                    gp += pg.grad_numerator[b][t][dv] * self.phi_q[b][t][r] * self.v_batch[b][tau][dv];
                                }
                                gp += pg.grad_denominator[b][t] * self.phi_q[b][t][r];
                                gk += gp * dphi * scale;
                            }
                            for j in 0..d_model {
                                local_wk[j][i] += (gk * self.input_batch[b][tau][j]).conj();
                                local_in[tau][j] += (gk * self.weights_k[j][i]).conj();
                            }
                        }
                    }
                }

                (local_wq, local_wk, local_wv, local_in)
            })
            .collect();

        // Merge scratch into global
        for (b, (lwq, lwk, lwv, lin)) in scratch.into_iter().enumerate() {
            for j in 0..d_model {
                for i in 0..d_k {
                    grad_wq[b][j][i] += lwq[j][i];
                    grad_wk[b][j][i] += lwk[j][i];
                    grad_wv[b][j][i] += lwv[j][i];
                }
            }
            for t in 0..seq {
                for j in 0..d_model {
                    grad_in[b][t][j] += lin[t][j];
                }
            }
        }

        // Build and return gradient struct
        let mut grad = Gradient::new_default();
        grad.set_gradient_weights_q_batch(grad_wq);
        grad.set_gradient_weights_k_batch(grad_wk);
        grad.set_gradient_weights_v_batch(grad_wv);
        grad.set_gradient_input_batch(grad_in);

        self.gradient = Some(grad.clone());

        grad
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in attention head layer");
        let (mut grad_w_q, mut grad_w_v, mut grad_w_k) = (gradient.get_gradient_weights_q(), gradient.get_gradient_weights_v(), gradient.get_gradient_weights_k());
        let input_batch = gradient.get_gradient_input_batch();
        let mut batch_size = input_batch.len() as f64;

        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }

        grad_w_q = average_matrix_by_scalar(&grad_w_q, batch_size);
        grad_w_v = average_matrix_by_scalar(&grad_w_v, batch_size);
        grad_w_k = average_matrix_by_scalar(&grad_w_k, batch_size);

        clip_all_gradients_by_global_norm_2d(&mut grad_w_q, &mut vec![], self.global_norm, self.max_norm);

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
