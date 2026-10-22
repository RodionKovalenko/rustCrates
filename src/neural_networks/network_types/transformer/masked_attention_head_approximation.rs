use num_complex::ComplexFloat;
use rand::RngExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::layer::LayerType,
    utils::{
        adam_w::calculate_adam_w,
        dtype::{r, Real, C, ONE, ZERO},
        matrix::{average_matrix_by_scalar, clip_all_gradients_by_global_norm_2d, normalize_gradients, RowMajorMatrix},
        weights_initializer::initialize_weights_complex,
    },
};

// Precomputed gradient caches for optimization
#[derive(Clone)]
struct PrecomputedGradients {
    numerator: Vec<Vec<Vec<C>>>,      // [batch][time][dv]
    denominator: Vec<Vec<C>>,         // [batch][time]
    grad_numerator: Vec<Vec<Vec<C>>>, // [batch][time][dv]
    grad_denominator: Vec<Vec<C>>,    // [batch][time]
    grad_phi_q: Vec<Vec<Vec<C>>>,     // [batch][time][r]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedAttentionHeadApproximation {
    pub weights_q: Vec<Vec<C>>,
    pub weights_k: Vec<Vec<C>>,
    pub weights_v: Vec<Vec<C>>,
    pub bias_pos: Vec<Vec<C>>,
    pub bias_q: Vec<C>,
    pub bias_k: Vec<C>,
    pub bias_v: Vec<C>,
    pub layer_type: LayerType,
    pub learning_rate: f64,
    pub smoothing: Real,
    pub ema: Real,
    pub batch_size: usize,
    pub total_valid_tokens: usize,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch: Vec<Vec<Vec<C>>>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub padding_mask: Vec<Vec<u32>>,
    #[serde(skip)]
    pub q_scaled: Vec<Vec<Vec<C>>>,
    #[serde(skip)]
    pub k_scaled: Vec<Vec<Vec<C>>>,
    #[serde(skip)]
    pub v_batch: Vec<Vec<Vec<C>>>,
    #[serde(skip)]
    pub phi_q: Vec<Vec<Vec<C>>>,
    #[serde(skip)]
    pub phi_k: Vec<Vec<Vec<C>>>,
    #[serde(skip)]
    pub prefix_phi_k: Vec<Vec<Vec<C>>>,
    #[serde(skip)]
    pub prefix_phi_kv: Vec<Vec<Vec<Vec<C>>>>,
    pub w: Vec<Vec<C>>,
    pub b: Vec<C>,
    pub m1: Vec<Vec<C>>,
    pub v1: Vec<Vec<C>>,
}

impl MaskedAttentionHeadApproximation {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut rng = rand::rng();
        let mut weights_q = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let mut weights_k = weights_q.clone();
        let mut weights_v = weights_q.clone();
        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);
        let mut bias_pos = vec![vec![C::new(ZERO, ZERO); 5]; 5];
        initialize_weights_complex(5, 5, &mut bias_pos);
        let bias_q = vec![C::new(0.001, ZERO); cols];
        let bias_k = bias_q.clone();
        let bias_v = bias_q.clone();
        let w = (0..cols).map(|_| (0..cols).map(|_| C::new(rng.random::<Real>(), ZERO)).collect()).collect();
        let b = (0..cols).map(|_| C::new(rng.random::<Real>(), ZERO)).collect();

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
            smoothing: r(0.99),
            ema: ZERO,
            batch_size: 0,
            total_valid_tokens: 1,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            input_batch: vec![],
            input_batch_rm: None,
            output_batch: None,
            output_batch_rm: None,
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
            m1: vec![vec![C::new(ZERO, ZERO); cols]; rows],
            v1: vec![vec![C::new(ZERO, ZERO); cols]; rows],
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, learning_rate: f64) -> MaskedAttentionHeadApproximation {
        let mut attention_layer: MaskedAttentionHeadApproximation = MaskedAttentionHeadApproximation::new(rows, cols, learning_rate);
        attention_layer.layer_type = layer_type;

        attention_layer
    }

    fn phi(&self, x: &[C], omega: &[C]) -> C {
        let dot = omega.iter().zip(x).map(|(w, x)| w.conj() * *x).sum::<C>();
        let norm_sq = x.iter().map(|c| c.norm_sqr()).sum::<Real>();
        (dot - C::new(norm_sq / r(2.0), ZERO)).exp()
    }

    /// Forward for Vec and RM input; both run the same row-major implementation.
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_rm_given = layer_input.get_input_batch_rm_ref().filter(|b| !b.is_empty()).map(|b| b.to_vec());
        let is_rm = input_rm_given.is_some();
        let input_rm: Vec<RowMajorMatrix<C>> = input_rm_given.unwrap_or_else(|| layer_input.get_input_batch().iter().map(|m| RowMajorMatrix::from_rows(m)).collect());

        self.padding_mask = layer_input.get_padding_mask_batch();
        self.batch_size = input_rm.len();
        self.total_valid_tokens = layer_input.get_total_valid_tokens();
        self.time_step = layer_input.get_time_step();

        let out_rm = self.forward_rm(&input_rm);

        self.input_batch = if is_rm { vec![] } else { input_rm.iter().map(|m| m.to_rows()).collect() };
        self.input_batch_rm = Some(input_rm);

        let mut lo = LayerOutput::new_default();
        if is_rm {
            self.output_batch = None;
            self.output_batch_rm = Some(out_rm.clone());
            lo.set_output_batch_rm(out_rm);
        } else {
            let out: Vec<Vec<Vec<C>>> = out_rm.iter().map(|m| m.to_rows()).collect();
            self.output_batch_rm = None;
            self.output_batch = Some(out.clone());
            lo.set_output_batch(out);
        }
        lo
    }

    /// Linear-attention forward on row-major input; caches everything needed for backward.
    fn forward_rm(&mut self, input_rm: &[RowMajorMatrix<C>]) -> Vec<RowMajorMatrix<C>> {
        let pad = self.padding_mask.clone();
        let bsz = input_rm.len();
        let seq = input_rm[0].rows;
        let d_model = input_rm[0].cols;
        let d_k = self.weights_k[0].len();

        let mut q = vec![vec![vec![C::new(ZERO, ZERO); d_k]; seq]; bsz];
        let mut k = q.clone();
        let mut v = q.clone();

        for b in 0..bsz {
            assert_eq!(input_rm[b].rows, seq);
            assert_eq!(input_rm[b].cols, d_model);
            for t in 0..seq {
                let in_row = input_rm[b].row_range(t);
                for i in 0..d_model {
                    let x = input_rm[b].data[in_row.start + i];
                    for j in 0..d_k {
                        q[b][t][j] += x * self.weights_q[i][j];
                        k[b][t][j] += x * self.weights_k[i][j];
                        v[b][t][j] += x * self.weights_v[i][j];
                    }
                }
            }
        }

        self.q_scaled = q.clone();
        self.k_scaled = k.clone();
        self.v_batch = v.clone();
        let scale: Real = ONE / r(d_k as f64).sqrt();
        for b in 0..bsz {
            for t in 0..seq {
                for j in 0..d_k {
                    self.q_scaled[b][t][j] = (q[b][t][j] + self.bias_pos[t % self.bias_pos.len()][j % self.bias_pos[0].len()]) * scale;
                    self.k_scaled[b][t][j] = (k[b][t][j] + self.bias_pos[t % self.bias_pos.len()][j % self.bias_pos[0].len()]) * scale;
                }
            }
        }

        self.phi_q = vec![vec![vec![C::new(ZERO, ZERO); self.w.len()]; seq]; bsz];
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
        self.prefix_phi_kv = vec![vec![vec![vec![C::new(ZERO, ZERO); d_k]; self.w.len()]; seq]; bsz];
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

        let mut out_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(bsz);
        for b in 0..bsz {
            let mut out = RowMajorMatrix::from_data(seq, d_k, vec![C::new(ZERO, ZERO); seq * d_k]);
            for t in 0..seq {
                if pad[b][t] == 0 {
                    continue;
                }
                let mut num = vec![C::new(ZERO, ZERO); d_k];
                let mut den = C::new(ZERO, ZERO);
                for r in 0..self.w.len() {
                    for dv in 0..d_k {
                        num[dv] += self.prefix_phi_kv[b][t][r][dv] * self.phi_q[b][t][r];
                    }
                    den += self.prefix_phi_k[b][t][r] * self.phi_q[b][t][r];
                }
                if den.abs() < r(1e-8) {
                    den += C::new(r(1e-8), ZERO);
                }
                let row = out.row_range(t);
                for dv in 0..d_k {
                    out.data[row.start + dv] = num[dv] / den;
                }
            }
            out_rm.push(out);
        }
        out_rm
    }

    pub fn backward_rm(&mut self, grad_output_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("RM input batch missing in attention head approximation backward_rm");
        if grad_output_rm.is_empty() {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let bsz = input_batch_rm.len();
        let seq = input_batch_rm[0].rows;
        let d_model = input_batch_rm[0].cols;
        let d_k = self.weights_k[0].len();
        let scale: Real = ONE / r(d_k as f64).sqrt();

        let pg = self.precompute_gradients(bsz, seq, grad_output_rm);

        let mut grad_in_rm: Vec<RowMajorMatrix<C>> = vec![RowMajorMatrix::from_data(seq, d_model, vec![C::new(ZERO, ZERO); seq * d_model]); bsz];
        let mut grad_wq = vec![vec![vec![C::new(ZERO, ZERO); d_k]; d_model]; bsz];
        let mut grad_wk = grad_wq.clone();
        let mut grad_wv = grad_wq.clone();

        let scratch: Vec<_> = (0..bsz)
            .into_par_iter()
            .map(|b| {
                let mut local_wq = vec![vec![C::new(ZERO, ZERO); d_k]; d_model];
                let mut local_wk = vec![vec![C::new(ZERO, ZERO); d_k]; d_model];
                let mut local_wv = vec![vec![C::new(ZERO, ZERO); d_k]; d_model];
                let mut local_in = RowMajorMatrix::from_data(seq, d_model, vec![C::new(ZERO, ZERO); seq * d_model]);

                // Q path
                for t in 0..seq {
                    if self.padding_mask[b][t] == 0 {
                        continue;
                    }
                    let in_row = input_batch_rm[b].row_range(t);
                    for r in 0..self.w.len() {
                        let phi = self.phi_q[b][t][r];
                        for i in 0..d_k {
                            let dphi = phi * (self.w[r][i].conj() - self.q_scaled[b][t][i]);
                            let gq = pg.grad_phi_q[b][t][r] * dphi * scale;
                            for j in 0..d_model {
                                let x = input_batch_rm[b].data[in_row.start + j];
                                local_wq[j][i] += (gq * x).conj();
                                let idx = local_in.idx(t, j);
                                local_in.data[idx] += (gq * self.weights_q[j][i]).conj();
                            }
                        }
                    }
                    // V path
                    for tau in 0..=t {
                        let in_row_tau = input_batch_rm[b].row_range(tau);
                        for dv in 0..d_k {
                            let mut gv = C::new(ZERO, ZERO);
                            for r in 0..self.w.len() {
                                gv += pg.grad_numerator[b][t][dv] * self.phi_q[b][t][r] * self.phi_k[b][tau][r];
                            }
                            for j in 0..d_model {
                                let x = input_batch_rm[b].data[in_row_tau.start + j];
                                local_wv[j][dv] += (gv * x).conj();
                                let idx = local_in.idx(tau, j);
                                local_in.data[idx] += (gv * self.weights_v[j][dv]).conj();
                            }
                        }
                    }
                }

                // K path
                for tau in 0..seq {
                    let in_row_tau = input_batch_rm[b].row_range(tau);
                    for r in 0..self.w.len() {
                        let phi = self.phi_k[b][tau][r];
                        for i in 0..d_k {
                            let dphi = phi * (self.w[r][i].conj() - self.k_scaled[b][tau][i]);
                            let mut gk = C::new(ZERO, ZERO);
                            for t in tau..seq {
                                if self.padding_mask[b][t] == 0 {
                                    continue;
                                }
                                let mut gp = C::new(ZERO, ZERO);
                                for dv in 0..d_k {
                                    gp += pg.grad_numerator[b][t][dv] * self.phi_q[b][t][r] * self.v_batch[b][tau][dv];
                                }
                                gp += pg.grad_denominator[b][t] * self.phi_q[b][t][r];
                                gk += gp * dphi * scale;
                            }
                            for j in 0..d_model {
                                let x = input_batch_rm[b].data[in_row_tau.start + j];
                                local_wk[j][i] += (gk * x).conj();
                                let idx = local_in.idx(tau, j);
                                local_in.data[idx] += (gk * self.weights_k[j][i]).conj();
                            }
                        }
                    }
                }

                (local_wq, local_wk, local_wv, local_in)
            })
            .collect();

        for (b, (lwq, lwk, lwv, lin)) in scratch.into_iter().enumerate() {
            for j in 0..d_model {
                for i in 0..d_k {
                    grad_wq[b][j][i] += lwq[j][i];
                    grad_wk[b][j][i] += lwk[j][i];
                    grad_wv[b][j][i] += lwv[j][i];
                }
            }
            for idx in 0..lin.data.len() {
                grad_in_rm[b].data[idx] += lin.data[idx];
            }
        }

        let mut grad = Gradient::new_default();
        grad.set_gradient_weights_q_batch(grad_wq);
        grad.set_gradient_weights_k_batch(grad_wk);
        grad.set_gradient_weights_v_batch(grad_wv);
        grad.set_gradient_input_batch_rm(grad_in_rm);
        self.gradient = Some(grad.clone());
        grad
    }

    /// Precompute numerator, denominator, and local gradient components for all (b,t)
    fn precompute_gradients(&self, batch_size: usize, seq_len: usize, grad_output: &[RowMajorMatrix<C>]) -> PrecomputedGradients {
        let d_v = self.weights_v[0].len();
        let mut pg = PrecomputedGradients {
            numerator: vec![vec![vec![C::new(ZERO, ZERO); d_v]; seq_len]; batch_size],
            denominator: vec![vec![C::new(ZERO, ZERO); seq_len]; batch_size],
            grad_numerator: vec![vec![vec![C::new(ZERO, ZERO); d_v]; seq_len]; batch_size],
            grad_denominator: vec![vec![C::new(ZERO, ZERO); seq_len]; batch_size],
            grad_phi_q: vec![vec![vec![C::new(ZERO, ZERO); self.w.len()]; seq_len]; batch_size],
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
                if pg.denominator[b][t].abs() < r(1e-8) {
                    pg.denominator[b][t] += C::new(r(1e-8), ZERO);
                }
                // Compute grad_numerator & grad_denominator
                for dv in 0..d_v {
                    let go = grad_output[b].data[grad_output[b].idx(t, dv)];
                    pg.grad_numerator[b][t][dv] = go / pg.denominator[b][t];
                    pg.grad_denominator[b][t] += -go * pg.numerator[b][t][dv] / (pg.denominator[b][t] * pg.denominator[b][t]);
                }
                // Compute grad_phi_q
                for r in 0..self.w.len() {
                    let mut sum = C::new(ZERO, ZERO);
                    for dv in 0..d_v {
                        sum += pg.grad_numerator[b][t][dv] * self.prefix_phi_kv[b][t][r][dv];
                    }
                    pg.grad_phi_q[b][t][r] = sum + pg.grad_denominator[b][t] * self.prefix_phi_k[b][t][r];
                }
            }
        }
        pg
    }

    /// Vec counterpart of `backward_rm`.
    pub fn backward(&mut self, grad_output: &Vec<Vec<Vec<C>>>) -> Gradient {
        let grad_output_rm: Vec<RowMajorMatrix<C>> = grad_output.iter().map(|m| RowMajorMatrix::from_rows(m)).collect();
        let mut grad = self.backward_rm(&grad_output_rm);
        let grad_in: Vec<Vec<Vec<C>>> = grad.get_gradient_input_batch_rm().iter().map(|m| m.to_rows()).collect();
        grad.set_gradient_input_batch(grad_in);
        grad.clear_gradient_input_batch_rm();
        self.gradient = Some(grad.clone());
        grad
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in attention head layer");
        let (mut grad_w_q, mut grad_w_v, mut grad_w_k) = (gradient.get_gradient_weights_q(), gradient.get_gradient_weights_v(), gradient.get_gradient_weights_k());

        let total_valid_tokens: Real = r(self.total_valid_tokens.max(1) as f64);

        grad_w_q = average_matrix_by_scalar(&grad_w_q, total_valid_tokens);
        grad_w_v = average_matrix_by_scalar(&grad_w_v, total_valid_tokens);
        grad_w_k = average_matrix_by_scalar(&grad_w_k, total_valid_tokens);

        clip_all_gradients_by_global_norm_2d(&mut grad_w_q, &mut vec![], self.global_norm, self.max_norm);

        normalize_gradients(&mut grad_w_q);
        normalize_gradients(&mut grad_w_v);
        normalize_gradients(&mut grad_w_k);

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
