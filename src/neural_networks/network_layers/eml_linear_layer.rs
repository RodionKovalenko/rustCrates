//! Hierarchical EML output head — first cut (see `documentation/eml.txt`).
//!
//! A two-level coordinate-matching language-model head that replaces the dense
//! `[d x V]` projection with an `O(sqrt(V))` cluster->word scheme. The score is the
//! numerically stable surrogate of the EML operator `eml(u, v) = exp(u) - ln(v)`:
//!
//! ```text
//!   S = alpha * U  -  beta * log( softplus(V) + eps )
//! ```
//!
//! Scope of THIS cut (deliberately tight, per the design discussion):
//!   * Option C coordinates — fixed orthogonal cluster coords (Modified Gram-Schmidt),
//!     random row-normalized word coords. No pretrained / spectral init (Options A/D).
//!   * Real-valued: the transformer hidden state arrives as complex `C`; we read `.re`
//!     and propagate a real gradient back (imaginary part stays zero).
//!   * Teacher-forced training path ONLY. The target token `v*` selects its cluster
//!     `c* = v*/M` and word `w* = v* % M`; the word softmax is evaluated against the
//!     gathered slab of cluster `c*` alone. No inference / top-k routing yet.
//!   * Correctness is validated by a finite-difference gradient check
//!     (`tests/test_eml_linear_layer.rs`), mirroring the dense `LinearLayer` check.
//!
//! The trainable parameters are the four projection matrices (`W_uc, W_vc, W_uw, W_vw`,
//! each `[d_model x d_coord]`) and the four per-level temperatures
//! (`alpha_c, beta_c, alpha_w, beta_w`). The coordinate matrices are frozen after init.

use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::default_layer::LayerInterface,
    utils::dtype::{r, Real, C, ONE, ZERO},
};

/// Numerical floor inside `log(softplus(V) + eps)` so the log never sees zero.
const EPS: f64 = 1e-6;
/// Number of top clusters whose words are scored during inference (greedy hierarchical
/// decode). The joint argmax may live just outside the single best cluster, so we widen
/// the beam slightly; the cost is `CLUSTERS_SEARCHED * M` extra word scores per token.
const CLUSTERS_SEARCHED: usize = 8;
/// AdamW hyper-parameters for the trainable projections and temperatures.
const ADAM_B1: f64 = 0.9;
const ADAM_B2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;
/// Fixed scale applied to the (bounded) cosine EML score before the softmax — the standard
/// cosine-classifier / ArcFace recipe. Both queries are L2-normalized, so each similarity
/// lives in `[-1, 1]`; multiplying by `LOGIT_SCALE` makes the softmax sharp enough to drive
/// CE to ~0 (it must beat `ln(K) ≈ 5.4`) while the logits stay **bounded**. Bounded logits
/// are what kills the loss spikes: an unnormalized query lets the logits grow without bound,
/// carving cliffs into the loss surface where a single AdamW step flips the argmax on many
/// tokens at once. With `s·cos` the surface is smooth, so the minimum is sharp but stable.
/// 16 is the empirical sweet spot here: large enough to drive CE low, small enough that the
/// minimum stays smooth. Raising it to 24 measurably re-introduced the loss spikes (a sharper
/// minimum is more cliff-prone), so it is deliberately kept at 16.
const LOGIT_SCALE: f64 = 16.0;
/// Global L2-norm cap applied to the (per-token-averaged) gradient before the AdamW step.
/// Cheap insurance against acute gradient transients.
const MAX_GRAD_NORM: f64 = 1.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmlLinearLayer {
    // ── dimensions ───────────────────────────────────────────────────────────
    pub d_model: usize,
    pub d_coord: usize,
    pub k_clusters: usize,
    pub m_words: usize,
    pub vocab_size: usize,

    // ── trainable projections [d_model x d_coord] ────────────────────────────
    pub w_uc: Vec<Vec<Real>>,
    pub w_vc: Vec<Vec<Real>>,
    pub w_uw: Vec<Vec<Real>>,
    pub w_vw: Vec<Vec<Real>>,

    // ── trainable per-level temperatures (scalars) ───────────────────────────
    pub alpha_c: Real,
    pub beta_c: Real,
    pub alpha_w: Real,
    pub beta_w: Real,

    // ── frozen coordinates ───────────────────────────────────────────────────
    /// Cluster coords, one unit row per cluster: `[K x d_coord]`.
    pub c_u: Vec<Vec<Real>>,
    pub c_v: Vec<Vec<Real>>,
    /// Word coords, one unit row per slot: `[K x M x d_coord]`.
    pub w_u: Vec<Vec<Vec<Real>>>,
    pub w_v: Vec<Vec<Vec<Real>>>,
    /// `valid[k][m]` is false for padding slots (`k*M + m >= V`).
    pub valid: Vec<Vec<bool>>,

    pub learning_rate: f64,
    pub eps: Real,

    // ── AdamW optimizer state (lazily sized; skipped in serialization) ────────
    #[serde(skip)]
    opt: Option<AdamState>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,

    // ── per-forward caches (training) ────────────────────────────────────────
    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    /// Analytical gradients of the trainable params from the last training forward.
    #[serde(skip)]
    grads: Option<ParamGrads>,
}

/// Accumulated analytical gradients for the trainable parameters.
#[derive(Debug, Clone)]
struct ParamGrads {
    w_uc: Vec<Vec<Real>>,
    w_vc: Vec<Vec<Real>>,
    w_uw: Vec<Vec<Real>>,
    w_vw: Vec<Vec<Real>>,
    alpha_c: Real,
    beta_c: Real,
    alpha_w: Real,
    beta_w: Real,
}

impl ParamGrads {
    /// Multiply every gradient (matrices and scalars) by `f` in place.
    fn scale(&mut self, f: Real) {
        for mat in [&mut self.w_uc, &mut self.w_vc, &mut self.w_uw, &mut self.w_vw] {
            for row in mat.iter_mut() {
                for g in row.iter_mut() {
                    *g *= f;
                }
            }
        }
        self.alpha_c *= f;
        self.beta_c *= f;
        self.alpha_w *= f;
        self.beta_w *= f;
    }

    /// L2 norm over all gradient entries (the four matrices and four scalars), in f64.
    fn global_norm(&self) -> f64 {
        let mut sum = 0.0_f64;
        for mat in [&self.w_uc, &self.w_vc, &self.w_uw, &self.w_vw] {
            for row in mat {
                for &g in row {
                    sum += (g as f64) * (g as f64);
                }
            }
        }
        for s in [self.alpha_c, self.beta_c, self.alpha_w, self.beta_w] {
            sum += (s as f64) * (s as f64);
        }
        sum.sqrt()
    }
}

/// First/second moment buffers for AdamW over the four matrices and four scalars.
#[derive(Debug, Clone)]
struct AdamState {
    m_w: [Vec<Vec<Real>>; 4],
    v_w: [Vec<Vec<Real>>; 4],
    m_s: [Real; 4],
    v_s: [Real; 4],
}

impl EmlLinearLayer {
    /// Build a head for `vocab_size` tokens over a `d_model`-wide backbone.
    ///
    /// `M = ceil(sqrt(V))`, `K = ceil(V / M)`, and `d_coord = K` so the cluster
    /// coordinate matrix is square and can be made exactly orthogonal (Option C).
    pub fn new(learning_rate: f64, d_model: usize, vocab_size: usize) -> Self {
        let vocab = vocab_size.max(1);
        let m_words = (vocab as f64).sqrt().ceil() as usize;
        let m_words = m_words.max(1);
        let k_clusters = vocab.div_ceil(m_words).max(1);
        let d_coord = k_clusters; // square -> exact orthogonal cluster coords

        let mut rng = SplitMix64::new(0x5eed_1234_abcd_ef01);

        // Trainable projections: small Gaussian scaled by 1/sqrt(d_model).
        let scale = 1.0 / (d_model.max(1) as f64).sqrt();
        let w_uc = random_matrix(d_model, d_coord, scale, &mut rng);
        let w_vc = random_matrix(d_model, d_coord, scale, &mut rng);
        let w_uw = random_matrix(d_model, d_coord, scale, &mut rng);
        let w_vw = random_matrix(d_model, d_coord, scale, &mut rng);

        // Frozen cluster coords: two independent orthogonal K x d_coord matrices.
        let c_u = orthogonal_matrix(k_clusters, d_coord, &mut rng);
        let c_v = orthogonal_matrix(k_clusters, d_coord, &mut rng);

        // Frozen word coords: random unit rows, one slab per cluster.
        let mut w_u = Vec::with_capacity(k_clusters);
        let mut w_v = Vec::with_capacity(k_clusters);
        for _ in 0..k_clusters {
            w_u.push(unit_rows(m_words, d_coord, &mut rng));
            w_v.push(unit_rows(m_words, d_coord, &mut rng));
        }

        // Validity mask: slot k*M + m is real iff its global index < V.
        let mut valid = vec![vec![false; m_words]; k_clusters];
        for k in 0..k_clusters {
            for m in 0..m_words {
                valid[k][m] = k * m_words + m < vocab;
            }
        }

        Self {
            d_model,
            d_coord,
            k_clusters,
            m_words,
            vocab_size: vocab,
            w_uc,
            w_vc,
            w_uw,
            w_vw,
            alpha_c: r(1.0),
            beta_c: r(1.0),
            alpha_w: r(1.0),
            beta_w: r(1.0),
            c_u,
            c_v,
            w_u,
            w_v,
            valid,
            learning_rate,
            eps: r(EPS),
            opt: None,
            time_step: 0,
            batch_size: 0,
            input_batch: None,
            gradient: None,
            grads: None,
        }
    }

    /// Forward pass.
    ///
    /// In training mode (`target_batch_ids` set and `calculate_gradient` true) this
    /// also computes the teacher-forced cross-entropy loss and the full analytical
    /// gradient (w.r.t. the hidden input and every trainable parameter), stashing them
    /// so the backward pass can reuse them — the same contract `AdaptiveLinearLayer` uses.
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = input.get_input_batch();
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();
        self.input_batch = Some(input_batch.clone());

        let target_tokens = input.get_target_batch_ids();
        let padding_mask_batch = input.get_padding_mask_batch();
        let total_valid_tokens = input.get_total_valid_tokens();
        let is_training = !target_tokens.is_empty() && input.get_calculate_gradient();

        self.gradient = None;
        self.grads = None;

        let batch_len = input_batch.len();
        let seq_len = if batch_len > 0 { input_batch[0].len() } else { 0 };
        let d_model = self.d_model;

        let zero_c = C::new(ZERO, ZERO);
        let mut ce_loss_batch = vec![vec![vec![zero_c]; seq_len]; batch_len];
        let mut grad_input = vec![vec![vec![zero_c; d_model]; seq_len]; batch_len];
        let mut grads = self.zero_param_grads();

        if is_training {
            for batch_idx in 0..batch_len {
                let offset = Self::target_offset(batch_idx, &target_tokens, &padding_mask_batch);
                let input_seq = &input_batch[batch_idx];

                for (row_idx, input_row) in input_seq.iter().enumerate() {
                    let target_id = Self::target_id(row_idx, batch_idx, offset, &target_tokens, &padding_mask_batch);
                    let Some(v_star) = target_id else { continue };
                    if v_star >= self.vocab_size {
                        continue;
                    }

                    // Hidden state h = Re(input_row).
                    let h: Vec<Real> = input_row.iter().map(|c| c.re).collect();

                    let (loss, dh) = self.token_forward_backward(&h, v_star, &mut grads);

                    ce_loss_batch[batch_idx][row_idx][0] = C::new(loss, ZERO);
                    for d in 0..d_model.min(dh.len()) {
                        grad_input[batch_idx][row_idx][d] = C::new(dh[d], ZERO);
                    }
                }
            }
        }

        let mut layer_output = LayerOutput::new_default();

        if is_training {
            // Dense logits are not materialized on the training path (teacher-forced only).
            layer_output.set_output_batch(vec![]);
            layer_output.set_cross_entropy_loss_batch(ce_loss_batch);

            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(grad_input);
            gradient.set_total_valid_tokens(total_valid_tokens);
            self.gradient = Some(gradient);
            self.grads = Some(grads);
        } else {
            // Inference: emit sparse top-k token scores per position in the same
            // `(values, indices)` layout the greedy decoder expects from the other heads.
            let top_k = input.get_top_k_size().max(1);
            let mut values_batch: Vec<Vec<Vec<C>>> = Vec::with_capacity(batch_len);
            let mut indices_batch: Vec<Vec<Vec<usize>>> = Vec::with_capacity(batch_len);

            for input_seq in input_batch.iter() {
                let mut seq_values: Vec<Vec<C>> = Vec::with_capacity(input_seq.len());
                let mut seq_indices: Vec<Vec<usize>> = Vec::with_capacity(input_seq.len());

                for input_row in input_seq.iter() {
                    let h: Vec<Real> = input_row.iter().map(|c| c.re).collect();
                    let cands = self.predict_token(&h, top_k);
                    seq_values.push(cands.iter().map(|&(_, s)| C::new(s, ZERO)).collect());
                    seq_indices.push(cands.iter().map(|&(id, _)| id).collect());
                }

                values_batch.push(seq_values);
                indices_batch.push(seq_indices);
            }

            layer_output.set_output_batch(values_batch);
            layer_output.set_output_indices(indices_batch);
        }

        layer_output
    }

    /// Inference-only greedy hierarchical decode for one hidden state.
    ///
    /// Scores all clusters, keeps the `CLUSTERS_SEARCHED` most probable, then scores their
    /// valid words and ranks tokens by the joint log-prob `log P(c) + log P(w|c)`. Returns
    /// up to `top_k` `(global_token_id, score)` pairs, highest score first. Uses the exact
    /// same scoring function as training so the train/predict argmax cannot disagree.
    fn predict_token(&self, h: &[Real], top_k: usize) -> Vec<(usize, Real)> {
        // Cluster level: all K clusters are real, so no mask.
        let sc = self.level_scores(h, &self.w_uc, &self.w_vc, &self.c_u, &self.c_v, self.alpha_c, self.beta_c);
        let log_pc = log_softmax(&sc, None);

        // Pick the top clusters by log P(c).
        let mut cluster_order: Vec<usize> = (0..self.k_clusters).collect();
        cluster_order.sort_by(|&a, &b| log_pc[b].partial_cmp(&log_pc[a]).unwrap_or(std::cmp::Ordering::Equal));
        cluster_order.truncate(CLUSTERS_SEARCHED.min(self.k_clusters));

        let mut cands: Vec<(usize, Real)> = Vec::new();
        for &c in &cluster_order {
            let sw = self.level_scores(h, &self.w_uw, &self.w_vw, &self.w_u[c], &self.w_v[c], self.alpha_w, self.beta_w);
            let log_pw = log_softmax(&sw, Some(&self.valid[c]));
            for m in 0..self.m_words {
                if self.valid[c][m] {
                    cands.push((c * self.m_words + m, log_pc[c] + log_pw[m]));
                }
            }
        }

        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(top_k.max(1));
        cands
    }

    /// EML scores `S[k] = alpha*U[k] - beta*log(softplus(V[k]) + eps)` for every coordinate
    /// row at one level. Mirrors the forward half of `level_forward_backward` exactly.
    #[allow(clippy::too_many_arguments)]
    fn level_scores(
        &self,
        h: &[Real],
        w_u_proj: &[Vec<Real>],
        w_v_proj: &[Vec<Real>],
        coord_u: &[Vec<Real>],
        coord_v: &[Vec<Real>],
        alpha: Real,
        beta: Real,
    ) -> Vec<Real> {
        let n = coord_u.len();
        let qu = project(h, w_u_proj, self.d_coord);
        let qv = project(h, w_v_proj, self.d_coord);
        let (qu_hat, _) = l2_normalize(&qu);
        let (qv_hat, _) = l2_normalize(&qv);
        let mut s = vec![ZERO; n];
        for k in 0..n {
            let u = dot(&qu_hat, &coord_u[k]);
            let v = dot(&qv_hat, &coord_v[k]);
            let rr = (softplus(v) + self.eps).ln();
            s[k] = r(LOGIT_SCALE) * (alpha * u - beta * rr);
        }
        s
    }

    /// Per-token teacher-forced forward + backward.
    ///
    /// Returns `(L_c + L_w, dL/dh)` and accumulates the parameter gradients into `grads`.
    /// All steps follow `documentation/eml.txt` (FORWARD/BACKWARD sections).
    fn token_forward_backward(&self, h: &[Real], v_star: usize, grads: &mut ParamGrads) -> (Real, Vec<Real>) {
        let c_star = v_star / self.m_words;
        let w_star = v_star % self.m_words;
        let mut dh = vec![ZERO; self.d_model];

        // ── Cluster level ────────────────────────────────────────────────────
        let (lc, dh_uc, dh_vc, dac, dbc) = self.level_forward_backward(
            h,
            &self.w_uc,
            &self.w_vc,
            &self.c_u,
            &self.c_v,
            None,
            c_star,
            self.alpha_c,
            self.beta_c,
            &mut grads.w_uc,
            &mut grads.w_vc,
        );
        grads.alpha_c += dac;
        grads.beta_c += dbc;
        accumulate(&mut dh, &dh_uc);
        accumulate(&mut dh, &dh_vc);

        // ── Word level (gathered slab of cluster c*) ─────────────────────────
        let (lw, dh_uw, dh_vw, daw, dbw) = self.level_forward_backward(
            h,
            &self.w_uw,
            &self.w_vw,
            &self.w_u[c_star],
            &self.w_v[c_star],
            Some(&self.valid[c_star]),
            w_star,
            self.alpha_w,
            self.beta_w,
            &mut grads.w_uw,
            &mut grads.w_vw,
        );
        grads.alpha_w += daw;
        grads.beta_w += dbw;
        accumulate(&mut dh, &dh_uw);
        accumulate(&mut dh, &dh_vw);

        (lc + lw, dh)
    }

    /// One EML scoring level (used for both the cluster and word steps).
    ///
    /// Projects `h` through `w_u_proj`/`w_v_proj`, scores against the `coord_u`/`coord_v`
    /// rows (optionally masked), forms the stable EML score, takes a softmax cross-entropy
    /// against `target`, and back-propagates. Returns `(loss, dH_u, dH_v, d_alpha, d_beta)`
    /// and accumulates `dW_u`/`dW_v`.
    ///
    /// Both queries are L2-normalized (cosine similarities in `[-1, 1]`) and the score is
    /// multiplied by the fixed `LOGIT_SCALE`. This is the cosine-classifier recipe: the scale
    /// makes the softmax sharp enough to reach ~0 CE, while bounded logits keep the loss
    /// surface smooth so AdamW cannot fall off a cliff (the source of the training spikes).
    /// The trainable temperatures fine-tune the relative U/V weighting on top of the scale.
    #[allow(clippy::too_many_arguments)]
    fn level_forward_backward(
        &self,
        h: &[Real],
        w_u_proj: &[Vec<Real>],
        w_v_proj: &[Vec<Real>],
        coord_u: &[Vec<Real>],
        coord_v: &[Vec<Real>],
        mask: Option<&[bool]>,
        target: usize,
        alpha: Real,
        beta: Real,
        dw_u: &mut [Vec<Real>],
        dw_v: &mut [Vec<Real>],
    ) -> (Real, Vec<Real>, Vec<Real>, Real, Real) {
        let n = coord_u.len();

        // Step 2/3 — project then L2-normalize both queries (bounded cosine similarities).
        let qu = project(h, w_u_proj, self.d_coord);
        let qv = project(h, w_v_proj, self.d_coord);
        let (qu_hat, qu_inv) = l2_normalize(&qu);
        let (qv_hat, qv_inv) = l2_normalize(&qv);

        // Step 4/5 — similarities and stable EML score, scaled by the fixed LOGIT_SCALE.
        let mut u = vec![ZERO; n];
        let mut v = vec![ZERO; n];
        let mut rr = vec![ZERO; n];
        let mut s = vec![ZERO; n];
        for k in 0..n {
            u[k] = dot(&qu_hat, &coord_u[k]);
            v[k] = dot(&qv_hat, &coord_v[k]);
            rr[k] = (softplus(v[k]) + self.eps).ln();
            s[k] = r(LOGIT_SCALE) * (alpha * u[k] - beta * rr[k]);
        }

        // Step 6/11 — masked softmax.
        let active = |k: usize| mask.map_or(true, |m| m[k]);
        let mut max_s = Real::NEG_INFINITY;
        for k in 0..n {
            if active(k) && s[k] > max_s {
                max_s = s[k];
            }
        }
        let mut sum_exp = ZERO;
        let mut p = vec![ZERO; n];
        for k in 0..n {
            if active(k) {
                let e = (s[k] - max_s).exp();
                p[k] = e;
                sum_exp += e;
            }
        }
        let inv_sum = ONE / sum_exp;
        for k in 0..n {
            p[k] *= inv_sum;
        }

        let loss = -(p[target].max(r(1e-30))).ln();

        // ── Backward ─────────────────────────────────────────────────────────
        // dS = P - one_hot(target)  (masked entries already have P = 0).
        let mut d_alpha = ZERO;
        let mut d_beta = ZERO;
        let mut dqu_hat = vec![ZERO; self.d_coord];
        let mut dqv_hat = vec![ZERO; self.d_coord];
        for k in 0..n {
            if !active(k) {
                continue;
            }
            // s[k] = LOGIT_SCALE*(alpha*u - beta*rr); fold the scale into every downstream grad.
            let ds = r(LOGIT_SCALE) * (p[k] - if k == target { ONE } else { ZERO });
            d_alpha += u[k] * ds;
            d_beta += -rr[k] * ds;

            let du = alpha * ds;
            let dr = -beta * ds;
            // dR/dV = sigmoid(V) / (softplus(V) + eps)  — bounded for all V.
            let dv = dr * sigmoid(v[k]) / (softplus(v[k]) + self.eps);

            for j in 0..self.d_coord {
                dqu_hat[j] += du * coord_u[k][j];
                dqv_hat[j] += dv * coord_v[k][j];
            }
        }

        // Both queries go back through the L2 normalization Jacobian:
        // dq = (dq_hat - (q_hat·dq_hat) q_hat) / ||q||.
        let dqu = l2_normalize_backward(&qu_hat, &dqu_hat, qu_inv);
        let dqv = l2_normalize_backward(&qv_hat, &dqv_hat, qv_inv);

        // Through the linear projections: dW[i][j] += h[i]*dq[j]; dH[i] += sum_j W[i][j]*dq[j].
        let dh_u = project_backward(h, w_u_proj, &dqu, dw_u);
        let dh_v = project_backward(h, w_v_proj, &dqv, dw_v);

        (loss, dh_u, dh_v, d_alpha, d_beta)
    }

    /// Backward pass for the terminal-head case: the gradient was already computed in
    /// `forward()`, so return it. Mirrors `AdaptiveLinearLayer`'s stored-gradient path.
    pub fn backward(&mut self, _previous_gradient: &Gradient) -> Gradient {
        self.gradient
            .clone()
            .expect("EmlLinearLayer::backward called without a training forward pass")
    }

    /// AdamW update of the four projection matrices and four temperatures.
    ///
    /// The temperatures stay trainable on purpose: empirically they act as a fast global
    /// stabilizer — when the softmax starts to over-sharpen they adapt to pull it back, so
    /// freezing them (tested) made the loss spikes markedly worse, not better.
    pub fn update_parameters(&mut self) {
        let mut grads = match self.grads.take() {
            Some(g) => g,
            None => return,
        };

        // Gradients are summed over every token in the batch; average them so weight decay
        // and the gradient-norm clip act at a per-token scale (matching the other heads).
        let n_tokens = self.gradient.as_ref().map(|g| g.get_total_valid_tokens()).unwrap_or(0).max(1) as f64;
        grads.scale(r(1.0 / n_tokens));

        // Global-norm clip — the primary defense against the memorization-regime loss spikes.
        let gnorm = grads.global_norm();
        if gnorm > MAX_GRAD_NORM {
            grads.scale(r(MAX_GRAD_NORM / gnorm));
        }

        self.time_step += 1;
        let t = self.time_step as f64;
        let lr = self.learning_rate;
        let bc1 = 1.0 - ADAM_B1.powf(t);
        let bc2 = 1.0 - ADAM_B2.powf(t);

        let (d_model, d_coord) = (self.d_model, self.d_coord);
        if self.opt.is_none() {
            let zero_mat = || vec![vec![ZERO; d_coord]; d_model];
            self.opt = Some(AdamState {
                m_w: [zero_mat(), zero_mat(), zero_mat(), zero_mat()],
                v_w: [zero_mat(), zero_mat(), zero_mat(), zero_mat()],
                m_s: [ZERO; 4],
                v_s: [ZERO; 4],
            });
        }
        let opt = self.opt.as_mut().unwrap();

        let mats: [&mut Vec<Vec<Real>>; 4] = [&mut self.w_uc, &mut self.w_vc, &mut self.w_uw, &mut self.w_vw];
        let grad_mats: [&Vec<Vec<Real>>; 4] = [&grads.w_uc, &grads.w_vc, &grads.w_uw, &grads.w_vw];
        for idx in 0..4 {
            adam_matrix(mats[idx], grad_mats[idx], &mut opt.m_w[idx], &mut opt.v_w[idx], lr, bc1, bc2);
        }

        let scalars: [&mut Real; 4] = [&mut self.alpha_c, &mut self.beta_c, &mut self.alpha_w, &mut self.beta_w];
        let grad_scalars = [grads.alpha_c, grads.beta_c, grads.alpha_w, grads.beta_w];
        for idx in 0..4 {
            adam_scalar(scalars[idx], grad_scalars[idx], &mut opt.m_s[idx], &mut opt.v_s[idx], lr, bc1, bc2);
        }

        self.gradient = None;
    }

    // ── Test / diagnostic accessors (analytical gradients) ───────────────────
    pub fn grad_w_uc(&self) -> Option<&Vec<Vec<Real>>> {
        self.grads.as_ref().map(|g| &g.w_uc)
    }
    pub fn grad_w_vc(&self) -> Option<&Vec<Vec<Real>>> {
        self.grads.as_ref().map(|g| &g.w_vc)
    }
    pub fn grad_w_uw(&self) -> Option<&Vec<Vec<Real>>> {
        self.grads.as_ref().map(|g| &g.w_uw)
    }
    pub fn grad_w_vw(&self) -> Option<&Vec<Vec<Real>>> {
        self.grads.as_ref().map(|g| &g.w_vw)
    }
    /// `(d_alpha_c, d_beta_c, d_alpha_w, d_beta_w)`.
    pub fn grad_scalars(&self) -> Option<(Real, Real, Real, Real)> {
        self.grads.as_ref().map(|g| (g.alpha_c, g.beta_c, g.alpha_w, g.beta_w))
    }

    fn zero_param_grads(&self) -> ParamGrads {
        let zero_proj = vec![vec![ZERO; self.d_coord]; self.d_model];
        ParamGrads {
            w_uc: zero_proj.clone(),
            w_vc: zero_proj.clone(),
            w_uw: zero_proj.clone(),
            w_vw: zero_proj,
            alpha_c: ZERO,
            beta_c: ZERO,
            alpha_w: ZERO,
            beta_w: ZERO,
        }
    }

    /// Teacher-forcing offset: targets are the last `len(targets)` unpadded rows.
    fn target_offset(batch_idx: usize, target_batch: &[Vec<u32>], padding_mask_batch: &[Vec<u32>]) -> Option<usize> {
        if batch_idx >= target_batch.len() || target_batch[batch_idx].is_empty() || batch_idx >= padding_mask_batch.len() {
            return None;
        }
        let unpadded = padding_mask_batch[batch_idx].iter().filter(|&&m| m != 0).count();
        Some(unpadded.saturating_sub(target_batch[batch_idx].len()))
    }

    fn target_id(
        row_idx: usize,
        batch_idx: usize,
        offset: Option<usize>,
        target_batch: &[Vec<u32>],
        padding_mask_batch: &[Vec<u32>],
    ) -> Option<usize> {
        let offset = offset?;
        if row_idx < offset {
            return None;
        }
        let mask = padding_mask_batch.get(batch_idx)?;
        if row_idx >= mask.len() || mask[row_idx] == 0 {
            return None;
        }
        target_batch.get(batch_idx)?.get(row_idx - offset).map(|&id| id as usize)
    }
}

impl LayerInterface for EmlLinearLayer {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        EmlLinearLayer::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        EmlLinearLayer::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        EmlLinearLayer::update_parameters(self)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// `q[j] = sum_i h[i] * w[i][j]`, shape `[d_coord]`.
fn project(h: &[Real], w: &[Vec<Real>], d_coord: usize) -> Vec<Real> {
    let mut q = vec![ZERO; d_coord];
    for (i, &hi) in h.iter().enumerate() {
        if i >= w.len() {
            break;
        }
        let row = &w[i];
        for j in 0..d_coord.min(row.len()) {
            q[j] += hi * row[j];
        }
    }
    q
}

/// Back-propagate through `q = W^T h`. Accumulates `dW[i][j] += h[i]*dq[j]` and returns
/// `dH[i] = sum_j W[i][j]*dq[j]`.
fn project_backward(h: &[Real], w: &[Vec<Real>], dq: &[Real], dw: &mut [Vec<Real>]) -> Vec<Real> {
    let mut dh = vec![ZERO; h.len()];
    for (i, &hi) in h.iter().enumerate() {
        if i >= w.len() {
            break;
        }
        let row = &w[i];
        let drow = &mut dw[i];
        let mut acc = ZERO;
        for j in 0..dq.len().min(row.len()) {
            drow[j] += hi * dq[j];
            acc += row[j] * dq[j];
        }
        dh[i] = acc;
    }
    dh
}

fn dot(a: &[Real], b: &[Real]) -> Real {
    let mut s = ZERO;
    for i in 0..a.len().min(b.len()) {
        s += a[i] * b[i];
    }
    s
}

/// Returns the unit vector `q/||q||` and `1/||q||` (0 if the vector is degenerate).
fn l2_normalize(q: &[Real]) -> (Vec<Real>, Real) {
    let norm = dot(q, q).sqrt();
    if norm <= r(1e-20) {
        return (vec![ZERO; q.len()], ZERO);
    }
    let inv = ONE / norm;
    (q.iter().map(|&x| x * inv).collect(), inv)
}

/// Back-propagate through L2 normalization: `dq = (dq_hat - (q_hat·dq_hat) q_hat) / ||q||`.
fn l2_normalize_backward(q_hat: &[Real], dq_hat: &[Real], inv_norm: Real) -> Vec<Real> {
    let s = dot(q_hat, dq_hat);
    q_hat
        .iter()
        .zip(dq_hat.iter())
        .map(|(&qh, &dh)| (dh - s * qh) * inv_norm)
        .collect()
}

/// Numerically stable masked log-softmax: `out[k] = s[k] - logsumexp(s)` over active `k`,
/// `-inf` for masked entries. Active set defaults to all when `mask` is `None`.
fn log_softmax(s: &[Real], mask: Option<&[bool]>) -> Vec<Real> {
    let active = |k: usize| mask.map_or(true, |m| m[k]);
    let mut max_s = Real::NEG_INFINITY;
    for k in 0..s.len() {
        if active(k) && s[k] > max_s {
            max_s = s[k];
        }
    }
    let mut sum = ZERO;
    for k in 0..s.len() {
        if active(k) {
            sum += (s[k] - max_s).exp();
        }
    }
    let lse = max_s + sum.ln();
    (0..s.len()).map(|k| if active(k) { s[k] - lse } else { Real::NEG_INFINITY }).collect()
}

fn accumulate(dst: &mut [Real], src: &[Real]) {
    for i in 0..dst.len().min(src.len()) {
        dst[i] += src[i];
    }
}

/// `softplus(x) = log(1 + exp(x))`, evaluated stably.
fn softplus(x: Real) -> Real {
    let xf = x as f64;
    let v = xf.max(0.0) + (-(xf.abs())).exp().ln_1p();
    r(v)
}

fn sigmoid(x: Real) -> Real {
    r(1.0 / (1.0 + (-(x as f64)).exp()))
}

/// Adam update of a matrix parameter in place. No weight decay: the queries are L2-normalized
/// so the loss is invariant to `‖W‖`, and decay would shrink the projections toward zero
/// unopposed (collapsing the cosine direction). Stability comes from the bounded cosine
/// logits + gradient-norm clip instead.
fn adam_matrix(w: &mut [Vec<Real>], g: &[Vec<Real>], m: &mut [Vec<Real>], v: &mut [Vec<Real>], lr: f64, bc1: f64, bc2: f64) {
    for i in 0..w.len() {
        for j in 0..w[i].len() {
            let grad = g[i][j] as f64;
            let mij = ADAM_B1 * m[i][j] as f64 + (1.0 - ADAM_B1) * grad;
            let vij = ADAM_B2 * v[i][j] as f64 + (1.0 - ADAM_B2) * grad * grad;
            m[i][j] = r(mij);
            v[i][j] = r(vij);
            let m_hat = mij / bc1;
            let v_hat = vij / bc2;
            w[i][j] = r(w[i][j] as f64 - lr * m_hat / (v_hat.sqrt() + ADAM_EPS));
        }
    }
}

/// Adam update of a single scalar parameter in place.
fn adam_scalar(w: &mut Real, g: Real, m: &mut Real, v: &mut Real, lr: f64, bc1: f64, bc2: f64) {
    let grad = g as f64;
    let mv = ADAM_B1 * *m as f64 + (1.0 - ADAM_B1) * grad;
    let vv = ADAM_B2 * *v as f64 + (1.0 - ADAM_B2) * grad * grad;
    *m = r(mv);
    *v = r(vv);
    *w = r(*w as f64 - lr * (mv / bc1) / ((vv / bc2).sqrt() + ADAM_EPS));
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate / weight initialization (dependency-free, deterministic)
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal SplitMix64 PRNG — enough for reproducible coordinate/weight init.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in (0, 1).
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (((1u64 << 53) as f64) + 1.0)
    }
    /// Standard normal via Box-Muller.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn random_matrix(rows: usize, cols: usize, scale: f64, rng: &mut SplitMix64) -> Vec<Vec<Real>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| r(rng.next_gaussian() * scale)).collect())
        .collect()
}

/// `rows` unit-length random rows of width `cols`.
fn unit_rows(rows: usize, cols: usize, rng: &mut SplitMix64) -> Vec<Vec<Real>> {
    (0..rows)
        .map(|_| {
            let row: Vec<Real> = (0..cols).map(|_| r(rng.next_gaussian())).collect();
            let (hat, _) = l2_normalize(&row);
            hat
        })
        .collect()
}

/// `rows x cols` matrix with mutually orthogonal unit rows (Option C, Modified
/// Gram-Schmidt). Requires `rows <= cols`; rows beyond `cols` fall back to unit rows.
fn orthogonal_matrix(rows: usize, cols: usize, rng: &mut SplitMix64) -> Vec<Vec<Real>> {
    let mut q: Vec<Vec<Real>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut v: Vec<Real> = (0..cols).map(|_| r(rng.next_gaussian())).collect();
        if i < cols {
            for qj in q.iter() {
                let proj = dot(&v, qj);
                for k in 0..cols {
                    v[k] -= proj * qj[k];
                }
            }
        }
        let (hat, _) = l2_normalize(&v);
        q.push(hat);
    }
    q
}
