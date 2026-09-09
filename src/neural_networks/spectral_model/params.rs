//! Parameter storage, initialisation, AdamW and (de)serialisation.
//!
//! Every learned tensor lives in one of two flat vectors — `cplx` for complex
//! tensors, `real` for real ones — indexed through the accessors below. That
//! makes the optimiser, gradient clipping, EMA and serialisation two short
//! loops instead of one branch per field, and it keeps a gradient buffer
//! structurally identical to the weights it belongs to.
//!
//! Blueprint section A4 for the initialisation scheme, B14 for the optimiser.

use crate::neural_networks::spectral_model::bands::{BandTable, D, D_MERGE, N, NBANDS, NH};
use crate::neural_networks::utils::dtype::{Real, C};
use num::Complex;
use rand::RngExt;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// S-blocks applied per stage.
pub const N_SBLOCKS: usize = 2;
/// Stage repeats. All stages share the same weights (blueprint C4).
pub const N_STAGES: usize = 3;
/// Width of the sinusoidal time features.
pub const TIME_FEAT: usize = 32;
/// Width of the shared time-conditioning trunk.
pub const TIME_HIDDEN: usize = 32;
/// Hidden width of the pixel-space MLP inside the R-block.
pub const RMLP_HIDDEN: usize = 32;
/// Soft-shrink threshold (blueprint C4a).
pub const SHRINK_LAMBDA: Real = 0.02;
/// Width of the concatenated band-summary vector.
pub const SUM_WIDTH: usize = NBANDS * D_MERGE;

/// Global gradient-norm clip.
///
/// The blueprint says 1.0 (B14), but that is far below where this model's
/// gradients actually sit. Measured over a four-band run: min 1.19, median
/// 6.19, p90 11.5, max 25.8 — **every single step was clipped**, and the clip
/// factor varied 22x across steps.
///
/// That is worse than it sounds. Adam is invariant to a *constant* rescale of
/// the gradient, so uniform clipping would be harmless; it is the *variation*
/// that does damage, because rescaling every step to the same norm discards
/// exactly the information about which steps the optimiser should trust. The
/// effect is to silently turn AdamW into normalised-gradient descent.
///
/// Set near the p90 of the observed distribution so clipping is what it is
/// meant to be — a safety valve for rare spikes, not the common path. Re-measure
/// with [`NormTracker::percentiles`] if the architecture or loss changes;
/// this number is empirical, not universal.
pub const DEFAULT_CLIP_NORM: Real = 12.0;

// ---------------------------------------------------------------------------
// Flat tensor indices
// ---------------------------------------------------------------------------

const C_W_IN: usize = 0; // 4 entries, one per band
const C_POS: usize = 4; // 4
const C_R_MAT: usize = 8; // N_SBLOCKS * NBANDS = 8
const C_FILT_A: usize = 16; // 8
const C_FILT_B: usize = 24; // 8
const C_PROJ_DOWN: usize = 32; // 4
const C_PROJ_UP: usize = 36; // 4
// Neighbour stencil: 3 taps along k1 then 3 along k2, per S-block per band.
const C_STENCIL_A: usize = 40; // 8, layout (3, d)
const C_STENCIL_B: usize = 48; // 8, layout (3, d)
// Band summary tokens: per-band down/up projections plus one shared mixer.
const C_SUM_DOWN: usize = 56; // 4
const C_SUM_UP: usize = 60; // 4
const C_SUM_MIX: usize = 64; // 1
const N_CPLX: usize = 65;

const R_TRUNK_W: usize = 0;
const R_TRUNK_B: usize = 1;
const R_FILM_W: usize = 2; // 8
const R_FILM_B: usize = 10; // 8
const R_RMLP_W1: usize = 18;
const R_RMLP_B1: usize = 19;
const R_RMLP_W2: usize = 20;
const R_RMLP_B2: usize = 21;
const R_HEAD_W: usize = 22; // 4
const R_HEAD_B: usize = 26; // 4
const N_REAL: usize = 30;

/// The full parameter set. A zeroed instance doubles as a gradient buffer and
/// as AdamW moment state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralParams {
    pub cplx: Vec<Vec<C>>,
    pub real: Vec<Vec<Real>>,
}

impl SpectralParams {
    /// Allocate every tensor at the right size, filled with zeros.
    pub fn zeros(bands: &BandTable) -> Self {
        let mut cplx = vec![Vec::new(); N_CPLX];
        let mut real = vec![Vec::new(); N_REAL];

        for bd in 0..NBANDS {
            let d = D[bd];
            let n = bands.counts[bd];
            cplx[C_W_IN + bd] = vec![C::new(0.0, 0.0); d];
            cplx[C_POS + bd] = vec![C::new(0.0, 0.0); n * d];
            cplx[C_PROJ_DOWN + bd] = vec![C::new(0.0, 0.0); d * D_MERGE];
            cplx[C_PROJ_UP + bd] = vec![C::new(0.0, 0.0); D_MERGE * d];
            real[R_HEAD_W + bd] = vec![0.0 as Real; 2 * d * 2];
            real[R_HEAD_B + bd] = vec![0.0 as Real; 2];

            cplx[C_SUM_DOWN + bd] = vec![C::new(0.0, 0.0); d * D_MERGE];
            cplx[C_SUM_UP + bd] = vec![C::new(0.0, 0.0); D_MERGE * d];

            for sb in 0..N_SBLOCKS {
                let s = sb * NBANDS + bd;
                cplx[C_R_MAT + s] = vec![C::new(0.0, 0.0); d * d];
                cplx[C_FILT_A + s] = vec![C::new(0.0, 0.0); N * d];
                cplx[C_FILT_B + s] = vec![C::new(0.0, 0.0); NH * d];
                cplx[C_STENCIL_A + s] = vec![C::new(0.0, 0.0); 3 * d];
                cplx[C_STENCIL_B + s] = vec![C::new(0.0, 0.0); 3 * d];
                real[R_FILM_W + s] = vec![0.0 as Real; TIME_HIDDEN * 3 * d];
                real[R_FILM_B + s] = vec![0.0 as Real; 3 * d];
            }
        }

        real[R_TRUNK_W] = vec![0.0 as Real; TIME_FEAT * TIME_HIDDEN];
        real[R_TRUNK_B] = vec![0.0 as Real; TIME_HIDDEN];
        real[R_RMLP_W1] = vec![0.0 as Real; D_MERGE * RMLP_HIDDEN];
        real[R_RMLP_B1] = vec![0.0 as Real; RMLP_HIDDEN];
        real[R_RMLP_W2] = vec![0.0 as Real; RMLP_HIDDEN * D_MERGE];
        real[R_RMLP_B2] = vec![0.0 as Real; D_MERGE];
        cplx[C_SUM_MIX] = vec![C::new(0.0, 0.0); (NBANDS * D_MERGE) * (NBANDS * D_MERGE)];

        Self { cplx, real }
    }

    /// A zero buffer with the same shape as `self`.
    pub fn zeros_like(&self) -> Self {
        Self {
            cplx: self.cplx.iter().map(|t| vec![C::new(0.0, 0.0); t.len()]).collect(),
            real: self.real.iter().map(|t| vec![0.0 as Real; t.len()]).collect(),
        }
    }

    pub fn zero_out(&mut self) {
        for t in self.cplx.iter_mut() {
            t.iter_mut().for_each(|v| *v = C::new(0.0, 0.0));
        }
        for t in self.real.iter_mut() {
            t.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// Number of real floats in the model.
    pub fn num_scalars(&self) -> usize {
        self.cplx.iter().map(|t| 2 * t.len()).sum::<usize>()
            + self.real.iter().map(|t| t.len()).sum::<usize>()
    }

    /// Randomly initialise. Complex tensors get independent Gaussian real and
    /// imaginary parts, which yields a Rayleigh magnitude and uniform phase
    /// automatically (blueprint A4).
    pub fn init<R: RngExt + ?Sized>(bands: &BandTable, rng: &mut R) -> Self {
        let mut p = Self::zeros(bands);

        for bd in 0..NBANDS {
            let d = D[bd];
            fill_cplx(&mut p.cplx[C_W_IN + bd], sigma(1, d), rng);
            fill_cplx(&mut p.cplx[C_POS + bd], 0.02, rng);
            fill_cplx(&mut p.cplx[C_PROJ_DOWN + bd], sigma(d, D_MERGE), rng);
            // The R-block starts nearly inert so early training is driven by
            // the S-blocks, but not exactly zero or its inputs get no gradient.
            fill_cplx(&mut p.cplx[C_PROJ_UP + bd], 0.1 * sigma(D_MERGE, d), rng);
            fill_real(&mut p.real[R_HEAD_W + bd], sigma(2 * d, 2), rng);
            fill_cplx(&mut p.cplx[C_SUM_DOWN + bd], sigma(d, D_MERGE), rng);
            // Same reasoning as proj_up: small but non-zero, so the summary
            // path starts near-inert without its inputs going gradient-dead.
            fill_cplx(&mut p.cplx[C_SUM_UP + bd], 0.1 * sigma(D_MERGE, d), rng);

            for sb in 0..N_SBLOCKS {
                let s = sb * NBANDS + bd;
                fill_cplx(&mut p.cplx[C_R_MAT + s], sigma(d, d), rng);
                // Separable filter starts at ~1 so the S-block begins as a
                // plain matrix multiply rather than a random reweighting.
                fill_cplx_around_one(&mut p.cplx[C_FILT_A + s], 0.02, rng);
                fill_cplx_around_one(&mut p.cplx[C_FILT_B + s], 0.02, rng);
                // Stencil starts as identity: centre tap 1, neighbours 0, so
                // the S-block begins exactly as it did without the stencil.
                init_identity_stencil(&mut p.cplx[C_STENCIL_A + s], d, 0.02, rng);
                init_identity_stencil(&mut p.cplx[C_STENCIL_B + s], d, 0.02, rng);
                // FiLM output layer starts at zero => gamma = 1, beta = 0.
            }
        }

        let w = NBANDS * D_MERGE;
        fill_cplx(&mut p.cplx[C_SUM_MIX], sigma(w, w), rng);
        fill_real(&mut p.real[R_TRUNK_W], (2.0 / TIME_FEAT as f64).sqrt() as Real, rng);
        fill_real(&mut p.real[R_RMLP_W1], (2.0 / D_MERGE as f64).sqrt() as Real, rng);
        fill_real(&mut p.real[R_RMLP_W2], (2.0 / RMLP_HIDDEN as f64).sqrt() as Real, rng);

        p
    }

    // -- accessors ---------------------------------------------------------

    pub fn w_in(&self, bd: usize) -> &[C] {
        &self.cplx[C_W_IN + bd]
    }
    pub fn pos(&self, bd: usize) -> &[C] {
        &self.cplx[C_POS + bd]
    }
    pub fn r_mat(&self, sb: usize, bd: usize) -> &[C] {
        &self.cplx[C_R_MAT + sb * NBANDS + bd]
    }
    pub fn filt_a(&self, sb: usize, bd: usize) -> &[C] {
        &self.cplx[C_FILT_A + sb * NBANDS + bd]
    }
    pub fn filt_b(&self, sb: usize, bd: usize) -> &[C] {
        &self.cplx[C_FILT_B + sb * NBANDS + bd]
    }
    pub fn proj_down(&self, bd: usize) -> &[C] {
        &self.cplx[C_PROJ_DOWN + bd]
    }
    pub fn proj_up(&self, bd: usize) -> &[C] {
        &self.cplx[C_PROJ_UP + bd]
    }
    pub fn stencil_a(&self, sb: usize, bd: usize) -> &[C] {
        &self.cplx[C_STENCIL_A + sb * NBANDS + bd]
    }
    pub fn stencil_b(&self, sb: usize, bd: usize) -> &[C] {
        &self.cplx[C_STENCIL_B + sb * NBANDS + bd]
    }
    pub fn sum_down(&self, bd: usize) -> &[C] {
        &self.cplx[C_SUM_DOWN + bd]
    }
    pub fn sum_up(&self, bd: usize) -> &[C] {
        &self.cplx[C_SUM_UP + bd]
    }
    pub fn sum_mix(&self) -> &[C] {
        &self.cplx[C_SUM_MIX]
    }
    pub fn trunk_w(&self) -> &[Real] {
        &self.real[R_TRUNK_W]
    }
    pub fn trunk_b(&self) -> &[Real] {
        &self.real[R_TRUNK_B]
    }
    pub fn film_w(&self, sb: usize, bd: usize) -> &[Real] {
        &self.real[R_FILM_W + sb * NBANDS + bd]
    }
    pub fn film_b(&self, sb: usize, bd: usize) -> &[Real] {
        &self.real[R_FILM_B + sb * NBANDS + bd]
    }
    pub fn rmlp_w1(&self) -> &[Real] {
        &self.real[R_RMLP_W1]
    }
    pub fn rmlp_b1(&self) -> &[Real] {
        &self.real[R_RMLP_B1]
    }
    pub fn rmlp_w2(&self) -> &[Real] {
        &self.real[R_RMLP_W2]
    }
    pub fn rmlp_b2(&self) -> &[Real] {
        &self.real[R_RMLP_B2]
    }
    pub fn head_w(&self, bd: usize) -> &[Real] {
        &self.real[R_HEAD_W + bd]
    }
    pub fn head_b(&self, bd: usize) -> &[Real] {
        &self.real[R_HEAD_B + bd]
    }

    // -- mutable accessors, used only when accumulating gradients ----------

    pub fn w_in_mut(&mut self, bd: usize) -> &mut [C] {
        &mut self.cplx[C_W_IN + bd]
    }
    pub fn pos_mut(&mut self, bd: usize) -> &mut [C] {
        &mut self.cplx[C_POS + bd]
    }
    pub fn r_mat_mut(&mut self, sb: usize, bd: usize) -> &mut [C] {
        &mut self.cplx[C_R_MAT + sb * NBANDS + bd]
    }
    pub fn filt_a_mut(&mut self, sb: usize, bd: usize) -> &mut [C] {
        &mut self.cplx[C_FILT_A + sb * NBANDS + bd]
    }
    pub fn filt_b_mut(&mut self, sb: usize, bd: usize) -> &mut [C] {
        &mut self.cplx[C_FILT_B + sb * NBANDS + bd]
    }
    pub fn proj_down_mut(&mut self, bd: usize) -> &mut [C] {
        &mut self.cplx[C_PROJ_DOWN + bd]
    }
    pub fn proj_up_mut(&mut self, bd: usize) -> &mut [C] {
        &mut self.cplx[C_PROJ_UP + bd]
    }
    pub fn stencil_a_mut(&mut self, sb: usize, bd: usize) -> &mut [C] {
        &mut self.cplx[C_STENCIL_A + sb * NBANDS + bd]
    }
    pub fn stencil_b_mut(&mut self, sb: usize, bd: usize) -> &mut [C] {
        &mut self.cplx[C_STENCIL_B + sb * NBANDS + bd]
    }
    pub fn sum_down_mut(&mut self, bd: usize) -> &mut [C] {
        &mut self.cplx[C_SUM_DOWN + bd]
    }
    pub fn sum_up_mut(&mut self, bd: usize) -> &mut [C] {
        &mut self.cplx[C_SUM_UP + bd]
    }
    pub fn sum_mix_mut(&mut self) -> &mut [C] {
        &mut self.cplx[C_SUM_MIX]
    }
    pub fn trunk_w_mut(&mut self) -> &mut [Real] {
        &mut self.real[R_TRUNK_W]
    }
    pub fn trunk_b_mut(&mut self) -> &mut [Real] {
        &mut self.real[R_TRUNK_B]
    }
    pub fn film_w_mut(&mut self, sb: usize, bd: usize) -> &mut [Real] {
        &mut self.real[R_FILM_W + sb * NBANDS + bd]
    }
    pub fn film_b_mut(&mut self, sb: usize, bd: usize) -> &mut [Real] {
        &mut self.real[R_FILM_B + sb * NBANDS + bd]
    }
    pub fn rmlp_w1_mut(&mut self) -> &mut [Real] {
        &mut self.real[R_RMLP_W1]
    }
    pub fn rmlp_b1_mut(&mut self) -> &mut [Real] {
        &mut self.real[R_RMLP_B1]
    }
    pub fn rmlp_w2_mut(&mut self) -> &mut [Real] {
        &mut self.real[R_RMLP_W2]
    }
    pub fn rmlp_b2_mut(&mut self) -> &mut [Real] {
        &mut self.real[R_RMLP_B2]
    }
    pub fn head_w_mut(&mut self, bd: usize) -> &mut [Real] {
        &mut self.real[R_HEAD_W + bd]
    }
    pub fn head_b_mut(&mut self, bd: usize) -> &mut [Real] {
        &mut self.real[R_HEAD_B + bd]
    }

    // -- whole-model operations -------------------------------------------

    /// Euclidean norm over every real component.
    pub fn global_norm(&self) -> Real {
        let mut acc = 0.0f64;
        for t in &self.cplx {
            for v in t {
                acc += (v.re as f64) * (v.re as f64) + (v.im as f64) * (v.im as f64);
            }
        }
        for t in &self.real {
            for v in t {
                acc += (*v as f64) * (*v as f64);
            }
        }
        acc.sqrt() as Real
    }

    pub fn scale(&mut self, f: Real) {
        for t in self.cplx.iter_mut() {
            t.iter_mut().for_each(|v| *v = *v * f);
        }
        for t in self.real.iter_mut() {
            t.iter_mut().for_each(|v| *v *= f);
        }
    }

    /// `self <- self * (1 - rate) + other * rate`, used for the sampling EMA.
    pub fn lerp_from(&mut self, other: &Self, rate: Real) {
        for (a, b) in self.cplx.iter_mut().zip(other.cplx.iter()) {
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x = *x * (1.0 - rate) + *y * rate;
            }
        }
        for (a, b) in self.real.iter_mut().zip(other.real.iter()) {
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x = *x * (1.0 - rate) + *y * rate;
            }
        }
    }

    pub fn has_non_finite(&self) -> bool {
        self.cplx.iter().flatten().any(|v| !v.re.is_finite() || !v.im.is_finite())
            || self.real.iter().flatten().any(|v| !v.is_finite())
    }
}

fn sigma(fan_in: usize, fan_out: usize) -> Real {
    (1.0 / (fan_in + fan_out) as f64).sqrt() as Real
}

fn fill_real<R: RngExt + ?Sized>(t: &mut [Real], sd: Real, rng: &mut R) {
    let dist = Normal::new(0.0f64, sd as f64).unwrap();
    for v in t.iter_mut() {
        *v = dist.sample(rng) as Real;
    }
}

fn fill_cplx<R: RngExt + ?Sized>(t: &mut [C], sd: Real, rng: &mut R) {
    let dist = Normal::new(0.0f64, sd as f64).unwrap();
    for v in t.iter_mut() {
        *v = Complex::new(dist.sample(rng) as Real, dist.sample(rng) as Real);
    }
}

/// Centre tap 1, side taps 0, all with a little noise.
fn init_identity_stencil<R: RngExt + ?Sized>(t: &mut [C], d: usize, sd: Real, rng: &mut R) {
    let dist = Normal::new(0.0f64, sd as f64).unwrap();
    for tap in 0..3 {
        let centre = if tap == 1 { 1.0 as Real } else { 0.0 as Real };
        for j in 0..d {
            t[tap * d + j] =
                Complex::new(centre + dist.sample(rng) as Real, dist.sample(rng) as Real);
        }
    }
}

fn fill_cplx_around_one<R: RngExt + ?Sized>(t: &mut [C], sd: Real, rng: &mut R) {
    let dist = Normal::new(0.0f64, sd as f64).unwrap();
    for v in t.iter_mut() {
        *v = Complex::new(1.0 as Real + dist.sample(rng) as Real, dist.sample(rng) as Real);
    }
}

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

/// AdamW with decoupled weight decay, global gradient clipping and a linear
/// warm-up. Complex tensors are optimised as independent real pairs, which is
/// exactly the correct update for a real-valued loss (blueprint B13).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamW {
    pub m: SpectralParams,
    pub v: SpectralParams,
    pub step: usize,
    pub lr: Real,
    pub beta1: Real,
    pub beta2: Real,
    pub eps: Real,
    pub weight_decay: Real,
    pub clip_norm: Real,
    pub warmup: usize,
}

/// How many recent gradient norms to retain.
const NORM_WINDOW: usize = 512;

/// Rolling gradient-norm telemetry.
///
/// Deliberately **not** part of [`AdamW`]: the optimiser's serialised state
/// lives in the positional-bincode half of a checkpoint, where appending a
/// field invalidates every file ever written. This is diagnostics, it has no
/// effect on the update, and it does not belong in that half. Keeping it here
/// is what lets the clipping instrumentation be added mid-run without
/// orphaning the checkpoints.
#[derive(Debug, Clone, Default)]
pub struct NormTracker {
    recent: Vec<Real>,
    clipped: usize,
    seen: usize,
}

impl NormTracker {
    pub fn record(&mut self, norm: Real, clip_at: Real) {
        if !norm.is_finite() {
            return;
        }
        self.seen += 1;
        if norm > clip_at {
            self.clipped += 1;
        }
        if self.recent.len() >= NORM_WINDOW {
            self.recent.remove(0);
        }
        self.recent.push(norm);
    }

    /// `(p50, p90, p99, fraction_clipped)` over the retained window.
    ///
    /// If the clipped fraction is not small, the threshold is not a safety
    /// valve and the optimiser is losing step-size information.
    pub fn percentiles(&self) -> (Real, Real, Real, Real) {
        if self.recent.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let mut v = self.recent.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |q: f64| v[((q * (v.len() - 1) as f64) as usize).min(v.len() - 1)];
        let frac = if self.seen > 0 { self.clipped as Real / self.seen as Real } else { 0.0 };
        (at(0.50), at(0.90), at(0.99), frac)
    }
}

impl AdamW {
    pub fn new(params: &SpectralParams, lr: Real) -> Self {
        Self {
            m: params.zeros_like(),
            v: params.zeros_like(),
            step: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            clip_norm: DEFAULT_CLIP_NORM,
            warmup: 500,
        }
    }

    fn current_lr(&self) -> Real {
        if self.step < self.warmup {
            self.lr * (self.step as Real + 1.0) / self.warmup as Real
        } else {
            self.lr
        }
    }

    /// One update. `grads` is left untouched apart from the clipping rescale.
    /// Returns the pre-clip gradient norm, which is the single most useful
    /// number to watch during training.
    pub fn update(&mut self, params: &mut SpectralParams, grads: &mut SpectralParams) -> Real {
        let norm = grads.global_norm();
        if norm.is_finite() && norm > self.clip_norm {
            grads.scale(self.clip_norm / norm);
        }

        self.step += 1;
        let t = self.step as i32;
        let lr = self.current_lr();
        let bc1 = 1.0 - self.beta1.powi(t);
        let bc2 = 1.0 - self.beta2.powi(t);
        let hp = StepHyper {
            lr,
            bc1,
            bc2,
            wd: lr * self.weight_decay,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
        };

        for i in 0..params.cplx.len() {
            let (p, g) = (&mut params.cplx[i], &grads.cplx[i]);
            let (m, v) = (&mut self.m.cplx[i], &mut self.v.cplx[i]);
            for k in 0..p.len() {
                p[k].re = step_scalar(p[k].re, g[k].re, &mut m[k].re, &mut v[k].re, &hp);
                p[k].im = step_scalar(p[k].im, g[k].im, &mut m[k].im, &mut v[k].im, &hp);
            }
        }
        for i in 0..params.real.len() {
            let (p, g) = (&mut params.real[i], &grads.real[i]);
            let (m, v) = (&mut self.m.real[i], &mut self.v.real[i]);
            for k in 0..p.len() {
                p[k] = step_scalar(p[k], g[k], &mut m[k], &mut v[k], &hp);
            }
        }

        norm
    }
}

struct StepHyper {
    lr: Real,
    bc1: Real,
    bc2: Real,
    wd: Real,
    beta1: Real,
    beta2: Real,
    eps: Real,
}

#[inline]
fn step_scalar(p: Real, g: Real, m: &mut Real, v: &mut Real, hp: &StepHyper) -> Real {
    let g = if g.is_finite() { g } else { 0.0 };
    *m = hp.beta1 * *m + (1.0 - hp.beta1) * g;
    *v = hp.beta2 * *v + (1.0 - hp.beta2) * g * g;
    let m_hat = *m / hp.bc1;
    let v_hat = *v / hp.bc2;
    p - hp.wd * p - hp.lr * m_hat / (v_hat.sqrt() + hp.eps)
}
