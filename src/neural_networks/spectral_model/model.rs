//! The network: forward pass (blueprint section C) and its hand-written
//! adjoint (blueprint B13).
//!
//! One `forward` call processes a *time group*: a set of images that share the
//! same global time `t`, and therefore the same band schedule, the same active
//! set and the same R-block grid size. That keeps every tensor rectangular and
//! turns each S-block into a single large GEMM, while the trainer still gets
//! several independent `t` values per optimiser step by running several groups.
//!
//! Gradients w.r.t. the inputs are not computed — nothing upstream of `ut` and
//! `phit` is learned.

use crate::neural_networks::spectral_model::bands::{
    clamp_u, highest_active_band, roundtrip_grid_size, BandTable, D, D_MERGE, EPS_MAG, N, NBANDS,
    NCOEF, NH, NO_NEIGHBOUR,
};
use crate::neural_networks::spectral_model::fft::Rfft2Plan;
use crate::neural_networks::spectral_model::gemm::{gemm, gemm_acc, gemm_grad_a, gemm_grad_b};
use crate::neural_networks::spectral_model::params::{
    SpectralParams, N_SBLOCKS, N_STAGES, RMLP_HIDDEN, SHRINK_LAMBDA, SUM_WIDTH, TIME_FEAT,
    TIME_HIDDEN,
};
use crate::neural_networks::utils::dtype::{Real, C};
use num::Complex;

/// Geometry and FFT plans. Immutable once built; safe to share across threads.
#[derive(Debug, Clone)]
pub struct SpectralNet {
    pub bands: BandTable,
    /// Plans for the R-block round trip, indexed by `log2(size) - 3` (8/16/32).
    plans: [Rfft2Plan; 3],
    /// Run the R-block round trip at the lowest resolution that still
    /// represents every active band (blueprint F, "the fix").
    pub adaptive_roundtrip: bool,
}

impl Default for SpectralNet {
    fn default() -> Self {
        Self::new(true)
    }
}

impl SpectralNet {
    pub fn new(adaptive_roundtrip: bool) -> Self {
        Self {
            bands: BandTable::new(),
            plans: [Rfft2Plan::new(8), Rfft2Plan::new(16), Rfft2Plan::new(32)],
            adaptive_roundtrip,
        }
    }

    fn plan_for(&self, ng: usize) -> &Rfft2Plan {
        match ng {
            8 => &self.plans[0],
            16 => &self.plans[1],
            32 => &self.plans[2],
            _ => panic!("unsupported round-trip grid size {ng}"),
        }
    }

    fn grid_size(&self, active: &[bool; NBANDS]) -> usize {
        match (self.adaptive_roundtrip, highest_active_band(active)) {
            (true, Some(h)) => roundtrip_grid_size(h),
            _ => N,
        }
    }
}

/// How hard the optional mixing mechanisms are actually working.
///
/// Identity initialisation means the stencil and the summary tokens *can* stay
/// inert — it does not mean they will. Tracking their contribution against the
/// residual stream they feed answers that directly: a ratio pinned near zero
/// means dead weight (and a cheap negative result), a ratio that grows tells
/// you they are being used and roughly when they started to matter.
///
/// Accumulated as sums of squares over one forward pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct Diagnostics {
    /// `|| y_stencil - y_matmul ||^2` — how much the stencil moved its input.
    pub stencil_delta_sq: f64,
    /// `|| y_matmul ||^2`, the scale the stencil delta is measured against.
    pub y_mm_sq: f64,
    /// `|| summary contribution ||^2` before it is broadcast-added.
    pub summary_sq: f64,
    /// `|| h ||^2` at the point of that add.
    pub residual_sq: f64,
}

impl Diagnostics {
    /// Stencil movement as a fraction of its input's magnitude.
    pub fn stencil_ratio(&self) -> f64 {
        if self.y_mm_sq > 0.0 {
            (self.stencil_delta_sq / self.y_mm_sq).sqrt()
        } else {
            0.0
        }
    }

    /// Summary-token contribution as a fraction of the residual stream.
    pub fn summary_ratio(&self) -> f64 {
        if self.residual_sq > 0.0 {
            (self.summary_sq / self.residual_sq).sqrt()
        } else {
            0.0
        }
    }

    pub fn accumulate(&mut self, o: &Diagnostics) {
        self.stencil_delta_sq += o.stencil_delta_sq;
        self.y_mm_sq += o.y_mm_sq;
        self.summary_sq += o.summary_sq;
        self.residual_sq += o.residual_sq;
    }
}

#[inline]
fn norm_sq(v: &[C]) -> f64 {
    v.iter().map(|z| (z.re as f64) * (z.re as f64) + (z.im as f64) * (z.im as f64)).sum()
}

/// One time group's input state.
#[derive(Debug, Clone)]
pub struct GroupInput {
    /// Images in the group.
    pub b: usize,
    pub tau: [Real; NBANDS],
    pub active: [bool; NBANDS],
    /// `b * NCOEF`, band-major.
    pub ut: Vec<Real>,
    pub phit: Vec<Real>,
}

/// Predicted velocities plus everything the backward pass needs.
pub struct Cache {
    pub b: usize,
    pub active: [bool; NBANDS],
    pub ng: usize,
    /// `b * NCOEF`, band-major.
    pub v_u: Vec<Real>,
    pub v_phi: Vec<Real>,
    pub diag: Diagnostics,

    c_lift: Vec<C>,
    feats: Vec<Real>,
    trunk_pre: Vec<Real>,
    trunk_out: Vec<Real>,
    gamma: Vec<Vec<Real>>,
    beta: Vec<Vec<C>>,

    s_h_in: Vec<Vec<C>>,
    s_y_mm: Vec<Vec<C>>,
    /// After the k1 pass of the stencil, before the k2 pass.
    s_y_sa: Vec<Vec<C>>,
    /// After the full stencil; this is what the filter multiplies.
    s_y_st: Vec<Vec<C>>,
    s_y_filt: Vec<Vec<C>>,
    s_y_film: Vec<Vec<C>>,
    /// Per-band coefficient means feeding the summary tokens, `(b, d)`.
    t_mean: Vec<Vec<C>>,
    /// Concatenated, projected summaries `(b, SUM_WIDTH)`.
    t_cat: Vec<Vec<C>>,
    /// Mixer output before the shrink, `(b, SUM_WIDTH)`.
    t_mix_pre: Vec<Vec<C>>,

    r_h_in: Vec<Vec<C>>,
    r_gathered: Vec<Vec<C>>,
    r_sp_in: Vec<Vec<Real>>,
    r_hidden: Vec<Vec<Real>>,

    h_final: Vec<Vec<C>>,
}

#[inline]
fn s_idx(stage: usize, sb: usize, bd: usize) -> usize {
    (stage * N_SBLOCKS + sb) * NBANDS + bd
}

#[inline]
fn r_idx(stage: usize, bd: usize) -> usize {
    stage * NBANDS + bd
}

#[inline]
fn silu(x: Real) -> Real {
    x / (1.0 + (-x).exp())
}

#[inline]
fn silu_grad(x: Real) -> Real {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

/// Shrinks magnitude and leaves phase exactly unchanged — the right
/// nonlinearity for a domain where phase carries the signal.
#[inline]
fn soft_shrink(z: C, lambda: Real) -> C {
    let m = z.norm();
    if m <= lambda {
        C::new(0.0, 0.0)
    } else {
        z * (1.0 - lambda / m)
    }
}

/// Adjoint of [`soft_shrink`]. The Jacobian is
/// `(1 - lambda/m) I + (lambda/m^3) z z^T`, which is symmetric.
#[inline]
fn soft_shrink_grad(z: C, g: C, lambda: Real) -> C {
    let m = z.norm();
    if m <= lambda {
        C::new(0.0, 0.0)
    } else {
        let dot = z.re * g.re + z.im * g.im;
        g * (1.0 - lambda / m) + z * (lambda / (m * m * m) * dot)
    }
}

/// Map a stored `(k1, k2)` onto a round-trip grid of side `ng`.
///
/// Returns `None` when the coefficient does not fit, which cannot happen for
/// the active bands `ng` was chosen from but keeps forward and backward
/// consistent if it ever does.
#[inline]
fn grid_pos(k1: usize, k2: usize, ng: usize) -> Option<(usize, usize)> {
    let k1s = if k1 <= N / 2 { k1 as i32 } else { k1 as i32 - N as i32 };
    let half = (ng / 2) as i32;
    if k1s.abs() > half || k2 as i32 > half {
        return None;
    }
    if ng < N && (k1s.abs() == half || k2 as i32 == half) {
        // Nyquist row/column of a reduced grid is its own mirror; refuse it
        // rather than fold two distinct frequencies together.
        return None;
    }
    Some((k1s.rem_euclid(ng as i32) as usize, k2))
}


/// One 3-tap pass of the neighbour stencil along a single frequency axis.
///
/// `slot` selects which pair of `neighbours` entries to use: 0 for the `k1`
/// axis (slots 0 and 1), 2 for the `k2` axis (slots 2 and 3). Taps are laid out
/// `(3, d)` as `[minus, centre, plus]`. A neighbour outside the band
/// contributes nothing, which is the same as treating it as zero.
#[allow(clippy::too_many_arguments)]
fn stencil_pass(
    src: &[C],
    dst: &mut [C],
    nbrs: &[[usize; 4]],
    taps: &[C],
    b: usize,
    n: usize,
    d: usize,
    slot: usize,
) {
    for bi in 0..b {
        let base = bi * n;
        for i in 0..n {
            let row = (base + i) * d;
            let (m, pl) = (nbrs[i][slot], nbrs[i][slot + 1]);
            for j in 0..d {
                let mut acc = taps[d + j] * src[row + j];
                if m != NO_NEIGHBOUR {
                    acc += taps[j] * src[(base + m) * d + j];
                }
                if pl != NO_NEIGHBOUR {
                    acc += taps[2 * d + j] * src[(base + pl) * d + j];
                }
                dst[row + j] = acc;
            }
        }
    }
}

/// Test-only re-export of [`stencil_pass`].
#[allow(clippy::too_many_arguments)]
pub fn stencil_pass_for_test(
    src: &[C],
    dst: &mut [C],
    nbrs: &[[usize; 4]],
    taps: &[C],
    b: usize,
    n: usize,
    d: usize,
    slot: usize,
) {
    stencil_pass(src, dst, nbrs, taps, b, n, d, slot)
}

/// Adjoint of [`stencil_pass`]: scatters `d_dst` back to `d_src` and
/// accumulates the tap gradients.
#[allow(clippy::too_many_arguments)]
fn stencil_pass_backward(
    src: &[C],
    d_dst: &[C],
    d_src: &mut [C],
    d_taps: &mut [C],
    nbrs: &[[usize; 4]],
    taps: &[C],
    b: usize,
    n: usize,
    d: usize,
    slot: usize,
) {
    for bi in 0..b {
        let base = bi * n;
        for i in 0..n {
            let row = (base + i) * d;
            let (m, pl) = (nbrs[i][slot], nbrs[i][slot + 1]);
            for j in 0..d {
                let g = d_dst[row + j];
                if g.re == 0.0 && g.im == 0.0 {
                    continue;
                }
                // centre
                d_taps[d + j] += conj_mul(src[row + j], g);
                d_src[row + j] += g * taps[d + j].conj();
                if m != NO_NEIGHBOUR {
                    let sm = (base + m) * d + j;
                    d_taps[j] += conj_mul(src[sm], g);
                    d_src[sm] += g * taps[j].conj();
                }
                if pl != NO_NEIGHBOUR {
                    let sp = (base + pl) * d + j;
                    d_taps[2 * d + j] += conj_mul(src[sp], g);
                    d_src[sp] += g * taps[2 * d + j].conj();
                }
            }
        }
    }
}

/// `conj(a) * g` — the weight-side gradient of a complex product.
#[inline]
fn conj_mul(a: C, g: C) -> C {
    Complex::new(a.re * g.re + a.im * g.im, a.re * g.im - a.im * g.re)
}

impl SpectralNet {
    // -----------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------

    pub fn forward(&self, p: &SpectralParams, input: &GroupInput) -> Cache {
        let b = input.b;
        let bands = &self.bands;
        let active = input.active;
        let ng = self.grid_size(&active);

        let mut cache = Cache {
            b,
            active,
            ng,
            v_u: vec![0.0 as Real; b * NCOEF],
            v_phi: vec![0.0 as Real; b * NCOEF],
            diag: Diagnostics::default(),
            c_lift: vec![C::new(0.0, 0.0); b * NCOEF],
            feats: vec![0.0 as Real; TIME_FEAT],
            trunk_pre: vec![0.0 as Real; TIME_HIDDEN],
            trunk_out: vec![0.0 as Real; TIME_HIDDEN],
            gamma: vec![Vec::new(); N_SBLOCKS * NBANDS],
            beta: vec![Vec::new(); N_SBLOCKS * NBANDS],
            s_h_in: vec![Vec::new(); N_STAGES * N_SBLOCKS * NBANDS],
            s_y_mm: vec![Vec::new(); N_STAGES * N_SBLOCKS * NBANDS],
            s_y_sa: vec![Vec::new(); N_STAGES * N_SBLOCKS * NBANDS],
            s_y_st: vec![Vec::new(); N_STAGES * N_SBLOCKS * NBANDS],
            s_y_filt: vec![Vec::new(); N_STAGES * N_SBLOCKS * NBANDS],
            s_y_film: vec![Vec::new(); N_STAGES * N_SBLOCKS * NBANDS],
            t_mean: vec![Vec::new(); N_STAGES * N_SBLOCKS * NBANDS],
            t_cat: vec![Vec::new(); N_STAGES * N_SBLOCKS],
            t_mix_pre: vec![Vec::new(); N_STAGES * N_SBLOCKS],
            r_h_in: vec![Vec::new(); N_STAGES * NBANDS],
            r_gathered: vec![Vec::new(); N_STAGES * NBANDS],
            r_sp_in: vec![Vec::new(); N_STAGES],
            r_hidden: vec![Vec::new(); N_STAGES],
            h_final: vec![Vec::new(); NBANDS],
        };

        if !active.iter().any(|&a| a) {
            return cache;
        }

        // -- C1: rebuild the complex coefficients --------------------------
        // Clamping is a no-op on training inputs, which are convex
        // combinations of two non-negative `asinh` values; it only guards the
        // sampler, whose state is not constrained to the valid range.
        for i in 0..b * NCOEF {
            let mag = EPS_MAG * clamp_u(input.ut[i]).sinh();
            cache.c_lift[i] = Complex::new(mag * input.phit[i].cos(), mag * input.phit[i].sin());
        }

        // -- C3: time conditioning ----------------------------------------
        self.time_conditioning(p, &input.tau, &mut cache);

        // -- C2: lift to channels ------------------------------------------
        let mut h: Vec<Vec<C>> = vec![Vec::new(); NBANDS];
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
            let w_in = p.w_in(bd);
            let pos = p.pos(bd);
            let mut hb = vec![C::new(0.0, 0.0); b * n * d];
            for bi in 0..b {
                for i in 0..n {
                    let c = cache.c_lift[bi * NCOEF + off + i];
                    let row = (bi * n + i) * d;
                    for j in 0..d {
                        hb[row + j] = c * w_in[j] + pos[i * d + j];
                    }
                }
            }
            h[bd] = hb;
        }

        // -- C4: stage loop, shared weights --------------------------------
        for stage in 0..N_STAGES {
            for sb in 0..N_SBLOCKS {
                for bd in 0..NBANDS {
                    if !active[bd] {
                        continue;
                    }
                    self.s_block_forward(p, &mut cache, &mut h[bd], stage, sb, bd);
                }
                self.summary_forward(p, &mut cache, &mut h, stage, sb);
            }
            self.r_block_forward(p, &mut cache, &mut h, stage);
        }

        // -- C5: head -------------------------------------------------------
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
            let hw = p.head_w(bd);
            let hb2 = p.head_b(bd);
            for bi in 0..b {
                for i in 0..n {
                    let row = (bi * n + i) * d;
                    let mut acc_u = hb2[0];
                    let mut acc_p = hb2[1];
                    for j in 0..d {
                        let z = h[bd][row + j];
                        acc_u += z.re * hw[j * 2] + z.im * hw[(d + j) * 2];
                        acc_p += z.re * hw[j * 2 + 1] + z.im * hw[(d + j) * 2 + 1];
                    }
                    cache.v_u[bi * NCOEF + off + i] = acc_u;
                    cache.v_phi[bi * NCOEF + off + i] = acc_p;
                }
            }
            cache.h_final[bd] = std::mem::take(&mut h[bd]);
        }

        cache
    }

    fn time_conditioning(&self, p: &SpectralParams, tau: &[Real; NBANDS], cache: &mut Cache) {
        let pi = std::f64::consts::PI as Real;
        for bd in 0..NBANDS {
            for f in 0..4 {
                let freq = (1u32 << f) as Real * pi;
                let a = freq * tau[bd];
                cache.feats[bd * 8 + f * 2] = a.sin();
                cache.feats[bd * 8 + f * 2 + 1] = a.cos();
            }
        }

        let tw = p.trunk_w();
        let tb = p.trunk_b();
        for j in 0..TIME_HIDDEN {
            let mut acc = tb[j];
            for i in 0..TIME_FEAT {
                acc += cache.feats[i] * tw[i * TIME_HIDDEN + j];
            }
            cache.trunk_pre[j] = acc;
            cache.trunk_out[j] = silu(acc);
        }

        for sb in 0..N_SBLOCKS {
            for bd in 0..NBANDS {
                let d = D[bd];
                let fw = p.film_w(sb, bd);
                let fb = p.film_b(sb, bd);
                let out_w = 3 * d;
                let mut out = fb.to_vec();
                for i in 0..TIME_HIDDEN {
                    let x = cache.trunk_out[i];
                    if x == 0.0 {
                        continue;
                    }
                    let row = &fw[i * out_w..(i + 1) * out_w];
                    for (o, &w) in out.iter_mut().zip(row.iter()) {
                        *o += x * w;
                    }
                }
                let idx = sb * NBANDS + bd;
                cache.gamma[idx] = (0..d).map(|j| 1.0 + out[j]).collect();
                cache.beta[idx] = (0..d).map(|j| Complex::new(out[d + j], out[2 * d + j])).collect();
            }
        }
    }


    /// Band summary tokens: mean-pool each band, project to a common width,
    /// mix all four together, project back and broadcast-add.
    ///
    /// This is the cheap counterpart to the R-block. It costs `O(n*d)` against
    /// the S-block's `O(n*d^2)` and adds no FFT, yet it gives every coefficient
    /// global, cross-band context after *every* S-block rather than only three
    /// times per forward pass. It cannot create new frequency content the way
    /// the R-block does -- a mean is not a nonlinearity in pixel space -- so it
    /// complements the round trips rather than replacing them.
    fn summary_forward(
        &self,
        p: &SpectralParams,
        cache: &mut Cache,
        h: &mut [Vec<C>],
        stage: usize,
        sb: usize,
    ) {
        let bands = &self.bands;
        let b = cache.b;
        let ti = stage * N_SBLOCKS + sb;

        // 1. mean-pool each active band, then project to the merge width
        let mut cat = vec![C::new(0.0, 0.0); b * SUM_WIDTH];
        for bd in 0..NBANDS {
            if !cache.active[bd] {
                continue;
            }
            let (n, d) = (bands.counts[bd], D[bd]);
            let inv = 1.0 as Real / n as Real;
            let mut mean = vec![C::new(0.0, 0.0); b * d];
            for bi in 0..b {
                for i in 0..n {
                    let row = (bi * n + i) * d;
                    for j in 0..d {
                        mean[bi * d + j] += h[bd][row + j];
                    }
                }
                for j in 0..d {
                    mean[bi * d + j] = mean[bi * d + j] * inv;
                }
            }
            let mut proj = vec![C::new(0.0, 0.0); b * D_MERGE];
            gemm(&mean, p.sum_down(bd), &mut proj, b, d, D_MERGE);
            for bi in 0..b {
                let dst = bi * SUM_WIDTH + bd * D_MERGE;
                cat[dst..dst + D_MERGE]
                    .copy_from_slice(&proj[bi * D_MERGE..(bi + 1) * D_MERGE]);
            }
            cache.t_mean[s_idx(stage, sb, bd)] = mean;
        }

        // 2. one shared mixer across the concatenated summaries
        let mut mix_pre = vec![C::new(0.0, 0.0); b * SUM_WIDTH];
        gemm(&cat, p.sum_mix(), &mut mix_pre, b, SUM_WIDTH, SUM_WIDTH);

        // 3. project back per band and broadcast-add
        for bd in 0..NBANDS {
            if !cache.active[bd] {
                continue;
            }
            let (n, d) = (bands.counts[bd], D[bd]);
            let mut slice = vec![C::new(0.0, 0.0); b * D_MERGE];
            for bi in 0..b {
                let src = bi * SUM_WIDTH + bd * D_MERGE;
                for k in 0..D_MERGE {
                    slice[bi * D_MERGE + k] = soft_shrink(mix_pre[src + k], SHRINK_LAMBDA);
                }
            }
            let mut up = vec![C::new(0.0, 0.0); b * d];
            gemm(&slice, p.sum_up(bd), &mut up, b, D_MERGE, d);
            // Measured before the add: the contribution against what it joins.
            cache.diag.residual_sq += norm_sq(&h[bd]);
            cache.diag.summary_sq += norm_sq(&up) * n as f64;
            for bi in 0..b {
                for i in 0..n {
                    let row = (bi * n + i) * d;
                    for j in 0..d {
                        h[bd][row + j] += up[bi * d + j];
                    }
                }
            }
        }

        cache.t_cat[ti] = cat;
        cache.t_mix_pre[ti] = mix_pre;
    }

    /// Adjoint of [`SpectralNet::summary_forward`].
    fn summary_backward(
        &self,
        p: &SpectralParams,
        cache: &Cache,
        stage: usize,
        sb: usize,
        dh: &mut [Vec<C>],
        grads: &mut SpectralParams,
    ) {
        let bands = &self.bands;
        let b = cache.b;
        let ti = stage * N_SBLOCKS + sb;
        let mix_pre = &cache.t_mix_pre[ti];
        if mix_pre.is_empty() {
            return;
        }

        // 3'. broadcast-add and project up
        let mut d_mix_pre = vec![C::new(0.0, 0.0); b * SUM_WIDTH];
        for bd in 0..NBANDS {
            if !cache.active[bd] {
                continue;
            }
            let (n, d) = (bands.counts[bd], D[bd]);

            // A broadcast forward is a sum backward.
            let mut d_up = vec![C::new(0.0, 0.0); b * d];
            for bi in 0..b {
                for i in 0..n {
                    let row = (bi * n + i) * d;
                    for j in 0..d {
                        d_up[bi * d + j] += dh[bd][row + j];
                    }
                }
            }

            let mut slice = vec![C::new(0.0, 0.0); b * D_MERGE];
            for bi in 0..b {
                let src = bi * SUM_WIDTH + bd * D_MERGE;
                for k in 0..D_MERGE {
                    slice[bi * D_MERGE + k] = soft_shrink(mix_pre[src + k], SHRINK_LAMBDA);
                }
            }
            gemm_grad_b(&slice, &d_up, grads.sum_up_mut(bd), b, D_MERGE, d);

            let mut d_slice = vec![C::new(0.0, 0.0); b * D_MERGE];
            gemm_grad_a(&d_up, p.sum_up(bd), &mut d_slice, b, D_MERGE, d);
            for bi in 0..b {
                let dst = bi * SUM_WIDTH + bd * D_MERGE;
                for k in 0..D_MERGE {
                    d_mix_pre[dst + k] = soft_shrink_grad(
                        mix_pre[dst + k],
                        d_slice[bi * D_MERGE + k],
                        SHRINK_LAMBDA,
                    );
                }
            }
        }

        // 2'. the shared mixer
        let cat = &cache.t_cat[ti];
        gemm_grad_b(cat, &d_mix_pre, grads.sum_mix_mut(), b, SUM_WIDTH, SUM_WIDTH);
        let mut d_cat = vec![C::new(0.0, 0.0); b * SUM_WIDTH];
        gemm_grad_a(&d_mix_pre, p.sum_mix(), &mut d_cat, b, SUM_WIDTH, SUM_WIDTH);

        // 1'. project down and un-pool
        for bd in 0..NBANDS {
            if !cache.active[bd] {
                continue;
            }
            let (n, d) = (bands.counts[bd], D[bd]);
            let inv = 1.0 as Real / n as Real;

            let mut d_proj = vec![C::new(0.0, 0.0); b * D_MERGE];
            for bi in 0..b {
                let src = bi * SUM_WIDTH + bd * D_MERGE;
                d_proj[bi * D_MERGE..(bi + 1) * D_MERGE]
                    .copy_from_slice(&d_cat[src..src + D_MERGE]);
            }

            let mean = &cache.t_mean[s_idx(stage, sb, bd)];
            gemm_grad_b(mean, &d_proj, grads.sum_down_mut(bd), b, d, D_MERGE);
            let mut d_mean = vec![C::new(0.0, 0.0); b * d];
            gemm_grad_a(&d_proj, p.sum_down(bd), &mut d_mean, b, d, D_MERGE);

            for bi in 0..b {
                for i in 0..n {
                    let row = (bi * n + i) * d;
                    for j in 0..d {
                        dh[bd][row + j] += d_mean[bi * d + j] * inv;
                    }
                }
            }
        }
    }

    fn s_block_forward(
        &self,
        p: &SpectralParams,
        cache: &mut Cache,
        h: &mut [C],
        stage: usize,
        sb: usize,
        bd: usize,
    ) {
        let bands = &self.bands;
        let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
        let rows = cache.b * n;
        let si = s_idx(stage, sb, bd);
        let ci = sb * NBANDS + bd;

        let h_in = h.to_vec();
        let mut y_mm = vec![C::new(0.0, 0.0); rows * d];
        gemm(h, p.r_mat(sb, bd), &mut y_mm, rows, d, d);

        // Neighbour mixing: the GEMM mixes channels within a coefficient, this
        // mixes coefficients within a channel. Two 3-tap passes (k1 then k2)
        // give a 3x3 effective stencil for 6*d parameters.
        let nbrs = &bands.neighbours[off..off + n];
        let sa = p.stencil_a(sb, bd);
        let sb_t = p.stencil_b(sb, bd);
        let mut y_sa = vec![C::new(0.0, 0.0); rows * d];
        stencil_pass(&y_mm, &mut y_sa, nbrs, sa, cache.b, n, d, 0);
        let mut y_st = vec![C::new(0.0, 0.0); rows * d];
        stencil_pass(&y_sa, &mut y_st, nbrs, sb_t, cache.b, n, d, 2);

        cache.diag.y_mm_sq += norm_sq(&y_mm);
        cache.diag.stencil_delta_sq += y_st
            .iter()
            .zip(y_mm.iter())
            .map(|(a, b)| {
                let d = *a - *b;
                (d.re as f64) * (d.re as f64) + (d.im as f64) * (d.im as f64)
            })
            .sum::<f64>();

        let fa = p.filt_a(sb, bd);
        let fb = p.filt_b(sb, bd);
        let gamma = &cache.gamma[ci];
        let beta = &cache.beta[ci];

        let mut y_filt = vec![C::new(0.0, 0.0); rows * d];
        let mut y_film = vec![C::new(0.0, 0.0); rows * d];

        for bi in 0..cache.b {
            for i in 0..n {
                let k1 = bands.k1_of_pos[off + i];
                let k2 = bands.k2_of_pos[off + i];
                let row = (bi * n + i) * d;
                for j in 0..d {
                    let f = fa[k1 * d + j] * fb[k2 * d + j];
                    let yf = y_st[row + j] * f;
                    y_filt[row + j] = yf;
                    let yl = yf * gamma[j] + beta[j];
                    y_film[row + j] = yl;
                    h[row + j] = h[row + j] + soft_shrink(yl, SHRINK_LAMBDA);
                }
            }
        }

        cache.s_h_in[si] = h_in;
        cache.s_y_mm[si] = y_mm;
        cache.s_y_sa[si] = y_sa;
        cache.s_y_st[si] = y_st;
        cache.s_y_filt[si] = y_filt;
        cache.s_y_film[si] = y_film;
    }

    fn r_block_forward(
        &self,
        p: &SpectralParams,
        cache: &mut Cache,
        h: &mut [Vec<C>],
        stage: usize,
    ) {
        let bands = &self.bands;
        let b = cache.b;
        let ng = cache.ng;
        let plan = self.plan_for(ng);
        let ngh = plan.nh;
        let cells = ng * ngh;
        let pixels = ng * ng;
        let active = cache.active;

        // Reduced-resolution round trips are exact for band-limited content
        // once these two factors undo the resampling; see the module docs.
        let scale_down = (ng as Real / N as Real).powi(2);
        let scale_up = (N as Real / ng as Real).powi(2);

        // 1. project every band down to the common merge width
        let mut m_band: Vec<Vec<C>> = vec![Vec::new(); NBANDS];
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, d) = (bands.counts[bd], D[bd]);
            let rows = b * n;
            let mut m = vec![C::new(0.0, 0.0); rows * D_MERGE];
            gemm(&h[bd], p.proj_down(bd), &mut m, rows, d, D_MERGE);
            cache.r_h_in[r_idx(stage, bd)] = h[bd].clone();
            m_band[bd] = m;
        }

        // 2. scatter into one grid; frozen and inactive coefficients stay zero
        let mut grid = vec![C::new(0.0, 0.0); b * D_MERGE * cells];
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, off) = (bands.counts[bd], bands.offsets[bd]);
            for i in 0..n {
                let Some((r, c)) = grid_pos(bands.k1_of_pos[off + i], bands.k2_of_pos[off + i], ng)
                else {
                    continue;
                };
                let cell = r * ngh + c;
                for bi in 0..b {
                    let src = (bi * n + i) * D_MERGE;
                    for ch in 0..D_MERGE {
                        grid[(bi * D_MERGE + ch) * cells + cell] += m_band[bd][src + ch];
                    }
                }
            }
        }

        // 3. inverse FFT per channel: frequency -> pixels
        let mut sp_in = vec![0.0 as Real; b * pixels * D_MERGE];
        for bi in 0..b {
            for ch in 0..D_MERGE {
                let img = plan.irfft2(&grid[(bi * D_MERGE + ch) * cells..(bi * D_MERGE + ch + 1) * cells]);
                for (pix, v) in img.iter().enumerate() {
                    sp_in[(bi * pixels + pix) * D_MERGE + ch] = *v * scale_down;
                }
            }
        }

        // 4. nonlinearity in pixel space: this is where bands talk to each
        //    other, and where new frequency content is born
        let w1 = p.rmlp_w1();
        let b1 = p.rmlp_b1();
        let w2 = p.rmlp_w2();
        let b2 = p.rmlp_b2();
        let mut hidden = vec![0.0 as Real; b * pixels * RMLP_HIDDEN];
        let mut sp_out = vec![0.0 as Real; b * pixels * D_MERGE];
        for r in 0..b * pixels {
            let xin = &sp_in[r * D_MERGE..(r + 1) * D_MERGE];
            let hid = &mut hidden[r * RMLP_HIDDEN..(r + 1) * RMLP_HIDDEN];
            hid.copy_from_slice(b1);
            for (ch, &x) in xin.iter().enumerate() {
                let row = &w1[ch * RMLP_HIDDEN..(ch + 1) * RMLP_HIDDEN];
                for (o, &w) in hid.iter_mut().zip(row.iter()) {
                    *o += x * w;
                }
            }
            for v in hid.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let outr = &mut sp_out[r * D_MERGE..(r + 1) * D_MERGE];
            outr.copy_from_slice(b2);
            for (q, &x) in hid.iter().enumerate() {
                if x == 0.0 {
                    continue;
                }
                let row = &w2[q * D_MERGE..(q + 1) * D_MERGE];
                for (o, &w) in outr.iter_mut().zip(row.iter()) {
                    *o += x * w;
                }
            }
        }

        // 5. forward FFT per channel: pixels -> frequency
        let mut grid2 = vec![C::new(0.0, 0.0); b * D_MERGE * cells];
        let mut chan = vec![0.0 as Real; pixels];
        for bi in 0..b {
            for ch in 0..D_MERGE {
                for pix in 0..pixels {
                    chan[pix] = sp_out[(bi * pixels + pix) * D_MERGE + ch];
                }
                let spec = plan.rfft2(&chan);
                let dst = (bi * D_MERGE + ch) * cells;
                for (cell, v) in spec.iter().enumerate() {
                    grid2[dst + cell] = *v * scale_up;
                }
            }
        }

        // 6. gather back per band and project up
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
            let rows = b * n;
            let mut g = vec![C::new(0.0, 0.0); rows * D_MERGE];
            for i in 0..n {
                let Some((r, c)) = grid_pos(bands.k1_of_pos[off + i], bands.k2_of_pos[off + i], ng)
                else {
                    continue;
                };
                let cell = r * ngh + c;
                for bi in 0..b {
                    let dst = (bi * n + i) * D_MERGE;
                    for ch in 0..D_MERGE {
                        g[dst + ch] = grid2[(bi * D_MERGE + ch) * cells + cell];
                    }
                }
            }
            gemm_acc(&g, p.proj_up(bd), &mut h[bd], rows, D_MERGE, d);
            cache.r_gathered[r_idx(stage, bd)] = g;
        }

        cache.r_sp_in[stage] = sp_in;
        cache.r_hidden[stage] = hidden;
    }

    // -----------------------------------------------------------------
    // Backward
    // -----------------------------------------------------------------

    /// Accumulate parameter gradients into `grads` given `dL/dv_u` and
    /// `dL/dv_phi` (both `b * NCOEF`, band-major).
    pub fn backward(
        &self,
        p: &SpectralParams,
        cache: &Cache,
        d_v_u: &[Real],
        d_v_phi: &[Real],
        grads: &mut SpectralParams,
    ) {
        let bands = &self.bands;
        let b = cache.b;
        let active = cache.active;
        if !active.iter().any(|&a| a) {
            return;
        }

        let mut dh: Vec<Vec<C>> = vec![Vec::new(); NBANDS];
        let mut d_gamma: Vec<Vec<Real>> = (0..N_SBLOCKS * NBANDS)
            .map(|i| vec![0.0 as Real; D[i % NBANDS]])
            .collect();
        let mut d_beta: Vec<Vec<C>> = (0..N_SBLOCKS * NBANDS)
            .map(|i| vec![C::new(0.0, 0.0); D[i % NBANDS]])
            .collect();

        // -- C5: head -------------------------------------------------------
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
            let hw = p.head_w(bd);
            let mut g_hw = vec![0.0 as Real; 2 * d * 2];
            let mut g_hb = [0.0 as Real; 2];
            let mut dhb = vec![C::new(0.0, 0.0); b * n * d];
            let hf = &cache.h_final[bd];
            for bi in 0..b {
                for i in 0..n {
                    let g0 = d_v_u[bi * NCOEF + off + i];
                    let g1 = d_v_phi[bi * NCOEF + off + i];
                    if g0 == 0.0 && g1 == 0.0 {
                        continue;
                    }
                    g_hb[0] += g0;
                    g_hb[1] += g1;
                    let row = (bi * n + i) * d;
                    for j in 0..d {
                        let z = hf[row + j];
                        g_hw[j * 2] += z.re * g0;
                        g_hw[j * 2 + 1] += z.re * g1;
                        g_hw[(d + j) * 2] += z.im * g0;
                        g_hw[(d + j) * 2 + 1] += z.im * g1;
                        dhb[row + j] = Complex::new(
                            g0 * hw[j * 2] + g1 * hw[j * 2 + 1],
                            g0 * hw[(d + j) * 2] + g1 * hw[(d + j) * 2 + 1],
                        );
                    }
                }
            }
            let dst = grads.head_w_mut(bd);
            for k in 0..g_hw.len() {
                dst[k] += g_hw[k];
            }
            let dst = grads.head_b_mut(bd);
            dst[0] += g_hb[0];
            dst[1] += g_hb[1];
            dh[bd] = dhb;
        }

        // -- C4: stages, in reverse ----------------------------------------
        for stage in (0..N_STAGES).rev() {
            self.r_block_backward(p, cache, stage, &mut dh, grads);
            for sb in (0..N_SBLOCKS).rev() {
                self.summary_backward(p, cache, stage, sb, &mut dh, grads);
                for bd in 0..NBANDS {
                    if !active[bd] {
                        continue;
                    }
                    self.s_block_backward(
                        p,
                        cache,
                        stage,
                        sb,
                        bd,
                        &mut dh[bd],
                        &mut d_gamma[sb * NBANDS + bd],
                        &mut d_beta[sb * NBANDS + bd],
                        grads,
                    );
                }
            }
        }

        // -- C2: lift -------------------------------------------------------
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
            let mut g_w = vec![C::new(0.0, 0.0); d];
            let mut g_pos = vec![C::new(0.0, 0.0); n * d];
            for bi in 0..b {
                for i in 0..n {
                    let c = cache.c_lift[bi * NCOEF + off + i];
                    let row = (bi * n + i) * d;
                    for j in 0..d {
                        let g = dh[bd][row + j];
                        // d/dw of (c * w): conj(c) * g
                        g_w[j] += Complex::new(
                            c.re * g.re + c.im * g.im,
                            c.re * g.im - c.im * g.re,
                        );
                        g_pos[i * d + j] += g;
                    }
                }
            }
            let dst = grads.w_in_mut(bd);
            for j in 0..d {
                dst[j] += g_w[j];
            }
            let dst = grads.pos_mut(bd);
            for k in 0..n * d {
                dst[k] += g_pos[k];
            }
        }

        // -- C3: time conditioning ------------------------------------------
        self.time_conditioning_backward(p, cache, &d_gamma, &d_beta, grads);
    }

    #[allow(clippy::too_many_arguments)]
    fn s_block_backward(
        &self,
        p: &SpectralParams,
        cache: &Cache,
        stage: usize,
        sb: usize,
        bd: usize,
        dh: &mut [C],
        d_gamma: &mut [Real],
        d_beta: &mut [C],
        grads: &mut SpectralParams,
    ) {
        let bands = &self.bands;
        let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
        let rows = cache.b * n;
        let si = s_idx(stage, sb, bd);
        let ci = sb * NBANDS + bd;

        let y_film = &cache.s_y_film[si];
        let y_filt = &cache.s_y_filt[si];
        let y_mm = &cache.s_y_mm[si];
        let y_sa = &cache.s_y_sa[si];
        let y_st = &cache.s_y_st[si];
        let h_in = &cache.s_h_in[si];
        let gamma = &cache.gamma[ci];
        let fa = p.filt_a(sb, bd);
        let fb = p.filt_b(sb, bd);

        let mut d_y_st = vec![C::new(0.0, 0.0); rows * d];
        let mut g_fa = vec![C::new(0.0, 0.0); N * d];
        let mut g_fb = vec![C::new(0.0, 0.0); NH * d];

        for bi in 0..cache.b {
            for i in 0..n {
                let k1 = bands.k1_of_pos[off + i];
                let k2 = bands.k2_of_pos[off + i];
                let row = (bi * n + i) * d;
                for j in 0..d {
                    let g_out = dh[row + j];
                    let d_yl = soft_shrink_grad(y_film[row + j], g_out, SHRINK_LAMBDA);
                    if d_yl.re == 0.0 && d_yl.im == 0.0 {
                        continue;
                    }

                    // FiLM
                    let yf = y_filt[row + j];
                    d_gamma[j] += d_yl.re * yf.re + d_yl.im * yf.im;
                    d_beta[j] += d_yl;
                    let d_yf = d_yl * gamma[j];

                    // separable filter
                    let a = fa[k1 * d + j];
                    let bb = fb[k2 * d + j];
                    let f = a * bb;
                    let ym = y_st[row + j];
                    d_y_st[row + j] = Complex::new(
                        d_yf.re * f.re + d_yf.im * f.im,
                        d_yf.im * f.re - d_yf.re * f.im,
                    );
                    let d_f = Complex::new(
                        ym.re * d_yf.re + ym.im * d_yf.im,
                        ym.re * d_yf.im - ym.im * d_yf.re,
                    );
                    g_fa[k1 * d + j] += Complex::new(
                        d_f.re * bb.re + d_f.im * bb.im,
                        d_f.im * bb.re - d_f.re * bb.im,
                    );
                    g_fb[k2 * d + j] += Complex::new(
                        d_f.re * a.re + d_f.im * a.im,
                        d_f.im * a.re - d_f.re * a.im,
                    );
                }
            }
        }

        // Stencil, in reverse: k2 pass then k1 pass.
        let nbrs = &bands.neighbours[off..off + n];
        let mut d_y_sa = vec![C::new(0.0, 0.0); rows * d];
        stencil_pass_backward(
            y_sa,
            &d_y_st,
            &mut d_y_sa,
            grads.stencil_b_mut(sb, bd),
            nbrs,
            p.stencil_b(sb, bd),
            cache.b,
            n,
            d,
            2,
        );
        let mut d_y_mm = vec![C::new(0.0, 0.0); rows * d];
        stencil_pass_backward(
            y_mm,
            &d_y_sa,
            &mut d_y_mm,
            grads.stencil_a_mut(sb, bd),
            nbrs,
            p.stencil_a(sb, bd),
            cache.b,
            n,
            d,
            0,
        );

        gemm_grad_b(h_in, &d_y_mm, grads.r_mat_mut(sb, bd), rows, d, d);
        gemm_grad_a(&d_y_mm, p.r_mat(sb, bd), dh, rows, d, d);

        let dst = grads.filt_a_mut(sb, bd);
        for k in 0..g_fa.len() {
            dst[k] += g_fa[k];
        }
        let dst = grads.filt_b_mut(sb, bd);
        for k in 0..g_fb.len() {
            dst[k] += g_fb[k];
        }
    }

    fn r_block_backward(
        &self,
        p: &SpectralParams,
        cache: &Cache,
        stage: usize,
        dh: &mut [Vec<C>],
        grads: &mut SpectralParams,
    ) {
        let bands = &self.bands;
        let b = cache.b;
        let ng = cache.ng;
        let plan = self.plan_for(ng);
        let ngh = plan.nh;
        let cells = ng * ngh;
        let pixels = ng * ng;
        let active = cache.active;
        let scale_down = (ng as Real / N as Real).powi(2);
        let scale_up = (N as Real / ng as Real).powi(2);

        // 6. project up
        let mut d_grid2 = vec![C::new(0.0, 0.0); b * D_MERGE * cells];
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
            let rows = b * n;
            let g = &cache.r_gathered[r_idx(stage, bd)];
            gemm_grad_b(g, &dh[bd], grads.proj_up_mut(bd), rows, D_MERGE, d);
            let mut d_g = vec![C::new(0.0, 0.0); rows * D_MERGE];
            gemm_grad_a(&dh[bd], p.proj_up(bd), &mut d_g, rows, D_MERGE, d);
            for i in 0..n {
                let Some((r, c)) = grid_pos(bands.k1_of_pos[off + i], bands.k2_of_pos[off + i], ng)
                else {
                    continue;
                };
                let cell = r * ngh + c;
                for bi in 0..b {
                    let src = (bi * n + i) * D_MERGE;
                    for ch in 0..D_MERGE {
                        d_grid2[(bi * D_MERGE + ch) * cells + cell] += d_g[src + ch];
                    }
                }
            }
        }

        // 5. adjoint of the forward FFT
        let mut d_sp_out = vec![0.0 as Real; b * pixels * D_MERGE];
        for bi in 0..b {
            for ch in 0..D_MERGE {
                let base = (bi * D_MERGE + ch) * cells;
                let scaled: Vec<C> =
                    d_grid2[base..base + cells].iter().map(|v| *v * scale_up).collect();
                let d_chan = plan.rfft2_adjoint(&scaled);
                for (pix, v) in d_chan.iter().enumerate() {
                    d_sp_out[(bi * pixels + pix) * D_MERGE + ch] = *v;
                }
            }
        }

        // 4. pixel-space MLP
        let w1 = p.rmlp_w1();
        let w2 = p.rmlp_w2();
        let sp_in = &cache.r_sp_in[stage];
        let hidden = &cache.r_hidden[stage];
        let mut g_w1 = vec![0.0 as Real; D_MERGE * RMLP_HIDDEN];
        let mut g_b1 = vec![0.0 as Real; RMLP_HIDDEN];
        let mut g_w2 = vec![0.0 as Real; RMLP_HIDDEN * D_MERGE];
        let mut g_b2 = vec![0.0 as Real; D_MERGE];
        let mut d_sp_in = vec![0.0 as Real; b * pixels * D_MERGE];
        let mut d_hid = vec![0.0 as Real; RMLP_HIDDEN];

        for r in 0..b * pixels {
            let go = &d_sp_out[r * D_MERGE..(r + 1) * D_MERGE];
            let hid = &hidden[r * RMLP_HIDDEN..(r + 1) * RMLP_HIDDEN];
            for (ch, &g) in go.iter().enumerate() {
                g_b2[ch] += g;
            }
            for (q, &x) in hid.iter().enumerate() {
                let row2 = &w2[q * D_MERGE..(q + 1) * D_MERGE];
                let mut acc = 0.0 as Real;
                for (ch, &g) in go.iter().enumerate() {
                    if x != 0.0 {
                        g_w2[q * D_MERGE + ch] += x * g;
                    }
                    acc += row2[ch] * g;
                }
                // ReLU: hidden is the post-activation value, so a stored zero
                // is exactly the clipped branch.
                d_hid[q] = if x > 0.0 { acc } else { 0.0 };
            }
            for (q, &g) in d_hid.iter().enumerate() {
                g_b1[q] += g;
            }
            let xin = &sp_in[r * D_MERGE..(r + 1) * D_MERGE];
            let din = &mut d_sp_in[r * D_MERGE..(r + 1) * D_MERGE];
            for ch in 0..D_MERGE {
                let row1 = &w1[ch * RMLP_HIDDEN..(ch + 1) * RMLP_HIDDEN];
                let x = xin[ch];
                let mut acc = 0.0 as Real;
                for (q, &g) in d_hid.iter().enumerate() {
                    g_w1[ch * RMLP_HIDDEN + q] += x * g;
                    acc += row1[q] * g;
                }
                din[ch] = acc;
            }
        }

        for k in 0..g_w1.len() {
            grads.rmlp_w1_mut()[k] += g_w1[k];
        }
        for k in 0..g_b1.len() {
            grads.rmlp_b1_mut()[k] += g_b1[k];
        }
        for k in 0..g_w2.len() {
            grads.rmlp_w2_mut()[k] += g_w2[k];
        }
        for k in 0..g_b2.len() {
            grads.rmlp_b2_mut()[k] += g_b2[k];
        }

        // 3. adjoint of the inverse FFT
        let mut d_grid = vec![C::new(0.0, 0.0); b * D_MERGE * cells];
        let mut chan = vec![0.0 as Real; pixels];
        for bi in 0..b {
            for ch in 0..D_MERGE {
                for pix in 0..pixels {
                    chan[pix] = d_sp_in[(bi * pixels + pix) * D_MERGE + ch] * scale_down;
                }
                let d_cells = plan.irfft2_adjoint(&chan);
                let dst = (bi * D_MERGE + ch) * cells;
                d_grid[dst..dst + cells].copy_from_slice(&d_cells);
            }
        }

        // 2 & 1. gather per band and project down
        for bd in 0..NBANDS {
            if !active[bd] {
                continue;
            }
            let (n, d, off) = (bands.counts[bd], D[bd], bands.offsets[bd]);
            let rows = b * n;
            let mut d_m = vec![C::new(0.0, 0.0); rows * D_MERGE];
            for i in 0..n {
                let Some((r, c)) = grid_pos(bands.k1_of_pos[off + i], bands.k2_of_pos[off + i], ng)
                else {
                    continue;
                };
                let cell = r * ngh + c;
                for bi in 0..b {
                    let dst = (bi * n + i) * D_MERGE;
                    for ch in 0..D_MERGE {
                        d_m[dst + ch] = d_grid[(bi * D_MERGE + ch) * cells + cell];
                    }
                }
            }
            let h_in = &cache.r_h_in[r_idx(stage, bd)];
            gemm_grad_b(h_in, &d_m, grads.proj_down_mut(bd), rows, d, D_MERGE);
            gemm_grad_a(&d_m, p.proj_down(bd), &mut dh[bd], rows, d, D_MERGE);
        }
    }

    fn time_conditioning_backward(
        &self,
        p: &SpectralParams,
        cache: &Cache,
        d_gamma: &[Vec<Real>],
        d_beta: &[Vec<C>],
        grads: &mut SpectralParams,
    ) {
        let mut d_trunk_out = vec![0.0 as Real; TIME_HIDDEN];

        for sb in 0..N_SBLOCKS {
            for bd in 0..NBANDS {
                if !cache.active[bd] {
                    continue;
                }
                let d = D[bd];
                let idx = sb * NBANDS + bd;
                let out_w = 3 * d;
                let mut dout = vec![0.0 as Real; out_w];
                for j in 0..d {
                    dout[j] = d_gamma[idx][j];
                    dout[d + j] = d_beta[idx][j].re;
                    dout[2 * d + j] = d_beta[idx][j].im;
                }

                let fw = p.film_w(sb, bd);
                for i in 0..TIME_HIDDEN {
                    let row = &fw[i * out_w..(i + 1) * out_w];
                    let mut acc = 0.0 as Real;
                    for (j, &g) in dout.iter().enumerate() {
                        acc += row[j] * g;
                    }
                    d_trunk_out[i] += acc;
                }

                let x = cache.trunk_out.clone();
                let g_fw = grads.film_w_mut(sb, bd);
                for i in 0..TIME_HIDDEN {
                    let xi = x[i];
                    if xi == 0.0 {
                        continue;
                    }
                    for (j, &g) in dout.iter().enumerate() {
                        g_fw[i * out_w + j] += xi * g;
                    }
                }
                let g_fb = grads.film_b_mut(sb, bd);
                for (j, &g) in dout.iter().enumerate() {
                    g_fb[j] += g;
                }
            }
        }

        let mut d_pre = vec![0.0 as Real; TIME_HIDDEN];
        for j in 0..TIME_HIDDEN {
            d_pre[j] = d_trunk_out[j] * silu_grad(cache.trunk_pre[j]);
        }
        let g_tb = grads.trunk_b_mut();
        for j in 0..TIME_HIDDEN {
            g_tb[j] += d_pre[j];
        }
        let feats = cache.feats.clone();
        let g_tw = grads.trunk_w_mut();
        for i in 0..TIME_FEAT {
            let f = feats[i];
            if f == 0.0 {
                continue;
            }
            for j in 0..TIME_HIDDEN {
                g_tw[i * TIME_HIDDEN + j] += f * d_pre[j];
            }
        }
    }
}
