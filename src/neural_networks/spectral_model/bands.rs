//! Radial band partition of the 32x17 half spectrum, plus the band-local time
//! schedule. Blueprint sections A2, B7 and B8.
//!
//! Everything in this module is pure index arithmetic on a fixed grid, so it is
//! computed once at start-up and shared by training and sampling.

use crate::neural_networks::utils::dtype::Real;
use serde::{Deserialize, Serialize};

/// Side length of the spectral grid (power of two, required by radix-2 FFT).
pub const N: usize = 32;
/// Retained horizontal frequencies, `N/2 + 1`.
pub const NH: usize = N / 2 + 1;
/// Stored coefficients per image.
pub const NCOEF: usize = N * NH;

/// Number of radial bands.
pub const NBANDS: usize = 4;

/// Radial cut points; band `b` holds `RADII[b] <= rho < RADII[b+1]`.
pub const RADII: [f32; NBANDS + 1] = [0.0, 4.0, 8.0, 13.0, f32::INFINITY];

/// Channel width per band (tapered: high bands carry less information).
///
/// Band 3 holds `RADII[3]..inf`, i.e. every coefficient with `rho >= 13` —
/// half of all 544 coefficients in the 32x17 half spectrum, and all of the
/// fine edge/stroke detail that separates a sharp digit from a blurry one.
/// The original tapering gave it the fewest channels of any band, which
/// starves exactly the content that needs the most capacity; widened here
/// (and band 2, the second most content-heavy band, nudged up too) per the
/// blueprint's own guidance that more per-band mixing is the cheap lever to
/// pull for poor output quality. Bands 0-1 are untouched: they already carry
/// adequate quality and hold the least content, so widening them buys the
/// least.
pub const D: [usize; NBANDS] = [64, 48, 40, 32];
/// Common width inside the R-block.
pub const D_MERGE: usize = 16;

/// Duration of one band's motion window.
pub const WINDOW: Real = 0.4;
/// Offset between consecutive band windows.
pub const STRIDE: Real = 0.2;

/// Softening constant for the `asinh` magnitude coordinate.
pub const EPS_MAG: Real = 0.1;

/// Upper bound on the magnitude coordinate `u`.
///
/// `u = asinh(|c|/eps)` is non-negative by construction, and during training
/// `u` is a convex combination of two such values, so the whole valid range is
/// `[0, U_MAX]` — real whitened coefficients sit near `asinh(10) ~ 3`, and
/// `U_MAX = 12` corresponds to a magnitude of 8135, far outside anything the
/// data produces.
///
/// Sampling has no such guarantee: an integrator following an imperfect
/// velocity field can walk `u` anywhere, and `sinh` overflows `f32` not far
/// beyond here, turning one bad step into an all-`NaN` image. Projecting back
/// onto the valid range is a correction, not a fudge — `u < 0` does not
/// describe any complex number.
pub const U_MAX: Real = 12.0;

#[inline]
pub fn clamp_u(u: Real) -> Real {
    if u.is_finite() {
        u.clamp(0.0, U_MAX)
    } else {
        0.0
    }
}

/// Band assignment, band-major permutation, and the per-coefficient metadata
/// derived from them.
#[derive(Debug, Clone)]
pub struct BandTable {
    /// `band[k]` for `k` in the natural `(k1, k2)` row-major layout.
    pub band: Vec<u8>,
    /// `order[p]` is the natural index stored at band-major position `p`.
    pub order: Vec<usize>,
    /// `inv_order[k]` is the band-major position of natural index `k`.
    pub inv_order: Vec<usize>,
    /// `offsets[b]..offsets[b+1]` is band `b`'s contiguous slice.
    pub offsets: [usize; NBANDS + 1],
    /// Coefficients per band.
    pub counts: [usize; NBANDS],
    /// Band of each band-major position (i.e. `band[order[p]]`).
    pub band_of_pos: Vec<u8>,
    /// `(k1, k2)` of each band-major position, `k1` in `0..N`, `k2` in `0..NH`.
    pub k1_of_pos: Vec<usize>,
    pub k2_of_pos: Vec<usize>,
    /// Band-*local* index of each position's neighbour along `-k1, +k1, -k2,
    /// +k2`, or `usize::MAX` when the neighbour falls outside the band.
    ///
    /// Adjacent modes are correlated in natural images, so a short stencil over
    /// these gives local mixing for a handful of parameters. `k1` is a circular
    /// frequency axis and wraps; `k2` indexes a half spectrum and does not.
    /// Neighbours in a *different* band are dropped — cross-band mixing is the
    /// summary tokens' and the R-block's job, and the bands have different
    /// channel widths anyway.
    pub neighbours: Vec<[usize; 4]>,
}

/// Sentinel for "this neighbour is not in the same band".
pub const NO_NEIGHBOUR: usize = usize::MAX;

impl Default for BandTable {
    fn default() -> Self {
        Self::new()
    }
}

impl BandTable {
    pub fn new() -> Self {
        let mut band = vec![0u8; NCOEF];
        for i in 0..N {
            for j in 0..NH {
                // Rows above N/2 hold negative vertical frequencies.
                let k1 = if i <= N / 2 { i as i32 } else { i as i32 - N as i32 };
                let k2 = j as i32;
                let rho = ((k1 * k1 + k2 * k2) as f32).sqrt();
                let mut b = (NBANDS - 1) as u8;
                for cand in 0..NBANDS {
                    if rho < RADII[cand + 1] {
                        b = cand as u8;
                        break;
                    }
                }
                band[i * NH + j] = b;
            }
        }

        // Stable sort by band keeps the natural order inside each band, which
        // makes the layout reproducible across runs.
        let mut order: Vec<usize> = (0..NCOEF).collect();
        order.sort_by_key(|&k| band[k]);

        let mut counts = [0usize; NBANDS];
        for &b in &band {
            counts[b as usize] += 1;
        }

        let mut offsets = [0usize; NBANDS + 1];
        for b in 0..NBANDS {
            offsets[b + 1] = offsets[b] + counts[b];
        }

        let mut inv_order = vec![0usize; NCOEF];
        let mut band_of_pos = vec![0u8; NCOEF];
        let mut k1_of_pos = vec![0usize; NCOEF];
        let mut k2_of_pos = vec![0usize; NCOEF];
        for (p, &k) in order.iter().enumerate() {
            inv_order[k] = p;
            band_of_pos[p] = band[k];
            k1_of_pos[p] = k / NH;
            k2_of_pos[p] = k % NH;
        }

        let mut neighbours = vec![[NO_NEIGHBOUR; 4]; NCOEF];
        for p in 0..NCOEF {
            let bd = band_of_pos[p];
            let (k1, k2) = (k1_of_pos[p], k2_of_pos[p]);
            let cand = [
                Some(((k1 + N - 1) % N, k2)),
                Some(((k1 + 1) % N, k2)),
                if k2 > 0 { Some((k1, k2 - 1)) } else { None },
                if k2 + 1 < NH { Some((k1, k2 + 1)) } else { None },
            ];
            for (slot, c) in cand.iter().enumerate() {
                if let Some((a, b)) = *c {
                    let q = inv_order[a * NH + b];
                    if band_of_pos[q] == bd {
                        neighbours[p][slot] = q - offsets[bd as usize];
                    }
                }
            }
        }

        Self {
            band,
            order,
            inv_order,
            offsets,
            counts,
            band_of_pos,
            k1_of_pos,
            k2_of_pos,
            neighbours,
        }
    }

    /// Coefficients in band `b`.
    #[inline]
    pub fn n_band(&self, b: usize) -> usize {
        self.counts[b]
    }

    /// Reorder a natural-layout array into band-major layout.
    pub fn permute<T: Copy>(&self, natural: &[T]) -> Vec<T> {
        self.order.iter().map(|&k| natural[k]).collect()
    }

    /// Inverse of [`BandTable::permute`].
    pub fn unpermute<T: Copy + Default>(&self, band_major: &[T]) -> Vec<T> {
        let mut out = vec![T::default(); band_major.len()];
        for (p, &k) in self.order.iter().enumerate() {
            out[k] = band_major[p];
        }
        out
    }
}

/// How a coefficient travels from the noise endpoint to the data endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowPath {
    /// The blueprint's path: magnitude on a straight line in `u`, phase along
    /// the shortest arc. No singularity anywhere, but at `tau = 0` the state is
    /// pure noise and a single MNIST coefficient's phase is marginally uniform,
    /// so `E[Delta phi | state] = 0` — the optimal early prediction is *no
    /// rotation*, and the flow has no drift to get started with.
    Polar,
    /// Straight line in the complex plane, `c_t = (1-tau) c0 + tau c1`.
    ///
    /// `E[c1 - c0 | c_t]` is informative from `tau = 0` onward, which is the
    /// property ordinary flow matching relies on and the polar path lacks.
    ///
    /// The velocity is expressed in the **rotating frame**: the head's two
    /// outputs are the radial and tangential components of `dc/dtau` measured
    /// against the current phase, i.e. `(v_r, v_t)` with
    /// `dc/dtau = e^{i phi_t} (v_r + i v_t)`.
    ///
    /// That framing matters. Writing the path in `(u, phi)` directly would give
    /// `dphi/dtau = Im(e^{-i phi} dc) / |c|`, which blows up whenever the
    /// straight line passes near the origin — and it does, often. The rotating
    /// frame drops the division entirely: both components stay bounded by
    /// `|c1| + |c0|`, while `v_t` remains exactly the rotational part, so the
    /// phase-preserving character of the design survives.
    ComplexLinear,
    /// The same straight line as [`FlowPath::ComplexLinear`], but the head
    /// predicts `dc/dtau` in **absolute** coordinates rather than in the
    /// state's rotating frame.
    ///
    /// This exists to isolate one question. Rotation is orthogonal, so
    /// equal-weighted L2 in the rotating frame is *identically* Cartesian L2 —
    /// `ComplexLinear` and `Cartesian` therefore have the same path, the same
    /// targets up to a rotation, and the same loss geometry. The only thing
    /// that differs is the head parameterisation: predicting relative to the
    /// state's own phase versus relative to a fixed axis.
    ///
    /// Comparing all three paths separates two claims that a two-way
    /// comparison confounds. If `Polar` < `Cartesian` ~= `ComplexLinear`, the
    /// phase-transport hypothesis was wrong and the rotating frame is
    /// cosmetic. If `ComplexLinear` > `Cartesian`, the head reparameterisation
    /// is doing real work.
    Cartesian,
}

impl Default for FlowPath {
    fn default() -> Self {
        Self::ComplexLinear
    }
}

impl FlowPath {
    /// Whether the head's second output is an angle (and so needs `wrap` and
    /// the energy weight) or a plain Cartesian rate.
    pub fn second_output_is_angular(&self) -> bool {
        matches!(self, FlowPath::Polar)
    }

    /// Short label for logs and filenames.
    pub fn name(&self) -> &'static str {
        match self {
            FlowPath::Polar => "polar",
            FlowPath::ComplexLinear => "complex",
            FlowPath::Cartesian => "cartesian",
        }
    }
}

/// Which bands the model handles and how their windows are laid out in time.
///
/// The staggered schedule is the blueprint's (B7). Turning it off gives every
/// band the window `[0, 1]`, i.e. ordinary rectified flow — that is the cheap
/// gate in build-order step 7, before the schedule is introduced.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Schedule {
    pub staggered: bool,
    /// Highest band the model represents; bands above this are never active
    /// and their coefficients stay at the prior.
    pub max_band: usize,
}

impl Default for Schedule {
    fn default() -> Self {
        Self { staggered: true, max_band: NBANDS - 1 }
    }
}

impl Schedule {
    pub fn full() -> Self {
        Self::default()
    }

    /// Low-band-only, unstaggered: the fast smoke test.
    pub fn warmup(max_band: usize) -> Self {
        Self { staggered: false, max_band }
    }

    /// Per-band local times for a global time `t`. Blueprint B7.
    pub fn tau(&self, t: Real) -> [Real; NBANDS] {
        let mut tau = [0.0 as Real; NBANDS];
        for (b, slot) in tau.iter_mut().enumerate() {
            *slot = if self.staggered {
                ((t - b as Real * STRIDE) / WINDOW).clamp(0.0, 1.0)
            } else {
                t.clamp(0.0, 1.0)
            };
        }
        tau
    }

    /// How far each band advances over `[t0, t1]`. Telescopes to exactly 1
    /// across a full sweep, which is what keeps the sampler's integration
    /// consistent with the training interpolation.
    pub fn dtau(&self, t0: Real, t1: Real) -> [Real; NBANDS] {
        let a = self.tau(t0);
        let b = self.tau(t1);
        let mut d = [0.0 as Real; NBANDS];
        for i in 0..NBANDS {
            d[i] = b[i] - a[i];
        }
        d
    }

    /// Bands that are strictly mid-flight and within `max_band`. Blueprint B8.
    pub fn active(&self, tau: &[Real; NBANDS]) -> [bool; NBANDS] {
        let mut m = [false; NBANDS];
        for b in 0..=self.max_band.min(NBANDS - 1) {
            m[b] = tau[b] > 0.0 && tau[b] < 1.0;
        }
        m
    }

    /// The last `t` at which anything is still moving.
    pub fn end_time(&self) -> Real {
        if self.staggered {
            self.max_band.min(NBANDS - 1) as Real * STRIDE + WINDOW
        } else {
            1.0
        }
    }
}

/// Highest band index that is currently active, if any.
#[inline]
pub fn highest_active_band(active: &[bool; NBANDS]) -> Option<usize> {
    (0..NBANDS).rev().find(|&b| active[b])
}

/// Grid side length sufficient to represent every coefficient up to band
/// `highest`, used to run the R-block round trip at reduced resolution.
///
/// Band `b` reaches at most `RADII[b+1]`, so any grid with Nyquist frequency
/// above that loses nothing.
pub fn roundtrip_grid_size(highest: usize) -> usize {
    match highest {
        0 => 8,
        1 => 16,
        _ => N,
    }
}

/// Wrap an angle into `(-pi, pi]`.
#[inline]
pub fn wrap(x: Real) -> Real {
    let pi = std::f64::consts::PI as Real;
    let two_pi = 2.0 * pi;
    let mut y = (x + pi) % two_pi;
    if y <= 0.0 {
        y += two_pi;
    }
    y - pi
}
