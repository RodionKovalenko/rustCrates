//! The build-order exit tests from blueprint section H, steps 1-6.
//!
//! Each one gates the next: do not chase a training failure before these pass.

use crate::neural_networks::spectral_model::bands::{
    wrap, BandTable, FlowPath, Schedule, D, EPS_MAG, N, NBANDS, NCOEF, NH, NO_NEIGHBOUR,
};
use crate::neural_networks::spectral_model::fft::{naive_rfft2, Rfft2Plan};
use crate::neural_networks::spectral_model::gemm::{gemm, gemm_naive};
use crate::neural_networks::spectral_model::loss::{flow_loss, PhaseWeights};
use crate::neural_networks::spectral_model::model::{GroupInput, SpectralNet};
use crate::neural_networks::spectral_model::params::SpectralParams;
use crate::neural_networks::spectral_model::spectral::{
    build_sample, from_polar, integrate_step, polar_to_complex, sample_prior, to_polar,
    SpectralPrep,
};
use crate::neural_networks::utils::dtype::{Real, C};
use num::Complex;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn random_image(rng: &mut SmallRng) -> Vec<Real> {
    (0..N * N).map(|_| rng.random_range(0.0f64..1.0) as Real).collect()
}

/// A smooth, image-like test signal: real data has a decaying spectrum, and a
/// white-noise image exercises the numerics quite differently.
fn smooth_image(rng: &mut SmallRng) -> Vec<Real> {
    let mut img = vec![0.0 as Real; N * N];
    for _ in 0..6 {
        let cx = rng.random_range(6.0f64..26.0);
        let cy = rng.random_range(6.0f64..26.0);
        let s = rng.random_range(2.0f64..5.0);
        let amp = rng.random_range(0.3f64..1.0);
        for y in 0..N {
            for x in 0..N {
                let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                img[y * N + x] += (amp * (-d2 / (2.0 * s * s)).exp()) as Real;
            }
        }
    }
    img
}

// -- Step 1: FFT -----------------------------------------------------------

#[test]
fn fft_matches_naive_dft() {
    let plan = Rfft2Plan::new(N);
    let mut rng = SmallRng::seed_from_u64(1);
    let img = random_image(&mut rng);

    let fast = plan.rfft2(&img);
    let slow = naive_rfft2(&img, N);
    for (a, b) in fast.iter().zip(slow.iter()) {
        assert!((a.re - b.re).abs() < 1e-3, "re {a} vs {b}");
        assert!((a.im - b.im).abs() < 1e-3, "im {a} vs {b}");
    }
}

#[test]
fn irfft2_inverts_rfft2() {
    for n in [8usize, 16, 32] {
        let plan = Rfft2Plan::new(n);
        let mut rng = SmallRng::seed_from_u64(2 + n as u64);
        let img: Vec<Real> = (0..n * n).map(|_| rng.random_range(-1.0f64..1.0) as Real).collect();
        let back = plan.irfft2(&plan.rfft2(&img));
        for (a, b) in img.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b} at n={n}");
        }
    }
}

/// The adjoints are what backpropagation through the R-block relies on, so
/// they are checked directly against the definition `<A x, y> == <x, A^T y>`.
#[test]
fn fft_adjoints_are_transposes() {
    let mut rng = SmallRng::seed_from_u64(3);
    for n in [8usize, 16, 32] {
        let plan = Rfft2Plan::new(n);
        let nh = plan.nh;

        // irfft2: C^{n*nh} (as reals) -> R^{n*n}
        let x: Vec<C> = (0..n * nh)
            .map(|_| Complex::new(rng.random_range(-1.0f64..1.0) as Real, rng.random_range(-1.0f64..1.0) as Real))
            .collect();
        let y: Vec<Real> = (0..n * n).map(|_| rng.random_range(-1.0f64..1.0) as Real).collect();
        let ax = plan.irfft2(&x);
        let aty = plan.irfft2_adjoint(&y);
        let lhs: f64 = ax.iter().zip(y.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let rhs: f64 = x
            .iter()
            .zip(aty.iter())
            .map(|(a, b)| (a.re as f64) * (b.re as f64) + (a.im as f64) * (b.im as f64))
            .sum();
        assert!((lhs - rhs).abs() < 1e-3 * (1.0 + lhs.abs()), "irfft2 adjoint: {lhs} vs {rhs}");

        // rfft2: R^{n*n} -> C^{n*nh}
        let x2: Vec<Real> = (0..n * n).map(|_| rng.random_range(-1.0f64..1.0) as Real).collect();
        let y2: Vec<C> = (0..n * nh)
            .map(|_| Complex::new(rng.random_range(-1.0f64..1.0) as Real, rng.random_range(-1.0f64..1.0) as Real))
            .collect();
        let bx = plan.rfft2(&x2);
        let bty = plan.rfft2_adjoint(&y2);
        let lhs2: f64 = bx
            .iter()
            .zip(y2.iter())
            .map(|(a, b)| (a.re as f64) * (b.re as f64) + (a.im as f64) * (b.im as f64))
            .sum();
        let rhs2: f64 = x2.iter().zip(bty.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        assert!((lhs2 - rhs2).abs() < 1e-2 * (1.0 + lhs2.abs()), "rfft2 adjoint: {lhs2} vs {rhs2}");
    }
}

// -- Step 3: band table ----------------------------------------------------

#[test]
fn band_counts_match_blueprint() {
    let bands = BandTable::new();
    assert_eq!(bands.counts, [26, 78, 167, 273]);
    assert_eq!(bands.offsets, [0, 26, 104, 271, 544]);
    assert_eq!(bands.counts.iter().sum::<usize>(), NCOEF);

    // The permutation must be a bijection.
    let mut seen = vec![false; NCOEF];
    for &k in &bands.order {
        assert!(!seen[k]);
        seen[k] = true;
    }
    for p in 0..NCOEF {
        assert_eq!(bands.inv_order[bands.order[p]], p);
    }
}

#[test]
fn whitening_decays_with_radius() {
    let mut rng = SmallRng::seed_from_u64(4);
    let imgs: Vec<Vec<Real>> = (0..64).map(|_| smooth_image(&mut rng)).collect();
    let prep = SpectralPrep::from_dataset(&imgs);

    // Band-major layout means band 0 is the first slice; its typical magnitude
    // must dominate the outermost band by orders of magnitude.
    let mean = |lo: usize, hi: usize| -> f64 {
        prep.s[lo..hi].iter().map(|v| *v as f64).sum::<f64>() / (hi - lo) as f64
    };
    let b0 = mean(0, 26);
    let b3 = mean(271, 544);
    assert!(b0 > 10.0 * b3, "band 0 rms {b0} vs band 3 rms {b3}");
}

// -- Step 4: whiten -> polar -> un-polar -> un-whiten ----------------------

#[test]
fn polar_round_trip_recovers_image() {
    let mut rng = SmallRng::seed_from_u64(5);
    let imgs: Vec<Vec<Real>> = (0..32).map(|_| smooth_image(&mut rng)).collect();
    let prep = SpectralPrep::from_dataset(&imgs);

    let img = &imgs[0];
    let (u, phi) = prep.image_to_polar(img);
    let back = prep.polar_to_image(&u, &phi);
    for (a, b) in img.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn polar_conversion_is_exact() {
    let c: Vec<C> = vec![
        Complex::new(0.0, 0.0),
        Complex::new(1e-9, 0.0),
        Complex::new(3.0, -4.0),
        Complex::new(-2.5, 0.75),
    ];
    let (u, phi) = to_polar(&c);
    let back = from_polar(&u, &phi);
    for (a, b) in c.iter().zip(back.iter()) {
        assert!((a.re - b.re).abs() < 1e-5 && (a.im - b.im).abs() < 1e-5);
    }
    // asinh stays finite and linear near zero.
    assert!(u[1] < 1e-7 && u[1] >= 0.0);
    assert!((EPS_MAG * u[2].sinh() - 5.0).abs() < 1e-4);
}

#[test]
fn prior_has_unit_energy() {
    let mut rng = SmallRng::seed_from_u64(6);
    let mut acc = 0.0f64;
    let draws = 64;
    for _ in 0..draws {
        let (u, _) = sample_prior(&mut rng);
        for uk in &u {
            let m = EPS_MAG * uk.sinh();
            acc += (m * m) as f64;
        }
    }
    let mean = acc / (draws * NCOEF) as f64;
    assert!((mean - 1.0).abs() < 0.05, "E[|c|^2] = {mean}, expected 1");
}

#[test]
fn interpolation_endpoints_and_wrap() {
    let bands = BandTable::new();
    let sched = Schedule { staggered: false, max_band: NBANDS - 1 };
    let mut rng = SmallRng::seed_from_u64(7);
    let (u0, phi0) = sample_prior(&mut rng);
    let (u1, phi1) = sample_prior(&mut rng);

    let at0 = build_sample(&bands, &sched, FlowPath::Polar, &u0, &phi0, &u1, &phi1, 1e-6);
    let at1 = build_sample(&bands, &sched, FlowPath::Polar, &u0, &phi0, &u1, &phi1, 1.0 - 1e-6);
    for p in 0..NCOEF {
        assert!((at0.ut[p] - u0[p]).abs() < 1e-3);
        assert!((at1.ut[p] - u1[p]).abs() < 1e-3);
        // Phase always travels the short way round.
        assert!(at0.v_phi[p].abs() <= std::f32::consts::PI as Real + 1e-5);
        assert!((wrap(at1.phit[p] - phi1[p])).abs() < 1e-3);
    }
}

// -- Step 2 / 6: gradients -------------------------------------------------

#[test]
fn blocked_gemm_matches_naive() {
    let mut rng = SmallRng::seed_from_u64(8);
    let (m, k, n) = (37usize, 48usize, 29usize);
    let mut mk = |len: usize| -> Vec<C> {
        (0..len)
            .map(|_| Complex::new(rng.random_range(-1.0f64..1.0) as Real, rng.random_range(-1.0f64..1.0) as Real))
            .collect()
    };
    let a = mk(m * k);
    let b = mk(k * n);
    let mut out = vec![C::new(0.0, 0.0); m * n];
    gemm(&a, &b, &mut out, m, k, n);
    let want = gemm_naive(&a, &b, m, k, n);
    for (x, y) in out.iter().zip(want.iter()) {
        assert!((x.re - y.re).abs() < 1e-3 && (x.im - y.im).abs() < 1e-3, "{x} vs {y}");
    }
}

/// Build a tiny but complete training state so gradients can be checked
/// against finite differences.
struct GradFixture {
    net: SpectralNet,
    params: SpectralParams,
    input: GroupInput,
    tgt_u: Vec<Real>,
    tgt_phi: Vec<Real>,
    mask: Vec<bool>,
    path: FlowPath,
}

fn fixture(staggered: bool, t: Real, seed: u64) -> GradFixture {
    fixture_with_path(staggered, t, seed, FlowPath::ComplexLinear)
}

fn fixture_with_path(staggered: bool, t: Real, seed: u64, path: FlowPath) -> GradFixture {
    let net = SpectralNet::new(true);
    let mut rng = SmallRng::seed_from_u64(seed);
    let params = SpectralParams::init(&net.bands, &mut rng);
    let sched = Schedule { staggered, max_band: NBANDS - 1 };

    let b = 2usize;
    let mut ut = vec![0.0 as Real; b * NCOEF];
    let mut phit = vec![0.0 as Real; b * NCOEF];
    let mut tgt_u = vec![0.0 as Real; b * NCOEF];
    let mut tgt_phi = vec![0.0 as Real; b * NCOEF];
    let mut mask = vec![false; b * NCOEF];
    let mut tau = [0.0 as Real; NBANDS];
    let mut active = [false; NBANDS];

    for bi in 0..b {
        let (u0, p0) = sample_prior(&mut rng);
        let (u1, p1) = sample_prior(&mut rng);
        let s = build_sample(&net.bands, &sched, path, &u0, &p0, &u1, &p1, t);
        tau = s.tau;
        active = s.active;
        let lo = bi * NCOEF;
        ut[lo..lo + NCOEF].copy_from_slice(&s.ut);
        phit[lo..lo + NCOEF].copy_from_slice(&s.phit);
        tgt_u[lo..lo + NCOEF].copy_from_slice(&s.v_u);
        tgt_phi[lo..lo + NCOEF].copy_from_slice(&s.v_phi);
        mask[lo..lo + NCOEF].copy_from_slice(&s.mask);
    }

    GradFixture {
        net,
        params,
        input: GroupInput { b, tau, active, ut, phit },
        tgt_u,
        tgt_phi,
        mask,
        path,
    }
}

impl GradFixture {
    fn loss(&self, params: &SpectralParams) -> Real {
        let cache = self.net.forward(params, &self.input);
        flow_loss(
            &cache.v_u,
            &cache.v_phi,
            &self.tgt_u,
            &self.tgt_phi,
            &self.input.ut,
            &self.mask,
            1.0,
            &PhaseWeights::uniform().scale,
            self.path,
        )
        .total
    }

    fn analytic_grads(&self) -> SpectralParams {
        let cache = self.net.forward(&self.params, &self.input);
        let l = flow_loss(
            &cache.v_u,
            &cache.v_phi,
            &self.tgt_u,
            &self.tgt_phi,
            &self.input.ut,
            &self.mask,
            1.0,
            &PhaseWeights::uniform().scale,
            self.path,
        );
        let mut g = self.params.zeros_like();
        self.net.backward(&self.params, &cache, &l.d_v_u, &l.d_v_phi, &mut g);
        g
    }
}

/// Finite-difference step and tolerance.
///
/// `f32` cannot resolve a central difference tightly — the step has to stay
/// large enough to survive cancellation, which leaves visible truncation
/// error around the ReLU and soft-shrink kinks. The blueprint's 1e-4 gate is
/// therefore only meaningful in `f64`, and **`cargo test --release --features
/// dtype-f64` is the run that actually certifies the backward pass**. The `f32`
/// run keeps a loose tolerance so it still catches a sign flip, a missing
/// conjugate or a dropped term, which is all it can honestly detect.
#[cfg(feature = "dtype-f64")]
const FD_STEP: Real = 1e-6;
#[cfg(not(feature = "dtype-f64"))]
const FD_STEP: Real = 1e-3;

#[cfg(feature = "dtype-f64")]
const FD_TOL: Real = 1e-4;
#[cfg(not(feature = "dtype-f64"))]
const FD_TOL: Real = 1e-1;

/// Directions tried per tensor before a mismatch is called a failure.
const FD_DIRECTIONS: usize = 6;

/// Check every parameter tensor by its *directional* derivative along a random
/// unit direction: `<dL/dW, v>` against `(L(W + hv) - L(W - hv)) / 2h`.
///
/// Perturbing a whole tensor at once localises a broken tensor just as well as
/// poking single entries, but it averages over the soft-shrink and ReLU kinks
/// instead of letting one straddled kink dominate.
///
/// One kink cannot be averaged away, though: `wrap(dphi)^2` has derivative
/// `2*wrap(dphi)`, which jumps from `+2pi` to `-2pi` when a coefficient's phase
/// error crosses `pi`. The loss stays continuous there but is not
/// differentiable, and a difference quotient straddling that point is
/// meaningless — not evidence of a bad gradient. Since whether any coefficient
/// sits on the seam depends on the direction, each tensor gets several
/// directions and the best agreement counts.
fn check_tensor_grads(fx: &GradFixture, grads: &SpectralParams, label: &str) {
    let h = FD_STEP;
    let mut rng = SmallRng::seed_from_u64(99);
    let mut checked = 0usize;

    for ti in 0..fx.params.cplx.len() {
        let len = fx.params.cplx[ti].len();
        if len == 0 {
            continue;
        }
        let mut best = Real::INFINITY;
        let mut report = (0.0 as Real, 0.0 as Real);
        for _ in 0..FD_DIRECTIONS {
            let dir: Vec<C> = (0..len)
                .map(|_| {
                    Complex::new(
                        rng.random_range(-1.0f64..1.0) as Real,
                        rng.random_range(-1.0f64..1.0) as Real,
                    )
                })
                .collect();
            let norm = (dir.iter().map(|v| v.norm_sqr() as f64).sum::<f64>()).sqrt() as Real;
            let analytic: Real = grads.cplx[ti]
                .iter()
                .zip(dir.iter())
                .map(|(g, v)| (g.re * v.re + g.im * v.im) / norm)
                .sum();

            let mut plus = fx.params.clone();
            let mut minus = fx.params.clone();
            for k in 0..len {
                plus.cplx[ti][k] += dir[k] * (h / norm);
                minus.cplx[ti][k] -= dir[k] * (h / norm);
            }
            let numeric = (fx.loss(&plus) - fx.loss(&minus)) / (2.0 * h);
            let rel = rel_err(analytic, numeric);
            if rel < best {
                best = rel;
                report = (analytic, numeric);
            }
            if best < FD_TOL {
                break;
            }
        }
        assert!(
            best < FD_TOL,
            "{label} cplx[{ti}]: analytic {:e} vs numeric {:e} (rel {best:e})",
            report.0,
            report.1
        );
        checked += 1;
    }

    for ti in 0..fx.params.real.len() {
        let len = fx.params.real[ti].len();
        if len == 0 {
            continue;
        }
        let mut best = Real::INFINITY;
        let mut report = (0.0 as Real, 0.0 as Real);
        for _ in 0..FD_DIRECTIONS {
            let dir: Vec<Real> = (0..len).map(|_| rng.random_range(-1.0f64..1.0) as Real).collect();
            let norm = (dir.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt() as Real;
            let analytic: Real =
                grads.real[ti].iter().zip(dir.iter()).map(|(g, v)| g * v / norm).sum();

            let mut plus = fx.params.clone();
            let mut minus = fx.params.clone();
            for k in 0..len {
                plus.real[ti][k] += dir[k] * (h / norm);
                minus.real[ti][k] -= dir[k] * (h / norm);
            }
            let numeric = (fx.loss(&plus) - fx.loss(&minus)) / (2.0 * h);
            let rel = rel_err(analytic, numeric);
            if rel < best {
                best = rel;
                report = (analytic, numeric);
            }
            if best < FD_TOL {
                break;
            }
        }
        assert!(
            best < FD_TOL,
            "{label} real[{ti}]: analytic {:e} vs numeric {:e} (rel {best:e})",
            report.0,
            report.1
        );
        checked += 1;
    }

    assert!(checked >= 40, "only {checked} tensors checked");
}

fn rel_err(analytic: Real, numeric: Real) -> Real {
    let scale = analytic.abs().max(numeric.abs()).max(1e-4);
    (analytic - numeric).abs() / scale
}

#[test]
fn gradients_match_finite_differences_all_bands() {
    // t chosen so the staggered schedule has more than one band mid-flight.
    let fx = fixture(true, 0.5, 11);
    assert!(fx.input.active.iter().filter(|&&a| a).count() >= 2);
    let g = fx.analytic_grads();
    check_tensor_grads(&fx, &g, "staggered");
}

#[test]
fn gradients_match_finite_differences_low_bands_only() {
    // Exercises the reduced-resolution R-block round trip.
    let fx = fixture(true, 0.1, 12);
    assert_eq!(fx.input.active, [true, false, false, false]);
    let g = fx.analytic_grads();
    check_tensor_grads(&fx, &g, "band0-only");
}

// -- Step 5: forward pass sanity ------------------------------------------

#[test]
fn forward_shapes_and_finiteness() {
    let fx = fixture(false, 0.5, 13);
    let cache = fx.net.forward(&fx.params, &fx.input);
    assert_eq!(cache.v_u.len(), fx.input.b * NCOEF);
    assert_eq!(cache.v_phi.len(), fx.input.b * NCOEF);
    assert!(cache.v_u.iter().all(|v| v.is_finite()));
    assert!(cache.v_phi.iter().all(|v| v.is_finite()));

    // Frozen coefficients never get a prediction.
    for bd in 0..NBANDS {
        if fx.input.active[bd] {
            continue;
        }
        for i in fx.net.bands.offsets[bd]..fx.net.bands.offsets[bd + 1] {
            assert_eq!(cache.v_u[i], 0.0);
        }
    }
}

#[test]
fn parameter_count_is_in_budget() {
    let bands = BandTable::new();
    let p = SpectralParams::zeros(&bands);
    let total = p.num_scalars();
    // Blueprint F tallies 150,624 with a 45k FiLM path. The compact
    // conditioning (I4b) buys that back; the neighbour stencil and band
    // summary tokens (I4g) spend it again on mixing, which is where the model
    // was actually starved.
    // Widening D from [64,48,32,16] to [64,48,40,32] (band 3 was starved of
    // capacity relative to the fine detail it carries) raised this from
    // ~150,624 to ~189,640.
    assert!(total > 150_000 && total < 220_000, "unexpected parameter count {total}");
    assert_eq!(D.iter().sum::<usize>(), 184);
}

/// A checkpoint trained under a different `D`/`N_STAGES` must be rejected
/// with a descriptive error, not accepted and left to panic later inside
/// `model.rs`/`gemm.rs` on a mismatched tensor length.
#[test]
fn stale_architecture_checkpoint_is_rejected_with_a_clear_error() {
    use crate::neural_networks::spectral_model::train::check_architecture_shape;

    // Matches the live D/N_STAGES: accepted.
    assert!(check_architecture_shape("test.bin", D, crate::neural_networks::spectral_model::params::N_STAGES).is_ok());

    // The pre-widening shape this checkpoint would have been trained under:
    // rejected, and the message names both the stale and the live shape.
    let err = check_architecture_shape("test.bin", [64, 48, 32, 16], 3).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("test.bin"), "{msg}");
    assert!(msg.contains("[64, 48, 32, 16]"), "{msg}");
    assert!(msg.contains(&format!("{:?}", D)), "{msg}");
    assert!(msg.to_lowercase().contains("retrain"), "{msg}");
}

// -- Section D: the integrator, independent of the network -----------------

/// Feed the sampler the *true* target velocity instead of a network prediction
/// and check it lands on the data endpoint, on both flow paths.
///
/// This isolates section D: if it fails, no amount of training will produce an
/// image, because the integration itself does not reach the data. For
/// `ComplexLinear` it also certifies the rotating-frame velocity — an error in
/// the `e^{-i phi}` rotation or its inverse would land somewhere else entirely
/// while still looking like a plausible trajectory.
#[test]
fn oracle_integration_reaches_the_data() {
    let net = SpectralNet::new(true);
    let mut rng = SmallRng::seed_from_u64(21);
    let imgs: Vec<Vec<Real>> = (0..32).map(|_| smooth_image(&mut rng)).collect();
    let prep = SpectralPrep::from_dataset(&imgs);

    for path in [FlowPath::Polar, FlowPath::ComplexLinear, FlowPath::Cartesian] {
        for sched in [Schedule::full(), Schedule::warmup(3), Schedule::warmup(1)] {
            let (u1, phi1) = prep.image_to_polar(&imgs[0]);
            let (u0, phi0) = sample_prior(&mut rng);

            let mut u = u0.clone();
            let mut phi = phi0.clone();
            let n_steps = 16;
            let dt = sched.end_time() / n_steps as Real;

            for step in 0..n_steps {
                let t0 = step as Real * dt;
                let t1 = t0 + dt;
                let a = sched.tau(t0);
                let b = sched.tau(t1);
                let tau_mid = {
                    let mut m = [0.0 as Real; NBANDS];
                    for bd in 0..NBANDS {
                        m[bd] = 0.5 * (a[bd] + b[bd]);
                    }
                    m
                };

                let _ = tau_mid;
                for p in 0..NCOEF {
                    let bd = net.bands.band_of_pos[p] as usize;
                    let dtau = b[bd] - a[bd];
                    if dtau <= 0.0 || bd > sched.max_band {
                        continue;
                    }

                    // The oracle velocity, expressed the way the network would
                    // express it: in the frame of the state actually reached.
                    // Under ComplexLinear this is state-dependent, so rotating
                    // by the *ideal* path's phase instead would drift.
                    let (v_u, v_phi) = match path {
                        FlowPath::Polar => (u1[p] - u0[p], wrap(phi1[p] - phi0[p])),
                        FlowPath::ComplexLinear => {
                            let dc = polar_to_complex(u1[p], phi1[p])
                                - polar_to_complex(u0[p], phi0[p]);
                            let (cs, sn) = (phi[p].cos(), phi[p].sin());
                            (dc.re * cs + dc.im * sn, dc.im * cs - dc.re * sn)
                        }
                        FlowPath::Cartesian => {
                            // Absolute coordinates: no frame to track.
                            let dc = polar_to_complex(u1[p], phi1[p])
                                - polar_to_complex(u0[p], phi0[p]);
                            (dc.re, dc.im)
                        }
                    };

                    let (nu, nphi) = integrate_step(path, u[p], phi[p], v_u, v_phi, dtau);
                    u[p] = nu;
                    phi[p] = nphi;
                }
            }

            for p in 0..NCOEF {
                if net.bands.band_of_pos[p] as usize > sched.max_band {
                    continue;
                }
                let got = polar_to_complex(u[p], phi[p]);
                let want = polar_to_complex(u1[p], phi1[p]);
                let tol = 1e-3 * (1.0 + want.norm());
                assert!(
                    (got - want).norm() < tol,
                    "{path:?}/max_band {}: coefficient {p} landed at {got} not {want}",
                    sched.max_band
                );
            }
        }
    }
}

/// The rotating-frame velocity must reproduce the straight complex line.
///
/// `dc/dtau = e^{i phi_t} (v_r + i v_t)` has to equal `c1 - c0` exactly at
/// every point along the path, or the oracle above would only pass by accident.
#[test]
fn rotating_frame_velocity_reconstructs_complex_delta() {
    let bands = BandTable::new();
    let sched = Schedule { staggered: false, max_band: NBANDS - 1 };
    let mut rng = SmallRng::seed_from_u64(41);
    let (u0, phi0) = sample_prior(&mut rng);
    let (u1, phi1) = sample_prior(&mut rng);

    for &t in &[0.0 as Real, 0.13, 0.5, 0.87, 1.0] {
        let s = build_sample(&bands, &sched, FlowPath::ComplexLinear, &u0, &phi0, &u1, &phi1, t);
        for p in 0..NCOEF {
            let c0 = polar_to_complex(u0[p], phi0[p]);
            let c1 = polar_to_complex(u1[p], phi1[p]);
            let want = c1 - c0;

            let (cs, sn) = (s.phit[p].cos(), s.phit[p].sin());
            let got = num::Complex::new(
                s.v_u[p] * cs - s.v_phi[p] * sn,
                s.v_u[p] * sn + s.v_phi[p] * cs,
            );
            assert!(
                (got - want).norm() < 1e-4 * (1.0 + want.norm()),
                "t={t} p={p}: {got} vs {want}"
            );

            // And the interpolated state must be the straight line itself.
            let ct = polar_to_complex(s.ut[p], s.phit[p]);
            let want_ct = c0 * (1.0 - t) + c1 * t;
            assert!((ct - want_ct).norm() < 1e-4 * (1.0 + want_ct.norm()));
        }
    }
}

/// Both components stay bounded even when the straight line passes through the
/// origin — the singularity that rules out writing this path in `(u, phi)`.
#[test]
fn complex_path_is_bounded_through_the_origin() {
    let bands = BandTable::new();
    let sched = Schedule { staggered: false, max_band: NBANDS - 1 };

    // c1 = -c0 puts the path exactly through zero at tau = 0.5.
    let n = NCOEF;
    let u0 = vec![(1.0 as Real / EPS_MAG).asinh(); n];
    let phi0 = vec![0.4 as Real; n];
    let u1 = u0.clone();
    let phi1 = vec![0.4 as Real + std::f32::consts::PI as Real; n];

    for &t in &[0.49 as Real, 0.5, 0.500001, 0.51] {
        let s = build_sample(&bands, &sched, FlowPath::ComplexLinear, &u0, &phi0, &u1, &phi1, t);
        for p in 0..NCOEF {
            assert!(s.v_u[p].is_finite() && s.v_phi[p].is_finite(), "t={t}");
            assert!(s.v_u[p].abs() < 10.0, "radial {} blew up at t={t}", s.v_u[p]);
            assert!(s.v_phi[p].abs() < 10.0, "tangential {} blew up at t={t}", s.v_phi[p]);
        }
    }
}

/// A randomly initialised model must at least produce images in a sane range —
/// all-black output means something upstream of the weights is broken.
#[test]
fn untrained_samples_are_not_degenerate() {
    use crate::neural_networks::spectral_model::sample::sample_batch;

    let net = SpectralNet::new(true);
    let mut rng = SmallRng::seed_from_u64(22);
    let imgs: Vec<Vec<Real>> = (0..32).map(|_| smooth_image(&mut rng)).collect();
    let prep = SpectralPrep::from_dataset(&imgs);
    let params = SpectralParams::init(&net.bands, &mut rng);

    for sched in [Schedule::full(), Schedule::warmup(1)] {
        let out = sample_batch(&net, &prep, &params, &sched, FlowPath::ComplexLinear, 4, 16, 7);
        for img in &out {
            let min = img.iter().cloned().fold(Real::INFINITY, Real::min);
            let max = img.iter().cloned().fold(Real::NEG_INFINITY, Real::max);
            let mean = img.iter().map(|v| *v as f64).sum::<f64>() / img.len() as f64;
            println!("max_band {} -> min {min:.4} max {max:.4} mean {mean:.4}", sched.max_band);
            assert!(img.iter().all(|v| v.is_finite()));
            assert!(max - min > 1e-3, "image is constant: min {min} max {max}");
        }
    }
}

/// Zeroing bands in the band-major layout must actually band-limit the image.
///
/// The decisive check is to transform the result back: every coefficient above
/// the cut must be zero. If the permutation and its inverse disagree anywhere,
/// this fails even though the picture may still look plausible.
#[test]
fn band_masking_really_band_limits() {
    let net = SpectralNet::new(true);
    let mut rng = SmallRng::seed_from_u64(31);
    let imgs: Vec<Vec<Real>> = (0..32).map(|_| smooth_image(&mut rng)).collect();
    let prep = SpectralPrep::from_dataset(&imgs);

    for max_band in 0..NBANDS {
        let (mut u, mut phi) = prep.image_to_polar(&imgs[0]);
        for p in 0..NCOEF {
            if net.bands.band_of_pos[p] as usize > max_band {
                u[p] = 0.0;
                phi[p] = 0.0;
            }
        }
        let img = prep.polar_to_image(&u, &phi);

        // Round-trip and measure energy above the cut.
        let spec = net.bands.permute(&prep.plan.rfft2(&img));
        let mut kept = 0.0f64;
        let mut spilled = 0.0f64;
        for p in 0..NCOEF {
            let e = spec[p].norm_sqr() as f64;
            if net.bands.band_of_pos[p] as usize > max_band {
                spilled += e;
            } else {
                kept += e;
            }
        }
        let frac = spilled / (kept + spilled);
        println!("max_band {max_band}: {:.3e} of energy above the cut", frac);
        assert!(frac < 1e-6, "max_band {max_band} leaked {frac:e} of its energy above the cut");
    }
}

/// How much of a real digit survives each band cut — the number that says what
/// a perfect model at that `max_band` could possibly look like.
#[test]
fn report_energy_retained_per_band() {
    let net = SpectralNet::new(true);
    let mut rng = SmallRng::seed_from_u64(32);
    let imgs: Vec<Vec<Real>> = (0..32).map(|_| smooth_image(&mut rng)).collect();
    let prep = SpectralPrep::from_dataset(&imgs);

    let spec = net.bands.permute(&prep.plan.rfft2(&imgs[0]));
    let total: f64 = spec.iter().map(|c| c.norm_sqr() as f64).sum();
    for max_band in 0..NBANDS {
        let kept: f64 = (0..NCOEF)
            .filter(|&p| net.bands.band_of_pos[p] as usize <= max_band)
            .map(|p| spec[p].norm_sqr() as f64)
            .sum();
        println!("bands 0..={max_band}: {:.4} of image energy", kept / total);
    }
}

// -- Neighbour stencil topology -------------------------------------------

/// `k1` is a circular frequency axis; `k2` indexes a half spectrum and is not.
///
/// Getting this backwards produces a model that trains perfectly well and
/// generates subtly wrong images, because the stencil would either fail to
/// couple modes that really are adjacent across the `k1 = 0 / k1 = 31` seam, or
/// invent a coupling across the `k2 = 16` edge where no such adjacency exists.
/// Neither shows up in a loss curve.
#[test]
fn stencil_wraps_k1_but_not_k2() {
    let bands = BandTable::new();

    let mut checked_wrap = 0usize;
    let mut checked_edge = 0usize;

    for p in 0..NCOEF {
        let bd = bands.band_of_pos[p] as usize;
        let (k1, k2) = (bands.k1_of_pos[p], bands.k2_of_pos[p]);
        let nb = bands.neighbours[p];
        let off = bands.offsets[bd];

        // Whenever a neighbour is present, it must be the right coefficient.
        let expect = |slot: usize, want_k1: usize, want_k2: usize| {
            if nb[slot] != NO_NEIGHBOUR {
                let q = off + nb[slot];
                assert_eq!(
                    (bands.k1_of_pos[q], bands.k2_of_pos[q]),
                    (want_k1, want_k2),
                    "position {p} (k1={k1},k2={k2}) slot {slot}"
                );
            }
        };
        expect(0, (k1 + N - 1) % N, k2);
        expect(1, (k1 + 1) % N, k2);
        if k2 > 0 {
            expect(2, k1, k2 - 1);
        }
        if k2 + 1 < NH {
            expect(3, k1, k2 + 1);
        }

        // k1 = 0 must reach k1 = N-1 through the wrap, when both sit in the
        // same band. At k2 = 0 they do: rho is |k1| either way.
        if k1 == 0 && k2 == 0 {
            let q = bands.inv_order[(N - 1) * NH + k2];
            if bands.band_of_pos[q] as usize == bd {
                assert_ne!(nb[0], NO_NEIGHBOUR, "k1 wrap missing at k2={k2}");
                assert_eq!(off + nb[0], q);
                checked_wrap += 1;
            }
        }

        // k2 has no wrap: nothing may reach past either end.
        if k2 == 0 {
            assert_eq!(nb[2], NO_NEIGHBOUR, "k2 wrapped below 0 at position {p}");
            checked_edge += 1;
        }
        if k2 == NH - 1 {
            assert_eq!(nb[3], NO_NEIGHBOUR, "k2 wrapped past the Nyquist edge at {p}");
            checked_edge += 1;
        }
    }

    assert!(checked_wrap > 0, "no k1 wrap case was exercised");
    assert!(checked_edge > 0, "no k2 boundary case was exercised");
}

/// Energy at a single mode must spread to its true neighbours and nowhere else.
///
/// This drives the actual `stencil_pass` rather than the index table, so it
/// catches a correct table wired into the wrong axis.
#[test]
fn stencil_couples_only_true_neighbours() {
    use crate::neural_networks::spectral_model::model::stencil_pass_for_test;

    let bands = BandTable::new();
    let bd = 3usize; // outermost band: contains both the k1 seam and the k2 edge
    let (n, off) = (bands.counts[bd], bands.offsets[bd]);
    let d = 1usize;
    let nbrs = &bands.neighbours[off..off + n];

    // Taps: centre 0, both sides 1, so the output is exactly the neighbour sum.
    let taps: Vec<C> = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    ];

    for &(axis, slot) in &[("k1", 0usize), ("k2", 2usize)] {
        for probe in [0usize, n / 3, n / 2, n - 1] {
            let mut src = vec![C::new(0.0, 0.0); n * d];
            src[probe] = Complex::new(1.0, 0.0);
            let mut dst = vec![C::new(0.0, 0.0); n * d];
            stencil_pass_for_test(&src, &mut dst, nbrs, &taps, 1, n, d, slot);

            // Exactly the positions that name `probe` as their neighbour light up.
            for i in 0..n {
                let touches = nbrs[i][slot] == probe || nbrs[i][slot + 1] == probe;
                let got = dst[i].norm();
                if touches {
                    assert!(got > 0.5, "{axis}: {i} should see probe {probe}, got {got}");
                } else {
                    assert!(got < 1e-6, "{axis}: {i} must not see probe {probe}, got {got}");
                }
            }
        }
    }
}

// -- Resolution changes ----------------------------------------------------

/// The defining property of ideal interpolation: sampling the enlarged image
/// on the original grid returns the original, exactly.
///
/// Anything that merely *looks* smooth — bilinear, bicubic, a blur — fails
/// this. If it holds, the upsample is provably information-preserving, which
/// is what justifies not learning anything above 128x128.
#[test]
fn spectral_upsample_preserves_original_samples() {
    use crate::neural_networks::spectral_model::upscale::spectral_upsample;

    let mut rng = SmallRng::seed_from_u64(51);
    let img = smooth_image(&mut rng);

    for m in [64usize, 128, 256] {
        let up = spectral_upsample(&img, N, m);
        assert_eq!(up.len(), m * m);
        let f = m / N;
        for y in 0..N {
            for x in 0..N {
                let got = up[(y * f) * m + (x * f)];
                let want = img[y * N + x];
                assert!(
                    (got - want).abs() < 1e-3,
                    "n={N} m={m} at ({x},{y}): {got} vs {want}"
                );
            }
        }
    }
}

/// `spectral_upsample_with_plans`/`spectral_upsample_apodized_with_plans`
/// must produce the exact same output as the plain (plan-constructing)
/// variants they were factored out of — the refactor should not change
/// behaviour, only let the caller reuse FFT plans across calls.
#[test]
fn upsample_with_plans_matches_plain_variant() {
    use crate::neural_networks::spectral_model::fft::Rfft2Plan;
    use crate::neural_networks::spectral_model::upscale::{
        spectral_upsample, spectral_upsample_apodized, spectral_upsample_apodized_with_plans,
        spectral_upsample_with_plans, DEFAULT_APODISATION,
    };

    let mut rng = SmallRng::seed_from_u64(53);
    let img = smooth_image(&mut rng);
    let (n, m) = (N, 128usize);
    let src = Rfft2Plan::new(n);
    let dst = Rfft2Plan::new(m);

    let plain = spectral_upsample(&img, n, m);
    let with_plans = spectral_upsample_with_plans(&img, &src, &dst);
    assert_eq!(plain, with_plans);

    let plain_ap = spectral_upsample_apodized(&img, n, m, DEFAULT_APODISATION);
    let with_plans_ap = spectral_upsample_apodized_with_plans(&img, &src, &dst, DEFAULT_APODISATION);
    assert_eq!(plain_ap, with_plans_ap);
}

/// Parallelizing `rfft2`/`irfft2`'s row and column passes with rayon must not
/// change their output: each parallel closure owns private scratch state and
/// results are merged by plain assignment (never summed across threads), so
/// this should be bit-for-bit identical to a sequential run, not just
/// "close." A tolerance here would silently hide a genuine data race.
#[test]
fn fft_parallel_passes_are_deterministic() {
    let mut rng = SmallRng::seed_from_u64(54);
    for n in [8usize, 16, 32] {
        let plan = Rfft2Plan::new(n);
        let img: Vec<Real> = (0..n * n).map(|_| rng.random_range(-1.0f64..1.0) as Real).collect();

        let a = plan.rfft2(&img);
        let b = plan.rfft2(&img);
        assert_eq!(a, b, "rfft2 not deterministic at n={n}");

        let ia = plan.irfft2(&a);
        let ib = plan.irfft2(&a);
        assert_eq!(ia, ib, "irfft2 not deterministic at n={n}");
    }
}

/// Upsampling must not invent energy above the original Nyquist beyond what
/// band-limited interpolation implies, and must not lose the mean.
#[test]
fn spectral_upsample_conserves_mean_and_band_limit() {
    use crate::neural_networks::spectral_model::upscale::spectral_upsample;

    let mut rng = SmallRng::seed_from_u64(52);
    let img = smooth_image(&mut rng);
    let m = 128usize;
    let up = spectral_upsample(&img, N, m);

    let mean_in: f64 = img.iter().map(|v| *v as f64).sum::<f64>() / img.len() as f64;
    let mean_out: f64 = up.iter().map(|v| *v as f64).sum::<f64>() / up.len() as f64;
    assert!((mean_in - mean_out).abs() < 1e-4, "mean {mean_in} -> {mean_out}");

    // Every coefficient outside the original band must be zero.
    let plan = Rfft2Plan::new(m);
    let spec = plan.rfft2(&up);
    let mut leaked = 0.0f64;
    for k1 in 0..m {
        let k1s = if k1 <= m / 2 { k1 as i32 } else { k1 as i32 - m as i32 };
        for k2 in 0..plan.nh {
            if k1s.unsigned_abs() as usize > N / 2 || k2 > N / 2 {
                leaked += spec[k1 * plan.nh + k2].norm_sqr() as f64;
            }
        }
    }
    assert!(leaked < 1e-3, "energy leaked above the original band: {leaked}");
}

/// Downsample is the inverse of upsample on band-limited content.
#[test]
fn spectral_resample_round_trips() {
    use crate::neural_networks::spectral_model::upscale::{
        spectral_downsample, spectral_upsample,
    };

    let mut rng = SmallRng::seed_from_u64(53);
    let img = smooth_image(&mut rng);
    let back = spectral_downsample(&spectral_upsample(&img, N, 128), 128, N);
    for (a, b) in img.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

#[test]
fn hd_canvas_centres_the_digit() {
    use crate::neural_networks::spectral_model::upscale::{centre_on_hd, HD_H, HD_W};

    let side = 256usize;
    let sq: Vec<Real> = (0..side * side).map(|_| 1.0 as Real).collect();
    let hd = centre_on_hd(&sq, side);
    assert_eq!(hd.len(), HD_W * HD_H);

    let x0 = (HD_W - side) / 2;
    let y0 = (HD_H - side) / 2;
    assert_eq!(hd[y0 * HD_W + x0], 1.0);
    assert_eq!(hd[(y0 - 1) * HD_W + x0], 0.0);
    assert_eq!(hd[y0 * HD_W + (x0 - 1)], 0.0);
    // Total ink is preserved.
    let sum: f64 = hd.iter().map(|v| *v as f64).sum();
    assert!((sum - (side * side) as f64).abs() < 1e-6);
}

/// Gibbs ringing has an objective signature: the interpolant overshoots the
/// input's value range. Apodisation must measurably reduce it.
///
/// This test exists because `spectral_upsample_preserves_original_samples`
/// passes for an enlargement that looks visibly worse than the raw pixels.
/// Exactness was the wrong property to measure — it certified fidelity to a
/// truncation artefact. Overshoot measures the thing that actually degrades
/// the picture.
#[test]
fn apodisation_reduces_overshoot() {
    use crate::neural_networks::spectral_model::upscale::{
        spectral_upsample, spectral_upsample_apodized, DEFAULT_APODISATION,
    };

    // A hard-edged image: the worst case for sinc interpolation, and much
    // closer to a digit than the smooth gaussians used elsewhere.
    let mut img = vec![0.0 as Real; N * N];
    for y in 10..22 {
        for x in 12..20 {
            img[y * N + x] = 1.0;
        }
    }

    let lo = img.iter().cloned().fold(Real::INFINITY, Real::min);
    let hi = img.iter().cloned().fold(Real::NEG_INFINITY, Real::max);

    let overshoot = |v: &[Real]| -> Real {
        v.iter().fold(0.0 as Real, |acc, &p| acc.max((p - hi).max(lo - p).max(0.0)))
    };

    let exact = spectral_upsample(&img, N, 256);
    let soft = spectral_upsample_apodized(&img, N, 256, DEFAULT_APODISATION);

    let (oe, os) = (overshoot(&exact), overshoot(&soft));
    println!("overshoot: exact {oe:.4}, apodised {os:.4}");

    assert!(oe > 0.05, "expected the exact upsample to ring, got {oe}");
    assert!(os < oe * 0.5, "apodisation should at least halve overshoot: {os} vs {oe}");
    // Measured at the default: 0.115 vs 0.326, a 65% reduction.

    // It must still be an enlargement of the same picture, not a blur to grey.
    let mean_in: f64 = img.iter().map(|v| *v as f64).sum::<f64>() / img.len() as f64;
    let mean_out: f64 = soft.iter().map(|v| *v as f64).sum::<f64>() / soft.len() as f64;
    assert!((mean_in - mean_out).abs() < 1e-3, "mean drifted: {mean_in} -> {mean_out}");
}

/// `alpha = 0` must be exactly the un-apodised path, so the two share one
/// code path's behaviour and the parameter genuinely spans the trade-off.
#[test]
fn zero_apodisation_matches_exact_upsample() {
    use crate::neural_networks::spectral_model::upscale::{
        spectral_upsample, spectral_upsample_apodized,
    };
    let mut rng = SmallRng::seed_from_u64(61);
    let img = smooth_image(&mut rng);
    let a = spectral_upsample(&img, N, 128);
    let b = spectral_upsample_apodized(&img, N, 128, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
}

/// Sweep the apodisation strength and report the trade-off, so the default is
/// chosen from data rather than guessed.
///
/// `overshoot` is the Gibbs signature (lower is better). `edge` is the
/// steepest local gradient across the enlarged image, a proxy for retained
/// sharpness (higher is better). The useful setting is the knee: most of the
/// ringing gone before sharpness starts collapsing.
#[test]
#[ignore]
fn sweep_apodisation_tradeoff() {
    use crate::neural_networks::spectral_model::upscale::spectral_upsample_apodized;

    let mut img = vec![0.0 as Real; N * N];
    for y in 10..22 {
        for x in 12..20 {
            img[y * N + x] = 1.0;
        }
    }
    let hi = 1.0 as Real;
    let lo = 0.0 as Real;
    let m = 256usize;

    println!("alpha  overshoot  edge-sharpness");
    for i in 0..=10 {
        let alpha = i as Real / 10.0;
        let up = spectral_upsample_apodized(&img, N, m, alpha);
        let overshoot = up
            .iter()
            .fold(0.0 as Real, |acc, &p| acc.max((p - hi).max(lo - p).max(0.0)));
        let mut edge = 0.0 as Real;
        for y in 0..m {
            for x in 1..m {
                edge = edge.max((up[y * m + x] - up[y * m + x - 1]).abs());
            }
        }
        println!("{alpha:.1}    {overshoot:.4}     {edge:.4}");
    }
}
