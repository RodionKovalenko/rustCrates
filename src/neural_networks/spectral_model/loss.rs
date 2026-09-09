//! The flow-matching objective. Blueprint section B12.
//!
//! Rotating a coefficient that is nearly zero changes nothing visible, so the
//! phase term is weighted by the coefficient's energy. That weight is treated
//! as a constant (no gradient flows through it) and normalised over the active
//! set so the term stays scale-stable as bands freeze and thaw.
//!
//! **The energy weight has to be measured in image units, not whitened ones.**
//! The blueprint computes it from `EPS*sinh(u_t)` — but that is the *whitened*
//! magnitude, and whitening's whole purpose is to make every coefficient's
//! magnitude ≈ 1. The weight therefore comes out uniform and the term does
//! nothing at all: it is a correct idea cancelled by an earlier step.
//!
//! What a phase error actually costs the image is `s[k] * |c_k| * dphi`, so the
//! weight belongs on `(s[k]^alpha * mag)^2`. `alpha = 0` recovers the
//! blueprint's (inert) version; `alpha = 1` makes the term an exact proxy for
//! image-space MSE, which is principled but hands almost all the gradient to
//! band 0 and starves the outer bands the band design exists to serve. The
//! default splits the difference.

use crate::neural_networks::spectral_model::bands::{wrap, FlowPath, EPS_MAG, NCOEF};
use crate::neural_networks::utils::dtype::Real;

/// Relative weight of the phase term. Calibrated once at initialisation so
/// both terms start comparable (see [`calibrate_lambda_phi`]).
pub const DEFAULT_LAMBDA_PHI: Real = 1.0;

/// Exponent on the whitening scale in the phase energy weight.
pub const DEFAULT_PHASE_ALPHA: Real = 0.5;

/// Cap on how far one coefficient's weight may exceed the mean.
///
/// MNIST's `s` spans ~200x, so even at `alpha = 0.5` a handful of low
/// coefficients would otherwise carry the entire term.
pub const WEIGHT_CAP: Real = 50.0;

#[derive(Debug, Clone)]
pub struct LossOut {
    pub total: Real,
    pub mag_term: Real,
    pub phase_term: Real,
    pub n_active: usize,
    /// `dL/dv_u`, `b * NCOEF`.
    pub d_v_u: Vec<Real>,
    /// `dL/dv_phi`, `b * NCOEF`.
    pub d_v_phi: Vec<Real>,
}

/// Compute the loss and its gradient w.r.t. the predicted velocities.
///
/// All slices are `b * NCOEF` in band-major layout; `mask` marks the
/// coefficients whose band is mid-flight. `energy_scale[p]` is
/// `s[p]^phase_alpha`, precomputed once by [`PhaseWeights`].
#[allow(clippy::too_many_arguments)]
pub fn flow_loss(
    pred_u: &[Real],
    pred_phi: &[Real],
    tgt_u: &[Real],
    tgt_phi: &[Real],
    ut: &[Real],
    mask: &[bool],
    lambda_phi: Real,
    energy_scale: &[Real],
    path: FlowPath,
) -> LossOut {
    let len = pred_u.len();
    debug_assert_eq!(len % NCOEF, 0);
    debug_assert_eq!(energy_scale.len(), NCOEF);

    // Under `ComplexLinear` the second output is a Cartesian rate, not an
    // angle: it must not be wrapped, and it carries no energy weight.
    //
    // The energy weight existed because an *angular* velocity is scaled by
    // 1/|c| — rotating a near-zero coefficient is perceptually free, so its
    // phase error had to be discounted. A tangential velocity has no such
    // factor: it is already an amplitude rate, bounded by |c1| + |c0|, and an
    // error in it is an error in the image whatever the magnitude. With the
    // weight uniform and `lambda_phi = 1`, this reduces to plain Cartesian L2.
    let angular = path.second_output_is_angular();

    let mut w = vec![0.0 as Real; len];
    let mut w_sum = 0.0f64;
    let mut n_active = 0usize;
    for i in 0..len {
        if mask[i] {
            w[i] = if angular {
                let mag = energy_scale[i % NCOEF] * EPS_MAG * ut[i].sinh();
                mag * mag
            } else {
                1.0
            };
            w_sum += w[i] as f64;
            n_active += 1;
        }
    }

    let mut out = LossOut {
        total: 0.0,
        mag_term: 0.0,
        phase_term: 0.0,
        n_active,
        d_v_u: vec![0.0 as Real; len],
        d_v_phi: vec![0.0 as Real; len],
    };
    if n_active == 0 {
        return out;
    }

    let w_mean = (w_sum / n_active as f64).max(1e-12) as Real;
    let inv_n = 1.0 as Real / n_active as Real;

    let mut mag_acc = 0.0f64;
    let mut phase_acc = 0.0f64;
    for i in 0..len {
        if !mask[i] {
            continue;
        }
        let du = pred_u[i] - tgt_u[i];
        let raw = pred_phi[i] - tgt_phi[i];
        let dphi = if angular { wrap(raw) } else { raw };
        let wn = (w[i] / w_mean).min(WEIGHT_CAP);

        mag_acc += (du * du) as f64;
        phase_acc += (lambda_phi * wn * dphi * dphi) as f64;

        out.d_v_u[i] = 2.0 * du * inv_n;
        out.d_v_phi[i] = 2.0 * lambda_phi * wn * dphi * inv_n;
    }

    out.mag_term = (mag_acc / n_active as f64) as Real;
    out.phase_term = (phase_acc / n_active as f64) as Real;
    out.total = out.mag_term + out.phase_term;
    out
}

/// Precomputed `s[p]^alpha`, band-major. Built once and shared.
#[derive(Debug, Clone)]
pub struct PhaseWeights {
    pub scale: Vec<Real>,
}

impl PhaseWeights {
    /// `alpha = 0` gives the blueprint's uniform weight; `alpha = 1` weights by
    /// the coefficient's true contribution to the image.
    pub fn new(s: &[Real], alpha: Real) -> Self {
        Self { scale: s.iter().map(|v| v.powf(alpha)).collect() }
    }

    pub fn uniform() -> Self {
        Self { scale: vec![1.0 as Real; NCOEF] }
    }
}

/// Ratio that makes the phase term match the magnitude term at initialisation.
///
/// Blueprint B12 asks for exactly this: start at 1.0, measure both terms, then
/// rescale. Returns `lambda_phi` clamped to a sane range.
///
/// **Only meaningful for [`FlowPath::Polar`].** Under `ComplexLinear` the two
/// outputs are the components of one Cartesian vector in an orthonormal frame,
/// so equal weighting is not a coincidence to be tuned away — it is what makes
/// the loss rotation-invariant. See [`calibrate_for_path`].
pub fn calibrate_lambda_phi(mag_term: Real, phase_term: Real) -> Real {
    if phase_term <= 1e-9 {
        return DEFAULT_LAMBDA_PHI;
    }
    (mag_term / phase_term).clamp(0.01, 100.0)
}

/// Calibration that respects the path's geometry.
///
/// Rotation is orthogonal, so for `v = R(-phi)(c1 - c0)` and a prediction
/// `v_hat`,
///
/// ```text
/// ||v - v_hat||^2 = ||R(phi)(v - v_hat)||^2 = ||(c1 - c0) - R(phi) v_hat||^2
/// ```
///
/// Equal-weighted L2 in the rotating frame is *identically* Cartesian L2 on the
/// complex velocity — not approximately, term for term. Calibrating
/// `lambda_phi` away from 1 would deliberately break that isometry and make the
/// objective anisotropic in a frame that has no preferred axis, so under
/// `ComplexLinear` the calibration is skipped and the weight is exactly 1.
pub fn calibrate_for_path(path: FlowPath, mag_term: Real, phase_term: Real) -> Real {
    match path {
        FlowPath::Polar => calibrate_lambda_phi(mag_term, phase_term),
        FlowPath::ComplexLinear | FlowPath::Cartesian => 1.0,
    }
}
