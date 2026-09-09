//! The data pipeline between pixels and the model's polar state:
//! whitening statistics (A3), the polar endpoint `z1` (B4), the noise endpoint
//! `z0` (B5), the interpolated state (B9) and the target velocity (B10).
//!
//! Every array in this module is in **band-major** layout, i.e. index `p` runs
//! over [`BandTable::order`], so a band is a contiguous slice.

use crate::neural_networks::spectral_model::bands::{
    clamp_u, wrap, BandTable, FlowPath, Schedule, EPS_MAG, NBANDS, NCOEF,
};
use crate::neural_networks::spectral_model::fft::Rfft2Plan;
use crate::neural_networks::utils::dtype::{Real, C};
use num::Complex;
use rand::RngExt;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Fixed geometry plus the dataset whitening table. Built once, then shared
/// (read-only) by training and sampling.
#[derive(Debug, Clone)]
pub struct SpectralPrep {
    pub plan: Rfft2Plan,
    pub bands: BandTable,
    /// RMS magnitude per coefficient, band-major. Blueprint A3.
    pub s: Vec<Real>,
}

/// Serialisable form of the whitening table, written next to the weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteningTable {
    pub s: Vec<Real>,
}

impl SpectralPrep {
    /// Build the geometry with a whitening table of all ones (used before the
    /// statistics have been measured).
    pub fn unwhitened() -> Self {
        Self {
            plan: Rfft2Plan::new(crate::neural_networks::spectral_model::bands::N),
            bands: BandTable::new(),
            s: vec![1.0 as Real; NCOEF],
        }
    }

    /// Measure the RMS magnitude of every coefficient over `images`.
    ///
    /// Accumulated in `f64` because the low-frequency terms are ~1e4 and the
    /// sum runs over tens of thousands of images.
    pub fn from_dataset(images: &[Vec<Real>]) -> Self {
        let mut prep = Self::unwhitened();
        prep.s = prep.measure_s(images);
        prep
    }

    fn measure_s(&self, images: &[Vec<Real>]) -> Vec<Real> {
        assert!(!images.is_empty(), "cannot measure whitening statistics on an empty set");
        let mut acc = vec![0.0f64; NCOEF];
        for img in images {
            let x = self.plan.rfft2(img);
            for (k, xk) in x.iter().enumerate() {
                acc[k] += (xk.re as f64) * (xk.re as f64) + (xk.im as f64) * (xk.im as f64);
            }
        }
        let n = images.len() as f64;
        let natural: Vec<Real> =
            acc.iter().map(|v| (((v / n).sqrt()) as Real).max(1e-8 as Real)).collect();
        self.bands.permute(&natural)
    }

    pub fn table(&self) -> WhiteningTable {
        WhiteningTable { s: self.s.clone() }
    }

    pub fn with_table(table: WhiteningTable) -> Self {
        let mut prep = Self::unwhitened();
        assert_eq!(table.s.len(), NCOEF, "whitening table has the wrong length");
        prep.s = table.s;
        prep
    }

    /// Image -> whitened, band-major complex spectrum.
    pub fn whiten(&self, img: &[Real]) -> Vec<C> {
        let x = self.bands.permute(&self.plan.rfft2(img));
        x.iter().zip(self.s.iter()).map(|(&xk, &sk)| xk / sk).collect()
    }

    /// Inverse of [`SpectralPrep::whiten`].
    pub fn unwhiten_to_image(&self, c: &[C]) -> Vec<Real> {
        let scaled: Vec<C> = c.iter().zip(self.s.iter()).map(|(&ck, &sk)| ck * sk).collect();
        self.plan.irfft2(&self.bands.unpermute(&scaled))
    }

    /// Image -> the data endpoint `z1 = (u1, phi1)`. Blueprint B4.
    pub fn image_to_polar(&self, img: &[Real]) -> (Vec<Real>, Vec<Real>) {
        to_polar(&self.whiten(img))
    }

    /// The full inverse: polar state -> image.
    pub fn polar_to_image(&self, u: &[Real], phi: &[Real]) -> Vec<Real> {
        self.unwhiten_to_image(&from_polar(u, phi))
    }
}

/// Whitened complex spectrum -> `(u, phi)`.
///
/// `asinh` behaves like a logarithm for large magnitudes but stays linear and
/// finite near zero, so a coefficient that happens to be tiny cannot blow up
/// the regression target.
pub fn to_polar(c: &[C]) -> (Vec<Real>, Vec<Real>) {
    let mut u = vec![0.0 as Real; c.len()];
    let mut phi = vec![0.0 as Real; c.len()];
    for (k, ck) in c.iter().enumerate() {
        u[k] = (ck.norm() / EPS_MAG).asinh();
        phi[k] = ck.im.atan2(ck.re);
    }
    (u, phi)
}

/// `(u, phi)` -> whitened complex spectrum. Blueprint C1.
pub fn from_polar(u: &[Real], phi: &[Real]) -> Vec<C> {
    u.iter()
        .zip(phi.iter())
        .map(|(&uk, &pk)| {
            let mag = EPS_MAG * uk.sinh();
            Complex::new(mag * pk.cos(), mag * pk.sin())
        })
        .collect()
}

/// Draw the noise endpoint `z0`. Blueprint B5.
///
/// Sampled in Cartesian coordinates so that `E[|c|^2] = 1`, matching the
/// whitened data; un-whitening this prior gives an image with the correct
/// natural power spectrum but no structure.
pub fn sample_prior<R: RngExt + ?Sized>(rng: &mut R) -> (Vec<Real>, Vec<Real>) {
    let normal = Normal::new(0.0f64, 1.0f64).unwrap();
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut u = vec![0.0 as Real; NCOEF];
    let mut phi = vec![0.0 as Real; NCOEF];
    for k in 0..NCOEF {
        let a = normal.sample(rng) * inv_sqrt2;
        let b = normal.sample(rng) * inv_sqrt2;
        u[k] = ((a.hypot(b) / EPS_MAG as f64).asinh()) as Real;
        phi[k] = b.atan2(a) as Real;
    }
    (u, phi)
}

/// The state and target of one training example.
///
/// The two velocity fields mean different things per [`FlowPath`]:
///
/// | Path | `v_u` | `v_phi` |
/// |---|---|---|
/// | [`FlowPath::Polar`] | `du/dtau` | `dphi/dtau`, an **angle** rate |
/// | [`FlowPath::ComplexLinear`] | radial part of `dc/dtau` | tangential part |
///
/// Only in the polar case is the second component angular, and only then do
/// `wrap` and the energy weight apply to it.
#[derive(Debug, Clone)]
pub struct FlowSample {
    /// Global time in `[0, 1]`.
    pub t: Real,
    /// Per-band local times.
    pub tau: [Real; NBANDS],
    /// Which bands are mid-flight.
    pub active: [bool; NBANDS],
    /// Interpolated magnitude coordinate.
    pub ut: Vec<Real>,
    /// Interpolated phase.
    pub phit: Vec<Real>,
    /// First target component; see the table above.
    pub v_u: Vec<Real>,
    /// Second target component; see the table above.
    pub v_phi: Vec<Real>,
    /// Per-coefficient active mask.
    pub mask: Vec<bool>,
}

/// Build the interpolated state and the constant target velocity for one
/// endpoint pair. Blueprint B6-B10.
#[allow(clippy::too_many_arguments)]
pub fn build_sample(
    bands: &BandTable,
    schedule: &Schedule,
    path: FlowPath,
    u0: &[Real],
    phi0: &[Real],
    u1: &[Real],
    phi1: &[Real],
    t: Real,
) -> FlowSample {
    let tau = schedule.tau(t);
    let active = schedule.active(&tau);

    let mut ut = vec![0.0 as Real; NCOEF];
    let mut phit = vec![0.0 as Real; NCOEF];
    let mut v_u = vec![0.0 as Real; NCOEF];
    let mut v_phi = vec![0.0 as Real; NCOEF];
    let mut mask = vec![false; NCOEF];

    for p in 0..NCOEF {
        let b = bands.band_of_pos[p] as usize;
        let tau_k = tau[b];
        mask[p] = active[b];

        match path {
            FlowPath::Polar => {
                // Magnitude on a straight line; phase along the shortest arc.
                // Without the wrap the model is asked to rotate the long way
                // round about half the time, giving one visual outcome two
                // different targets.
                let delta = wrap(phi1[p] - phi0[p]);
                ut[p] = (1.0 - tau_k) * u0[p] + tau_k * u1[p];
                phit[p] = phi0[p] + tau_k * delta;
                v_u[p] = u1[p] - u0[p];
                v_phi[p] = delta;
            }
            FlowPath::ComplexLinear => {
                let c0 = polar_to_complex(u0[p], phi0[p]);
                let c1 = polar_to_complex(u1[p], phi1[p]);
                let ct = c0 * (1.0 - tau_k) + c1 * tau_k;
                let dc = c1 - c0;

                let m = ct.norm();
                // Exactly at the origin the phase is undefined; inherit the
                // start phase so the frame stays continuous.
                let ph = if m > 0.0 { ct.im.atan2(ct.re) } else { phi0[p] };

                ut[p] = (m / EPS_MAG).asinh();
                phit[p] = ph;

                // Rotate dc into the frame carried by the current phase. No
                // division by |c|, so this is bounded even through the origin.
                let (cs, sn) = (ph.cos(), ph.sin());
                v_u[p] = dc.re * cs + dc.im * sn;
                v_phi[p] = dc.im * cs - dc.re * sn;
            }
            FlowPath::Cartesian => {
                // Identical path, but the target is stated against a fixed
                // axis instead of the state's own phase.
                let c0 = polar_to_complex(u0[p], phi0[p]);
                let c1 = polar_to_complex(u1[p], phi1[p]);
                let ct = c0 * (1.0 - tau_k) + c1 * tau_k;
                let dc = c1 - c0;

                let m = ct.norm();
                ut[p] = (m / EPS_MAG).asinh();
                phit[p] = if m > 0.0 { ct.im.atan2(ct.re) } else { phi0[p] };
                v_u[p] = dc.re;
                v_phi[p] = dc.im;
            }
        }
    }

    FlowSample { t, tau, active, ut, phit, v_u, v_phi, mask }
}

/// `(u, phi)` -> complex, for a single coefficient.
#[inline]
pub fn polar_to_complex(u: Real, phi: Real) -> C {
    let mag = EPS_MAG * u.sinh();
    Complex::new(mag * phi.cos(), mag * phi.sin())
}

/// Advance one coefficient by `dtau` along the model's predicted velocity.
///
/// Returns the new `(u, phi)`. For [`FlowPath::ComplexLinear`] the step is
/// taken in the complex plane and converted back, which makes the oracle
/// integration exact rather than merely first-order.
#[inline]
pub fn integrate_step(
    path: FlowPath,
    u: Real,
    phi: Real,
    v_u: Real,
    v_phi: Real,
    dtau: Real,
) -> (Real, Real) {
    match path {
        FlowPath::Polar => (clamp_u(u + dtau * v_u), wrap(phi + dtau * v_phi)),
        FlowPath::ComplexLinear => {
            let c = polar_to_complex(u, phi);
            let (cs, sn) = (phi.cos(), phi.sin());
            // Undo the rotating frame: dc = e^{i phi} (v_u + i v_phi)
            let dc = Complex::new(v_u * cs - v_phi * sn, v_u * sn + v_phi * cs);
            let c = c + dc * dtau;
            let m = c.norm();
            let ph = if m > 0.0 { c.im.atan2(c.re) } else { phi };
            (clamp_u((m / EPS_MAG).asinh()), ph)
        }
        FlowPath::Cartesian => {
            let c = polar_to_complex(u, phi) + Complex::new(v_u, v_phi) * dtau;
            let m = c.norm();
            let ph = if m > 0.0 { c.im.atan2(c.re) } else { phi };
            (clamp_u((m / EPS_MAG).asinh()), ph)
        }
    }
}

/// Draw a uniform global time.
pub fn sample_time<R: RngExt + ?Sized>(rng: &mut R) -> Real {
    rng.random_range(0.0f64..1.0f64) as Real
}
