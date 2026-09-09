//! Radix-2 FFT and the real 2-D transforms used by the spectral flow model.
//!
//! Blueprint section H, step 1. Everything here is pure numerics: no learned
//! parameters, no autodiff. The adjoints (`irfft2_adjoint`, `rfft2_adjoint`)
//! live here too because they are the transposes of these linear maps and are
//! needed to backpropagate through the R-block round trip.
//!
//! Layout convention for a real `n x n` image:
//!   * `rfft2` returns `n * (n/2+1)` complex coefficients, row-major in
//!     `(k1, k2)` with `k1` the vertical frequency (rows `> n/2` hold negative
//!     frequencies) and `k2` the horizontal frequency in `0..=n/2`.
//!   * For `n = 32` that is `32 * 17 = 544`, the count the blueprint uses.

use crate::neural_networks::utils::dtype::{Real, C};
use num::Complex;
use rayon::prelude::*;

/// Precomputed twiddle factors and bit-reversal table for one transform length.
///
/// `n` must be a power of two.
#[derive(Debug, Clone)]
pub struct FftPlan {
    pub n: usize,
    /// Forward twiddles `exp(-2*pi*i*j/n)` for `j in 0..n/2`.
    tw_fwd: Vec<C>,
    /// Inverse twiddles `exp(+2*pi*i*j/n)` for `j in 0..n/2`.
    tw_inv: Vec<C>,
    /// Bit-reversal permutation.
    rev: Vec<u32>,
}

impl FftPlan {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2, "FFT length must be a power of two, got {n}");

        let half = n / 2;
        let mut tw_fwd = Vec::with_capacity(half);
        let mut tw_inv = Vec::with_capacity(half);
        for j in 0..half {
            let theta = -2.0 * std::f64::consts::PI * j as f64 / n as f64;
            tw_fwd.push(Complex::new(theta.cos() as Real, theta.sin() as Real));
            tw_inv.push(Complex::new(theta.cos() as Real, -theta.sin() as Real));
        }

        let bits = n.trailing_zeros();
        let mut rev = vec![0u32; n];
        for i in 0..n {
            rev[i] = (i as u32).reverse_bits() >> (32 - bits);
        }

        Self { n, tw_fwd, tw_inv, rev }
    }

    /// In-place decimation-in-time Cooley-Tukey. Unnormalised in both
    /// directions; the `1/n` for the inverse is applied by the caller.
    fn transform(&self, buf: &mut [C], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(buf.len(), n);

        for i in 0..n {
            let j = self.rev[i] as usize;
            if i < j {
                buf.swap(i, j);
            }
        }

        let tw = if inverse { &self.tw_inv } else { &self.tw_fwd };

        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            let mut base = 0;
            while base < n {
                for k in 0..half {
                    let w = tw[k * step];
                    let a = buf[base + k];
                    let b = buf[base + k + half] * w;
                    buf[base + k] = a + b;
                    buf[base + k + half] = a - b;
                }
                base += len;
            }
            len <<= 1;
        }
    }

    /// Unnormalised forward DFT of a complex signal.
    pub fn fft(&self, buf: &mut [C]) {
        self.transform(buf, false);
    }

    /// Inverse DFT including the `1/n` normalisation.
    pub fn ifft(&self, buf: &mut [C]) {
        self.transform(buf, true);
        let scale = 1.0 as Real / self.n as Real;
        for v in buf.iter_mut() {
            *v = *v * scale;
        }
    }
}

/// Plan for the real 2-D transforms on an `n x n` grid.
#[derive(Debug, Clone)]
pub struct Rfft2Plan {
    pub n: usize,
    /// `n / 2 + 1`, the number of retained horizontal frequencies.
    pub nh: usize,
    /// `n * nh`, the number of stored coefficients.
    pub ncoef: usize,
    plan: FftPlan,
}

impl Rfft2Plan {
    pub fn new(n: usize) -> Self {
        let nh = n / 2 + 1;
        Self { n, nh, ncoef: n * nh, plan: FftPlan::new(n) }
    }

    #[inline]
    pub fn idx(&self, k1: usize, k2: usize) -> usize {
        k1 * self.nh + k2
    }

    /// Weight of stored coefficient column `k2` in the inverse transform.
    ///
    /// Columns `0` and `n/2` are their own Hermitian mirror and appear once;
    /// every other column stands for a conjugate pair and appears twice.
    #[inline]
    pub fn column_weight(&self, k2: usize) -> Real {
        if k2 == 0 || k2 == self.n / 2 { 1.0 } else { 2.0 }
    }

    /// Real image (`n*n`, row-major) -> half spectrum (`n*nh`).
    ///
    /// The row and column passes are each independent per row/column, so both
    /// are parallelised with rayon; every closure owns its own scratch buffer,
    /// so there is no shared mutable state and no risk of a data race.
    pub fn rfft2(&self, img: &[Real]) -> Vec<C> {
        assert_eq!(img.len(), self.n * self.n);
        let (n, nh) = (self.n, self.nh);

        // Rows first: real -> complex, keep the non-redundant half.
        let mut tmp = vec![Complex::new(0.0 as Real, 0.0 as Real); n * nh];
        img.par_chunks(n).zip(tmp.par_chunks_mut(nh)).for_each(|(img_row, tmp_row)| {
            let mut row = vec![Complex::new(0.0 as Real, 0.0 as Real); n];
            for n2 in 0..n {
                row[n2] = Complex::new(img_row[n2], 0.0 as Real);
            }
            self.plan.fft(&mut row);
            tmp_row.copy_from_slice(&row[..nh]);
        });

        // Then columns (strided, so gather each column into an owned buffer
        // rather than trying to slice it out of the row-major `tmp`).
        let cols: Vec<Vec<C>> = (0..nh)
            .into_par_iter()
            .map(|k2| {
                let mut col = vec![Complex::new(0.0 as Real, 0.0 as Real); n];
                for n1 in 0..n {
                    col[n1] = tmp[n1 * nh + k2];
                }
                self.plan.fft(&mut col);
                col
            })
            .collect();
        let mut out = vec![Complex::new(0.0 as Real, 0.0 as Real); n * nh];
        for (k2, col) in cols.iter().enumerate() {
            for k1 in 0..n {
                out[k1 * nh + k2] = col[k1];
            }
        }
        out
    }

    /// Half spectrum (`n*nh`) -> real image (`n*n`).
    ///
    /// Coefficients that violate the Hermitian constraint in columns `0` and
    /// `n/2` are silently projected onto the valid subspace, exactly as any
    /// standard `irfft2` does.
    pub fn irfft2(&self, coef: &[C]) -> Vec<Real> {
        assert_eq!(coef.len(), self.n * self.nh);
        let (n, nh) = (self.n, self.nh);

        // Inverse along k1 for every retained column (strided, so gather into
        // an owned per-column buffer, same reasoning as in `rfft2`).
        let cols: Vec<Vec<C>> = (0..nh)
            .into_par_iter()
            .map(|k2| {
                let mut col = vec![Complex::new(0.0 as Real, 0.0 as Real); n];
                for k1 in 0..n {
                    col[k1] = coef[k1 * nh + k2];
                }
                self.plan.ifft(&mut col);
                col
            })
            .collect();
        let mut tmp = vec![Complex::new(0.0 as Real, 0.0 as Real); n * nh];
        for (k2, col) in cols.iter().enumerate() {
            for n1 in 0..n {
                tmp[n1 * nh + k2] = col[n1];
            }
        }

        // Then rebuild each row from its half spectrum; rows of `tmp`/`out`
        // are contiguous, so this pass parallelises directly.
        let mut out = vec![0.0 as Real; n * n];
        tmp.par_chunks(nh).zip(out.par_chunks_mut(n)).for_each(|(tmp_row, out_row)| {
            let mut row = vec![Complex::new(0.0 as Real, 0.0 as Real); n];
            row[..nh].copy_from_slice(&tmp_row[..nh]);
            for k2 in nh..n {
                row[k2] = tmp_row[n - k2].conj();
            }
            self.plan.ifft(&mut row);
            for n2 in 0..n {
                out_row[n2] = row[n2].re;
            }
        });
        out
    }

    /// Transpose of [`Rfft2Plan::irfft2`], viewing the stored coefficients as
    /// `2 * n * nh` independent reals.
    ///
    /// `d_img` is the gradient w.r.t. the image; the result is the gradient
    /// w.r.t. the half spectrum.
    pub fn irfft2_adjoint(&self, d_img: &[Real]) -> Vec<C> {
        let mut g = self.rfft2(d_img);
        let inv_n2 = 1.0 as Real / (self.n * self.n) as Real;
        for k1 in 0..self.n {
            for k2 in 0..self.nh {
                g[k1 * self.nh + k2] = g[k1 * self.nh + k2] * (self.column_weight(k2) * inv_n2);
            }
        }
        g
    }

    /// Transpose of [`Rfft2Plan::rfft2`].
    ///
    /// `d_coef` is the gradient w.r.t. the half spectrum; the result is the
    /// gradient w.r.t. the real image.
    pub fn rfft2_adjoint(&self, d_coef: &[C]) -> Vec<Real> {
        let mut scaled = d_coef.to_vec();
        for k1 in 0..self.n {
            for k2 in 0..self.nh {
                scaled[k1 * self.nh + k2] = scaled[k1 * self.nh + k2] / self.column_weight(k2);
            }
        }
        let mut out = self.irfft2(&scaled);
        let n2 = (self.n * self.n) as Real;
        for v in out.iter_mut() {
            *v = *v * n2;
        }
        out
    }
}

/// Naive `O(N^2)` 2-D DFT, used only to validate the fast path in tests.
pub fn naive_rfft2(img: &[Real], n: usize) -> Vec<C> {
    let nh = n / 2 + 1;
    let mut out = vec![Complex::new(0.0 as Real, 0.0 as Real); n * nh];
    for k1 in 0..n {
        for k2 in 0..nh {
            let mut acc = Complex::new(0.0f64, 0.0f64);
            for n1 in 0..n {
                for n2 in 0..n {
                    let theta = -2.0 * std::f64::consts::PI
                        * ((k1 * n1) as f64 + (k2 * n2) as f64)
                        / n as f64;
                    acc += Complex::new(theta.cos(), theta.sin()) * img[n1 * n + n2] as f64;
                }
            }
            out[k1 * nh + k2] = Complex::new(acc.re as Real, acc.im as Real);
        }
    }
    out
}
