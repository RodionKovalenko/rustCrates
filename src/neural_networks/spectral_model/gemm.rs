//! Cache-blocked complex GEMM. Blueprint section G.
//!
//! Every coefficient in a band shares the same `R` matrix, so an S-block is one
//! `(n, d) x (d, d)` product. That single routine is roughly half of training
//! compute, so it gets the `ikj` loop order (contiguous inner loop over the
//! output row, B row streamed once per `k`) plus tiling so the working set of
//! each tile stays inside L1.
//!
//! All three variants accumulate into `out` rather than overwriting, which is
//! what gradient accumulation across the weight-shared stages needs.

use crate::neural_networks::utils::dtype::C;

/// Rows of A processed per tile.
const MC: usize = 64;
/// Columns of B processed per tile; `NC * size_of::<C>()` should stay well
/// inside L1 for one row of the tile.
const NC: usize = 64;
/// Depth processed per tile.
const KC: usize = 64;

/// `out[m, n] += a[m, k] * b[k, n]`, all row-major.
pub fn gemm_acc(a: &[C], b: &[C], out: &mut [C], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + MC).min(m);
        let mut p0 = 0;
        while p0 < k {
            let p1 = (p0 + KC).min(k);
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + NC).min(n);
                for i in i0..i1 {
                    let out_row = &mut out[i * n + j0..i * n + j1];
                    for p in p0..p1 {
                        let av = a[i * k + p];
                        if av.re == 0.0 && av.im == 0.0 {
                            continue;
                        }
                        let b_row = &b[p * n + j0..p * n + j1];
                        for (o, &bv) in out_row.iter_mut().zip(b_row.iter()) {
                            o.re += av.re * bv.re - av.im * bv.im;
                            o.im += av.re * bv.im + av.im * bv.re;
                        }
                    }
                }
                j0 = j1;
            }
            p0 = p1;
        }
        i0 = i1;
    }
}

/// `out[m, n] = a[m, k] * b[k, n]`, overwriting `out`.
pub fn gemm(a: &[C], b: &[C], out: &mut [C], m: usize, k: usize, n: usize) {
    for v in out.iter_mut() {
        *v = C::new(0.0, 0.0);
    }
    gemm_acc(a, b, out, m, k, n);
}

/// `d_a[m, k] += d_out[m, n] * conj(b[k, n])^T`.
///
/// This is the input-side adjoint of [`gemm_acc`]: for a real loss the gradient
/// of `c = a * b` w.r.t. `a` is `d_c * conj(b)`.
pub fn gemm_grad_a(d_out: &[C], b: &[C], d_a: &mut [C], m: usize, k: usize, n: usize) {
    debug_assert_eq!(d_out.len(), m * n);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(d_a.len(), m * k);

    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + MC).min(m);
        for i in i0..i1 {
            let g_row = &d_out[i * n..(i + 1) * n];
            for p in 0..k {
                let b_row = &b[p * n..(p + 1) * n];
                let mut acc_re = 0.0;
                let mut acc_im = 0.0;
                for (&g, &bv) in g_row.iter().zip(b_row.iter()) {
                    // g * conj(b)
                    acc_re += g.re * bv.re + g.im * bv.im;
                    acc_im += g.im * bv.re - g.re * bv.im;
                }
                d_a[i * k + p].re += acc_re;
                d_a[i * k + p].im += acc_im;
            }
        }
        i0 = i1;
    }
}

/// `d_b[k, n] += conj(a[m, k])^T * d_out[m, n]`.
pub fn gemm_grad_b(a: &[C], d_out: &[C], d_b: &mut [C], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(d_out.len(), m * n);
    debug_assert_eq!(d_b.len(), k * n);

    for i in 0..m {
        let g_row = &d_out[i * n..(i + 1) * n];
        for p in 0..k {
            let av = a[i * k + p];
            if av.re == 0.0 && av.im == 0.0 {
                continue;
            }
            let d_row = &mut d_b[p * n..(p + 1) * n];
            for (o, &g) in d_row.iter_mut().zip(g_row.iter()) {
                // conj(a) * g
                o.re += av.re * g.re + av.im * g.im;
                o.im += av.re * g.im - av.im * g.re;
            }
        }
    }
}

/// Reference implementation used to validate the blocked kernel in tests.
pub fn gemm_naive(a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Vec<C> {
    let mut out = vec![C::new(0.0, 0.0); m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                out[i * n + j] = out[i * n + j] + a[i * k + p] * b[p * n + j];
            }
        }
    }
    out
}
