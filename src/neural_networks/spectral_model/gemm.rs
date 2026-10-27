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
//!
//! With the `cuda` feature every product of at least [`GPU_MIN_MACS`] multiply-adds runs on the GPU (cuBLAS,
//! the conjugate transposes of the backward pass included); smaller ones and every product without a GPU use
//! the blocked CPU kernels below.

use crate::neural_networks::utils::dtype::C;
use crate::neural_networks::utils::matrix_rm::MatOp;

/// Smallest GEMM (complex multiply-adds `m*k*n`) sent to the GPU. Below it, copying the operands costs more than
/// the product: the band-summary matrices (`group x d x 16`, ~16k) stay on the CPU, every S-block and R-block
/// projection (`group * coefficients x d x d`, millions) goes to the GPU.
pub const GPU_MIN_MACS: usize = 1 << 16;

/// `op_a(A) (m x k) * op_b(B) (k x n)` on the GPU, or `None` (too small, no GPU, or the GPU call failed).
#[cfg(feature = "cuda")]
fn gpu_product(a: &[C], op_a: MatOp, b: &[C], op_b: MatOp, m: usize, k: usize, n: usize) -> Option<Vec<C>> {
    if m * k * n < GPU_MIN_MACS {
        return None;
    }
    crate::neural_networks::utils::gpu_matmul::try_multiply_complex_gpu(a, op_a, b, op_b, m, k, n, false)
}

#[cfg(not(feature = "cuda"))]
fn gpu_product(_a: &[C], _op_a: MatOp, _b: &[C], _op_b: MatOp, _m: usize, _k: usize, _n: usize) -> Option<Vec<C>> {
    None
}

fn add_into(out: &mut [C], product: &[C]) {
    for (o, p) in out.iter_mut().zip(product.iter()) {
        *o += *p;
    }
}

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
    if let Some(product) = gpu_product(a, MatOp::None, b, MatOp::None, m, k, n) {
        add_into(out, &product);
        return;
    }
    gemm_acc_cpu(a, b, out, m, k, n);
}

/// CPU kernel of [`gemm_acc`].
pub fn gemm_acc_cpu(a: &[C], b: &[C], out: &mut [C], m: usize, k: usize, n: usize) {
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
    if let Some(product) = gpu_product(a, MatOp::None, b, MatOp::None, m, k, n) {
        out.copy_from_slice(&product);
        return;
    }
    for v in out.iter_mut() {
        *v = C::new(0.0, 0.0);
    }
    gemm_acc_cpu(a, b, out, m, k, n);
}

/// `d_a[m, k] += d_out[m, n] * conj(b[k, n])^T`.
///
/// This is the input-side adjoint of [`gemm_acc`]: for a real loss the gradient
/// of `c = a * b` w.r.t. `a` is `d_c * conj(b)`.
pub fn gemm_grad_a(d_out: &[C], b: &[C], d_a: &mut [C], m: usize, k: usize, n: usize) {
    debug_assert_eq!(d_out.len(), m * n);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(d_a.len(), m * k);
    // d_out (m x n) * b^H (n x k)
    if let Some(product) = gpu_product(d_out, MatOp::None, b, MatOp::ConjTranspose, m, n, k) {
        add_into(d_a, &product);
        return;
    }
    gemm_grad_a_cpu(d_out, b, d_a, m, k, n);
}

/// CPU kernel of [`gemm_grad_a`].
pub fn gemm_grad_a_cpu(d_out: &[C], b: &[C], d_a: &mut [C], m: usize, k: usize, n: usize) {
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
    // a^H (k x m) * d_out (m x n)
    if let Some(product) = gpu_product(a, MatOp::ConjTranspose, d_out, MatOp::None, k, m, n) {
        add_into(d_b, &product);
        return;
    }
    gemm_grad_b_cpu(a, d_out, d_b, m, k, n);
}

/// CPU kernel of [`gemm_grad_b`].
pub fn gemm_grad_b_cpu(a: &[C], d_out: &[C], d_b: &mut [C], m: usize, k: usize, n: usize) {
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
