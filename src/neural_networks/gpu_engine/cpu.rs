//! Reference implementation of the engine operations on the CPU (`f32`, rayon over rows). It defines what every
//! CUDA kernel has to compute and makes the whole model testable without a GPU.

use rayon::prelude::*;

use super::backend::{Backend, Gemm};

pub struct CpuBackend;

fn sigmoid(t: f32) -> f32 {
    if t >= 0.0 {
        1.0 / (1.0 + (-t).exp())
    } else {
        let e = t.exp();
        e / (1.0 + e)
    }
}

fn silu(t: f32) -> f32 {
    t * sigmoid(t)
}

fn silu_derivative(t: f32) -> f32 {
    let s = sigmoid(t);
    s * (1.0 + t * (1.0 - s))
}

/// RoPE angle of the pair `i` of a head at `position` (as the RM layer, including the clamp to [-1, 1]).
pub fn rope_angle(position: usize, i: usize, head_dim: usize) -> f32 {
    (position as f64 / super::ROPE_BASE.powf(2.0 * i as f64 / head_dim as f64)).clamp(-1.0, 1.0) as f32
}

impl Backend for CpuBackend {
    type Buf = Vec<f32>;
    type Ids = Vec<u32>;

    fn zeros(&self, len: usize) -> Vec<f32> {
        vec![0.0; len]
    }
    fn upload(&self, data: &[f32]) -> Vec<f32> {
        data.to_vec()
    }
    fn download(&self, buf: &Vec<f32>) -> Vec<f32> {
        buf.clone()
    }
    fn upload_ids(&self, ids: &[u32]) -> Vec<u32> {
        ids.to_vec()
    }
    fn len(&self, buf: &Vec<f32>) -> usize {
        buf.len()
    }

    fn fill(&self, buf: &mut Vec<f32>, off: usize, len: usize, value: f32) {
        buf[off..off + len].fill(value);
    }
    fn copy(&self, src: &Vec<f32>, src_off: usize, dst: &mut Vec<f32>, dst_off: usize, len: usize) {
        dst[dst_off..dst_off + len].copy_from_slice(&src[src_off..src_off + len]);
    }
    fn axpy(&self, alpha: f32, src: &Vec<f32>, dst: &mut Vec<f32>, len: usize) {
        dst[..len].par_iter_mut().zip(&src[..len]).for_each(|(d, s)| *d += alpha * s);
    }
    fn scale(&self, buf: &mut Vec<f32>, len: usize, alpha: f32) {
        buf[..len].par_iter_mut().for_each(|v| *v *= alpha);
    }
    fn sum_squares(&self, buf: &Vec<f32>, len: usize) -> f64 {
        buf[..len].par_iter().map(|v| (*v as f64) * (*v as f64)).sum()
    }
    fn all_finite(&self, buf: &Vec<f32>, len: usize) -> bool {
        buf[..len].par_iter().all(|v| v.is_finite())
    }

    fn gemm(&self, g: &Gemm, a: &Vec<f32>, b: &Vec<f32>, c: &mut Vec<f32>) {
        // Row-major element (i, j) of a stored matrix is at i * ld + j; a transposed operand swaps the strides.
        let (rsa, csa) = if g.trans_a { (1, g.lda as isize) } else { (g.lda as isize, 1) };
        let (rsb, csb) = if g.trans_b { (1, g.ldb as isize) } else { (g.ldb as isize, 1) };
        for i in 0..g.batch {
            let (ao, bo, co) = (g.a_off + i * g.stride_a, g.b_off + i * g.stride_b, g.c_off + i * g.stride_c);
            // The extent each operand touches, so that the slices below are bounds-checked once.
            let a_end = ao + if g.trans_a { (g.k - 1) * g.lda + g.m } else { (g.m - 1) * g.lda + g.k };
            let b_end = bo + if g.trans_b { (g.n - 1) * g.ldb + g.k } else { (g.k - 1) * g.ldb + g.n };
            let c_end = co + (g.m - 1) * g.ldc + g.n;
            assert!(a_end <= a.len() && b_end <= b.len() && c_end <= c.len(), "gemm out of bounds: {g:?}");
            unsafe {
                matrixmultiply::sgemm(
                    g.m,
                    g.k,
                    g.n,
                    g.alpha,
                    a.as_ptr().add(ao),
                    rsa,
                    csa,
                    b.as_ptr().add(bo),
                    rsb,
                    csb,
                    g.beta,
                    c.as_mut_ptr().add(co),
                    g.ldc as isize,
                    1,
                );
            }
        }
    }

    fn build_block(&self, params: &Vec<f32>, re_off: usize, im_off: usize, k: usize, n: usize, out: &mut Vec<f32>) {
        let (wr, wi) = (&params[re_off..re_off + k * n], &params[im_off..im_off + k * n]);
        out[..4 * k * n].par_chunks_mut(2 * n).enumerate().for_each(|(row, o)| {
            let (src, top) = (row % k, row < k);
            for j in 0..n {
                let (r, i) = (wr[src * n + j], wi[src * n + j]);
                // [[Wr, Wi], [-Wi, Wr]]
                (o[j], o[n + j]) = if top { (r, i) } else { (-i, r) };
            }
        });
    }

    fn fold_block_grad(&self, block_grad: &Vec<f32>, k: usize, n: usize, grads: &mut Vec<f32>, re_off: usize, im_off: usize) {
        let g = block_grad;
        for row in 0..k {
            for j in 0..n {
                let (g11, g12) = (g[row * 2 * n + j], g[row * 2 * n + n + j]);
                let (g21, g22) = (g[(k + row) * 2 * n + j], g[(k + row) * 2 * n + n + j]);
                grads[re_off + row * n + j] += g11 + g22;
                grads[im_off + row * n + j] += g12 - g21;
            }
        }
    }

    fn add_row_vector(&self, y: &mut Vec<f32>, rows: usize, cols: usize, params: &Vec<f32>, off: usize) {
        let v = &params[off..off + cols];
        y[..rows * cols].par_chunks_mut(cols).for_each(|row| row.iter_mut().zip(v).for_each(|(y, b)| *y += b));
    }

    fn column_sums(&self, dy: &Vec<f32>, rows: usize, cols: usize, grads: &mut Vec<f32>, off: usize) {
        let sums = dy[..rows * cols].par_chunks(cols).fold(|| vec![0.0f64; cols], |mut acc, row| {
            acc.iter_mut().zip(row).for_each(|(a, v)| *a += *v as f64);
            acc
        });
        let total = sums.reduce(|| vec![0.0f64; cols], |mut a, b| {
            a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            a
        });
        grads[off..off + cols].iter_mut().zip(total).for_each(|(g, s)| *g += s as f32);
    }

    fn layer_norm(&self, x: &Vec<f32>, rows: usize, w: usize, eps: f32, params: &Vec<f32>, gamma_off: usize, beta_off: usize, y: &mut Vec<f32>, stats: &mut Vec<f32>) {
        let (gamma, beta) = (&params[gamma_off..gamma_off + 2 * w], &params[beta_off..beta_off + 2 * w]);
        y[..rows * 2 * w].par_chunks_mut(2 * w).zip(stats[..rows * 3].par_chunks_mut(3)).enumerate().for_each(|(r, (yr, st))| {
            let xr = &x[r * 2 * w..(r + 1) * 2 * w];
            let (mut mre, mut mim) = (0.0f32, 0.0f32);
            for j in 0..w {
                mre += xr[j];
                mim += xr[w + j];
            }
            (mre, mim) = (mre / w as f32, mim / w as f32);
            let mut var = 0.0f32;
            for j in 0..w {
                let (dr, di) = (xr[j] - mre, xr[w + j] - mim);
                var += dr * dr + di * di;
            }
            let rstd = 1.0 / (var / w as f32 + eps).sqrt();
            for j in 0..w {
                let (hr, hi) = ((xr[j] - mre) * rstd, (xr[w + j] - mim) * rstd);
                let (gr, gi) = (gamma[j], gamma[w + j]);
                yr[j] = hr * gr - hi * gi + beta[j];
                yr[w + j] = hr * gi + hi * gr + beta[w + j];
            }
            st.copy_from_slice(&[mre, mim, rstd]);
        });
    }

    fn layer_norm_backward(&self, x: &Vec<f32>, stats: &Vec<f32>, dy: &Vec<f32>, rows: usize, w: usize, params: &Vec<f32>, gamma_off: usize, dx: &mut Vec<f32>, grads: &mut Vec<f32>, dgamma_off: usize, dbeta_off: usize) {
        // With x_hat = (x - mu) / sigma and u = G conj(gamma) (complex_layer_norm::layer_norm_row_backward):
        // dx = (u - mean(u)) / sigma - x_hat * S / (n sigma) with S = sum Re(conj(u) x_hat),
        // dgamma = G conj(x_hat), dbeta = G.
        let gamma = &params[gamma_off..gamma_off + 2 * w];
        let partial = dx[..rows * 2 * w]
            .par_chunks_mut(2 * w)
            .enumerate()
            .fold(
                || vec![0.0f64; 4 * w],
                |mut acc, (r, dxr)| {
                    let (xr, g) = (&x[r * 2 * w..(r + 1) * 2 * w], &dy[r * 2 * w..(r + 1) * 2 * w]);
                    let (mre, mim, rstd) = (stats[3 * r], stats[3 * r + 1], stats[3 * r + 2]);
                    let (mut ure_mean, mut uim_mean, mut s) = (0.0f32, 0.0f32, 0.0f32);
                    for j in 0..w {
                        let (gr, gi, gmr, gmi) = (g[j], g[w + j], gamma[j], gamma[w + j]);
                        let (ur, ui) = (gr * gmr + gi * gmi, gi * gmr - gr * gmi);
                        let (hr, hi) = ((xr[j] - mre) * rstd, (xr[w + j] - mim) * rstd);
                        ure_mean += ur;
                        uim_mean += ui;
                        s += ur * hr + ui * hi;
                        // dgamma = G conj(x_hat), dbeta = G
                        acc[j] += (gr * hr + gi * hi) as f64;
                        acc[w + j] += (gi * hr - gr * hi) as f64;
                        acc[2 * w + j] += gr as f64;
                        acc[3 * w + j] += gi as f64;
                    }
                    (ure_mean, uim_mean) = (ure_mean / w as f32, uim_mean / w as f32);
                    let c = s / w as f32;
                    for j in 0..w {
                        let (gr, gi, gmr, gmi) = (g[j], g[w + j], gamma[j], gamma[w + j]);
                        let (ur, ui) = (gr * gmr + gi * gmi, gi * gmr - gr * gmi);
                        let (hr, hi) = ((xr[j] - mre) * rstd, (xr[w + j] - mim) * rstd);
                        dxr[j] = (ur - ure_mean - hr * c) * rstd;
                        dxr[w + j] = (ui - uim_mean - hi * c) * rstd;
                    }
                    acc
                },
            )
            .reduce(|| vec![0.0f64; 4 * w], |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                a
            });
        for j in 0..2 * w {
            grads[dgamma_off + j] += partial[j] as f32;
            grads[dbeta_off + j] += partial[2 * w + j] as f32;
        }
    }

    fn rope_to_heads(&self, src: &Vec<f32>, src_ld: usize, re_off: usize, im_off: usize, rows: usize, seq_len: usize, heads: usize, head_dim: usize, dst: &mut Vec<f32>) {
        let dst_ld = 2 * heads * head_dim;
        dst[..rows * dst_ld].par_chunks_mut(dst_ld).enumerate().for_each(|(r, out)| {
            let s = &src[r * src_ld..(r + 1) * src_ld];
            let position = r % seq_len;
            for h in 0..heads {
                for i in 0..head_dim / 2 {
                    let (cos, sin) = { let a = rope_angle(position, i, head_dim); (a.cos(), a.sin()) };
                    for (part, off) in [(0, re_off), (1, im_off)] {
                        let (e, o) = (s[off + h * head_dim + 2 * i], s[off + h * head_dim + 2 * i + 1]);
                        let base = h * 2 * head_dim + part * head_dim + 2 * i;
                        out[base] = e * cos - o * sin;
                        out[base + 1] = e * sin + o * cos;
                    }
                }
            }
        });
    }

    fn rope_from_heads(&self, src: &Vec<f32>, rows: usize, seq_len: usize, heads: usize, head_dim: usize, dst: &mut Vec<f32>, dst_ld: usize, re_off: usize, im_off: usize) {
        let src_ld = 2 * heads * head_dim;
        dst[..rows * dst_ld].par_chunks_mut(dst_ld).enumerate().for_each(|(r, out)| {
            let s = &src[r * src_ld..(r + 1) * src_ld];
            let position = r % seq_len;
            for h in 0..heads {
                for i in 0..head_dim / 2 {
                    let (cos, sin) = { let a = rope_angle(position, i, head_dim); (a.cos(), a.sin()) };
                    for (part, off) in [(0, re_off), (1, im_off)] {
                        let base = h * 2 * head_dim + part * head_dim + 2 * i;
                        let (e, o) = (s[base], s[base + 1]);
                        // The transpose of the rotation: rotation by -angle.
                        out[off + h * head_dim + 2 * i] = e * cos + o * sin;
                        out[off + h * head_dim + 2 * i + 1] = -e * sin + o * cos;
                    }
                }
            }
        });
    }

    fn softmax_window(&self, s: &mut Vec<f32>, groups: usize, t: usize, window: usize) {
        s[..groups * t * t].par_chunks_mut(t).enumerate().for_each(|(row, values)| {
            let i = row % t;
            let first = i.saturating_sub(window);
            let max = values[first..=i].iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let mut sum = 0.0f32;
            for (j, v) in values.iter_mut().enumerate() {
                *v = if j >= first && j <= i { (*v - max).exp() } else { 0.0 };
                sum += *v;
            }
            values.iter_mut().for_each(|v| *v /= sum);
        });
    }

    fn softmax_window_backward(&self, p: &Vec<f32>, dp: &mut Vec<f32>, groups: usize, t: usize) {
        dp[..groups * t * t].par_chunks_mut(t).enumerate().for_each(|(row, d)| {
            let pr = &p[row * t..(row + 1) * t];
            let dot: f32 = pr.iter().zip(d.iter()).map(|(p, d)| p * d).sum();
            d.iter_mut().zip(pr).for_each(|(d, p)| *d = p * (*d - dot));
        });
    }

    fn attention_residual(&self, x: &Vec<f32>, att: &Vec<f32>, rows: usize, w: usize, beta: f32, y: &mut Vec<f32>) {
        y[..rows * 2 * w].par_chunks_mut(2 * w).enumerate().for_each(|(r, yr)| {
            for j in 0..w {
                yr[j] = x[r * 2 * w + j] + beta * att[r * w + j];
                yr[w + j] = x[r * 2 * w + w + j];
            }
        });
    }

    fn real_part_scaled(&self, dy: &Vec<f32>, rows: usize, w: usize, beta: f32, out: &mut Vec<f32>) {
        out[..rows * w].par_chunks_mut(w).enumerate().for_each(|(r, o)| {
            for j in 0..w {
                o[j] = beta * dy[r * 2 * w + j];
            }
        });
    }

    fn gated(&self, h: &Vec<f32>, rows: usize, hidden: usize, g: &mut Vec<f32>) {
        g[..rows * 2 * hidden].par_chunks_mut(2 * hidden).enumerate().for_each(|(r, gr)| {
            let hr = &h[r * 4 * hidden..(r + 1) * 4 * hidden];
            for j in 0..hidden {
                let (ar, br, ai, bi) = (hr[j], hr[hidden + j], hr[2 * hidden + j], hr[3 * hidden + j]);
                let (sr, si) = (silu(br), silu(bi));
                gr[j] = ar * sr - ai * si;
                gr[hidden + j] = ar * si + ai * sr;
            }
        });
    }

    fn gated_backward(&self, h: &Vec<f32>, dg: &Vec<f32>, rows: usize, hidden: usize, dh: &mut Vec<f32>) {
        // complex_layer_norm::gated_backward: da = g conj(s), db = (Re(u) silu'(br), Im(u) silu'(bi)), u = g conj(a).
        dh[..rows * 4 * hidden].par_chunks_mut(4 * hidden).enumerate().for_each(|(r, d)| {
            let hr = &h[r * 4 * hidden..(r + 1) * 4 * hidden];
            let gr = &dg[r * 2 * hidden..(r + 1) * 2 * hidden];
            for j in 0..hidden {
                let (ar, br, ai, bi) = (hr[j], hr[hidden + j], hr[2 * hidden + j], hr[3 * hidden + j]);
                let (gre, gim) = (gr[j], gr[hidden + j]);
                let (sr, si) = (silu(br), silu(bi));
                d[j] = gre * sr + gim * si;
                d[2 * hidden + j] = gim * sr - gre * si;
                let (ur, ui) = (gre * ar + gim * ai, gim * ar - gre * ai);
                d[hidden + j] = ur * silu_derivative(br);
                d[3 * hidden + j] = ui * silu_derivative(bi);
            }
        });
    }

    fn leaky_residual(&self, o: &Vec<f32>, x: &Vec<f32>, rows: usize, w: usize, beta: f32, slope: f32, y: &mut Vec<f32>) {
        y[..rows * 2 * w].par_chunks_mut(2 * w).enumerate().for_each(|(r, yr)| {
            for j in 0..w {
                let (or, oi) = (o[r * 2 * w + j], o[r * 2 * w + w + j]);
                let f = if or > 0.0 { beta } else { beta * slope };
                yr[j] = x[r * 2 * w + j] + f * or;
                yr[w + j] = x[r * 2 * w + w + j] + f * oi;
            }
        });
    }

    fn leaky_residual_backward(&self, o: &Vec<f32>, dy: &Vec<f32>, rows: usize, w: usize, beta: f32, slope: f32, d_o: &mut Vec<f32>) {
        d_o[..rows * 2 * w].par_chunks_mut(2 * w).enumerate().for_each(|(r, d)| {
            for j in 0..w {
                let f = if o[r * 2 * w + j] > 0.0 { beta } else { beta * slope };
                d[j] = f * dy[r * 2 * w + j];
                d[w + j] = f * dy[r * 2 * w + w + j];
            }
        });
    }

    fn cross_entropy(&self, logits: &mut Vec<f32>, rows: usize, vocab: usize, targets: &Vec<u32>, t_off: usize, scale: f32) -> f64 {
        logits[..rows * vocab]
            .par_chunks_mut(vocab)
            .enumerate()
            .map(|(r, l)| {
                let target = targets[t_off + r] as usize;
                let max = l.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
                let sum: f64 = l.iter().map(|v| ((*v - max) as f64).exp()).sum();
                let loss = sum.ln() - (l[target] - max) as f64;
                for v in l.iter_mut() {
                    *v = ((((*v - max) as f64).exp() / sum) as f32) * scale;
                }
                l[target] -= scale;
                loss
            })
            .sum()
    }

    fn embed(&self, params: &Vec<f32>, re_off: usize, im_off: usize, ids: &Vec<u32>, ids_off: usize, rows: usize, w: usize, out: &mut Vec<f32>) {
        out[..rows * 2 * w].par_chunks_mut(2 * w).enumerate().for_each(|(r, o)| {
            let id = ids[ids_off + r] as usize;
            o[..w].copy_from_slice(&params[re_off + id * w..re_off + (id + 1) * w]);
            o[w..].copy_from_slice(&params[im_off + id * w..im_off + (id + 1) * w]);
        });
    }

    fn embed_backward(&self, dx: &Vec<f32>, ids: &Vec<u32>, ids_off: usize, rows: usize, w: usize, grads: &mut Vec<f32>, re_off: usize, im_off: usize) {
        for r in 0..rows {
            let id = ids[ids_off + r] as usize;
            for j in 0..w {
                grads[re_off + id * w + j] += dx[r * 2 * w + j];
                grads[im_off + id * w + j] += dx[r * 2 * w + w + j];
            }
        }
    }

    fn adamw(&self, params: &mut Vec<f32>, grads: &Vec<f32>, m: &mut Vec<f32>, v: &mut Vec<f32>, len: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, grad_scale: f32, step: u64) {
        let (c1, c2) = (1.0 - beta1.powi(step as i32), 1.0 - beta2.powi(step as i32));
        params[..len].par_iter_mut().zip(m[..len].par_iter_mut()).zip(v[..len].par_iter_mut()).zip(&grads[..len]).for_each(|(((p, m), v), g)| {
            let g = g * grad_scale;
            *m = beta1 * *m + (1.0 - beta1) * g;
            *v = beta2 * *v + (1.0 - beta2) * g * g;
            // Decoupled weight decay, as candle's AdamW: p *= 1 - lr * wd, then the Adam step.
            *p *= 1.0 - lr * weight_decay;
            *p -= lr * (*m / c1) / ((*v / c2).sqrt() + eps);
        });
    }
}
