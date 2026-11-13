//! The operations the engine is built from. `cpu::CpuBackend` implements them in Rust (reference and tests),
//! `cuda::CudaBackend` with cuBLAS and own CUDA kernels. The model (forward pass and hand-written backward pass)
//! is written once against this trait.
//!
//! Layouts (all `f32`, row-major):
//! - a complex matrix with `rows` rows and `n` columns is stored *planar per row*: every row is
//!   `[re_0 .. re_{n-1} | im_0 .. im_{n-1}]` (2n floats). A complex product X W is then one real product
//!   `[Xr | Xi] · [[Wr, Wi], [-Wi, Wr]]` whose result is again planar (see `block` in `model.rs`).
//! - buffer offsets and leading dimensions are counted in floats.

/// Batched, strided GEMM on row-major matrices:
/// `C_i = alpha * op(A_i) * op(B_i) + beta * C_i` for `i < batch`, with `A_i` starting at `a_off + i * stride_a`
/// (likewise B, C). `op(A)` is `m x k`, `op(B)` is `k x n`, `C` is `m x n`; `trans_*` transposes the stored matrix.
#[derive(Clone, Copy, Debug)]
pub struct Gemm {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub trans_a: bool,
    pub trans_b: bool,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub a_off: usize,
    pub b_off: usize,
    pub c_off: usize,
    pub batch: usize,
    pub stride_a: usize,
    pub stride_b: usize,
    pub stride_c: usize,
    pub alpha: f32,
    pub beta: f32,
}

impl Gemm {
    /// A single product of contiguous matrices: C (m x n) = op(A) op(B), stored without padding.
    pub fn new(m: usize, n: usize, k: usize, trans_a: bool, trans_b: bool) -> Self {
        Self {
            m,
            n,
            k,
            trans_a,
            trans_b,
            lda: if trans_a { m } else { k },
            ldb: if trans_b { k } else { n },
            ldc: n,
            a_off: 0,
            b_off: 0,
            c_off: 0,
            batch: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// `beta = 1`: the product is added to C.
    pub fn accumulate(mut self) -> Self {
        self.beta = 1.0;
        self
    }

    pub fn ld(mut self, lda: usize, ldb: usize, ldc: usize) -> Self {
        (self.lda, self.ldb, self.ldc) = (lda, ldb, ldc);
        self
    }

    pub fn offsets(mut self, a_off: usize, b_off: usize, c_off: usize) -> Self {
        (self.a_off, self.b_off, self.c_off) = (a_off, b_off, c_off);
        self
    }

    pub fn batched(mut self, batch: usize, stride_a: usize, stride_b: usize, stride_c: usize) -> Self {
        (self.batch, self.stride_a, self.stride_b, self.stride_c) = (batch, stride_a, stride_b, stride_c);
        self
    }
}

pub trait Backend {
    /// A buffer of `f32` (host memory for the CPU, device memory for CUDA).
    type Buf;
    /// A buffer of `u32` token ids.
    type Ids;

    fn zeros(&self, len: usize) -> Self::Buf;
    fn upload(&self, data: &[f32]) -> Self::Buf;
    fn download(&self, buf: &Self::Buf) -> Vec<f32>;
    fn upload_ids(&self, ids: &[u32]) -> Self::Ids;
    fn len(&self, buf: &Self::Buf) -> usize;

    /// `buf[off .. off + len] = value`.
    fn fill(&self, buf: &mut Self::Buf, off: usize, len: usize, value: f32);
    /// `dst[dst_off ..][.. len] = src[src_off ..][.. len]`.
    fn copy(&self, src: &Self::Buf, src_off: usize, dst: &mut Self::Buf, dst_off: usize, len: usize);
    /// `dst[..len] += alpha * src[..len]`.
    fn axpy(&self, alpha: f32, src: &Self::Buf, dst: &mut Self::Buf, len: usize);
    /// `buf[..len] *= alpha`.
    fn scale(&self, buf: &mut Self::Buf, len: usize, alpha: f32);
    /// Sum of squares of `buf[..len]` (accumulated in f64 on the CPU, f32 per block on the GPU).
    fn sum_squares(&self, buf: &Self::Buf, len: usize) -> f64;
    /// Whether all values of `buf[..len]` are finite.
    fn all_finite(&self, buf: &Self::Buf, len: usize) -> bool;

    fn gemm(&self, g: &Gemm, a: &Self::Buf, b: &Self::Buf, c: &mut Self::Buf);

    /// Block matrix of a complex weight W (k x n) with Wr at `re_off`, Wi at `im_off` of `params`:
    /// `out` (2k x 2n) = [[Wr, Wi], [-Wi, Wr]].
    fn build_block(&self, params: &Self::Buf, re_off: usize, im_off: usize, k: usize, n: usize, out: &mut Self::Buf);
    /// Folds the gradient of a block matrix, G (2k x 2n) = [[G11, G12], [G21, G22]], into the gradients of
    /// Wr and Wi: `grads[re_off] += G11 + G22`, `grads[im_off] += G12 - G21`.
    fn fold_block_grad(&self, block_grad: &Self::Buf, k: usize, n: usize, grads: &mut Self::Buf, re_off: usize, im_off: usize);

    /// `y[r, :] += params[off .. off + cols]` for every row r of the `rows x cols` matrix y.
    fn add_row_vector(&self, y: &mut Self::Buf, rows: usize, cols: usize, params: &Self::Buf, off: usize);
    /// `grads[off + j] += sum_r dy[r, j]`.
    fn column_sums(&self, dy: &Self::Buf, rows: usize, cols: usize, grads: &mut Self::Buf, off: usize);

    /// Complex layer norm (real variance) of `rows` planar rows of width `w`. gamma and beta are planar vectors
    /// (2w floats each) at `gamma_off` / `beta_off` of `params`. `stats` receives (mean_re, mean_im, 1 / sigma)
    /// per row.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(&self, x: &Self::Buf, rows: usize, w: usize, eps: f32, params: &Self::Buf, gamma_off: usize, beta_off: usize, y: &mut Self::Buf, stats: &mut Self::Buf);
    /// Backward pass of `layer_norm`: writes `dx`, adds the gamma / beta gradients to `grads`.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_backward(&self, x: &Self::Buf, stats: &Self::Buf, dy: &Self::Buf, rows: usize, w: usize, params: &Self::Buf, gamma_off: usize, dx: &mut Self::Buf, grads: &mut Self::Buf, dgamma_off: usize, dbeta_off: usize);

    /// RoPE of the queries and keys: reads the planar complex columns `[re_off + c]` / `[im_off + c]`
    /// (c < heads * head_dim) of `src` (row stride `src_ld`), rotates the (even, odd) pairs of every head by the
    /// angle of the row's position (row % seq_len) and writes them *head-planar* to `dst` (row stride
    /// 2 * heads * head_dim, head h at `h * 2 * head_dim`: `[re | im]`).
    #[allow(clippy::too_many_arguments)]
    fn rope_to_heads(&self, src: &Self::Buf, src_ld: usize, re_off: usize, im_off: usize, rows: usize, seq_len: usize, heads: usize, head_dim: usize, dst: &mut Self::Buf);
    /// Backward pass of `rope_to_heads`: rotates the head-planar gradient back and writes it to the planar
    /// columns of `dst` (other columns untouched).
    #[allow(clippy::too_many_arguments)]
    fn rope_from_heads(&self, src: &Self::Buf, rows: usize, seq_len: usize, heads: usize, head_dim: usize, dst: &mut Self::Buf, dst_ld: usize, re_off: usize, im_off: usize);

    /// Row-wise softmax of `groups` blocks of `t x t` scores, in place; key j is visible to query i when
    /// `i - window <= j <= i`, all other weights are 0.
    fn softmax_window(&self, s: &mut Self::Buf, groups: usize, t: usize, window: usize);
    /// Backward pass: turns `dp` (gradient of the weights P) into the gradient of the scores,
    /// `ds = P * (dp - rowsum(P * dp))`, in place.
    fn softmax_window_backward(&self, p: &Self::Buf, dp: &mut Self::Buf, groups: usize, t: usize);

    /// Attention residual: `y = x`, `y_re += beta * att` (att is real, `rows x w`; the imaginary part passes).
    fn attention_residual(&self, x: &Self::Buf, att: &Self::Buf, rows: usize, w: usize, beta: f32, y: &mut Self::Buf);
    /// `out = beta * dy_re` (`rows x w`): the gradient of the attention output.
    fn real_part_scaled(&self, dy: &Self::Buf, rows: usize, w: usize, beta: f32, out: &mut Self::Buf);

    /// Gated unit on planar rows `[ar | br | ai | bi]` (each `hidden` wide): `g = a * split_silu(b)`, planar
    /// `rows x hidden`.
    fn gated(&self, h: &Self::Buf, rows: usize, hidden: usize, g: &mut Self::Buf);
    fn gated_backward(&self, h: &Self::Buf, dg: &Self::Buf, rows: usize, hidden: usize, dh: &mut Self::Buf);

    /// `y = x + beta * leaky_relu(o)` on planar rows (the complex leaky ReLU scales the whole value by `slope`
    /// where `re(o) <= 0`).
    #[allow(clippy::too_many_arguments)]
    fn leaky_residual(&self, o: &Self::Buf, x: &Self::Buf, rows: usize, w: usize, beta: f32, slope: f32, y: &mut Self::Buf);
    /// `d_o = beta * leaky_relu'(o) * dy`.
    #[allow(clippy::too_many_arguments)]
    fn leaky_residual_backward(&self, o: &Self::Buf, dy: &Self::Buf, rows: usize, w: usize, beta: f32, slope: f32, d_o: &mut Self::Buf);

    /// Softmax cross-entropy of `rows x vocab` logits against `targets[t_off ..]`: returns the summed loss and
    /// replaces the logits by `scale * (softmax - onehot)` (the gradient of `scale * loss`).
    fn cross_entropy(&self, logits: &mut Self::Buf, rows: usize, vocab: usize, targets: &Self::Ids, t_off: usize, scale: f32) -> f64;

    /// Embedding lookup: planar rows from the tables Er (`re_off`) and Ei (`im_off`), each `vocab x w`.
    #[allow(clippy::too_many_arguments)]
    fn embed(&self, params: &Self::Buf, re_off: usize, im_off: usize, ids: &Self::Ids, ids_off: usize, rows: usize, w: usize, out: &mut Self::Buf);
    /// Backward pass: adds the planar rows of `dx` to the gradient rows of their tokens.
    #[allow(clippy::too_many_arguments)]
    fn embed_backward(&self, dx: &Self::Buf, ids: &Self::Ids, ids_off: usize, rows: usize, w: usize, grads: &mut Self::Buf, re_off: usize, im_off: usize);

    /// AdamW step on `params[..len]` with the gradients `grads` and the moments `m`, `v`
    /// (`grad_scale` multiplies the gradients first; `step` counts from 1 for the bias correction).
    #[allow(clippy::too_many_arguments)]
    fn adamw(&self, params: &mut Self::Buf, grads: &Self::Buf, m: &mut Self::Buf, v: &mut Self::Buf, len: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, grad_scale: f32, step: u64);
}
