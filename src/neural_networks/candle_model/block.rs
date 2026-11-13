//! One transformer block of the RM transformer, written with candle tensors so that it runs batched on the GPU
//! and is differentiated by candle's autograd:
//!
//!   x -> x + beta * concat_h attention_h(norm(x))          (SparseSelfAttentionLayerRm)
//!     -> x + beta * leaky(swiglu(norm(x) W1 + b1) W2 + b2)  (FeedForwardLayerRm)
//!
//! The arithmetic follows the RM layers step by step (complex norm with the real variance mean(|x - mean|^2), RoPE,
//! ComplexToLinear on queries and keys, causal local-window softmax on real scores, real parts of V, gated unit
//! a * split SiLU(b) and leaky ReLU),
//! only the attention window is a parameter (`window`, the RM model uses 2). Sequences are not padded.

use candle_core::{DType, Device, Result, Tensor, Var, D};

use super::complex::{CTensor, CVar};

/// RoPE base of the RM model (`INITIAL_BASE * SCALING_FAKTOR`).
pub const ROPE_BASE: f64 = 1250.0;

/// Complex layer norm over the last dimension: (x - mean) / sqrt(var + eps) * gamma + beta with the real variance
/// var = mean(|x - mean|^2).
#[derive(Clone, Debug)]
pub struct ComplexNorm {
    pub gamma: CVar,
    pub beta: CVar,
    pub epsilon: f64,
}

impl ComplexNorm {
    pub fn forward(&self, x: &CTensor) -> Result<CTensor> {
        // Real variance mean(|x - mu|^2), as in complex_layer_norm::layer_norm_row.
        let mean = x.mean_keepdim_last()?;
        let diff = x.sub(&mean)?;
        let std = (diff.mean_norm_sqr_keepdim_last()? + self.epsilon)?.sqrt()?;
        diff.div_real(&std)?.mul(&self.gamma.tensor())?.add(&self.beta.tensor())
    }

    pub fn vars(&self) -> Vec<Var> {
        [self.gamma.vars(), self.beta.vars()].concat().into_iter().cloned().collect()
    }
}

/// One attention head: Q, K, V projections (complex), ComplexToLinear on Q and K (real outputs).
#[derive(Clone, Debug)]
pub struct AttentionHead {
    pub w_q: CVar,
    pub w_k: CVar,
    pub w_v: CVar,
    /// ComplexToLinear weights (only their real parts are used by the RM layer): out = Re(x) W1 + Im(x) W2.
    pub ctl_q_1: Var,
    pub ctl_q_2: Var,
    pub ctl_k_1: Var,
    pub ctl_k_2: Var,
}

impl AttentionHead {
    pub fn vars(&self) -> Vec<Var> {
        let mut v: Vec<Var> = [self.w_q.vars(), self.w_k.vars(), self.w_v.vars()].concat().into_iter().cloned().collect();
        v.extend([self.ctl_q_1.clone(), self.ctl_q_2.clone(), self.ctl_k_1.clone(), self.ctl_k_2.clone()]);
        v
    }
}

/// Self-attention layer: pre-norm, heads, concatenation, scaled residual.
#[derive(Clone, Debug)]
pub struct AttentionLayer {
    pub norm: ComplexNorm,
    pub heads: Vec<AttentionHead>,
    pub beta: f64,
}

/// Feed-forward layer: pre-norm, dense SwiGLU (D -> 2H, gated to H), dense leaky ReLU (H -> D), scaled residual.
#[derive(Clone, Debug)]
pub struct FeedForwardLayer {
    pub norm: ComplexNorm,
    pub w_1: CVar,
    pub b_1: CVar,
    pub w_2: CVar,
    pub b_2: CVar,
    pub beta: f64,
}

/// Position-dependent constants shared by all layers for one sequence length.
pub struct BlockContext {
    /// RoPE cos/sin tables, shape (T, d_head / 2).
    pub rope_cos: Tensor,
    pub rope_sin: Tensor,
    /// Additive attention mask (T, T): 0 where key j may be attended by query i (j <= i, i - j <= window), -inf elsewhere.
    pub mask: Tensor,
}

impl BlockContext {
    pub fn new(seq_len: usize, head_dim: usize, window: usize, dtype: DType, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(seq_len * half);
        let mut sin = Vec::with_capacity(seq_len * half);
        for position in 0..seq_len {
            for i in 0..half {
                // As rope_sin_cos of the RM layer, including its clamp of the angle to [-1, 1].
                let theta = (position as f64 / ROPE_BASE.powf(2.0 * i as f64 / head_dim as f64)).clamp(-1.0, 1.0);
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }
        let mut mask = vec![f64::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in i.saturating_sub(window)..=i {
                mask[i * seq_len + j] = 0.0;
            }
        }
        Ok(Self {
            rope_cos: Tensor::from_vec(cos, (seq_len, half), device)?.to_dtype(dtype)?,
            rope_sin: Tensor::from_vec(sin, (seq_len, half), device)?.to_dtype(dtype)?,
            mask: Tensor::from_vec(mask, (seq_len, seq_len), device)?.to_dtype(dtype)?,
        })
    }
}

/// Rotates (even, odd) feature pairs of a real tensor (B, T, d) by the RoPE angles of their position.
fn rope_real(x: &Tensor, ctx: &BlockContext) -> Result<Tensor> {
    let (b, t, d) = x.dims3()?;
    let pairs = x.reshape((b, t, d / 2, 2))?;
    let even = pairs.narrow(3, 0, 1)?.squeeze(3)?;
    let odd = pairs.narrow(3, 1, 1)?.squeeze(3)?;
    let (cos, sin) = (&ctx.rope_cos, &ctx.rope_sin);
    let out_even = even.broadcast_mul(cos)?.sub(&odd.broadcast_mul(sin)?)?;
    let out_odd = even.broadcast_mul(sin)?.add(&odd.broadcast_mul(cos)?)?;
    Tensor::stack(&[out_even, out_odd], 3)?.reshape((b, t, d))
}

/// RoPE of a complex tensor: real and imaginary parts are rotated with the same angles.
fn rope(x: &CTensor, ctx: &BlockContext) -> Result<CTensor> {
    Ok(CTensor::new(rope_real(&x.re, ctx)?, rope_real(&x.im, ctx)?))
}

/// ComplexToLinear: Re(x) W1 + Im(x) W2 (real).
fn complex_to_linear(x: &CTensor, w_1: &Var, w_2: &Var) -> Result<Tensor> {
    x.re.broadcast_matmul(w_1.as_tensor())?.add(&x.im.broadcast_matmul(w_2.as_tensor())?)
}

impl AttentionLayer {
    pub fn forward(&self, x: &CTensor, ctx: &BlockContext) -> Result<CTensor> {
        let normed = self.norm.forward(x)?;
        let mut outputs = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            let q = rope(&normed.matmul(&head.w_q.tensor())?, ctx)?;
            let k = rope(&normed.matmul(&head.w_k.tensor())?, ctx)?;
            let v = normed.matmul(&head.w_v.tensor())?;
            let q_real = complex_to_linear(&q, &head.ctl_q_1, &head.ctl_q_2)?;
            let k_real = complex_to_linear(&k, &head.ctl_k_1, &head.ctl_k_2)?;

            let d_k = q_real.dim(D::Minus1)? as f64;
            let scores = (q_real.matmul(&k_real.t()?)? / (d_k.sqrt() + 1e-12))?;
            let weights = softmax_last_dim(&scores.broadcast_add(&ctx.mask)?)?;
            // Only the real parts of V reach the output (the RM layer keeps Re(a * v) with real a).
            outputs.push(weights.matmul(&v.re)?);
        }
        let attention = Tensor::cat(&outputs, D::Minus1)?;
        Ok(CTensor::new((x.re.clone() + (attention * self.beta)?)?, x.im.clone()))
    }

    pub fn vars(&self) -> Vec<Var> {
        let mut v = self.norm.vars();
        for head in &self.heads {
            v.extend(head.vars());
        }
        v
    }
}

impl FeedForwardLayer {
    pub fn forward(&self, x: &CTensor) -> Result<CTensor> {
        let normed = self.norm.forward(x)?;
        let hidden = normed.matmul(&self.w_1.tensor())?.add(&self.b_1.tensor())?;
        let (a, b) = hidden.split_half_last()?;
        let gated = a.mul(&b.split_silu()?)?; // as in complex_layer_norm::gated
        let out = gated.matmul(&self.w_2.tensor())?.add(&self.b_2.tensor())?.leaky_relu(0.01)?;
        out.scale(self.beta)?.add(x)
    }

    pub fn vars(&self) -> Vec<Var> {
        let mut v = self.norm.vars();
        for p in [&self.w_1, &self.b_1, &self.w_2, &self.b_2] {
            v.extend(p.vars().into_iter().cloned());
        }
        v
    }
}

/// Numerically stable softmax over the last dimension (rows that are entirely -inf do not occur: the diagonal is
/// always allowed).
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    e.broadcast_div(&e.sum_keepdim(D::Minus1)?)
}
