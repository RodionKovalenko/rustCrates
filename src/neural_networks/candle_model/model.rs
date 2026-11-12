//! The full candle language model:
//!
//!   token ids -> complex embedding (wavelet-initialized) -> complex norm
//!     -> layers x (attention layer + feed-forward layer)      (block.rs, same arithmetic as the RM layers)
//!     -> ComplexToLinear (real, width x width) -> linear head (width x vocab) -> softmax cross-entropy
//!
//! Unlike the RM model the output is a full softmax over the vocabulary (exact, and cheap on a GPU; the EML head
//! exists to make the softmax affordable on the CPU) and the attention window is a parameter.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{backprop::GradStore, DType, Device, Result, Tensor, Var, D};
use rand::{Rng, RngExt};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    block::{AttentionHead, AttentionLayer, BlockContext, ComplexNorm, FeedForwardLayer},
    complex::{CTensor, CVar},
};
use crate::neural_networks::network_layers::wavelet_network::decompose_in_wavelet_2d_levels;

/// Size of the raw token vectors that the wavelet decomposition compresses to the model width.
pub const EMBEDDING_DIM_ORIGINAL: usize = 512;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn_hidden: usize,
    /// Each token attends to itself and the `window` tokens before it.
    pub window: usize,
    /// Training sequence length (tokens per sample).
    pub seq_len: usize,
}

impl Default for ModelConfig {
    /// ~25 M parameters in the transformer layers, ~64 M in total.
    fn default() -> Self {
        Self { vocab_size: 50282, width: 256, layers: 12, heads: 4, ffn_hidden: 1024, window: 256, seq_len: 512 }
    }
}

impl ModelConfig {
    /// Residual scale of the RM model: 1 / (3 * layers)^(1/4).
    pub fn residual_beta(&self) -> f64 {
        1.0 / (3.0 * self.layers as f64).powf(0.25)
    }

    /// Wavelet levels from EMBEDDING_DIM_ORIGINAL down to the width (512 -> 256 is one level).
    pub fn wavelet_levels(&self) -> i32 {
        assert!(EMBEDDING_DIM_ORIGINAL % self.width == 0 && (EMBEDDING_DIM_ORIGINAL / self.width).is_power_of_two(), "width must be 512 / 2^k");
        (EMBEDDING_DIM_ORIGINAL / self.width).trailing_zeros() as i32
    }
}

pub struct Model {
    pub config: ModelConfig,
    pub embedding: CVar,
    pub input_norm: ComplexNorm,
    pub blocks: Vec<(AttentionLayer, FeedForwardLayer)>,
    /// ComplexToLinear: Re(x) W1 + Im(x) W2.
    pub ctl_1: Var,
    pub ctl_2: Var,
    /// Output head (width x vocab) and bias.
    pub head: Var,
    pub head_bias: Var,
}

/// Uniform Xavier values (as `xavier_init` of the RM model).
fn xavier(rows: usize, cols: usize, n: usize, rng: &mut impl Rng) -> Vec<f64> {
    let limit = (6.0 / (rows + cols) as f64).sqrt();
    (0..n).map(|_| rng.random_range(-limit..limit)).collect()
}

fn complex_xavier(rows: usize, cols: usize, dtype: DType, device: &Device, rng: &mut impl Rng) -> Result<CVar> {
    let n = rows * cols;
    let pairs: Vec<(f64, f64)> = xavier(rows, cols, n, rng).into_iter().zip(xavier(rows, cols, n, rng)).collect();
    CVar::from_pairs(&pairs, &[rows, cols], dtype, device)
}

fn real_xavier(rows: usize, cols: usize, dtype: DType, device: &Device, rng: &mut impl Rng) -> Result<Var> {
    Var::from_tensor(&Tensor::from_vec(xavier(rows, cols, rows * cols, rng), (rows, cols), device)?.to_dtype(dtype)?)
}

fn complex_constant(value: f64, n: usize, dtype: DType, device: &Device) -> Result<CVar> {
    CVar::from_pairs(&vec![(value, 0.0); n], &[n], dtype, device)
}

/// One token vector: a random raw vector (uniform in [-0.4, 0.5), as `create_embedding` of the RM model) compressed
/// to the model width by the wavelet decomposition.
pub(crate) fn wavelet_token_vector(levels: i32, width: usize, seed: u64) -> Vec<(f64, f64)> {
    let decomposed = decompose_in_wavelet_2d_levels(&raw_token_vector(seed), levels);
    let row = &decomposed[0][0];
    assert_eq!(row.len(), width, "wavelet decomposition gave {} values, expected the width {}", row.len(), width);
    row.iter().map(|c| (c.re, c.im)).collect()
}

fn raw_token_vector(seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..EMBEDDING_DIM_ORIGINAL).map(|_| rng.random_range(-0.4..0.5)).collect()
}

fn token_seed(seed: u64, token: usize) -> u64 {
    seed ^ (token as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// The wavelet decomposition is linear, so the whole embedding table is the raw vectors times the decomposition of
/// the unit vectors (512 decompositions instead of one per token: seconds instead of minutes for 50k tokens).
pub(crate) fn wavelet_embedding_table(vocab_size: usize, levels: i32, width: usize, seed: u64) -> Vec<(f64, f64)> {
    let basis: Vec<Vec<(f64, f64)>> = (0..EMBEDDING_DIM_ORIGINAL)
        .into_par_iter()
        .map(|i| {
            let mut unit = vec![0.0; EMBEDDING_DIM_ORIGINAL];
            unit[i] = 1.0;
            let decomposed = decompose_in_wavelet_2d_levels(&unit, levels);
            assert_eq!(decomposed[0][0].len(), width, "wavelet decomposition gave {} values, expected the width {}", decomposed[0][0].len(), width);
            decomposed[0][0].iter().map(|c| (c.re, c.im)).collect()
        })
        .collect();
    (0..vocab_size)
        .into_par_iter()
        .flat_map_iter(|token| {
            let raw = raw_token_vector(token_seed(seed, token));
            let mut row = vec![(0.0, 0.0); width];
            for (x, b) in raw.iter().zip(&basis) {
                for (r, c) in row.iter_mut().zip(b) {
                    r.0 += x * c.0;
                    r.1 += x * c.1;
                }
            }
            row
        })
        .collect()
}

impl Model {
    pub fn new(config: ModelConfig, dtype: DType, device: &Device, seed: u64) -> Result<Self> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (w, h, heads) = (config.width, config.ffn_hidden, config.heads);
        let head_dim = w / heads;
        let beta = config.residual_beta();
        let norm = |dtype, device: &Device| -> Result<ComplexNorm> {
            Ok(ComplexNorm { gamma: complex_constant(1.0, w, dtype, device)?, beta: complex_constant(0.0, w, dtype, device)?, epsilon: 1e-8 })
        };

        let levels = config.wavelet_levels();
        let table = wavelet_embedding_table(config.vocab_size, levels, w, seed);
        let embedding = CVar::from_pairs(&table, &[config.vocab_size, w], dtype, device)?;

        let mut blocks = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            let attention_heads = (0..heads)
                .map(|_| {
                    Ok(AttentionHead {
                        w_q: complex_xavier(w, head_dim, dtype, device, &mut rng)?,
                        w_k: complex_xavier(w, head_dim, dtype, device, &mut rng)?,
                        w_v: complex_xavier(w, head_dim, dtype, device, &mut rng)?,
                        ctl_q_1: real_xavier(head_dim, head_dim, dtype, device, &mut rng)?,
                        ctl_q_2: real_xavier(head_dim, head_dim, dtype, device, &mut rng)?,
                        ctl_k_1: real_xavier(head_dim, head_dim, dtype, device, &mut rng)?,
                        ctl_k_2: real_xavier(head_dim, head_dim, dtype, device, &mut rng)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let attention = AttentionLayer { norm: norm(dtype, device)?, heads: attention_heads, beta };
            let ffn = FeedForwardLayer {
                norm: norm(dtype, device)?,
                w_1: complex_xavier(w, 2 * h, dtype, device, &mut rng)?,
                b_1: complex_bias(w, 2 * h, dtype, device, &mut rng)?,
                w_2: complex_xavier(h, w, dtype, device, &mut rng)?,
                b_2: complex_bias(h, w, dtype, device, &mut rng)?,
                beta,
            };
            blocks.push((attention, ffn));
        }

        Ok(Self {
            embedding,
            input_norm: norm(dtype, device)?,
            blocks,
            ctl_1: real_xavier(w, w, dtype, device, &mut rng)?,
            ctl_2: real_xavier(w, w, dtype, device, &mut rng)?,
            head: real_xavier(w, config.vocab_size, dtype, device, &mut rng)?,
            head_bias: Var::from_tensor(&Tensor::zeros(config.vocab_size, dtype, device)?)?,
            config,
        })
    }

    /// Logits (B, T, vocab) for token ids (B, T).
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (b, t) = tokens.dims2()?;
        let ctx = BlockContext::new(t, self.config.width / self.config.heads, self.config.window, self.embedding.re.dtype(), tokens.device())?;
        let flat = tokens.flatten_all()?;
        let w = self.config.width;
        let mut x = CTensor::new(self.embedding.re.index_select(&flat, 0)?.reshape((b, t, w))?, self.embedding.im.index_select(&flat, 0)?.reshape((b, t, w))?);
        x = self.input_norm.forward(&x)?;
        for (attention, ffn) in &self.blocks {
            x = attention.forward(&x, &ctx)?;
            x = ffn.forward(&x)?;
        }
        let real = x.re.broadcast_matmul(self.ctl_1.as_tensor())?.add(&x.im.broadcast_matmul(self.ctl_2.as_tensor())?)?;
        real.broadcast_matmul(self.head.as_tensor())?.broadcast_add(self.head_bias.as_tensor())
    }

    /// Mean next-token cross-entropy of a batch of token windows (B, T + 1): positions 0..T predict 1..=T.
    pub fn loss(&self, windows: &Tensor) -> Result<Tensor> {
        let t = windows.dim(1)? - 1;
        let inputs = windows.narrow(1, 0, t)?;
        let targets = windows.narrow(1, 1, t)?;
        let logits = self.forward(&inputs)?;
        cross_entropy(&logits.flatten_to(1)?, &targets.flatten_all()?)
    }

    /// Embedding lookup and input norm: (B, T) ids -> (B, T, width).
    fn embed(&self, tokens: &Tensor) -> Result<CTensor> {
        let (b, t) = tokens.dims2()?;
        let flat = tokens.flatten_all()?;
        let w = self.config.width;
        let x = CTensor::new(self.embedding.re.index_select(&flat, 0)?.reshape((b, t, w))?, self.embedding.im.index_select(&flat, 0)?.reshape((b, t, w))?);
        self.input_norm.forward(&x)
    }

    /// ComplexToLinear + output head: (..., width) -> logits (..., vocab).
    fn head_logits(&self, x: &CTensor) -> Result<Tensor> {
        let real = x.re.broadcast_matmul(self.ctl_1.as_tensor())?.add(&x.im.broadcast_matmul(self.ctl_2.as_tensor())?)?;
        real.broadcast_matmul(self.head.as_tensor())?.broadcast_add(self.head_bias.as_tensor())
    }

    /// Mean next-token loss of `windows` (B, T + 1) and the gradients of all parameters, computed with gradient
    /// checkpointing: the forward pass keeps only the input of every block; the backward pass recomputes one block
    /// at a time and back-propagates the incoming gradient through it (surrogate loss <output, gradient>). The output
    /// head runs in chunks of `head_chunk` token positions. Memory: one block instead of all, and a slice of the logits.
    pub fn loss_and_grads(&self, windows: &Tensor, head_chunk: usize) -> Result<(f64, GradStore)> {
        let t = windows.dim(1)? - 1;
        let inputs = windows.narrow(1, 0, t)?;
        let targets = windows.narrow(1, 1, t)?.flatten_all()?;
        let (b, w) = (inputs.dim(0)?, self.config.width);
        let ctx = BlockContext::new(t, w / self.config.heads, self.config.window, self.embedding.re.dtype(), windows.device())?;

        // Forward without keeping graphs: only the block inputs are stored.
        let mut block_inputs = Vec::with_capacity(self.blocks.len());
        let mut x = detach(&self.embed(&inputs)?);
        for (attention, ffn) in &self.blocks {
            block_inputs.push(x.clone());
            x = detach(&ffn.forward(&attention.forward(&x, &ctx)?)?);
        }

        // Output head and loss, chunk by chunk over the B * T positions.
        let n = b * t;
        let flat = CTensor::new(x.re.reshape((n, w))?, x.im.reshape((n, w))?);
        let mut total = zero_grads(&self.vars())?;
        let mut loss_sum = 0.0;
        let (mut grad_re, mut grad_im) = (Vec::new(), Vec::new());
        for start in (0..n).step_by(head_chunk.max(1)) {
            let len = head_chunk.min(n - start);
            let leaf = leaf(&CTensor::new(flat.re.narrow(0, start, len)?, flat.im.narrow(0, start, len)?))?;
            let logits = self.head_logits(&leaf.1)?;
            let chunk_loss = (cross_entropy(&logits, &targets.narrow(0, start, len)?)? * (len as f64 / n as f64))?;
            loss_sum += chunk_loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
            let mut grads = chunk_loss.backward()?;
            grad_re.push(grads.remove(leaf.0.re.as_tensor()).unwrap_or(leaf.0.re.zeros_like()?).detach());
            grad_im.push(grads.remove(leaf.0.im.as_tensor()).unwrap_or(leaf.0.im.zeros_like()?).detach());
            add_grads(&mut total, &grads, &self.head_vars())?;
        }
        let mut grad = CTensor::new(Tensor::cat(&grad_re, 0)?.reshape((b, t, w))?, Tensor::cat(&grad_im, 0)?.reshape((b, t, w))?);

        // Blocks in reverse, one recomputed graph at a time.
        for ((attention, ffn), input) in self.blocks.iter().zip(block_inputs.iter()).rev() {
            let (input_var, input_t) = leaf(input)?;
            let out = ffn.forward(&attention.forward(&input_t, &ctx)?)?;
            let surrogate = out.re.mul(&grad.re)?.sum_all()?.add(&out.im.mul(&grad.im)?.sum_all()?)?;
            let mut grads = surrogate.backward()?;
            // Gradients computed by candle keep their graph alive; detached, only the values are kept.
            grad = CTensor::new(
                grads.remove(input_var.re.as_tensor()).unwrap_or(input_t.re.zeros_like()?).detach(),
                grads.remove(input_var.im.as_tensor()).unwrap_or(input_t.im.zeros_like()?).detach(),
            );
            let mut block_vars = attention.vars();
            block_vars.extend(ffn.vars());
            add_grads(&mut total, &grads, &block_vars)?;
        }

        // Embedding and input norm.
        let x0 = self.embed(&inputs)?;
        let surrogate = x0.re.mul(&grad.re)?.sum_all()?.add(&x0.im.mul(&grad.im)?.sum_all()?)?;
        let grads = surrogate.backward()?;
        let mut embed_vars = vec![self.embedding.re.clone(), self.embedding.im.clone()];
        embed_vars.extend(self.input_norm.vars());
        add_grads(&mut total, &grads, &embed_vars)?;
        Ok((loss_sum, total))
    }

    /// Mean next-token loss without keeping any graph (validation): blocks one at a time, head in chunks.
    pub fn loss_no_grad(&self, windows: &Tensor, head_chunk: usize) -> Result<f64> {
        let t = windows.dim(1)? - 1;
        let inputs = windows.narrow(1, 0, t)?;
        let targets = windows.narrow(1, 1, t)?.flatten_all()?;
        let (b, w) = (inputs.dim(0)?, self.config.width);
        let ctx = BlockContext::new(t, w / self.config.heads, self.config.window, self.embedding.re.dtype(), windows.device())?;
        let mut x = detach(&self.embed(&inputs)?);
        for (attention, ffn) in &self.blocks {
            x = detach(&ffn.forward(&attention.forward(&x, &ctx)?)?);
        }
        let n = b * t;
        let flat = CTensor::new(x.re.reshape((n, w))?, x.im.reshape((n, w))?);
        let mut loss_sum = 0.0;
        for start in (0..n).step_by(head_chunk.max(1)) {
            let len = head_chunk.min(n - start);
            let logits = self.head_logits(&CTensor::new(flat.re.narrow(0, start, len)?, flat.im.narrow(0, start, len)?))?.detach();
            loss_sum += cross_entropy(&logits, &targets.narrow(0, start, len)?)?.to_dtype(DType::F64)?.to_scalar::<f64>()? * len as f64;
        }
        Ok(loss_sum / n as f64)
    }

    fn head_vars(&self) -> Vec<Var> {
        vec![self.ctl_1.clone(), self.ctl_2.clone(), self.head.clone(), self.head_bias.clone()]
    }

    pub fn named_vars(&self) -> Vec<(String, Var)> {
        let mut out = vec![("embedding.re".to_string(), self.embedding.re.clone()), ("embedding.im".to_string(), self.embedding.im.clone())];
        let push_norm = |out: &mut Vec<(String, Var)>, prefix: &str, norm: &ComplexNorm| {
            out.push((format!("{prefix}.gamma.re"), norm.gamma.re.clone()));
            out.push((format!("{prefix}.gamma.im"), norm.gamma.im.clone()));
            out.push((format!("{prefix}.beta.re"), norm.beta.re.clone()));
            out.push((format!("{prefix}.beta.im"), norm.beta.im.clone()));
        };
        push_norm(&mut out, "input_norm", &self.input_norm);
        for (l, (attention, ffn)) in self.blocks.iter().enumerate() {
            push_norm(&mut out, &format!("layer{l}.attention.norm"), &attention.norm);
            for (h, head) in attention.heads.iter().enumerate() {
                let p = format!("layer{l}.attention.head{h}");
                for (name, var) in [("w_q", &head.w_q), ("w_k", &head.w_k), ("w_v", &head.w_v)] {
                    out.push((format!("{p}.{name}.re"), var.re.clone()));
                    out.push((format!("{p}.{name}.im"), var.im.clone()));
                }
                for (name, var) in [("ctl_q_1", &head.ctl_q_1), ("ctl_q_2", &head.ctl_q_2), ("ctl_k_1", &head.ctl_k_1), ("ctl_k_2", &head.ctl_k_2)] {
                    out.push((format!("{p}.{name}"), var.clone()));
                }
            }
            push_norm(&mut out, &format!("layer{l}.ffn.norm"), &ffn.norm);
            for (name, var) in [("w_1", &ffn.w_1), ("b_1", &ffn.b_1), ("w_2", &ffn.w_2), ("b_2", &ffn.b_2)] {
                out.push((format!("layer{l}.ffn.{name}.re"), var.re.clone()));
                out.push((format!("layer{l}.ffn.{name}.im"), var.im.clone()));
            }
        }
        out.push(("ctl_1".to_string(), self.ctl_1.clone()));
        out.push(("ctl_2".to_string(), self.ctl_2.clone()));
        out.push(("head".to_string(), self.head.clone()));
        out.push(("head_bias".to_string(), self.head_bias.clone()));
        out
    }

    pub fn vars(&self) -> Vec<Var> {
        self.named_vars().into_iter().map(|(_, v)| v).collect()
    }

    /// Number of real parameters (a complex weight counts twice), in total and in the transformer layers.
    pub fn parameter_counts(&self) -> (usize, usize) {
        let named = self.named_vars();
        let total = named.iter().map(|(_, v)| v.elem_count()).sum();
        let layers = named.iter().filter(|(n, _)| n.starts_with("layer")).map(|(_, v)| v.elem_count()).sum();
        (total, layers)
    }

    /// Saves all parameters (safetensors) and the configuration (`<path>.json`).
    pub fn save(&self, path: &Path) -> Result<()> {
        let tensors: HashMap<String, Tensor> = self.named_vars().into_iter().map(|(n, v)| (n, v.as_tensor().clone())).collect();
        candle_core::safetensors::save(&tensors, path)?;
        std::fs::write(config_path(path), serde_json::to_string_pretty(&self.config).map_err(candle_core::Error::wrap)?)?;
        Ok(())
    }

    /// Loads a model saved with `save` onto `device`.
    pub fn load(path: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let config: ModelConfig = serde_json::from_str(&std::fs::read_to_string(config_path(path))?).map_err(candle_core::Error::wrap)?;
        // Small random model of the same shape, then every parameter is overwritten from the file.
        let model = Self::new_uninitialized(config, dtype, device)?;
        let tensors = candle_core::safetensors::load(path, device)?;
        for (name, var) in model.named_vars() {
            let t = tensors.get(&name).ok_or_else(|| candle_core::Error::Msg(format!("{name} missing in {}", path.display())))?;
            var.set(&t.to_dtype(dtype)?)?;
        }
        Ok(model)
    }

    /// Same shapes as `new`, zero parameters (to be filled by `load`).
    fn new_uninitialized(config: ModelConfig, dtype: DType, device: &Device) -> Result<Self> {
        let (w, h, heads) = (config.width, config.ffn_hidden, config.heads);
        let hd = w / heads;
        let zeros_c = |shape: &[usize]| -> Result<CVar> { Ok(CVar { re: Var::zeros(shape, dtype, device)?, im: Var::zeros(shape, dtype, device)? }) };
        let zeros_r = |shape: &[usize]| Var::zeros(shape, dtype, device);
        let norm = || -> Result<ComplexNorm> { Ok(ComplexNorm { gamma: zeros_c(&[w])?, beta: zeros_c(&[w])?, epsilon: 1e-8 }) };
        let beta = config.residual_beta();
        let mut blocks = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            let heads_vec = (0..heads)
                .map(|_| {
                    Ok(AttentionHead {
                        w_q: zeros_c(&[w, hd])?,
                        w_k: zeros_c(&[w, hd])?,
                        w_v: zeros_c(&[w, hd])?,
                        ctl_q_1: zeros_r(&[hd, hd])?,
                        ctl_q_2: zeros_r(&[hd, hd])?,
                        ctl_k_1: zeros_r(&[hd, hd])?,
                        ctl_k_2: zeros_r(&[hd, hd])?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            blocks.push((
                AttentionLayer { norm: norm()?, heads: heads_vec, beta },
                FeedForwardLayer { norm: norm()?, w_1: zeros_c(&[w, 2 * h])?, b_1: zeros_c(&[2 * h])?, w_2: zeros_c(&[h, w])?, b_2: zeros_c(&[w])?, beta },
            ));
        }
        Ok(Self {
            embedding: zeros_c(&[config.vocab_size, w])?,
            input_norm: norm()?,
            blocks,
            ctl_1: zeros_r(&[w, w])?,
            ctl_2: zeros_r(&[w, w])?,
            head: zeros_r(&[w, config.vocab_size])?,
            head_bias: zeros_r(&[config.vocab_size])?,
            config,
        })
    }
}

fn detach(x: &CTensor) -> CTensor {
    CTensor::new(x.re.detach(), x.im.detach())
}

/// A complex tensor as new leaf variables (its gradient can be read from the grad store).
fn leaf(x: &CTensor) -> Result<(CVar, CTensor)> {
    let var = CVar { re: Var::from_tensor(&x.re)?, im: Var::from_tensor(&x.im)? };
    let t = var.tensor();
    Ok((var, t))
}

/// Adds the gradients of `vars` from `grads` into `total`.
pub fn add_grads(total: &mut GradStore, grads: &GradStore, vars: &[Var]) -> Result<()> {
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            // Detached: a gradient computed by candle references its whole graph, which would stay in memory.
            let sum = match total.get(var.as_tensor()) {
                Some(existing) => existing.add(g)?.detach(),
                None => g.detach(),
            };
            total.insert(var.as_tensor(), sum);
        }
    }
    Ok(())
}

/// Multiplies the gradients of `vars` in `grads` by `factor`.
pub fn scale_grads(grads: &mut GradStore, vars: &[Var], factor: f64) -> Result<()> {
    for var in vars {
        if let Some(g) = grads.remove(var.as_tensor()) {
            grads.insert(var.as_tensor(), (g * factor)?.detach());
        }
    }
    Ok(())
}

/// A grad store with zero gradients for exactly `vars` (candle has no public constructor; the backward pass of
/// sum(0 * v) touches only the variables and scalar intermediates).
fn zero_grads(vars: &[Var]) -> Result<GradStore> {
    let mut sum = Tensor::new(0f32, vars[0].device())?.to_dtype(vars[0].dtype())?;
    for var in vars {
        sum = sum.add(&(var.as_tensor().sum_all()? * 0.0)?)?;
    }
    sum.backward()
}

/// Bias like `initialize_bias` of the RM model: Xavier with fan_in = fan_out = rows of the layer.
fn complex_bias(rows: usize, n: usize, dtype: DType, device: &Device, rng: &mut impl Rng) -> Result<CVar> {
    let pairs: Vec<(f64, f64)> = xavier(rows, rows, n, rng).into_iter().zip(xavier(rows, rows, n, rng)).collect();
    CVar::from_pairs(&pairs, &[n], dtype, device)
}

fn config_path(path: &Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".json");
    p.into()
}

/// Mean cross-entropy of logits (N, V) against target ids (N).
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let max = logits.max_keepdim(D::Minus1)?;
    let shifted = logits.broadcast_sub(&max)?;
    let log_sum = shifted.exp()?.sum_keepdim(D::Minus1)?.log()?;
    let log_probs = shifted.broadcast_sub(&log_sum)?;
    let picked = log_probs.gather(&targets.unsqueeze(1)?, 1)?;
    picked.neg()?.mean_all()
}
