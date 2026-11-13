//! The model: parameter layout, initialization, forward pass and hand-written backward pass.
//!
//! All parameters live in one flat buffer (`params`), their gradients in a second one of the same layout (`grads`),
//! so that clipping and AdamW are single passes. Complex weights W (k x n) are two segments Wr, Wi; a complex
//! product uses the block matrix W' = [[Wr, Wi], [-Wi, Wr]] (2k x 2n), rebuilt from the parameters after every
//! update (`refresh_blocks`):
//!
//!   forward   Y' = X' W'          with X' = [Xr | Xi] (planar rows), Y' = [Yr | Yi]
//!   backward  dX' = dY' W'^T,  G = X'^T dY'  ->  dWr = G11 + G22, dWi = G12 - G21
//!
//! Gradients are dL/d(real part) + i dL/d(imaginary part) throughout (the convention of the RM layers).

use rand::{Rng, RngExt, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::backend::{Backend, Gemm};
use crate::neural_networks::network_layers::wavelet_network::decompose_in_wavelet_2d_levels;

/// Size of the raw token vectors that the wavelet decomposition compresses to the model width.
pub const EMBEDDING_DIM_ORIGINAL: usize = 512;
pub const NORM_EPSILON: f32 = 1e-8;
pub const LEAKY_SLOPE: f32 = 0.01;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EngineConfig {
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

impl Default for EngineConfig {
    fn default() -> Self {
        Self { vocab_size: 50282, width: 256, layers: 12, heads: 4, ffn_hidden: 1024, window: 256, seq_len: 512 }
    }
}

impl EngineConfig {
    /// Residual scale of the RM model: 1 / (3 * layers)^(1/4).
    pub fn residual_beta(&self) -> f32 {
        (1.0 / (3.0 * self.layers as f64).powf(0.25)) as f32
    }

    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }

    pub fn wavelet_levels(&self) -> i32 {
        assert!(EMBEDDING_DIM_ORIGINAL % self.width == 0 && (EMBEDDING_DIM_ORIGINAL / self.width).is_power_of_two(), "width must be 512 / 2^k");
        (EMBEDDING_DIM_ORIGINAL / self.width).trailing_zeros() as i32
    }
}

/// A named segment of the parameter buffer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub name: String,
    pub offset: usize,
    pub shape: Vec<usize>,
}

impl Segment {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Offsets of a complex layer norm: gamma (re then im, 2w) and beta (2w).
#[derive(Clone, Copy, Debug)]
pub struct NormOffsets {
    pub gamma: usize,
    pub beta: usize,
}

/// Offsets of a complex weight (re and im segments, each k x n).
#[derive(Clone, Copy, Debug)]
pub struct ComplexWeight {
    pub re: usize,
    pub im: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Clone, Debug)]
pub struct LayerOffsets {
    pub attention_norm: NormOffsets,
    /// Q, K and V of all heads: w x 3w, columns [q heads | k heads | v heads].
    pub wqkv: ComplexWeight,
    /// ComplexToLinear of the queries / keys, per head a (2 head_dim x head_dim) block [W1; W2].
    pub ctl_q: usize,
    pub ctl_k: usize,
    pub ffn_norm: NormOffsets,
    pub w1: ComplexWeight,
    /// Planar bias (re then im).
    pub b1: usize,
    pub w2: ComplexWeight,
    pub b2: usize,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub segments: Vec<Segment>,
    pub total: usize,
    pub embedding_re: usize,
    pub embedding_im: usize,
    pub input_norm: NormOffsets,
    pub layers: Vec<LayerOffsets>,
    /// ComplexToLinear before the head: (2w x w) = [W1; W2].
    pub ctl: usize,
    pub head: usize,
    pub head_bias: usize,
}

impl Layout {
    pub fn new(c: &EngineConfig) -> Self {
        let mut segments = Vec::new();
        let mut total = 0;
        let mut add = |name: String, shape: Vec<usize>| -> usize {
            let offset = total;
            total += shape.iter().product::<usize>();
            segments.push(Segment { name, offset, shape });
            offset
        };
        let (w, hd, hf) = (c.width, c.head_dim(), c.ffn_hidden);
        let norm = |add: &mut dyn FnMut(String, Vec<usize>) -> usize, prefix: &str| -> NormOffsets {
            let gamma = add(format!("{prefix}.gamma.re"), vec![w]);
            add(format!("{prefix}.gamma.im"), vec![w]);
            let beta = add(format!("{prefix}.beta.re"), vec![w]);
            add(format!("{prefix}.beta.im"), vec![w]);
            NormOffsets { gamma, beta }
        };
        let complex = |add: &mut dyn FnMut(String, Vec<usize>) -> usize, name: &str, k: usize, n: usize| -> ComplexWeight {
            let re = add(format!("{name}.re"), vec![k, n]);
            let im = add(format!("{name}.im"), vec![k, n]);
            ComplexWeight { re, im, k, n }
        };
        let embedding_re = add("embedding.re".into(), vec![c.vocab_size, w]);
        let embedding_im = add("embedding.im".into(), vec![c.vocab_size, w]);
        let input_norm = norm(&mut add, "input_norm");
        let mut layers = Vec::new();
        for l in 0..c.layers {
            let attention_norm = norm(&mut add, &format!("layer{l}.attention.norm"));
            let wqkv = complex(&mut add, &format!("layer{l}.attention.wqkv"), w, 3 * w);
            let ctl_q = add(format!("layer{l}.attention.ctl_q"), vec![c.heads, 2 * hd, hd]);
            let ctl_k = add(format!("layer{l}.attention.ctl_k"), vec![c.heads, 2 * hd, hd]);
            let ffn_norm = norm(&mut add, &format!("layer{l}.ffn.norm"));
            let w1 = complex(&mut add, &format!("layer{l}.ffn.w_1"), w, 2 * hf);
            let b1 = add(format!("layer{l}.ffn.b_1.re"), vec![2 * hf]);
            add(format!("layer{l}.ffn.b_1.im"), vec![2 * hf]);
            let w2 = complex(&mut add, &format!("layer{l}.ffn.w_2"), hf, w);
            let b2 = add(format!("layer{l}.ffn.b_2.re"), vec![w]);
            add(format!("layer{l}.ffn.b_2.im"), vec![w]);
            layers.push(LayerOffsets { attention_norm, wqkv, ctl_q, ctl_k, ffn_norm, w1, b1, w2, b2 });
        }
        let ctl = add("ctl".into(), vec![2 * w, w]);
        let head = add("head".into(), vec![w, c.vocab_size]);
        let head_bias = add("head_bias".into(), vec![c.vocab_size]);
        Self { segments, total, embedding_re, embedding_im, input_norm, layers, ctl, head, head_bias }
    }

    pub fn segment(&self, name: &str) -> Option<&Segment> {
        self.segments.iter().find(|s| s.name == name)
    }

    /// Real parameters in total and in the transformer layers.
    pub fn parameter_counts(&self) -> (usize, usize) {
        (self.total, self.segments.iter().filter(|s| s.name.starts_with("layer")).map(|s| s.len()).sum())
    }
}

/// Uniform Xavier values (as `xavier_init` of the RM model).
fn xavier(fan_in: usize, fan_out: usize, values: &mut [f32], rng: &mut impl Rng) {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    values.iter_mut().for_each(|v| *v = rng.random_range(-limit..limit) as f32);
}

fn raw_token_vector(seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..EMBEDDING_DIM_ORIGINAL).map(|_| rng.random_range(-0.4..0.5)).collect()
}

/// Wavelet-initialized embedding table (the decomposition is linear: raw vectors times the decomposition of the
/// unit vectors), as `wavelet_embedding_table` of the candle model.
pub fn wavelet_embedding_table(vocab_size: usize, levels: i32, width: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let basis: Vec<Vec<(f64, f64)>> = (0..EMBEDDING_DIM_ORIGINAL)
        .into_par_iter()
        .map(|i| {
            let mut unit = vec![0.0; EMBEDDING_DIM_ORIGINAL];
            unit[i] = 1.0;
            let decomposed = decompose_in_wavelet_2d_levels(&unit, levels);
            assert_eq!(decomposed[0][0].len(), width, "wavelet decomposition gave {} values, expected {}", decomposed[0][0].len(), width);
            decomposed[0][0].iter().map(|c| (c.re as f64, c.im as f64)).collect()
        })
        .collect();
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..vocab_size)
        .into_par_iter()
        .map(|token| {
            let raw = raw_token_vector(seed ^ (token as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut row = vec![(0.0f64, 0.0f64); width];
            for (x, b) in raw.iter().zip(&basis) {
                for (r, c) in row.iter_mut().zip(b) {
                    r.0 += x * c.0;
                    r.1 += x * c.1;
                }
            }
            (row.iter().map(|v| v.0 as f32).collect(), row.iter().map(|v| v.1 as f32).collect())
        })
        .collect();
    (rows.iter().flat_map(|r| r.0.clone()).collect(), rows.iter().flat_map(|r| r.1.clone()).collect())
}

/// Initial parameters (as the candle model: wavelet embedding, Xavier weights, biases like `initialize_bias`,
/// norms gamma = 1, beta = 0, head bias 0).
pub fn initial_parameters(c: &EngineConfig, layout: &Layout, seed: u64) -> Vec<f32> {
    let mut p = vec![0.0f32; layout.total];
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (w, hd, hf) = (c.width, c.head_dim(), c.ffn_hidden);
    let (er, ei) = wavelet_embedding_table(c.vocab_size, c.wavelet_levels(), w, seed);
    p[layout.embedding_re..layout.embedding_re + er.len()].copy_from_slice(&er);
    p[layout.embedding_im..layout.embedding_im + ei.len()].copy_from_slice(&ei);
    let set_norm = |p: &mut [f32], n: NormOffsets| p[n.gamma..n.gamma + w].fill(1.0);
    set_norm(&mut p, layout.input_norm);
    for l in &layout.layers {
        set_norm(&mut p, l.attention_norm);
        set_norm(&mut p, l.ffn_norm);
        // Every head's Q, K, V is a w x head_dim Xavier matrix (fan-in w, fan-out head_dim).
        for part in [l.wqkv.re, l.wqkv.im] {
            for v in p[part..part + w * 3 * w].iter_mut() {
                *v = rng.random_range(-((6.0 / (w + hd) as f64).sqrt())..(6.0 / (w + hd) as f64).sqrt()) as f32;
            }
        }
        xavier(hd, hd, &mut p[l.ctl_q..l.ctl_q + c.heads * 2 * hd * hd], &mut rng);
        xavier(hd, hd, &mut p[l.ctl_k..l.ctl_k + c.heads * 2 * hd * hd], &mut rng);
        for (weight, bias, bias_len) in [(l.w1, l.b1, 2 * hf), (l.w2, l.b2, w)] {
            xavier(weight.k, weight.n, &mut p[weight.re..weight.re + weight.k * weight.n], &mut rng);
            xavier(weight.k, weight.n, &mut p[weight.im..weight.im + weight.k * weight.n], &mut rng);
            // initialize_bias: Xavier with fan_in = fan_out = rows of the layer.
            xavier(weight.k, weight.k, &mut p[bias..bias + 2 * bias_len], &mut rng);
        }
    }
    xavier(w, w, &mut p[layout.ctl..layout.ctl + 2 * w * w], &mut rng);
    xavier(w, c.vocab_size, &mut p[layout.head..layout.head + w * c.vocab_size], &mut rng);
    p
}

/// Activations of one layer kept for the backward pass.
struct LayerActivations<B: Backend> {
    input: B::Buf,
    h1: B::Buf,
    stats1: B::Buf,
    qkv: B::Buf,
    qh: B::Buf,
    kh: B::Buf,
    qr: B::Buf,
    kr: B::Buf,
    probs: B::Buf,
    att: B::Buf,
    mid: B::Buf,
    h2: B::Buf,
    stats2: B::Buf,
    hidden: B::Buf,
    gated: B::Buf,
    out: B::Buf,
}

/// Buffers for one micro-batch size (`rows` = batch * seq_len token positions).
struct Workspace<B: Backend> {
    rows: usize,
    batch: usize,
    embedded: B::Buf,
    stats0: B::Buf,
    layers: Vec<LayerActivations<B>>,
    final_x: B::Buf,
    real: B::Buf,
    d_real: B::Buf,
    logits: B::Buf,
    // backward temporaries
    dx: B::Buf,
    dtmp: B::Buf,
    d_out: B::Buf,
    d_gated: B::Buf,
    d_hidden: B::Buf,
    d_att: B::Buf,
    d_probs: B::Buf,
    d_qkv: B::Buf,
    d_qr: B::Buf,
    d_kr: B::Buf,
    d_qh: B::Buf,
    d_kh: B::Buf,
}

/// Token positions per head chunk: the logits of a chunk (`chunk x vocab`) are the largest buffer.
pub const HEAD_CHUNK: usize = 1024;

pub struct Model<B: Backend> {
    pub config: EngineConfig,
    pub layout: Layout,
    pub backend: B,
    pub params: B::Buf,
    pub grads: B::Buf,
    /// Block matrices of the complex weights, per layer (wqkv, w1, w2).
    blocks: Vec<[B::Buf; 3]>,
    /// Gradient of one block matrix (the largest one).
    block_grad: B::Buf,
    workspace: Option<Workspace<B>>,
}

impl<B: Backend> Model<B> {
    pub fn new(backend: B, config: EngineConfig, seed: u64) -> Self {
        let layout = Layout::new(&config);
        let params = initial_parameters(&config, &layout, seed);
        Self::from_parameters(backend, config, &params)
    }

    pub fn from_parameters(backend: B, config: EngineConfig, params: &[f32]) -> Self {
        let layout = Layout::new(&config);
        assert_eq!(params.len(), layout.total, "parameter count does not match the configuration");
        let params = backend.upload(params);
        let grads = backend.zeros(layout.total);
        let blocks = layout
            .layers
            .iter()
            .map(|l| [backend.zeros(4 * l.wqkv.k * l.wqkv.n), backend.zeros(4 * l.w1.k * l.w1.n), backend.zeros(4 * l.w2.k * l.w2.n)])
            .collect();
        let largest = layout.layers.iter().map(|l| [l.wqkv, l.w1, l.w2].iter().map(|c| 4 * c.k * c.n).max().unwrap()).max().unwrap_or(1);
        let block_grad = backend.zeros(largest);
        let mut model = Self { config, layout, backend, params, grads, blocks, block_grad, workspace: None };
        model.refresh_blocks();
        model
    }

    /// Rebuilds the block matrices from the parameters (after every parameter update).
    pub fn refresh_blocks(&mut self) {
        for (l, blocks) in self.layout.layers.iter().zip(self.blocks.iter_mut()) {
            for (weight, block) in [l.wqkv, l.w1, l.w2].iter().zip(blocks.iter_mut()) {
                self.backend.build_block(&self.params, weight.re, weight.im, weight.k, weight.n, block);
            }
        }
    }

    pub fn zero_grads(&mut self) {
        let len = self.layout.total;
        self.backend.fill(&mut self.grads, 0, len, 0.0);
    }

    fn workspace(&mut self, batch: usize) -> Workspace<B> {
        let rows = batch * self.config.seq_len;
        if let Some(ws) = self.workspace.take() {
            if ws.rows == rows {
                return ws;
            }
        }
        let b = &self.backend;
        let c = &self.config;
        let (w, hf, t) = (c.width, c.ffn_hidden, c.seq_len);
        let z = |len: usize| b.zeros(len.max(1));
        let layers = (0..c.layers)
            .map(|_| LayerActivations {
                input: z(rows * 2 * w),
                h1: z(rows * 2 * w),
                stats1: z(rows * 3),
                qkv: z(rows * 6 * w),
                qh: z(rows * 2 * w),
                kh: z(rows * 2 * w),
                qr: z(rows * w),
                kr: z(rows * w),
                probs: z(batch * c.heads * t * t),
                att: z(rows * w),
                mid: z(rows * 2 * w),
                h2: z(rows * 2 * w),
                stats2: z(rows * 3),
                hidden: z(rows * 4 * hf),
                gated: z(rows * 2 * hf),
                out: z(rows * 2 * w),
            })
            .collect();
        let chunk = HEAD_CHUNK.min(rows);
        Workspace {
            rows,
            batch,
            embedded: z(rows * 2 * w),
            stats0: z(rows * 3),
            layers,
            final_x: z(rows * 2 * w),
            real: z(rows * w),
            d_real: z(rows * w),
            logits: z(chunk * c.vocab_size),
            dx: z(rows * 2 * w),
            dtmp: z(rows * 2 * w),
            d_out: z(rows * 2 * w),
            d_gated: z(rows * 2 * hf),
            d_hidden: z(rows * 4 * hf),
            d_att: z(rows * w),
            d_probs: z(batch * c.heads * t * t),
            d_qkv: z(rows * 6 * w),
            d_qr: z(rows * w),
            d_kr: z(rows * w),
            d_qh: z(rows * 2 * w),
            d_kh: z(rows * 2 * w),
        }
    }

    /// Mean next-token loss of `windows` (batch x (seq_len + 1) token ids). With `backward`, adds
    /// `grad_scale * d(mean loss)` to `grads` (several micro-batches can be accumulated).
    pub fn forward_backward(&mut self, windows: &[u32], backward: bool, grad_scale: f32) -> f64 {
        let t = self.config.seq_len;
        assert_eq!(windows.len() % (t + 1), 0, "windows must be batch x (seq_len + 1) ids");
        let batch = windows.len() / (t + 1);
        let mut inputs = Vec::with_capacity(batch * t);
        let mut targets = Vec::with_capacity(batch * t);
        for wdw in windows.chunks(t + 1) {
            inputs.extend_from_slice(&wdw[..t]);
            targets.extend_from_slice(&wdw[1..]);
        }
        let inputs = self.backend.upload_ids(&inputs);
        let targets = self.backend.upload_ids(&targets);
        let mut ws = self.workspace(batch);
        self.forward(&mut ws, &inputs);
        let loss = self.head(&mut ws, &targets, backward, grad_scale);
        if backward {
            self.backward(&mut ws, &inputs);
        }
        self.workspace = Some(ws);
        loss
    }

    fn forward(&self, ws: &mut Workspace<B>, inputs: &B::Ids) {
        let (b, c, l) = (&self.backend, &self.config, &self.layout);
        let (rows, w) = (ws.rows, c.width);
        b.embed(&self.params, l.embedding_re, l.embedding_im, inputs, 0, rows, w, &mut ws.embedded);
        let first = &mut ws.layers[0].input;
        b.layer_norm(&ws.embedded, rows, w, NORM_EPSILON, &self.params, l.input_norm.gamma, l.input_norm.beta, first, &mut ws.stats0);
        let beta = c.residual_beta();
        for i in 0..c.layers {
            self.layer_forward(ws, i);
            // Layer output y = mid + beta leaky(o): the input of the next layer (or of the head).
            let (head, tail) = ws.layers.split_at_mut(i + 1);
            let next = if i + 1 < c.layers { &mut tail[0].input } else { &mut ws.final_x };
            b.leaky_residual(&head[i].out, &head[i].mid, rows, w, beta, LEAKY_SLOPE, next);
        }
    }

    /// One layer up to the feed-forward pre-activation o = g W2 + b2 (in `out`); `forward` adds the residual.
    fn layer_forward(&self, ws: &mut Workspace<B>, i: usize) {
        let (b, c) = (&self.backend, &self.config);
        let lo = &self.layout.layers[i];
        let [wqkv, w1, w2] = &self.blocks[i];
        let (rows, batch, w, t, heads, hd, hf) = (ws.rows, ws.batch, c.width, c.seq_len, c.heads, c.head_dim(), c.ffn_hidden);
        let beta = c.residual_beta();
        let a = &mut ws.layers[i];

        // Attention.
        b.layer_norm(&a.input, rows, w, NORM_EPSILON, &self.params, lo.attention_norm.gamma, lo.attention_norm.beta, &mut a.h1, &mut a.stats1);
        b.gemm(&Gemm::new(rows, 6 * w, 2 * w, false, false), &a.h1, wqkv, &mut a.qkv);
        b.rope_to_heads(&a.qkv, 6 * w, 0, 3 * w, rows, t, heads, hd, &mut a.qh);
        b.rope_to_heads(&a.qkv, 6 * w, w, 4 * w, rows, t, heads, hd, &mut a.kh);
        // ComplexToLinear per head: [Qr_h | Qi_h] (rows x 2hd) times [W1; W2] (2hd x hd).
        let ctl = |ctl_off: usize| Gemm::new(rows, hd, 2 * hd, false, false).ld(2 * w, hd, w).offsets(0, ctl_off, 0).batched(heads, 2 * hd, 2 * hd * hd, hd);
        b.gemm(&ctl(lo.ctl_q), &a.qh, &self.params, &mut a.qr);
        b.gemm(&ctl(lo.ctl_k), &a.kh, &self.params, &mut a.kr);
        let scale = 1.0 / ((hd as f64).sqrt() + 1e-12) as f32;
        for s in 0..batch {
            // Scores of all heads of sample s: q_h k_h^T (t x t).
            let g = Gemm::new(t, t, hd, false, true).ld(w, w, t).offsets(s * t * w, s * t * w, s * heads * t * t).batched(heads, hd, hd, t * t).alpha(scale);
            b.gemm(&g, &a.qr, &a.kr, &mut a.probs);
        }
        b.softmax_window(&mut a.probs, batch * heads, t, c.window);
        for s in 0..batch {
            // Weighted real parts of V (columns 2w + h hd of qkv).
            let g = Gemm::new(t, hd, t, false, false).ld(t, 6 * w, w).offsets(s * heads * t * t, s * t * 6 * w + 2 * w, s * t * w).batched(heads, t * t, hd, hd);
            b.gemm(&g, &a.probs, &a.qkv, &mut a.att);
        }
        b.attention_residual(&a.input, &a.att, rows, w, beta, &mut a.mid);

        // Feed-forward.
        b.layer_norm(&a.mid, rows, w, NORM_EPSILON, &self.params, lo.ffn_norm.gamma, lo.ffn_norm.beta, &mut a.h2, &mut a.stats2);
        b.gemm(&Gemm::new(rows, 4 * hf, 2 * w, false, false), &a.h2, w1, &mut a.hidden);
        b.add_row_vector(&mut a.hidden, rows, 4 * hf, &self.params, lo.b1);
        b.gated(&a.hidden, rows, hf, &mut a.gated);
        b.gemm(&Gemm::new(rows, 2 * w, 2 * hf, false, false), &a.gated, w2, &mut a.out);
        b.add_row_vector(&mut a.out, rows, 2 * w, &self.params, lo.b2);
    }

    /// Output head and loss in chunks of positions; with `backward`, the head gradients and d(final x).
    fn head(&mut self, ws: &mut Workspace<B>, targets: &B::Ids, backward: bool, grad_scale: f32) -> f64 {
        let (c, l) = (&self.config, &self.layout);
        let (rows, w, v) = (ws.rows, c.width, c.vocab_size);
        let b = &self.backend;
        b.gemm(&Gemm::new(rows, w, 2 * w, false, false).offsets(0, l.ctl, 0), &ws.final_x, &self.params, &mut ws.real);
        let mut loss = 0.0;
        let chunk = HEAD_CHUNK.min(rows);
        let scale = grad_scale / rows as f32;
        for start in (0..rows).step_by(chunk) {
            let len = chunk.min(rows - start);
            b.gemm(&Gemm::new(len, v, w, false, false).offsets(start * w, l.head, 0), &ws.real, &self.params, &mut ws.logits);
            b.add_row_vector(&mut ws.logits, len, v, &self.params, l.head_bias);
            loss += b.cross_entropy(&mut ws.logits, len, v, targets, start, scale);
            if backward {
                b.column_sums(&ws.logits, len, v, &mut self.grads, l.head_bias);
                b.gemm(&Gemm::new(w, v, len, true, false).ld(w, v, v).offsets(start * w, 0, l.head).accumulate(), &ws.real, &ws.logits, &mut self.grads);
                b.gemm(&Gemm::new(len, w, v, false, true).ld(v, v, w).offsets(0, l.head, start * w), &ws.logits, &self.params, &mut ws.d_real);
            }
        }
        if backward {
            b.gemm(&Gemm::new(2 * w, w, rows, true, false).offsets(0, 0, l.ctl).accumulate(), &ws.final_x, &ws.d_real, &mut self.grads);
            b.gemm(&Gemm::new(rows, 2 * w, w, false, true).offsets(0, l.ctl, 0), &ws.d_real, &self.params, &mut ws.dx);
        }
        loss / rows as f64
    }

    /// Complex linear backward: `d_in = d_out W'^T`, weight gradient folded from `in'^T d_out`.
    #[allow(clippy::too_many_arguments)]
    fn linear_backward(b: &B, input: &B::Buf, d_out: &B::Buf, block: &B::Buf, weight: ComplexWeight, rows: usize, block_grad: &mut B::Buf, grads: &mut B::Buf, d_in: &mut B::Buf) {
        let (k, n) = (weight.k, weight.n);
        b.gemm(&Gemm::new(2 * k, 2 * n, rows, true, false), input, d_out, block_grad);
        b.fold_block_grad(block_grad, k, n, grads, weight.re, weight.im);
        b.gemm(&Gemm::new(rows, 2 * k, 2 * n, false, true), d_out, block, d_in);
    }

    fn backward(&mut self, ws: &mut Workspace<B>, inputs: &B::Ids) {
        let c = self.config.clone();
        let (rows, batch, w, t, heads, hd, hf) = (ws.rows, ws.batch, c.width, c.seq_len, c.heads, c.head_dim(), c.ffn_hidden);
        let beta = c.residual_beta();
        let scale = 1.0 / ((hd as f64).sqrt() + 1e-12) as f32;
        let b = &self.backend;
        // ws.dx holds d(layer output) of the last layer.
        for i in (0..c.layers).rev() {
            let lo = self.layout.layers[i].clone();
            let [wqkv, w1, w2] = &self.blocks[i];
            let a = &ws.layers[i];

            // Feed-forward: y = mid + beta leaky(o), o = g W2 + b2.
            b.leaky_residual_backward(&a.out, &ws.dx, rows, w, beta, LEAKY_SLOPE, &mut ws.d_out);
            b.column_sums(&ws.d_out, rows, 2 * w, &mut self.grads, lo.b2);
            Self::linear_backward(b, &a.gated, &ws.d_out, w2, lo.w2, rows, &mut self.block_grad, &mut self.grads, &mut ws.d_gated);
            b.gated_backward(&a.hidden, &ws.d_gated, rows, hf, &mut ws.d_hidden);
            b.column_sums(&ws.d_hidden, rows, 4 * hf, &mut self.grads, lo.b1);
            Self::linear_backward(b, &a.h2, &ws.d_hidden, w1, lo.w1, rows, &mut self.block_grad, &mut self.grads, &mut ws.dtmp);
            // d(mid) = dy + norm backward of d(h2).
            b.layer_norm_backward(&a.mid, &a.stats2, &ws.dtmp, rows, w, &self.params, lo.ffn_norm.gamma, &mut ws.d_out, &mut self.grads, lo.ffn_norm.gamma, lo.ffn_norm.beta);
            b.axpy(1.0, &ws.d_out, &mut ws.dx, rows * 2 * w);

            // Attention: mid = input + beta [att | 0].
            b.real_part_scaled(&ws.dx, rows, w, beta, &mut ws.d_att);
            b.fill(&mut ws.d_qkv, 0, rows * 6 * w, 0.0);
            for s in 0..batch {
                // dP = d_att V_r^T, dV_r = P^T d_att.
                let g = Gemm::new(t, t, hd, false, true).ld(w, 6 * w, t).offsets(s * t * w, s * t * 6 * w + 2 * w, s * heads * t * t).batched(heads, hd, hd, t * t);
                b.gemm(&g, &ws.d_att, &a.qkv, &mut ws.d_probs);
                let g = Gemm::new(t, hd, t, true, false).ld(t, w, 6 * w).offsets(s * heads * t * t, s * t * w, s * t * 6 * w + 2 * w).batched(heads, t * t, hd, hd);
                b.gemm(&g, &a.probs, &ws.d_att, &mut ws.d_qkv);
            }
            b.softmax_window_backward(&a.probs, &mut ws.d_probs, batch * heads, t);
            for s in 0..batch {
                // dq = scale dS k, dk = scale dS^T q.
                let g = Gemm::new(t, hd, t, false, false).ld(t, w, w).offsets(s * heads * t * t, s * t * w, s * t * w).batched(heads, t * t, hd, hd).alpha(scale);
                b.gemm(&g, &ws.d_probs, &a.kr, &mut ws.d_qr);
                let g = Gemm::new(t, hd, t, true, false).ld(t, w, w).offsets(s * heads * t * t, s * t * w, s * t * w).batched(heads, t * t, hd, hd).alpha(scale);
                b.gemm(&g, &ws.d_probs, &a.qr, &mut ws.d_kr);
            }
            // ComplexToLinear per head: dW = [Qr|Qi]^T dq_real, d[Qr|Qi] = dq_real W^T.
            for (heads_buf, d_real, d_heads, ctl_off) in [(&a.qh, &ws.d_qr, &mut ws.d_qh, lo.ctl_q), (&a.kh, &ws.d_kr, &mut ws.d_kh, lo.ctl_k)] {
                let g = Gemm::new(2 * hd, hd, rows, true, false).ld(2 * w, w, hd).offsets(0, 0, ctl_off).batched(heads, 2 * hd, hd, 2 * hd * hd).accumulate();
                b.gemm(&g, heads_buf, d_real, &mut self.grads);
                let g = Gemm::new(rows, 2 * hd, hd, false, true).ld(w, hd, 2 * w).offsets(0, ctl_off, 0).batched(heads, hd, 2 * hd * hd, 2 * hd);
                b.gemm(&g, d_real, &self.params, d_heads);
            }
            b.rope_from_heads(&ws.d_qh, rows, t, heads, hd, &mut ws.d_qkv, 6 * w, 0, 3 * w);
            b.rope_from_heads(&ws.d_kh, rows, t, heads, hd, &mut ws.d_qkv, 6 * w, w, 4 * w);
            Self::linear_backward(b, &a.h1, &ws.d_qkv, wqkv, lo.wqkv, rows, &mut self.block_grad, &mut self.grads, &mut ws.dtmp);
            b.layer_norm_backward(&a.input, &a.stats1, &ws.dtmp, rows, w, &self.params, lo.attention_norm.gamma, &mut ws.d_out, &mut self.grads, lo.attention_norm.gamma, lo.attention_norm.beta);
            b.axpy(1.0, &ws.d_out, &mut ws.dx, rows * 2 * w);
        }
        // Input norm and embedding.
        let l = &self.layout;
        b.layer_norm_backward(&ws.embedded, &ws.stats0, &ws.dx, rows, w, &self.params, l.input_norm.gamma, &mut ws.d_out, &mut self.grads, l.input_norm.gamma, l.input_norm.beta);
        b.embed_backward(&ws.d_out, inputs, 0, rows, w, &mut self.grads, l.embedding_re, l.embedding_im);
    }
}
