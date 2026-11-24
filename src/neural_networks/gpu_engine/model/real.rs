//! Real-valued control model (`EngineConfig::real`): a standard pre-norm transformer with the same training,
//! checkpoint, generation and fine-tuning code as the complex model, to measure what the complex architecture
//! brings at the same parameter count.
//!
//!   token ids -> embedding (W)
//!     -> layers x (x += Wo attention(rope(Q), rope(K), V) of RMSNorm(x);  x += W2 swiglu(W1 RMSNorm(x)))
//!     -> RMSNorm -> linear head -> softmax cross-entropy
//!
//! The embedding table is stored as two halves (`embedding.re` / `embedding.im`, each vocab x W/2) so that the
//! planar embedding kernels of the complex model serve it unchanged: a row is `[first half | second half]`.

use rand::{RngExt, SeedableRng};

use super::{Backend, EngineConfig, Gemm, Layout, Model, Segment, HEAD_CHUNK, IGNORE_TARGET};

pub const RMS_EPSILON: f32 = 1e-6;

/// Parameter offsets of one real layer (all matrices row-major, input dimension first).
#[derive(Clone, Copy, Debug)]
pub struct RealLayerOffsets {
    /// RMS norm gamma before the attention (W).
    pub attention_norm: usize,
    /// W x 3W: [Q | K | V].
    pub wqkv: usize,
    /// Attention output projection, W x W.
    pub wo: usize,
    pub ffn_norm: usize,
    /// W x 2F: [gate a | value b] of SwiGLU.
    pub w1: usize,
    /// F x W.
    pub w2: usize,
}

pub(super) fn real_layout(c: &EngineConfig) -> Layout {
    assert!(c.width % 2 == 0 && c.width % c.heads == 0 && c.head_dim() % 2 == 0, "width must be even and split into heads of even size");
    let mut segments = Vec::new();
    let mut total = 0;
    let mut add = |name: String, shape: Vec<usize>| -> usize {
        let offset = total;
        total += shape.iter().product::<usize>();
        segments.push(Segment { name, offset, shape });
        offset
    };
    let (w, f) = (c.width, c.ffn_hidden);
    let embedding_re = add("embedding.re".into(), vec![c.vocab_size, w / 2]);
    let embedding_im = add("embedding.im".into(), vec![c.vocab_size, w / 2]);
    let mut real_layers = Vec::new();
    for l in 0..c.layers {
        let attention_norm = add(format!("layer{l}.attention.norm.gamma"), vec![w]);
        let wqkv = add(format!("layer{l}.attention.wqkv"), vec![w, 3 * w]);
        let wo = add(format!("layer{l}.attention.wo"), vec![w, w]);
        let ffn_norm = add(format!("layer{l}.ffn.norm.gamma"), vec![w]);
        let w1 = add(format!("layer{l}.ffn.w_1"), vec![w, 2 * f]);
        let w2 = add(format!("layer{l}.ffn.w_2"), vec![f, w]);
        real_layers.push(RealLayerOffsets { attention_norm, wqkv, wo, ffn_norm, w1, w2 });
    }
    let real_final_gamma = add("final_norm.gamma".into(), vec![w]);
    let head = add("head".into(), vec![w, c.vocab_size]);
    let head_bias = add("head_bias".into(), vec![c.vocab_size]);
    Layout {
        segments,
        total,
        embedding_re,
        embedding_im,
        input_norm: Default::default(),
        layers: Vec::new(),
        final_norm: None,
        ctl: 0,
        head,
        head_bias,
        real_layers,
        real_final_gamma,
    }
}

/// GPT-2 style: embedding and weights uniform with std 0.02, the residual output projections (Wo, W2) scaled by
/// 1 / sqrt(2 layers), norm gammas 1, biases 0.
pub(super) fn real_initial_parameters(c: &EngineConfig, layout: &Layout, seed: u64) -> Vec<f32> {
    let mut p = vec![0.0f32; layout.total];
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let limit = 0.02 * 3f64.sqrt();
    let output_scale = 1.0 / (2.0 * c.layers as f64).sqrt();
    let mut fill = |p: &mut [f32], scale: f64| p.iter_mut().for_each(|v| *v = (rng.random_range(-limit..limit) * scale) as f32);
    for s in &layout.segments {
        let range = s.offset..s.offset + s.len();
        if s.name.ends_with("gamma") {
            p[range].fill(1.0);
        } else if s.name == "head_bias" {
        } else if s.name.ends_with(".wo") || s.name.ends_with(".w_2") {
            fill(&mut p[range], output_scale);
        } else {
            fill(&mut p[range], 1.0);
        }
    }
    p
}

struct RealLayerActivations<B: Backend> {
    input: B::Buf,
    h1: B::Buf,
    stats1: B::Buf,
    qkv: B::Buf,
    q: B::Buf,
    k: B::Buf,
    probs: B::Buf,
    att: B::Buf,
    o: B::Buf,
    mid: B::Buf,
    h2: B::Buf,
    stats2: B::Buf,
    u: B::Buf,
    g: B::Buf,
    f: B::Buf,
}

pub(crate) struct RealWorkspace<B: Backend> {
    rows: usize,
    batch: usize,
    layers: Vec<RealLayerActivations<B>>,
    final_x: B::Buf,
    xn: B::Buf,
    stats_f: B::Buf,
    logits: B::Buf,
    d_xn: B::Buf,
    dx: B::Buf,
    dtmp: B::Buf,
    d_out: B::Buf,
    d_g: B::Buf,
    d_u: B::Buf,
    d_att: B::Buf,
    d_probs: B::Buf,
    d_qkv: B::Buf,
    d_q: B::Buf,
    d_k: B::Buf,
}

impl<B: Backend> Model<B> {
    fn real_workspace(&mut self, batch: usize) -> RealWorkspace<B> {
        let rows = batch * self.config.seq_len;
        if let Some(ws) = self.real_ws.take() {
            if ws.rows == rows {
                return ws;
            }
        }
        let b = &self.backend;
        let c = &self.config;
        let (w, f, t) = (c.width, c.ffn_hidden, c.seq_len);
        let z = |len: usize| b.zeros(len.max(1));
        let layers = (0..c.layers)
            .map(|_| RealLayerActivations {
                input: z(rows * w),
                h1: z(rows * w),
                stats1: z(rows),
                qkv: z(rows * 3 * w),
                q: z(rows * w),
                k: z(rows * w),
                probs: z(batch * c.heads * t * t),
                att: z(rows * w),
                o: z(rows * w),
                mid: z(rows * w),
                h2: z(rows * w),
                stats2: z(rows),
                u: z(rows * 2 * f),
                g: z(rows * f),
                f: z(rows * w),
            })
            .collect();
        RealWorkspace {
            rows,
            batch,
            layers,
            final_x: z(rows * w),
            xn: z(rows * w),
            stats_f: z(rows),
            logits: z(HEAD_CHUNK.min(rows) * c.vocab_size),
            d_xn: z(rows * w),
            dx: z(rows * w),
            dtmp: z(rows * w),
            d_out: z(rows * w),
            d_g: z(rows * f),
            d_u: z(rows * 2 * f),
            d_att: z(rows * w),
            d_probs: z(batch * c.heads * t * t),
            d_qkv: z(rows * 3 * w),
            d_q: z(rows * w),
            d_k: z(rows * w),
        }
    }

    fn real_forward(&self, ws: &mut RealWorkspace<B>, ids: &B::Ids) {
        let (b, c, l) = (&self.backend, &self.config, &self.layout);
        let (rows, batch, w, f, t, heads, hd) = (ws.rows, ws.batch, c.width, c.ffn_hidden, c.seq_len, c.heads, c.head_dim());
        let rope = c.rope();
        let scale = 1.0 / ((hd as f64).sqrt() + 1e-12) as f32;
        b.embed(&self.params, l.embedding_re, l.embedding_im, ids, 0, rows, w / 2, &mut ws.layers[0].input);
        for i in 0..c.layers {
            let lo = l.real_layers[i];
            let a = &mut ws.layers[i];
            b.rms_norm(&a.input, rows, w, RMS_EPSILON, &self.params, lo.attention_norm, &mut a.h1, &mut a.stats1);
            b.gemm(&Gemm::new(rows, 3 * w, w, false, false).offsets(0, lo.wqkv, 0), &a.h1, &self.params, &mut a.qkv);
            b.rope_rows(&a.qkv, 3 * w, 0, rows, t, heads, hd, rope, 1.0, &mut a.q, w, 0);
            b.rope_rows(&a.qkv, 3 * w, w, rows, t, heads, hd, rope, 1.0, &mut a.k, w, 0);
            for s in 0..batch {
                let g = Gemm::new(t, t, hd, false, true).ld(w, w, t).offsets(s * t * w, s * t * w, s * heads * t * t).batched(heads, hd, hd, t * t).alpha(scale);
                b.gemm(&g, &a.q, &a.k, &mut a.probs);
            }
            b.softmax_window(&mut a.probs, batch * heads, t, c.window);
            for s in 0..batch {
                let g = Gemm::new(t, hd, t, false, false).ld(t, 3 * w, w).offsets(s * heads * t * t, s * t * 3 * w + 2 * w, s * t * w).batched(heads, t * t, hd, hd);
                b.gemm(&g, &a.probs, &a.qkv, &mut a.att);
            }
            b.gemm(&Gemm::new(rows, w, w, false, false).offsets(0, lo.wo, 0), &a.att, &self.params, &mut a.o);
            b.copy(&a.input, 0, &mut a.mid, 0, rows * w);
            b.axpy(1.0, &a.o, &mut a.mid, rows * w);
            b.rms_norm(&a.mid, rows, w, RMS_EPSILON, &self.params, lo.ffn_norm, &mut a.h2, &mut a.stats2);
            b.gemm(&Gemm::new(rows, 2 * f, w, false, false).offsets(0, lo.w1, 0), &a.h2, &self.params, &mut a.u);
            b.swiglu(&a.u, rows, f, &mut a.g);
            b.gemm(&Gemm::new(rows, w, f, false, false).offsets(0, lo.w2, 0), &a.g, &self.params, &mut a.f);
            let (head, tail) = ws.layers.split_at_mut(i + 1);
            let next = if i + 1 < c.layers { &mut tail[0].input } else { &mut ws.final_x };
            b.copy(&head[i].mid, 0, next, 0, rows * w);
            b.axpy(1.0, &head[i].f, next, rows * w);
        }
        b.rms_norm(&ws.final_x, rows, w, RMS_EPSILON, &self.params, l.real_final_gamma, &mut ws.xn, &mut ws.stats_f);
    }

    /// Head and loss in chunks; with `backward`, the head gradients and `d_xn`.
    fn real_head(&mut self, ws: &mut RealWorkspace<B>, targets: &B::Ids, backward: bool, grad_scale: f32, counted: usize) -> f64 {
        let (l, b) = (&self.layout, &self.backend);
        let (rows, w, v) = (ws.rows, self.config.width, self.config.vocab_size);
        let chunk = HEAD_CHUNK.min(rows);
        let scale = grad_scale / counted as f32;
        let mut loss = 0.0;
        for start in (0..rows).step_by(chunk) {
            let len = chunk.min(rows - start);
            b.gemm(&Gemm::new(len, v, w, false, false).offsets(start * w, l.head, 0), &ws.xn, &self.params, &mut ws.logits);
            b.add_row_vector(&mut ws.logits, len, v, &self.params, l.head_bias);
            loss += b.cross_entropy(&mut ws.logits, len, v, targets, start, scale);
            if backward {
                b.column_sums(&ws.logits, len, v, &mut self.grads, l.head_bias);
                b.gemm(&Gemm::new(w, v, len, true, false).ld(w, v, v).offsets(start * w, 0, l.head).accumulate(), &ws.xn, &ws.logits, &mut self.grads);
                b.gemm(&Gemm::new(len, w, v, false, true).ld(v, v, w).offsets(0, l.head, start * w), &ws.logits, &self.params, &mut ws.d_xn);
            }
        }
        loss / counted as f64
    }

    fn real_backward(&mut self, ws: &mut RealWorkspace<B>, ids: &B::Ids) {
        let c = self.config.clone();
        let (rows, batch, w, f, t, heads, hd) = (ws.rows, ws.batch, c.width, c.ffn_hidden, c.seq_len, c.heads, c.head_dim());
        let rope = c.rope();
        let scale = 1.0 / ((hd as f64).sqrt() + 1e-12) as f32;
        let b = &self.backend;
        let l = &self.layout;
        b.rms_norm_backward(&ws.final_x, &ws.stats_f, &ws.d_xn, rows, w, &self.params, l.real_final_gamma, &mut ws.dx, &mut self.grads, l.real_final_gamma);
        for i in (0..c.layers).rev() {
            let lo = l.real_layers[i];
            let a = &ws.layers[i];
            // Feed-forward: next = mid + g W2, g = swiglu(h2 W1).
            b.gemm(&Gemm::new(f, w, rows, true, false).offsets(0, 0, lo.w2).accumulate(), &a.g, &ws.dx, &mut self.grads);
            b.gemm(&Gemm::new(rows, f, w, false, true).offsets(0, lo.w2, 0), &ws.dx, &self.params, &mut ws.d_g);
            b.swiglu_backward(&a.u, &ws.d_g, rows, f, &mut ws.d_u);
            b.gemm(&Gemm::new(w, 2 * f, rows, true, false).offsets(0, 0, lo.w1).accumulate(), &a.h2, &ws.d_u, &mut self.grads);
            b.gemm(&Gemm::new(rows, w, 2 * f, false, true).offsets(0, lo.w1, 0), &ws.d_u, &self.params, &mut ws.dtmp);
            b.rms_norm_backward(&a.mid, &a.stats2, &ws.dtmp, rows, w, &self.params, lo.ffn_norm, &mut ws.d_out, &mut self.grads, lo.ffn_norm);
            b.axpy(1.0, &ws.d_out, &mut ws.dx, rows * w);
            // Attention: mid = input + att Wo.
            b.gemm(&Gemm::new(w, w, rows, true, false).offsets(0, 0, lo.wo).accumulate(), &a.att, &ws.dx, &mut self.grads);
            b.gemm(&Gemm::new(rows, w, w, false, true).offsets(0, lo.wo, 0), &ws.dx, &self.params, &mut ws.d_att);
            b.fill(&mut ws.d_qkv, 0, rows * 3 * w, 0.0);
            for s in 0..batch {
                let g = Gemm::new(t, t, hd, false, true).ld(w, 3 * w, t).offsets(s * t * w, s * t * 3 * w + 2 * w, s * heads * t * t).batched(heads, hd, hd, t * t);
                b.gemm(&g, &ws.d_att, &a.qkv, &mut ws.d_probs);
                let g = Gemm::new(t, hd, t, true, false).ld(t, w, 3 * w).offsets(s * heads * t * t, s * t * w, s * t * 3 * w + 2 * w).batched(heads, t * t, hd, hd);
                b.gemm(&g, &a.probs, &ws.d_att, &mut ws.d_qkv);
            }
            b.softmax_window_backward(&a.probs, &mut ws.d_probs, batch * heads, t);
            for s in 0..batch {
                let g = Gemm::new(t, hd, t, false, false).ld(t, w, w).offsets(s * heads * t * t, s * t * w, s * t * w).batched(heads, t * t, hd, hd).alpha(scale);
                b.gemm(&g, &ws.d_probs, &a.k, &mut ws.d_q);
                let g = Gemm::new(t, hd, t, true, false).ld(t, w, w).offsets(s * heads * t * t, s * t * w, s * t * w).batched(heads, t * t, hd, hd).alpha(scale);
                b.gemm(&g, &ws.d_probs, &a.q, &mut ws.d_k);
            }
            b.rope_rows(&ws.d_q, w, 0, rows, t, heads, hd, rope, -1.0, &mut ws.d_qkv, 3 * w, 0);
            b.rope_rows(&ws.d_k, w, 0, rows, t, heads, hd, rope, -1.0, &mut ws.d_qkv, 3 * w, w);
            b.gemm(&Gemm::new(w, 3 * w, rows, true, false).offsets(0, 0, lo.wqkv).accumulate(), &a.h1, &ws.d_qkv, &mut self.grads);
            b.gemm(&Gemm::new(rows, w, 3 * w, false, true).offsets(0, lo.wqkv, 0), &ws.d_qkv, &self.params, &mut ws.dtmp);
            b.rms_norm_backward(&a.input, &a.stats1, &ws.dtmp, rows, w, &self.params, lo.attention_norm, &mut ws.d_out, &mut self.grads, lo.attention_norm);
            b.axpy(1.0, &ws.d_out, &mut ws.dx, rows * w);
        }
        b.embed_backward(&ws.dx, ids, 0, rows, w / 2, &mut self.grads, l.embedding_re, l.embedding_im);
    }

    pub(super) fn real_forward_backward(&mut self, inputs: &[u32], targets: &[u32], backward: bool, grad_scale: f32, counted: usize) -> f64 {
        debug_assert!(targets.iter().any(|&t| t != IGNORE_TARGET));
        let batch = inputs.len() / self.config.seq_len;
        let ids = self.backend.upload_ids(inputs);
        let target_ids = self.backend.upload_ids(targets);
        let mut ws = self.real_workspace(batch);
        self.real_forward(&mut ws, &ids);
        let loss = self.real_head(&mut ws, &target_ids, backward, grad_scale, counted);
        if backward {
            self.real_backward(&mut ws, &ids);
        }
        self.real_ws = Some(ws);
        loss
    }

    pub(super) fn real_forward_hidden(&mut self, inputs: &[u32]) -> Vec<f32> {
        let ids = self.backend.upload_ids(inputs);
        let mut ws = self.real_workspace(inputs.len() / self.config.seq_len);
        self.real_forward(&mut ws, &ids);
        let mut hidden = self.backend.download(&ws.xn);
        hidden.truncate(ws.rows * self.config.width);
        self.real_ws = Some(ws);
        hidden
    }

    pub(super) fn real_backward_hidden(&mut self, inputs: &[u32], d_hidden: &[f32]) {
        let mut ws = self.real_ws.take().expect("backward_hidden needs a forward_hidden first");
        assert_eq!(d_hidden.len(), ws.rows * self.config.width, "d_hidden must have the shape of the hidden states");
        let ids = self.backend.upload_ids(inputs);
        ws.d_xn = self.backend.upload(d_hidden);
        self.real_backward(&mut ws, &ids);
        self.real_ws = Some(ws);
    }

    /// `ids` has `t` (<= seq_len) tokens; the logits at `position`.
    pub(super) fn real_next_token_logits(&mut self, ids: &[u32], t: usize, position: usize) -> Vec<f32> {
        let full = self.config.seq_len;
        self.config.seq_len = t;
        let inputs = self.backend.upload_ids(ids);
        let mut ws = self.real_workspace(1);
        self.real_forward(&mut ws, &inputs);
        self.config.seq_len = full;
        let (b, l, w, v) = (&self.backend, &self.layout, self.config.width, self.config.vocab_size);
        b.gemm(&Gemm::new(1, v, w, false, false).offsets(position * w, l.head, 0), &ws.xn, &self.params, &mut ws.logits);
        b.add_row_vector(&mut ws.logits, 1, v, &self.params, l.head_bias);
        let mut logits = b.download(&ws.logits);
        logits.truncate(v);
        self.real_ws = Some(ws);
        logits
    }
}
