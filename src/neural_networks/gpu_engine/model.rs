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

use super::backend::{Backend, Gemm, Rope, IGNORE_TARGET};
use crate::neural_networks::network_layers::wavelet_network::decompose_in_wavelet_2d_levels;

pub mod real;

/// Scales (dilations 1, 2, 4, ... 2^(scales - 1)) and taps per scale of the wavelet mixer: reach
/// (taps - 1) 2^(scales - 1) + 1 = 449 positions.
pub const WAVELET_SCALES: usize = 7;
pub const WAVELET_TAPS: usize = 8;

/// Where the causal complex wavelet mixer goes. The mixer: [v | g] = h Wvg, z = sum over scales of the causal
/// dilated convolution of v (complex taps per channel, initialized as Morlet wavelets), u = z * split_silu(g),
/// out = u Wo (h: the normed layer input, shared with the attention).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WaveletMixer {
    #[default]
    None,
    /// Odd layers have the mixer instead of the attention.
    Hybrid,
    /// Every layer has the mixer as a second branch next to the attention.
    Parallel,
}


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
    /// Attention weights the complex values (real and imaginary part of V) and writes into both halves of the
    /// residual stream; false: only Re(V), written into the real half (the RM model).
    #[serde(default)]
    pub complex_values: bool,
    /// Plain residuals `x + attention`, `x + feed-forward` and a layer norm before the head; false: residuals scaled
    /// by `residual_beta` and a leaky ReLU on the feed-forward output (the RM model).
    #[serde(default)]
    pub standard_residual: bool,
    /// RoPE with base 10000 and unclamped angles; false: base 1250 with angles clamped to [-1, 1] (the RM model,
    /// where pair i cannot tell positions beyond 1250^(2i/head_dim) apart).
    #[serde(default)]
    pub standard_rope: bool,
    /// Real-valued control model (`model::real`): RMSNorm, attention with output projection, SwiGLU, no complex
    /// numbers; `width` and `ffn_hidden` are real sizes.
    /// Attention scores as the Hermitian product Re(q conj(k)) of the rotated complex queries and keys instead of
    /// ComplexToLinear projections to real vectors.
    #[serde(default)]
    pub hermitian_scores: bool,
    /// Complex output projection Wo (w x w) of the attention before the residual (needs `complex_values`).
    #[serde(default)]
    pub attention_output: bool,
    /// Norms divide by the RMS of the complex values without subtracting the mean (complex RMS norm).
    #[serde(default)]
    pub rms_norm: bool,
    /// Output projections (Wo, W2) initialized smaller by 1 / sqrt(2 layers), as in GPT-2.
    #[serde(default)]
    pub scaled_init: bool,
    /// Logits Re<x, e_v> of the final complex state with the embedding of token v (weight tying; no ComplexToLinear
    /// and no separate head matrix).
    #[serde(default)]
    pub tied_head: bool,
    /// Causal complex wavelet mixer (`WaveletMixer`): in place of every second attention or next to every attention.
    #[serde(default)]
    pub wavelet_mixer: WaveletMixer,
    #[serde(default)]
    pub real: bool,
    /// Tokenizer file the model was trained with (empty: the GPT-NeoX tokenizer of the project).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tokenizer: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        // New models: full attention over the sequence, complex values, plain residuals (checkpoints without these
        // fields keep the RM behaviour, see the serde defaults).
        Self { vocab_size: 50282, width: 256, layers: 12, heads: 4, ffn_hidden: 1024, window: 512, seq_len: 512, complex_values: true, standard_residual: true, standard_rope: true, hermitian_scores: false, attention_output: false, rms_norm: false, scaled_init: false, tied_head: false, wavelet_mixer: WaveletMixer::None, real: false, tokenizer: String::new() }
    }
}

impl EngineConfig {
    /// Residual scale: 1 / (3 * layers)^(1/4) in the RM model, 1 with plain residuals.
    pub fn residual_beta(&self) -> f32 {
        if self.standard_residual {
            return 1.0;
        }
        (1.0 / (3.0 * self.layers as f64).powf(0.25)) as f32
    }

    pub fn rope(&self) -> Rope {
        if self.standard_rope { Rope { base: 10000.0, clamp: false } } else { Rope { base: super::ROPE_BASE as f32, clamp: true } }
    }

    /// Makes the tokenizer of this model the one used for encoding and decoding (before the first use).
    pub fn select_tokenizer(&self) -> Result<(), String> {
        if self.tokenizer.is_empty() {
            return Ok(());
        }
        crate::neural_networks::utils::tokenizer::use_tokenizer(&self.tokenizer)
    }

    /// Features per position of the final hidden states (2w planar complex, w real).
    /// Whether layer `l` has an attention / a wavelet mixer.
    pub fn layer_has_attention(&self, l: usize) -> bool {
        !(self.wavelet_mixer == WaveletMixer::Hybrid && l % 2 == 1)
    }

    pub fn layer_has_mixer(&self, l: usize) -> bool {
        match self.wavelet_mixer {
            WaveletMixer::None => false,
            WaveletMixer::Hybrid => l % 2 == 1,
            WaveletMixer::Parallel => true,
        }
    }

    pub fn hidden_features(&self) -> usize {
        if self.real { self.width } else { 2 * self.width }
    }

    /// Columns of the attention output per row: w (Re(V) only) or 2w (planar complex).
    pub fn attention_width(&self) -> usize {
        if self.complex_values { 2 * self.width } else { self.width }
    }

    /// Named architectures for comparison runs: `rm` (pretraining runs 1-3), `rm-pos` (+ full attention window and
    /// standard RoPE), `complex-v` (+ complex values), `new` (+ plain residuals = the default).
    pub fn preset(name: &str) -> Option<Self> {
        let rm = Self::rm();
        match name {
            "rm" => Some(rm),
            "rm-pos" => Some(Self { window: rm.seq_len, standard_rope: true, ..rm }),
            "complex-v" => Some(Self { window: rm.seq_len, standard_rope: true, complex_values: true, ..rm }),
            "new" => Some(Self::default()),
            "herm" => Some(Self { hermitian_scores: true, ..Self::default() }),
            "wo" => Some(Self { attention_output: true, ..Self::default() }),
            "herm-wo" => Some(Self { hermitian_scores: true, attention_output: true, ..Self::default() }),
            "herm-wo-init" => Some(Self { hermitian_scores: true, attention_output: true, scaled_init: true, ..Self::default() }),
            "herm-wo-rms" => Some(Self { hermitian_scores: true, attention_output: true, rms_norm: true, ..Self::default() }),
            "herm-wo-tied" => Some(Self { hermitian_scores: true, attention_output: true, tied_head: true, ..Self::default() }),
            "herm-wo-wav-hybrid" => Some(Self { hermitian_scores: true, attention_output: true, wavelet_mixer: WaveletMixer::Hybrid, ..Self::default() }),
            "herm-wo-wav-parallel" => Some(Self { hermitian_scores: true, attention_output: true, wavelet_mixer: WaveletMixer::Parallel, ..Self::default() }),
            "herm-wo-all" => Some(Self { hermitian_scores: true, attention_output: true, scaled_init: true, rms_norm: true, tied_head: true, ..Self::default() }),
            // Real transformer with about the parameters of the complex model (21 M in the layers, 63 M in total).
            "real" => Some(Self { width: 384, heads: 6, ffn_hidden: 1024, complex_values: false, real: true, ..Self::default() }),
            _ => None,
        }
    }

    /// The RM model's architecture (the checkpoints of the pretraining runs 1 to 3).
    pub fn rm() -> Self {
        Self { window: 256, complex_values: false, standard_residual: false, standard_rope: false, real: false, ..Self::default() }
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
#[derive(Clone, Copy, Debug, Default)]
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
    /// Q, K and V of all heads: w x 3w, columns [q heads | k heads | v heads] (None: a mixer-only layer).
    pub wqkv: Option<ComplexWeight>,
    /// ComplexToLinear of the queries / keys, per head a (2 head_dim x head_dim) block [W1; W2].
    /// (0 with Hermitian scores: no such parameters.)
    pub ctl_q: usize,
    pub ctl_k: usize,
    /// Attention output projection (w x w), with `attention_output`.
    pub wo: Option<ComplexWeight>,
    /// Wavelet mixer of the layer.
    pub mixer: Option<MixerOffsets>,
    pub ffn_norm: NormOffsets,
    pub w1: ComplexWeight,
    /// Planar bias (re then im).
    pub b1: usize,
    pub w2: ComplexWeight,
    pub b2: usize,
}

/// Parameters of a wavelet mixer.
#[derive(Clone, Copy, Debug)]
pub struct MixerOffsets {
    /// w x 2w: [v | g].
    pub wvg: ComplexWeight,
    /// Taps (scales x taps x w each).
    pub taps_re: usize,
    pub taps_im: usize,
    /// Output projection, w x w.
    pub wo: ComplexWeight,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub segments: Vec<Segment>,
    pub total: usize,
    pub embedding_re: usize,
    pub embedding_im: usize,
    pub input_norm: NormOffsets,
    pub layers: Vec<LayerOffsets>,
    /// Layer norm before the head (plain residuals only).
    pub final_norm: Option<NormOffsets>,
    /// ComplexToLinear before the head: (2w x w) = [W1; W2].
    pub ctl: usize,
    /// Layers and final RMS norm gamma of the real model (empty / 0 for the complex model).
    pub real_layers: Vec<real::RealLayerOffsets>,
    pub real_final_gamma: usize,
    pub head: usize,
    pub head_bias: usize,
}

impl Layout {
    pub fn new(c: &EngineConfig) -> Self {
        if c.real {
            return real::real_layout(c);
        }
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
            let attention = c.layer_has_attention(l);
            let wqkv = attention.then(|| complex(&mut add, &format!("layer{l}.attention.wqkv"), w, 3 * w));
            let (ctl_q, ctl_k) = if c.hermitian_scores || !attention {
                (0, 0)
            } else {
                (add(format!("layer{l}.attention.ctl_q"), vec![c.heads, 2 * hd, hd]), add(format!("layer{l}.attention.ctl_k"), vec![c.heads, 2 * hd, hd]))
            };
            assert!(!c.attention_output || c.complex_values, "attention_output needs complex_values");
            let wo = (c.attention_output && attention).then(|| complex(&mut add, &format!("layer{l}.attention.wo"), w, w));
            assert!(c.wavelet_mixer == WaveletMixer::None || (c.complex_values && c.standard_residual), "the wavelet mixer needs complex_values and standard_residual");
            let mixer = c.layer_has_mixer(l).then(|| {
                let wvg = complex(&mut add, &format!("layer{l}.mixer.wvg"), w, 2 * w);
                let taps_re = add(format!("layer{l}.mixer.taps.re"), vec![WAVELET_SCALES, WAVELET_TAPS, w]);
                let taps_im = add(format!("layer{l}.mixer.taps.im"), vec![WAVELET_SCALES, WAVELET_TAPS, w]);
                let wo = complex(&mut add, &format!("layer{l}.mixer.wo"), w, w);
                MixerOffsets { wvg, taps_re, taps_im, wo }
            });
            let ffn_norm = norm(&mut add, &format!("layer{l}.ffn.norm"));
            let w1 = complex(&mut add, &format!("layer{l}.ffn.w_1"), w, 2 * hf);
            let b1 = add(format!("layer{l}.ffn.b_1.re"), vec![2 * hf]);
            add(format!("layer{l}.ffn.b_1.im"), vec![2 * hf]);
            let w2 = complex(&mut add, &format!("layer{l}.ffn.w_2"), hf, w);
            let b2 = add(format!("layer{l}.ffn.b_2.re"), vec![w]);
            add(format!("layer{l}.ffn.b_2.im"), vec![w]);
            layers.push(LayerOffsets { attention_norm, wqkv, ctl_q, ctl_k, wo, mixer, ffn_norm, w1, b1, w2, b2 });
        }
        let final_norm = c.standard_residual.then(|| norm(&mut add, "final_norm"));
        let (ctl, head) = if c.tied_head { (0, 0) } else { (add("ctl".into(), vec![2 * w, w]), add("head".into(), vec![w, c.vocab_size])) };
        let head_bias = add("head_bias".into(), vec![c.vocab_size]);
        Self { segments, total, embedding_re, embedding_im, input_norm, layers, final_norm, ctl, head, head_bias, real_layers: Vec::new(), real_final_gamma: 0 }
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
    if c.real {
        return real::real_initial_parameters(c, layout, seed);
    }
    let mut p = vec![0.0f32; layout.total];
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (w, hd, hf) = (c.width, c.head_dim(), c.ffn_hidden);
    let (er, ei) = wavelet_embedding_table(c.vocab_size, c.wavelet_levels(), w, seed);
    p[layout.embedding_re..layout.embedding_re + er.len()].copy_from_slice(&er);
    p[layout.embedding_im..layout.embedding_im + ei.len()].copy_from_slice(&ei);
    let set_norm = |p: &mut [f32], n: NormOffsets| p[n.gamma..n.gamma + w].fill(1.0);
    set_norm(&mut p, layout.input_norm);
    if let Some(n) = layout.final_norm {
        set_norm(&mut p, n);
    }
    for l in &layout.layers {
        set_norm(&mut p, l.attention_norm);
        set_norm(&mut p, l.ffn_norm);
        // Every head's Q, K, V is a w x head_dim Xavier matrix (fan-in w, fan-out head_dim).
        for part in l.wqkv.iter().flat_map(|q| [q.re, q.im]) {
            for v in p[part..part + w * 3 * w].iter_mut() {
                *v = rng.random_range(-((6.0 / (w + hd) as f64).sqrt())..(6.0 / (w + hd) as f64).sqrt()) as f32;
            }
        }
        if let Some(m) = l.mixer {
            for weight in [m.wvg, m.wo] {
                xavier(weight.k, weight.n, &mut p[weight.re..weight.re + weight.k * weight.n], &mut rng);
                xavier(weight.k, weight.n, &mut p[weight.im..weight.im + weight.k * weight.n], &mut rng);
            }
            if c.scaled_init {
                let factor = (1.0 / (2.0 * c.layers as f64).sqrt()) as f32;
                p[m.wo.re..m.wo.re + w * w].iter_mut().for_each(|v| *v *= factor);
                p[m.wo.im..m.wo.im + w * w].iter_mut().for_each(|v| *v *= factor);
            }
            // Morlet taps: Gaussian envelope over the taps, channel j oscillating at frequency pi j / w, the sum over
            // scales and taps normalized to about 1.
            let sigma = WAVELET_TAPS as f64 / 2.5;
            let envelope: Vec<f64> = (0..WAVELET_TAPS).map(|k| (-0.5 * (k as f64 / sigma).powi(2)).exp()).collect();
            let norm = WAVELET_SCALES as f64 * envelope.iter().sum::<f64>();
            for s in 0..WAVELET_SCALES {
                for (k, e) in envelope.iter().enumerate() {
                    for j in 0..w {
                        let omega = std::f64::consts::PI * j as f64 / w as f64;
                        let idx = (s * WAVELET_TAPS + k) * w + j;
                        p[m.taps_re + idx] = (e * (omega * k as f64).cos() / norm) as f32;
                        p[m.taps_im + idx] = (-e * (omega * k as f64).sin() / norm) as f32;
                    }
                }
            }
        }
        if !c.hermitian_scores && l.wqkv.is_some() {
            xavier(hd, hd, &mut p[l.ctl_q..l.ctl_q + c.heads * 2 * hd * hd], &mut rng);
            xavier(hd, hd, &mut p[l.ctl_k..l.ctl_k + c.heads * 2 * hd * hd], &mut rng);
        }
        if let Some(wo) = l.wo {
            xavier(w, w, &mut p[wo.re..wo.re + w * w], &mut rng);
            xavier(w, w, &mut p[wo.im..wo.im + w * w], &mut rng);
        }
        for (weight, bias, bias_len) in [(l.w1, l.b1, 2 * hf), (l.w2, l.b2, w)] {
            xavier(weight.k, weight.n, &mut p[weight.re..weight.re + weight.k * weight.n], &mut rng);
            xavier(weight.k, weight.n, &mut p[weight.im..weight.im + weight.k * weight.n], &mut rng);
            // initialize_bias: Xavier with fan_in = fan_out = rows of the layer.
            xavier(weight.k, weight.k, &mut p[bias..bias + 2 * bias_len], &mut rng);
        }
        if c.scaled_init {
            let factor = (1.0 / (2.0 * c.layers as f64).sqrt()) as f32;
            for weight in l.wo.iter().chain([&l.w2]) {
                p[weight.re..weight.re + weight.k * weight.n].iter_mut().for_each(|v| *v *= factor);
                p[weight.im..weight.im + weight.k * weight.n].iter_mut().for_each(|v| *v *= factor);
            }
        }
    }
    if !c.tied_head {
        xavier(w, w, &mut p[layout.ctl..layout.ctl + 2 * w * w], &mut rng);
        xavier(w, c.vocab_size, &mut p[layout.head..layout.head + w * c.vocab_size], &mut rng);
    }
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
    /// Attention output after Wo (with `attention_output`).
    att_o: B::Buf,
    /// Wavelet mixer: [vr | gr | vi | gi] = h Wvg, the same with z in place of v, u = z * silu(g), u Wo.
    vg: B::Buf,
    cg: B::Buf,
    u: B::Buf,
    mix_o: B::Buf,
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
    /// Layer norm of `final_x` (plain residuals) and its statistics.
    final_normed: B::Buf,
    stats_final: B::Buf,
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
    /// Gradient of the attention output after Wo.
    d_o: B::Buf,
    d_mo: B::Buf,
    d_u: B::Buf,
    d_cg: B::Buf,
    d_vg: B::Buf,
    d_probs: B::Buf,
    d_qkv: B::Buf,
    d_qr: B::Buf,
    d_kr: B::Buf,
    d_qh: B::Buf,
    d_kh: B::Buf,
}

/// Token positions per head chunk: the logits of a chunk (`chunk x vocab`) are the largest buffer.
pub const HEAD_CHUNK: usize = 1024;

/// Generation computes the context length rounded up to a multiple of this (fewer workspace reallocations).
const CONTEXT_BUCKET: usize = if cfg!(test) { 4 } else { 64 };

/// Block matrices [[Wr, Wi], [-Wi, Wr]] of a layer's complex weights.
struct LayerBlocks<B: Backend> {
    wqkv: Option<B::Buf>,
    w1: B::Buf,
    w2: B::Buf,
    wo: Option<B::Buf>,
    wvg: Option<B::Buf>,
    mixer_wo: Option<B::Buf>,
}

pub struct Model<B: Backend> {
    pub config: EngineConfig,
    pub layout: Layout,
    pub backend: B,
    pub params: B::Buf,
    pub grads: B::Buf,
    /// Block matrices of the complex weights, per layer.
    blocks: Vec<LayerBlocks<B>>,
    /// Gradient of one block matrix (the largest one).
    block_grad: B::Buf,
    workspace: Option<Workspace<B>>,
    real_ws: Option<real::RealWorkspace<B>>,
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
        let block = |c: &ComplexWeight| backend.zeros(4 * c.k * c.n);
        let blocks = layout
            .layers
            .iter()
            .map(|l| LayerBlocks {
                wqkv: l.wqkv.as_ref().map(block),
                w1: block(&l.w1),
                w2: block(&l.w2),
                wo: l.wo.as_ref().map(block),
                wvg: l.mixer.as_ref().map(|m| block(&m.wvg)),
                mixer_wo: l.mixer.as_ref().map(|m| block(&m.wo)),
            })
            .collect();
        let largest = layout.layers.iter().flat_map(Self::block_weights).map(|c| 4 * c.k * c.n).max().unwrap_or(1);
        let block_grad = backend.zeros(largest);
        let mut model = Self { config, layout, backend, params, grads, blocks, block_grad, workspace: None, real_ws: None };
        model.refresh_blocks();
        model
    }

    /// The complex weights of a layer that use block matrices.
    fn block_weights(l: &LayerOffsets) -> Vec<ComplexWeight> {
        let mut weights = vec![l.w1, l.w2];
        weights.extend(l.wqkv);
        weights.extend(l.wo);
        if let Some(m) = l.mixer {
            weights.extend([m.wvg, m.wo]);
        }
        weights
    }

    /// Rebuilds the block matrices from the parameters (after every parameter update).
    pub fn refresh_blocks(&mut self) {
        let b = &self.backend;
        for (l, blocks) in self.layout.layers.iter().zip(self.blocks.iter_mut()) {
            let build = |weight: &ComplexWeight, block: &mut B::Buf| b.build_block(&self.params, weight.re, weight.im, weight.k, weight.n, block);
            build(&l.w1, &mut blocks.w1);
            build(&l.w2, &mut blocks.w2);
            if let (Some(wt), Some(bl)) = (&l.wqkv, &mut blocks.wqkv) {
                build(wt, bl);
            }
            if let (Some(wt), Some(bl)) = (&l.wo, &mut blocks.wo) {
                build(wt, bl);
            }
            if let (Some(m), Some(bl)) = (&l.mixer, &mut blocks.wvg) {
                build(&m.wvg, bl);
            }
            if let (Some(m), Some(bl)) = (&l.mixer, &mut blocks.mixer_wo) {
                build(&m.wo, bl);
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
        let mixer = c.wavelet_mixer != WaveletMixer::None;
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
                att: z(rows * c.attention_width()),
                att_o: z(if c.attention_output { rows * 2 * w } else { 1 }),
                vg: z(if mixer { rows * 4 * w } else { 1 }),
                cg: z(if mixer { rows * 4 * w } else { 1 }),
                u: z(if mixer { rows * 2 * w } else { 1 }),
                mix_o: z(if mixer { rows * 2 * w } else { 1 }),
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
            final_normed: z(if c.standard_residual { rows * 2 * w } else { 1 }),
            stats_final: z(if c.standard_residual { rows * 3 } else { 1 }),
            real: z(rows * w),
            d_real: z(rows * w),
            logits: z(chunk * c.vocab_size),
            dx: z(rows * 2 * w),
            dtmp: z(rows * 2 * w),
            d_out: z(rows * 2 * w),
            d_gated: z(rows * 2 * hf),
            d_hidden: z(rows * 4 * hf),
            d_att: z(rows * c.attention_width()),
            d_o: z(if c.attention_output { rows * 2 * w } else { 1 }),
            d_mo: z(if mixer { rows * 2 * w } else { 1 }),
            d_u: z(if mixer { rows * 2 * w } else { 1 }),
            d_cg: z(if mixer { rows * 4 * w } else { 1 }),
            d_vg: z(if mixer { rows * 4 * w } else { 1 }),
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
        self.forward_backward_targets(&inputs, &targets, backward, grad_scale)
    }

    /// Mean loss over the counted positions: `inputs` and `targets` are batch x seq_len ids, a target of
    /// `IGNORE_TARGET` does not count (fine-tuning: only the answer). With `backward`, adds
    /// `grad_scale * d(mean loss)` to `grads`. Without any counted position the loss is 0 and nothing is added.
    pub fn forward_backward_targets(&mut self, inputs: &[u32], targets: &[u32], backward: bool, grad_scale: f32) -> f64 {
        let t = self.config.seq_len;
        assert!(inputs.len() == targets.len() && inputs.len() % t == 0, "inputs and targets must be batch x seq_len ids");
        let counted = targets.iter().filter(|&&id| id != IGNORE_TARGET).count();
        if counted == 0 {
            return 0.0;
        }
        let batch = inputs.len() / t;
        if self.config.real {
            return self.real_forward_backward(inputs, targets, backward, grad_scale, counted);
        }
        let inputs = self.backend.upload_ids(inputs);
        let targets = self.backend.upload_ids(targets);
        let mut ws = self.workspace(batch);
        self.forward(&mut ws, &inputs);
        let loss = self.head(&mut ws, &targets, backward, grad_scale, counted);
        if backward {
            self.backward(&mut ws, &inputs);
        }
        self.workspace = Some(ws);
        loss
    }

    /// Final hidden states (the head input: after the final layer norm if the model has one) of `inputs`
    /// (batch x seq_len ids), `rows x 2w` planar; the activations stay for `backward_hidden`.
    pub fn forward_hidden(&mut self, inputs: &[u32]) -> Vec<f32> {
        let t = self.config.seq_len;
        assert!(!inputs.is_empty() && inputs.len() % t == 0, "inputs must be batch x seq_len ids");
        if self.config.real {
            return self.real_forward_hidden(inputs);
        }
        let ids = self.backend.upload_ids(inputs);
        let mut ws = self.workspace(inputs.len() / t);
        self.forward(&mut ws, &ids);
        let x = if self.layout.final_norm.is_some() { &ws.final_normed } else { &ws.final_x };
        let mut hidden = self.backend.download(x);
        hidden.truncate(ws.rows * 2 * self.config.width);
        self.workspace = Some(ws);
        hidden
    }

    /// Adds to `grads` the parameter gradient for the gradient `d_hidden` of the hidden states of the last
    /// `forward_hidden` call (same `inputs`).
    pub fn backward_hidden(&mut self, inputs: &[u32], d_hidden: &[f32]) {
        if self.config.real {
            return self.real_backward_hidden(inputs, d_hidden);
        }
        let mut ws = self.workspace.take().expect("backward_hidden needs a forward_hidden first");
        assert_eq!(d_hidden.len(), ws.rows * 2 * self.config.width, "d_hidden must have the shape of the hidden states");
        let ids = self.backend.upload_ids(inputs);
        ws.dx = self.backend.upload(d_hidden);
        self.backward(&mut ws, &ids);
        self.workspace = Some(ws);
    }

    /// Logits of the token after `context` (inference, no gradients). Uses the last `seq_len` ids; a shorter context
    /// is padded at the end, which does not change the logits at its last position because attention is causal.
    /// Only the context length rounded up to `CONTEXT_BUCKET` is computed (positions start at 0 either way).
    pub fn next_token_logits(&mut self, context: &[u32]) -> Vec<f32> {
        let full = self.config.seq_len;
        let context = &context[context.len().saturating_sub(full)..];
        let last = *context.last().expect("context must not be empty");
        let t = context.len().div_ceil(CONTEXT_BUCKET).saturating_mul(CONTEXT_BUCKET).min(full);
        let mut ids = context.to_vec();
        ids.resize(t, last);
        if self.config.real {
            return self.real_next_token_logits(&ids, t, context.len() - 1);
        }
        let inputs = self.backend.upload_ids(&ids);
        self.config.seq_len = t;
        let mut ws = self.workspace(1);
        self.forward(&mut ws, &inputs);
        self.config.seq_len = full;
        let (b, l, w, v) = (&self.backend, &self.layout, self.config.width, self.config.vocab_size);
        let position = context.len() - 1;
        let x = if l.final_norm.is_some() { &ws.final_normed } else { &ws.final_x };
        if self.config.tied_head {
            // Re<x, e_v> = x_re . Er_v + x_im . Ei_v
            b.gemm(&Gemm::new(1, v, w, false, true).ld(2 * w, w, v).offsets(position * 2 * w, l.embedding_re, 0), x, &self.params, &mut ws.logits);
            b.gemm(&Gemm::new(1, v, w, false, true).ld(2 * w, w, v).offsets(position * 2 * w + w, l.embedding_im, 0).accumulate(), x, &self.params, &mut ws.logits);
        } else {
            b.gemm(&Gemm::new(1, w, 2 * w, false, false).offsets(position * 2 * w, l.ctl, 0), x, &self.params, &mut ws.real);
            b.gemm(&Gemm::new(1, v, w, false, false).offsets(0, l.head, 0), &ws.real, &self.params, &mut ws.logits);
        }
        b.add_row_vector(&mut ws.logits, 1, v, &self.params, l.head_bias);
        let mut logits = b.download(&ws.logits);
        logits.truncate(v);
        self.workspace = Some(ws);
        logits
    }

    fn forward(&self, ws: &mut Workspace<B>, inputs: &B::Ids) {
        let (b, c, l) = (&self.backend, &self.config, &self.layout);
        let (rows, w) = (ws.rows, c.width);
        b.embed(&self.params, l.embedding_re, l.embedding_im, inputs, 0, rows, w, &mut ws.embedded);
        let first = &mut ws.layers[0].input;
        b.layer_norm(&ws.embedded, rows, w, NORM_EPSILON, !c.rms_norm, &self.params, l.input_norm.gamma, l.input_norm.beta, first, &mut ws.stats0);
        let beta = c.residual_beta();
        for i in 0..c.layers {
            self.layer_forward(ws, i);
            // Layer output y = mid + beta leaky(o): the input of the next layer (or of the head).
            let (head, tail) = ws.layers.split_at_mut(i + 1);
            let next = if i + 1 < c.layers { &mut tail[0].input } else { &mut ws.final_x };
            if c.standard_residual {
                b.copy(&head[i].mid, 0, next, 0, rows * 2 * w);
                b.axpy(1.0, &head[i].out, next, rows * 2 * w);
            } else {
                b.leaky_residual(&head[i].out, &head[i].mid, rows, w, beta, LEAKY_SLOPE, next);
            }
        }
        if let Some(n) = l.final_norm {
            b.layer_norm(&ws.final_x, rows, w, NORM_EPSILON, !c.rms_norm, &self.params, n.gamma, n.beta, &mut ws.final_normed, &mut ws.stats_final);
        }
    }

    /// One layer up to the feed-forward pre-activation o = g W2 + b2 (in `out`); `forward` adds the residual.
    fn layer_forward(&self, ws: &mut Workspace<B>, i: usize) {
        let (b, c) = (&self.backend, &self.config);
        let lo = &self.layout.layers[i];
        let blk = &self.blocks[i];
        let (w1, w2) = (&blk.w1, &blk.w2);
        let (rows, batch, w, t, heads, hd, hf) = (ws.rows, ws.batch, c.width, c.seq_len, c.heads, c.head_dim(), c.ffn_hidden);
        let beta = c.residual_beta();
        let a = &mut ws.layers[i];

        // Attention.
        b.layer_norm(&a.input, rows, w, NORM_EPSILON, !c.rms_norm, &self.params, lo.attention_norm.gamma, lo.attention_norm.beta, &mut a.h1, &mut a.stats1);
        if let Some(wqkv) = &blk.wqkv {
            b.gemm(&Gemm::new(rows, 6 * w, 2 * w, false, false), &a.h1, wqkv, &mut a.qkv);
            b.rope_to_heads(&a.qkv, 6 * w, 0, 3 * w, rows, t, heads, hd, c.rope(), &mut a.qh);
            b.rope_to_heads(&a.qkv, 6 * w, w, 4 * w, rows, t, heads, hd, c.rope(), &mut a.kh);
            if c.hermitian_scores {
                // Re(q conj(k)) = [qr | qi] . [kr | ki]: the head-planar rows (2hd per head) directly.
                let scale = 1.0 / ((2.0 * hd as f64).sqrt() + 1e-12) as f32;
                for s in 0..batch {
                    let g = Gemm::new(t, t, 2 * hd, false, true).ld(2 * w, 2 * w, t).offsets(s * t * 2 * w, s * t * 2 * w, s * heads * t * t).batched(heads, 2 * hd, 2 * hd, t * t).alpha(scale);
                    b.gemm(&g, &a.qh, &a.kh, &mut a.probs);
                }
            } else {
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
            }
            b.softmax_window(&mut a.probs, batch * heads, t, c.window);
            let aw = c.attention_width();
            for s in 0..batch {
                // Weighted values: Re(V) at columns 2w + h hd of qkv (and Im(V) at 5w + h hd with complex values).
                for part in 0..aw / w {
                    let g = Gemm::new(t, hd, t, false, false).ld(t, 6 * w, aw).offsets(s * heads * t * t, s * t * 6 * w + 2 * w + part * 3 * w, s * t * aw + part * w).batched(heads, t * t, hd, hd);
                    b.gemm(&g, &a.probs, &a.qkv, &mut a.att);
                }
            }
            if let Some(wo) = &blk.wo {
                b.gemm(&Gemm::new(rows, 2 * w, 2 * w, false, false), &a.att, wo, &mut a.att_o);
            }
        }
        // mid = input + beta (attention output, Wo applied if present) + beta (mixer output).
        if c.complex_values {
            b.copy(&a.input, 0, &mut a.mid, 0, rows * 2 * w);
            if blk.wqkv.is_some() {
                b.axpy(beta, if blk.wo.is_some() { &a.att_o } else { &a.att }, &mut a.mid, rows * 2 * w);
            }
        } else {
            b.attention_residual(&a.input, &a.att, rows, w, beta, &mut a.mid);
        }
        if let (Some(m), Some(wvg), Some(mixer_wo)) = (&lo.mixer, &blk.wvg, &blk.mixer_wo) {
            // [v | g] = h Wvg as planar rows [vr | gr | vi | gi]; z = wavelet conv of v in place of v; u = z * silu(g).
            b.gemm(&Gemm::new(rows, 4 * w, 2 * w, false, false), &a.h1, wvg, &mut a.vg);
            b.copy(&a.vg, 0, &mut a.cg, 0, rows * 4 * w);
            b.wavelet_conv(&a.vg, 4 * w, 0, 2 * w, rows, t, w, &self.params, m.taps_re, m.taps_im, WAVELET_SCALES, WAVELET_TAPS, &mut a.cg);
            b.gated(&a.cg, rows, w, &mut a.u);
            b.gemm(&Gemm::new(rows, 2 * w, 2 * w, false, false), &a.u, mixer_wo, &mut a.mix_o);
            b.axpy(beta, &a.mix_o, &mut a.mid, rows * 2 * w);
        }

        // Feed-forward.
        b.layer_norm(&a.mid, rows, w, NORM_EPSILON, !c.rms_norm, &self.params, lo.ffn_norm.gamma, lo.ffn_norm.beta, &mut a.h2, &mut a.stats2);
        b.gemm(&Gemm::new(rows, 4 * hf, 2 * w, false, false), &a.h2, w1, &mut a.hidden);
        b.add_row_vector(&mut a.hidden, rows, 4 * hf, &self.params, lo.b1);
        b.gated(&a.hidden, rows, hf, &mut a.gated);
        b.gemm(&Gemm::new(rows, 2 * w, 2 * hf, false, false), &a.gated, w2, &mut a.out);
        b.add_row_vector(&mut a.out, rows, 2 * w, &self.params, lo.b2);
    }

    /// Output head and loss in chunks of positions; with `backward`, the head gradients and d(final x).
    fn head(&mut self, ws: &mut Workspace<B>, targets: &B::Ids, backward: bool, grad_scale: f32, counted: usize) -> f64 {
        let (c, l) = (&self.config, &self.layout);
        let (rows, w, v) = (ws.rows, c.width, c.vocab_size);
        let b = &self.backend;
        let x = if l.final_norm.is_some() { &ws.final_normed } else { &ws.final_x };
        if c.tied_head {
            return self.tied_head(ws, targets, backward, grad_scale, counted);
        }
        b.gemm(&Gemm::new(rows, w, 2 * w, false, false).offsets(0, l.ctl, 0), x, &self.params, &mut ws.real);
        let mut loss = 0.0;
        let chunk = HEAD_CHUNK.min(rows);
        let scale = grad_scale / counted as f32;
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
            let x = if l.final_norm.is_some() { &ws.final_normed } else { &ws.final_x };
            b.gemm(&Gemm::new(2 * w, w, rows, true, false).offsets(0, 0, l.ctl).accumulate(), x, &ws.d_real, &mut self.grads);
            b.gemm(&Gemm::new(rows, 2 * w, w, false, true).offsets(0, l.ctl, 0), &ws.d_real, &self.params, &mut ws.dx);
        }
        loss / counted as f64
    }

    /// Tied head: logits Re<x, e_v> = x_re Er^T + x_im Ei^T (+ bias), in chunks; with `backward`, the embedding and
    /// bias gradients and d(head input) in `dx`.
    fn tied_head(&mut self, ws: &mut Workspace<B>, targets: &B::Ids, backward: bool, grad_scale: f32, counted: usize) -> f64 {
        let (c, l) = (&self.config, &self.layout);
        let (rows, w, v) = (ws.rows, c.width, c.vocab_size);
        let b = &self.backend;
        let x = if l.final_norm.is_some() { &ws.final_normed } else { &ws.final_x };
        let mut loss = 0.0;
        let chunk = HEAD_CHUNK.min(rows);
        let scale = grad_scale / counted as f32;
        for start in (0..rows).step_by(chunk) {
            let len = chunk.min(rows - start);
            for (part, table) in [(0, l.embedding_re), (1, l.embedding_im)] {
                let mut g = Gemm::new(len, v, w, false, true).ld(2 * w, w, v).offsets(start * 2 * w + part * w, table, 0);
                if part == 1 {
                    g = g.accumulate();
                }
                b.gemm(&g, x, &self.params, &mut ws.logits);
            }
            b.add_row_vector(&mut ws.logits, len, v, &self.params, l.head_bias);
            loss += b.cross_entropy(&mut ws.logits, len, v, targets, start, scale);
            if backward {
                b.column_sums(&ws.logits, len, v, &mut self.grads, l.head_bias);
                for (part, table) in [(0, l.embedding_re), (1, l.embedding_im)] {
                    // dE += dlogits^T x_part (v x w), d x_part = dlogits E (len x w).
                    b.gemm(&Gemm::new(v, w, len, true, false).ld(v, 2 * w, w).offsets(0, start * 2 * w + part * w, table).accumulate(), &ws.logits, x, &mut self.grads);
                    b.gemm(&Gemm::new(len, w, v, false, false).ld(v, w, 2 * w).offsets(0, table, start * 2 * w + part * w), &ws.logits, &self.params, &mut ws.dx);
                }
            }
        }
        loss / counted as f64
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
        // ws.dx holds d(head input); through the final norm to d(layer output) of the last layer.
        if let Some(n) = self.layout.final_norm {
            b.layer_norm_backward(&ws.final_x, &ws.stats_final, &ws.dx, rows, w, !c.rms_norm, &self.params, n.gamma, &mut ws.d_out, &mut self.grads, n.gamma, n.beta);
            b.copy(&ws.d_out, 0, &mut ws.dx, 0, rows * 2 * w);
        }
        let aw = c.attention_width();
        for i in (0..c.layers).rev() {
            let lo = self.layout.layers[i].clone();
            let blk = &self.blocks[i];
            let (w1, w2) = (&blk.w1, &blk.w2);
            let a = &ws.layers[i];

            // Feed-forward: y = mid + beta leaky(o) (or y = mid + o), o = g W2 + b2.
            if c.standard_residual {
                b.copy(&ws.dx, 0, &mut ws.d_out, 0, rows * 2 * w);
            } else {
                b.leaky_residual_backward(&a.out, &ws.dx, rows, w, beta, LEAKY_SLOPE, &mut ws.d_out);
            }
            b.column_sums(&ws.d_out, rows, 2 * w, &mut self.grads, lo.b2);
            Self::linear_backward(b, &a.gated, &ws.d_out, w2, lo.w2, rows, &mut self.block_grad, &mut self.grads, &mut ws.d_gated);
            b.gated_backward(&a.hidden, &ws.d_gated, rows, hf, &mut ws.d_hidden);
            b.column_sums(&ws.d_hidden, rows, 4 * hf, &mut self.grads, lo.b1);
            Self::linear_backward(b, &a.h2, &ws.d_hidden, w1, lo.w1, rows, &mut self.block_grad, &mut self.grads, &mut ws.dtmp);
            // d(mid) = dy + norm backward of d(h2).
            b.layer_norm_backward(&a.mid, &a.stats2, &ws.dtmp, rows, w, !c.rms_norm, &self.params, lo.ffn_norm.gamma, &mut ws.d_out, &mut self.grads, lo.ffn_norm.gamma, lo.ffn_norm.beta);
            b.axpy(1.0, &ws.d_out, &mut ws.dx, rows * 2 * w);

            if let (Some(wqkv), Some(wqkv_block)) = (lo.wqkv, &blk.wqkv) {
                // Attention: mid = input + beta [att | 0] (or input + beta att with complex values).
                if let (Some(wo), Some(wo_block)) = (lo.wo, &blk.wo) {
                    // mid = input + beta att Wo: d(att Wo) = beta dx, then back through Wo.
                    b.copy(&ws.dx, 0, &mut ws.d_o, 0, rows * 2 * w);
                    b.scale(&mut ws.d_o, rows * 2 * w, beta);
                    Self::linear_backward(b, &a.att, &ws.d_o, wo_block, wo, rows, &mut self.block_grad, &mut self.grads, &mut ws.d_att);
                } else if c.complex_values {
                    b.copy(&ws.dx, 0, &mut ws.d_att, 0, rows * 2 * w);
                    b.scale(&mut ws.d_att, rows * 2 * w, beta);
                } else {
                    b.real_part_scaled(&ws.dx, rows, w, beta, &mut ws.d_att);
                }
                b.fill(&mut ws.d_qkv, 0, rows * 6 * w, 0.0);
                for s in 0..batch {
                    for part in 0..aw / w {
                        // dP = sum over parts of d_att V^T, dV = P^T d_att (V = Re(V), and Im(V) with complex values).
                        let v_off = s * t * 6 * w + 2 * w + part * 3 * w;
                        let mut g = Gemm::new(t, t, hd, false, true).ld(aw, 6 * w, t).offsets(s * t * aw + part * w, v_off, s * heads * t * t).batched(heads, hd, hd, t * t);
                        if part > 0 {
                            g = g.accumulate();
                        }
                        b.gemm(&g, &ws.d_att, &a.qkv, &mut ws.d_probs);
                        let g = Gemm::new(t, hd, t, true, false).ld(t, aw, 6 * w).offsets(s * heads * t * t, s * t * aw + part * w, v_off).batched(heads, t * t, hd, hd);
                        b.gemm(&g, &a.probs, &ws.d_att, &mut ws.d_qkv);
                    }
                }
                b.softmax_window_backward(&a.probs, &mut ws.d_probs, batch * heads, t);
                if c.hermitian_scores {
                    // d[qr|qi] = scale dS [kr|ki], d[kr|ki] = scale dS^T [qr|qi] (head-planar).
                    let hscale = 1.0 / ((2.0 * hd as f64).sqrt() + 1e-12) as f32;
                    for s in 0..batch {
                        let g = Gemm::new(t, 2 * hd, t, false, false).ld(t, 2 * w, 2 * w).offsets(s * heads * t * t, s * t * 2 * w, s * t * 2 * w).batched(heads, t * t, 2 * hd, 2 * hd).alpha(hscale);
                        b.gemm(&g, &ws.d_probs, &a.kh, &mut ws.d_qh);
                        let g = Gemm::new(t, 2 * hd, t, true, false).ld(t, 2 * w, 2 * w).offsets(s * heads * t * t, s * t * 2 * w, s * t * 2 * w).batched(heads, t * t, 2 * hd, 2 * hd).alpha(hscale);
                        b.gemm(&g, &ws.d_probs, &a.qh, &mut ws.d_kh);
                    }
                } else {
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
                }
                b.rope_from_heads(&ws.d_qh, rows, t, heads, hd, c.rope(), &mut ws.d_qkv, 6 * w, 0, 3 * w);
                b.rope_from_heads(&ws.d_kh, rows, t, heads, hd, c.rope(), &mut ws.d_qkv, 6 * w, w, 4 * w);
                Self::linear_backward(b, &a.h1, &ws.d_qkv, wqkv_block, wqkv, rows, &mut self.block_grad, &mut self.grads, &mut ws.dtmp);
            }
            if let (Some(m), Some(wvg), Some(mixer_wo)) = (lo.mixer, &blk.wvg, &blk.mixer_wo) {
                // Mixer: d(u Wo) = beta dx -> du -> d[z | g] -> d[v | g] (wavelet conv backward) -> dh.
                b.copy(&ws.dx, 0, &mut ws.d_mo, 0, rows * 2 * w);
                b.scale(&mut ws.d_mo, rows * 2 * w, beta);
                Self::linear_backward(b, &a.u, &ws.d_mo, mixer_wo, m.wo, rows, &mut self.block_grad, &mut self.grads, &mut ws.d_u);
                b.gated_backward(&a.cg, &ws.d_u, rows, w, &mut ws.d_cg);
                b.copy(&ws.d_cg, 0, &mut ws.d_vg, 0, rows * 4 * w);
                b.wavelet_conv_backward_input(&ws.d_cg, 4 * w, 0, 2 * w, rows, t, w, &self.params, m.taps_re, m.taps_im, WAVELET_SCALES, WAVELET_TAPS, &mut ws.d_vg);
                b.wavelet_conv_backward_taps(&ws.d_cg, &a.vg, 4 * w, 0, 2 * w, rows, t, w, WAVELET_SCALES, WAVELET_TAPS, &mut self.grads, m.taps_re, m.taps_im);
                Self::linear_backward(b, &a.h1, &ws.d_vg, wvg, m.wvg, rows, &mut self.block_grad, &mut self.grads, &mut ws.d_mo);
                if lo.wqkv.is_some() {
                    b.axpy(1.0, &ws.d_mo, &mut ws.dtmp, rows * 2 * w);
                } else {
                    b.copy(&ws.d_mo, 0, &mut ws.dtmp, 0, rows * 2 * w);
                }
            }
            b.layer_norm_backward(&a.input, &a.stats1, &ws.dtmp, rows, w, !c.rms_norm, &self.params, lo.attention_norm.gamma, &mut ws.d_out, &mut self.grads, lo.attention_norm.gamma, lo.attention_norm.beta);
            b.axpy(1.0, &ws.d_out, &mut ws.dx, rows * 2 * w);
        }
        // Input norm and embedding.
        let l = &self.layout;
        b.layer_norm_backward(&ws.embedded, &ws.stats0, &ws.dx, rows, w, !c.rms_norm, &self.params, l.input_norm.gamma, &mut ws.d_out, &mut self.grads, l.input_norm.gamma, l.input_norm.beta);
        b.embed_backward(&ws.d_out, inputs, 0, rows, w, &mut self.grads, l.embedding_re, l.embedding_im);
    }
}
