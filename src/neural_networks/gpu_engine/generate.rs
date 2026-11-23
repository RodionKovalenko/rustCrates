//! Text generation with a trained engine model: next-token logits from `Model::next_token_logits`, then greedy
//! decoding (temperature 0) or sampling with temperature and top-k.

use rand::{rngs::StdRng, Rng, RngExt, SeedableRng};

use super::{backend::Backend, model::Model};

#[derive(Clone, Debug)]
pub struct SamplingSettings {
    /// 0 = greedy (always the most likely token); higher values flatten the distribution.
    pub temperature: f32,
    /// Sample only among the `top_k` most likely tokens (0 = all).
    pub top_k: usize,
    /// Divides the logits of tokens already in the context (> 1 discourages repetition, 1 = off).
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingSettings {
    fn default() -> Self {
        Self { temperature: 0.8, top_k: 40, repetition_penalty: 1.1, seed: 0 }
    }
}

/// Lowers the logits of the tokens in `context` (positive logits divided, negative multiplied by `penalty`).
pub fn apply_repetition_penalty(logits: &mut [f32], context: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = vec![false; logits.len()];
    for &id in context {
        if let Some(s) = seen.get_mut(id as usize) {
            *s = true;
        }
    }
    for (logit, _) in logits.iter_mut().zip(&seen).filter(|(_, s)| **s) {
        *logit = if *logit > 0.0 { *logit / penalty } else { *logit * penalty };
    }
}

/// Picks the next token from `logits`.
pub fn sample(logits: &[f32], settings: &SamplingSettings, rng: &mut impl Rng) -> u32 {
    let argmax = || logits.iter().enumerate().fold(0, |best, (i, v)| if *v > logits[best] { i } else { best }) as u32;
    if settings.temperature <= 0.0 {
        return argmax();
    }
    let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    if settings.top_k > 0 && settings.top_k < candidates.len() {
        candidates.select_nth_unstable_by(settings.top_k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(settings.top_k);
    }
    let max = candidates.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = candidates.iter().map(|c| (((c.1 - max) / settings.temperature) as f64).exp()).collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return argmax();
    }
    let mut r = rng.random_range(0.0..total);
    for (c, w) in candidates.iter().zip(&weights) {
        if r < *w {
            return c.0 as u32;
        }
        r -= w;
    }
    candidates.last().map(|c| c.0 as u32).unwrap_or_else(argmax)
}

/// Continues `prompt` by up to `max_new_tokens` tokens (stops early at `eos`); `on_token` sees every new token.
pub fn generate<B: Backend>(model: &mut Model<B>, prompt: &[u32], max_new_tokens: usize, eos: Option<u32>, settings: &SamplingSettings, mut on_token: impl FnMut(u32)) -> Vec<u32> {
    assert!(!prompt.is_empty(), "the prompt must contain at least one token");
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let mut ids = prompt.to_vec();
    for _ in 0..max_new_tokens {
        let mut logits = model.next_token_logits(&ids);
        let window = &ids[ids.len().saturating_sub(model.config.seq_len)..];
        apply_repetition_penalty(&mut logits, window, settings.repetition_penalty);
        let next = sample(&logits, settings, &mut rng);
        if Some(next) == eos {
            break;
        }
        ids.push(next);
        on_token(next);
    }
    ids[prompt.len()..].to_vec()
}
