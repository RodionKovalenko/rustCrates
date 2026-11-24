//! Extractive question answering with a span head: the answer is a span of the context, predicted as a start and an
//! end position (as for SQuAD). Each sequence is
//!
//! ```text
//!   Frage: <question>\n\n<context window>        (padded to seq_len)
//! ```
//!
//! so that every context token has read the question (attention is causal). Two linear heads on the final hidden
//! state (the real and imaginary part as `2w` features) give a start and an end score per position; a softmax over
//! the context positions plus a "no answer here" position (the first token) is trained with cross-entropy for start
//! and end. Windows of a long context without the answer point both to the no-answer position. The answer of a
//! question is the best span (start <= end < start + `MAX_SPAN_TOKENS`) over all windows.
//!
//! The head itself is small and computed on the CPU from the hidden states of `Model::forward_hidden`; its gradient
//! goes back through `Model::backward_hidden`.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::{rngs::StdRng, seq::{IndexedRandom, SliceRandom}, RngExt, SeedableRng};

use super::backend::Backend;
use super::finetune::{encode_example_limited, evaluation_windows, exact_match_and_f1, load_squad, squad_files, EncodedQa, FinetuneSettings, QaExample};
use super::model::Model;
use super::train::{file, learning_rate, load_checkpoint, read_f32, save_checkpoint, EngineTrainState};
use crate::neural_networks::utils::tokenizer::{encode_with_offsets, token_id};

/// Longest answer span (in tokens) trained on and predicted.
pub const MAX_SPAN_TOKENS: usize = 96;
/// Share of the long contexts that also give a window without the answer, per epoch.
const NEGATIVE_SHARE: f64 = 0.5;
/// Position of the "no answer in this window" choice (the first prompt token).
const NULL_POSITION: usize = 0;

type TokenResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// One training or evaluation sequence: `seq_len` input ids, the context positions `context.0 .. context.1`, the
/// target start / end positions (`NULL_POSITION` when the answer is not in the window).
#[derive(Clone, Debug)]
pub struct SpanSequence {
    pub inputs: Vec<u32>,
    pub context: (usize, usize),
    pub start: usize,
    pub end: usize,
}

pub fn question_prefix(question: &str) -> TokenResult<Vec<u32>> {
    Ok(encode_with_offsets(&format!("Frage: {}\n\n", question.trim()))?.0)
}

/// `prefix + window`, padded with `pad` to `seq_len`.
pub fn span_sequence(prefix: &[u32], window: &[u32], answer: Option<(usize, usize)>, pad: u32, seq_len: usize) -> SpanSequence {
    let mut inputs: Vec<u32> = prefix.iter().chain(window).copied().collect();
    assert!(inputs.len() <= seq_len, "sequence of {} tokens does not fit {seq_len}", inputs.len());
    inputs.resize(seq_len, pad);
    let (start, end) = answer.map(|(a, b)| (prefix.len() + a, prefix.len() + b)).unwrap_or((NULL_POSITION, NULL_POSITION));
    SpanSequence { inputs, context: (prefix.len(), prefix.len() + window.len()), start, end }
}

/// The training sequences of one question for one epoch (answer span relative to the window, end inclusive).
pub fn span_training_sequences(e: &EncodedQa, pad: u32, seq_len: usize, rng: &mut StdRng) -> Vec<SpanSequence> {
    let budget = seq_len.saturating_sub(e.prefix.len());
    let (a0, a1) = e.answer_span;
    let n = e.context.len();
    if a1 - a0 > budget.min(MAX_SPAN_TOKENS) {
        return Vec::new();
    }
    if n <= budget {
        return vec![span_sequence(&e.prefix, &e.context, Some((a0, a1 - 1)), pad, seq_len)];
    }
    let lowest = a1.saturating_sub(budget);
    let highest = a0.min(n - budget);
    let start = rng.random_range(lowest..=highest.max(lowest));
    let mut out = vec![span_sequence(&e.prefix, &e.context[start..start + budget], Some((a0 - start, a1 - 1 - start)), pad, seq_len)];
    if rng.random_bool(NEGATIVE_SHARE) {
        let mut starts = Vec::new();
        if a0 >= budget {
            starts.push(rng.random_range(0..=a0 - budget));
        }
        if n >= a1 + budget {
            starts.push(rng.random_range(a1..=n - budget));
        }
        if let Some(&s) = starts.as_slice().choose(rng) {
            out.push(span_sequence(&e.prefix, &e.context[s..s + budget], None, pad, seq_len));
        }
    }
    out
}

/// Start and end heads: `[w_start (2w), b_start, w_end (2w), b_end]`.
#[derive(Clone, Debug)]
pub struct SpanHead {
    pub features: usize,
    pub params: Vec<f32>,
}

impl SpanHead {
    /// A head for hidden states of `features` values per position (`EngineConfig::hidden_features`).
    pub fn new(features: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let limit = (1.0 / features as f64).sqrt();
        let mut params = vec![0.0f32; 2 * (features + 1)];
        for (i, p) in params.iter_mut().enumerate() {
            if i % (features + 1) != features {
                *p = rng.random_range(-limit..limit) as f32;
            }
        }
        Self { features, params }
    }

    /// Start and end scores of every position of the hidden states `rows x features`.
    pub fn scores(&self, hidden: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let f = self.features;
        let (ws, bs) = (&self.params[..f], self.params[f]);
        let (we, be) = (&self.params[f + 1..2 * f + 1], self.params[2 * f + 1]);
        hidden
            .chunks(f)
            .map(|x| {
                let s: f32 = x.iter().zip(ws).map(|(a, b)| a * b).sum::<f32>() + bs;
                let e: f32 = x.iter().zip(we).map(|(a, b)| a * b).sum::<f32>() + be;
                (s, e)
            })
            .unzip()
    }
}

/// Candidate positions of a sequence: the no-answer position and the context.
fn candidates(seq: &SpanSequence) -> impl Iterator<Item = usize> + '_ {
    std::iter::once(NULL_POSITION).chain(seq.context.0..seq.context.1)
}

/// Mean over the sequences of (cross-entropy of the start + of the end) / 2 for the hidden states of `seqs`
/// (`seqs.len() * seq_len` rows); with `grads`, also the gradient of `weight * loss` with respect to the hidden
/// states (returned) and to the head (added to `grads`).
pub fn span_loss(head: &SpanHead, hidden: &[f32], seqs: &[&SpanSequence], seq_len: usize, weight: f32, grads: Option<&mut [f32]>) -> (f64, Vec<f32>) {
    let f = head.features;
    let (start_scores, end_scores) = head.scores(hidden);
    let mut d_hidden = vec![0.0f32; if grads.is_some() { hidden.len() } else { 0 }];
    let mut d_head = vec![0.0f64; head.params.len()];
    let mut loss = 0.0;
    for (n, seq) in seqs.iter().enumerate() {
        let row0 = n * seq_len;
        for (scores, target, part) in [(&start_scores, seq.start, 0usize), (&end_scores, seq.end, 1usize)] {
            let positions: Vec<usize> = candidates(seq).collect();
            let max = positions.iter().map(|&p| scores[row0 + p]).fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = positions.iter().map(|&p| (scores[row0 + p] as f64 - max).exp()).sum();
            loss += 0.5 * (sum.ln() + max - scores[row0 + target] as f64);
            if grads.is_some() {
                let scale = 0.5 * weight as f64 / seqs.len() as f64;
                let w = &head.params[part * (f + 1)..part * (f + 1) + f];
                for &p in &positions {
                    let prob = (scores[row0 + p] as f64 - max).exp() / sum;
                    let d = scale * (prob - if p == target { 1.0 } else { 0.0 });
                    let x = &hidden[(row0 + p) * f..(row0 + p + 1) * f];
                    let dx = &mut d_hidden[(row0 + p) * f..(row0 + p + 1) * f];
                    for k in 0..f {
                        dx[k] += (d * w[k] as f64) as f32;
                        d_head[part * (f + 1) + k] += d * x[k] as f64;
                    }
                    d_head[part * (f + 1) + f] += d;
                }
            }
        }
    }
    if let Some(g) = grads {
        g.iter_mut().zip(&d_head).for_each(|(g, d)| *g += *d as f32);
    }
    (loss / seqs.len().max(1) as f64, d_hidden)
}

/// Best span of one window: (score, start, end) over context positions with start <= end < start + max_len.
pub fn best_span(start_scores: &[f32], end_scores: &[f32], context: (usize, usize), max_len: usize) -> Option<(f32, usize, usize)> {
    let mut best: Option<(f32, usize, usize)> = None;
    for i in context.0..context.1 {
        for j in i..(i + max_len).min(context.1) {
            let s = start_scores[i] + end_scores[j];
            if best.is_none_or(|b| s > b.0) {
                best = Some((s, i, j));
            }
        }
    }
    best
}

/// The answer to a question: the best span over the context windows, as text.
pub fn answer_span<B: Backend>(model: &mut Model<B>, head: &SpanHead, question: &str, context: &str, pad: u32) -> TokenResult<(String, f32)> {
    let seq_len = model.config.seq_len;
    let prefix = question_prefix(question)?;
    let (ctx, offsets) = encode_with_offsets(context)?;
    let budget = seq_len.saturating_sub(prefix.len()).max(1);
    let windows = evaluation_windows(ctx.len(), budget);
    let seqs: Vec<SpanSequence> = windows.iter().map(|&(a, b)| span_sequence(&prefix, &ctx[a..b], None, pad, seq_len)).collect();
    let mut best: Option<(f32, usize, usize)> = None;
    for (chunk, wins) in seqs.chunks(16).zip(windows.chunks(16)) {
        let inputs: Vec<u32> = chunk.iter().flat_map(|s| s.inputs.iter().copied()).collect();
        let hidden = model.forward_hidden(&inputs);
        let (start_scores, end_scores) = head.scores(&hidden);
        for (n, (seq, &(w0, _))) in chunk.iter().zip(wins).enumerate() {
            let range = n * seq_len..(n + 1) * seq_len;
            if let Some((score, i, j)) = best_span(&start_scores[range.clone()], &end_scores[range], seq.context, MAX_SPAN_TOKENS) {
                // Positions -> context token indices.
                let (ti, tj) = (w0 + i - seq.context.0, w0 + j - seq.context.0);
                if best.is_none_or(|b| score > b.0) {
                    best = Some((score, ti, tj));
                }
            }
        }
    }
    let Some((score, i, j)) = best else { return Ok((String::new(), f32::NEG_INFINITY)) };
    // Text of the original context (keeps its exact spelling) from the token offsets.
    Ok((context[offsets[i].0..offsets[j].1].trim().to_string(), score))
}

/// Exact match and F1 (in %) over `examples`; prints the first `show` predictions.
pub fn evaluate_span<B: Backend>(model: &mut Model<B>, head: &SpanHead, examples: &[QaExample], pad: u32, show: usize) -> TokenResult<(f64, f64)> {
    let (mut em, mut f1) = (0.0, 0.0);
    for (i, e) in examples.iter().enumerate() {
        let (prediction, _) = answer_span(model, head, &e.question, &e.context, pad)?;
        let (a, b) = exact_match_and_f1(&prediction, &e.answer);
        em += a;
        f1 += b;
        if i < show {
            println!("  Q: {}\n     reference: {}\n     model:     {}", e.question, e.answer, prediction);
        }
    }
    let n = examples.len().max(1) as f64;
    Ok((100.0 * em / n, 100.0 * f1 / n))
}

/// Saves the span head next to the fine-tuned model (`<checkpoint>.span`).
pub fn save_head(head: &SpanHead, checkpoint: &std::path::Path) -> std::io::Result<()> {
    let bytes: Vec<u8> = head.params.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(file(checkpoint, ".span"), bytes)
}

pub fn load_head(checkpoint: &std::path::Path, features: usize) -> std::io::Result<SpanHead> {
    let params = read_f32(&file(checkpoint, ".span"))?;
    if params.len() != 2 * (features + 1) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "span head of a different width"));
    }
    Ok(SpanHead { features, params })
}

/// Mean span loss over fixed test sequences.
fn test_loss<B: Backend>(model: &mut Model<B>, head: &SpanHead, seqs: &[SpanSequence], micro: usize) -> f64 {
    let seq_len = model.config.seq_len;
    let mut total = 0.0;
    for chunk in seqs.chunks(micro) {
        let inputs: Vec<u32> = chunk.iter().flat_map(|s| s.inputs.iter().copied()).collect();
        let hidden = model.forward_hidden(&inputs);
        let refs: Vec<&SpanSequence> = chunk.iter().collect();
        total += span_loss(head, &hidden, &refs, seq_len, 1.0, None).0 * chunk.len() as f64;
    }
    total / seqs.len().max(1) as f64
}

/// Plain AdamW step on the CPU for the span head.
fn adamw_cpu(params: &mut [f32], grads: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, step: u64, clip: f32) {
    let (b1, b2, eps, wd) = (0.9f32, 0.95f32, 1e-8f32, 0.01f32);
    let (c1, c2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
    for i in 0..params.len() {
        let g = grads[i] * clip;
        m[i] = b1 * m[i] + (1.0 - b1) * g;
        v[i] = b2 * v[i] + (1.0 - b2) * g * g;
        params[i] -= lr * ((m[i] / c1) / ((v[i] / c2).sqrt() + eps) + wd * params[i]);
    }
}

pub fn finetune_span<B: Backend>(backend: B, settings: &FinetuneSettings) -> Result<(), Box<dyn std::error::Error>> {
    let pad = token_id("<eos>").ok_or("the tokenizer has no <eos> token")?;
    let train = load_squad(&squad_files(&settings.data_dir, &settings.train_prefix)?)?;
    let test = load_squad(&squad_files(&settings.data_dir, &settings.test_prefix)?)?;
    if train.is_empty() || test.is_empty() {
        return Err(format!("no {}* / {}* questions in {}", settings.train_prefix, settings.test_prefix, settings.data_dir.display()).into());
    }
    let (config, params, _, base_state) = load_checkpoint(&settings.base).map_err(|e| format!("{}: {e}", settings.base.display()))?;
    println!("Span fine-tuning of {} (pretraining step {}, validation loss {:?}): {:?}", settings.base.display(), base_state.step, base_state.best_val_loss, config);
    let mut model = Model::from_parameters(backend, config, &params);
    let seq_len = model.config.seq_len;
    let mut head = SpanHead::new(model.config.hidden_features(), settings.seed);

    let encode_all = |examples: &[QaExample]| -> TokenResult<Vec<EncodedQa>> {
        let mut out = Vec::new();
        for e in examples {
            if let Some(mut enc) = encode_example_limited(e, MAX_SPAN_TOKENS)? {
                enc.prefix = question_prefix(&e.question)?;
                out.push(enc);
            }
        }
        Ok(out)
    };
    let train_encoded = encode_all(&train).map_err(|e| e.to_string())?;
    let test_encoded = encode_all(&test).map_err(|e| e.to_string())?;
    let mut fixed = StdRng::seed_from_u64(7);
    let test_seqs: Vec<SpanSequence> = test_encoded.iter().filter_map(|e| span_training_sequences(e, pad, seq_len, &mut fixed).into_iter().next()).collect();
    println!("{} train questions ({} usable), {} test questions ({} for the loss)", train.len(), train_encoded.len(), test.len(), test_seqs.len());
    let mut eval_set = test.clone();
    eval_set.shuffle(&mut StdRng::seed_from_u64(3));
    eval_set.truncate(settings.eval_questions);

    let mut rng = StdRng::seed_from_u64(settings.seed);
    let epoch_sequences = |rng: &mut StdRng| {
        let mut s: Vec<SpanSequence> = train_encoded.iter().flat_map(|e| span_training_sequences(e, pad, seq_len, rng)).collect();
        s.shuffle(rng);
        s
    };
    let steps_per_epoch = epoch_sequences(&mut StdRng::seed_from_u64(settings.seed)).len() / settings.batch_size;
    let total_steps = steps_per_epoch * settings.epochs;
    println!("{} epochs x {} steps, batch {}, micro-batch {}, peak learning rate {} (head {})", settings.epochs, steps_per_epoch, settings.batch_size, settings.micro_batch, settings.learning_rate, 10.0 * settings.learning_rate);

    let total = model.layout.total;
    let (mut m, mut v) = (model.backend.zeros(total), model.backend.zeros(total));
    let (mut hm, mut hv) = (vec![0.0f32; head.params.len()], vec![0.0f32; head.params.len()]);
    let mut state = EngineTrainState { loss_scale: if settings.loss_scaling { 16384.0 } else { 1.0 }, ..EngineTrainState::default() };
    let micro = settings.micro_batch.clamp(1, settings.batch_size);
    let started = Instant::now();
    if let Some(parent) = settings.checkpoint.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut history = std::fs::OpenOptions::new().create(true).append(true).open(file(&settings.checkpoint, ".history.csv"))?;

    println!("before fine-tuning: test span loss {:.4} (random head)", test_loss(&mut model, &head, &test_seqs, micro));
    let (mut loss_sum, mut loss_count, mut skipped, mut clean) = (0.0, 0usize, 0usize, 0usize);
    let mut step = 0;
    for epoch in 1..=settings.epochs {
        let sequences = epoch_sequences(&mut rng);
        for batch in sequences.chunks_exact(settings.batch_size) {
            let lr = learning_rate(step, settings.learning_rate, settings.warmup_steps, total_steps.max(settings.warmup_steps + 1));
            model.zero_grads();
            let mut head_grads = vec![0.0f32; head.params.len()];
            let mut loss = 0.0;
            for chunk in batch.chunks(micro) {
                let weight = chunk.len() as f32 / batch.len() as f32;
                let inputs: Vec<u32> = chunk.iter().flat_map(|s| s.inputs.iter().copied()).collect();
                let hidden = model.forward_hidden(&inputs);
                let refs: Vec<&SpanSequence> = chunk.iter().collect();
                let (l, mut d_hidden) = span_loss(&head, &hidden, &refs, seq_len, weight, Some(&mut head_grads));
                d_hidden.iter_mut().for_each(|d| *d *= state.loss_scale);
                model.backward_hidden(&inputs, &d_hidden);
                loss += l * weight as f64;
            }
            step += 1;
            let model_squares = model.backend.sum_squares(&model.grads, total) / (state.loss_scale as f64).powi(2);
            let head_squares: f64 = head_grads.iter().map(|g| (*g as f64).powi(2)).sum();
            let norm = (model_squares + head_squares).sqrt();
            if !norm.is_finite() || !loss.is_finite() {
                skipped += 1;
                clean = 0;
                if settings.loss_scaling && state.loss_scale > 1.0 {
                    state.loss_scale /= 2.0;
                }
            } else {
                let clip = if norm > 1.0 { 1.0 / norm } else { 1.0 };
                state.adam_step += 1;
                model.backend.adamw(&mut model.params, &model.grads, &mut m, &mut v, total, lr as f32, 0.9, 0.95, 1e-8, 0.01, (clip / state.loss_scale as f64) as f32, state.adam_step);
                model.refresh_blocks();
                adamw_cpu(&mut head.params, &head_grads, &mut hm, &mut hv, 10.0 * lr as f32, state.adam_step, clip as f32);
                loss_sum += loss;
                loss_count += 1;
                clean += 1;
                if settings.loss_scaling && clean % 1000 == 0 && state.loss_scale < 1048576.0 {
                    state.loss_scale *= 2.0;
                }
            }
            if step % 50 == 0 && loss_count > 0 {
                println!("[{:5.2} h] epoch {epoch} step {step}/{total_steps}  span loss {:.4}  lr {lr:.2e}  grad norm {norm:.2}  skipped {skipped}", started.elapsed().as_secs_f64() / 3600.0, loss_sum / loss_count as f64);
                (loss_sum, loss_count) = (0.0, 0);
            }
            if step % settings.eval_every == 0 || step == total_steps {
                let loss = test_loss(&mut model, &head, &test_seqs, micro);
                writeln!(history, "{step},{loss:.6},{lr:.6}")?;
                state.step = step;
                if state.best_val_loss.is_none_or(|b| loss < b) {
                    state.best_val_loss = Some(loss);
                    save_checkpoint(&model, &settings.checkpoint, None, &state)?;
                    save_head(&head, &settings.checkpoint)?;
                    println!("step {step}: TEST SPAN LOSS {loss:.4}  (new best, saved {})", settings.checkpoint.display());
                } else {
                    println!("step {step}: TEST SPAN LOSS {loss:.4}  (best {:.4})", state.best_val_loss.unwrap_or(f64::NAN));
                }
            }
        }
        if settings.eval_questions > 0 {
            let questions = if epoch == settings.epochs { &test[..] } else { &eval_set[..] };
            let eval_started = Instant::now();
            let (em, f1) = evaluate_span(&mut model, &head, questions, pad, if epoch == settings.epochs { 10 } else { 3 }).map_err(|e| e.to_string())?;
            println!("epoch {epoch}: EXACT MATCH {em:.1} %  F1 {f1:.1} %  ({} test questions, {:.1} min)", questions.len(), eval_started.elapsed().as_secs_f64() / 60.0);
            writeln!(history, "# epoch {epoch} em {em:.2} f1 {f1:.2}")?;
        }
    }
    println!("done: best test span loss {:?}, model in {} (+ .span head)", state.best_val_loss, settings.checkpoint.display());
    Ok(())
}

/// Default location of the span model.
pub fn default_checkpoint() -> PathBuf {
    PathBuf::from("STORAGE/engine_qa")
}
