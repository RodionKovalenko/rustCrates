//! Fine-tuning of a pretrained engine model on extractive question answering (SQuAD format, e.g. GermanQuAD).
//!
//! Every question becomes one causal sequence; only the answer and the closing `<eos>` count in the loss:
//!
//! ```text
//!   Frage: <question>\n\n<context window>\n\nFrage: <question>\nAntwort: <answer><eos>
//! ```
//!
//! The question comes first so that the model reads the context knowing what is asked (attention is causal), and
//! again right before the answer. A context longer than the sequence is cut into a window around the answer; such
//! long contexts also give "negative" windows without the answer, whose target is `<eos>` right away (= "not in this
//! window"). Evaluation slides windows over the whole context, generates greedily in each and takes the non-empty
//! answer with the highest mean log-probability; metrics are exact match and token F1 as in SQuAD.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::{rngs::StdRng, seq::{IndexedRandom, SliceRandom}, RngExt, SeedableRng};
use serde::Deserialize;

use super::backend::{Backend, IGNORE_TARGET};
use super::model::Model;
use super::train::{learning_rate, load_checkpoint, save_checkpoint, EngineTrainState};
use crate::neural_networks::utils::tokenizer::{detokenize, encode_with_offsets, token_id};

/// Longest answer (in tokens) trained on and generated.
pub const MAX_ANSWER_TOKENS: usize = 64;
/// Share of the long contexts that also give a window without the answer, per epoch.
const NEGATIVE_SHARE: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct QaExample {
    pub context: String,
    pub question: String,
    pub answer: String,
    /// Byte offset of the answer in `context`.
    pub answer_start: usize,
}

#[derive(Deserialize)]
struct SquadFile {
    data: Vec<SquadArticle>,
}
#[derive(Deserialize)]
struct SquadArticle {
    paragraphs: Vec<SquadParagraph>,
}
#[derive(Deserialize)]
struct SquadParagraph {
    context: String,
    qas: Vec<SquadQa>,
}
#[derive(Deserialize)]
struct SquadQa {
    question: String,
    answers: Vec<SquadAnswer>,
}
#[derive(Deserialize)]
struct SquadAnswer {
    text: String,
    #[serde(default)]
    answer_start: Option<usize>,
}

/// Byte offset of the answer: `answer_start` (characters) if the text is there, else the first occurrence.
fn locate_answer(context: &str, answer: &str, char_start: Option<usize>) -> Option<usize> {
    if let Some(start) = char_start {
        let byte = context.char_indices().nth(start).map(|(b, _)| b).unwrap_or(context.len());
        if context[byte..].starts_with(answer) {
            return Some(byte);
        }
    }
    context.find(answer)
}

/// The answerable questions of SQuAD-format files (first answer of each question).
pub fn load_squad(paths: &[PathBuf]) -> Result<Vec<QaExample>, Box<dyn std::error::Error>> {
    let mut examples = Vec::new();
    for path in paths {
        let file: SquadFile = serde_json::from_str(&std::fs::read_to_string(path)?).map_err(|e| format!("{}: {e}", path.display()))?;
        for paragraph in file.data.into_iter().flat_map(|a| a.paragraphs) {
            for qa in paragraph.qas {
                let Some(answer) = qa.answers.into_iter().find(|a| !a.text.trim().is_empty()) else { continue };
                if let Some(answer_start) = locate_answer(&paragraph.context, &answer.text, answer.answer_start) {
                    examples.push(QaExample { context: paragraph.context.clone(), question: qa.question, answer: answer.text, answer_start });
                }
            }
        }
    }
    Ok(examples)
}

/// The SQuAD files of `dir` whose name starts with `prefix` (e.g. `germanquad_train`), sorted.
pub fn squad_files(dir: &Path, prefix: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json") && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(prefix)))
        .collect();
    files.sort();
    Ok(files)
}

type TokenResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// A question in tokens: the fixed parts of the prompt, the context with byte offsets, the answer.
#[derive(Clone, Debug)]
pub struct EncodedQa {
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
    pub context: Vec<u32>,
    /// Context tokens `answer_span.0 .. answer_span.1` contain the answer.
    pub answer_span: (usize, usize),
    pub answer: Vec<u32>,
}

pub fn prompt_parts(question: &str) -> TokenResult<(Vec<u32>, Vec<u32>)> {
    let question = question.trim();
    Ok((encode_with_offsets(&format!("Frage: {question}\n\n"))?.0, encode_with_offsets(&format!("\n\nFrage: {question}\nAntwort:"))?.0))
}

pub fn encode_example(example: &QaExample) -> TokenResult<Option<EncodedQa>> {
    let (prefix, suffix) = prompt_parts(&example.question)?;
    let (context, offsets) = encode_with_offsets(&example.context)?;
    let answer = encode_with_offsets(&format!(" {}", example.answer.trim()))?.0;
    let (start, end) = (example.answer_start, example.answer_start + example.answer.len());
    let first = offsets.iter().position(|o| o.1 > start);
    let last = offsets.iter().rposition(|o| o.0 < end);
    match (first, last) {
        (Some(first), Some(last)) if first <= last && !answer.is_empty() && answer.len() <= MAX_ANSWER_TOKENS => {
            Ok(Some(EncodedQa { prefix, suffix, context, answer_span: (first, last + 1), answer }))
        }
        _ => Ok(None),
    }
}

/// Inputs and targets (`seq_len` each) of `prefix + window + suffix + answer + eos`; only the answer and `eos` are
/// targets, the rest and the padding are `IGNORE_TARGET`.
pub fn sequence(prefix: &[u32], window: &[u32], suffix: &[u32], answer: &[u32], eos: u32, seq_len: usize) -> (Vec<u32>, Vec<u32>) {
    let mut ids: Vec<u32> = prefix.iter().chain(window).chain(suffix).chain(answer).copied().collect();
    ids.push(eos);
    assert!(ids.len() <= seq_len + 1, "sequence of {} tokens does not fit {seq_len}", ids.len());
    let prompt = prefix.len() + window.len() + suffix.len();
    let mut inputs = ids[..ids.len() - 1].to_vec();
    let mut targets = vec![IGNORE_TARGET; inputs.len()];
    for i in prompt - 1..inputs.len() {
        targets[i] = ids[i + 1];
    }
    inputs.resize(seq_len, eos);
    targets.resize(seq_len, IGNORE_TARGET);
    (inputs, targets)
}

/// Context tokens that fit next to the prompt and an answer of `answer_len` tokens.
fn context_budget(e: &EncodedQa, answer_len: usize, seq_len: usize) -> usize {
    seq_len.saturating_sub(e.prefix.len() + e.suffix.len() + answer_len + 1)
}

/// The training sequences of one question for one epoch: the window with the answer (random position when the
/// context is too long), and for long contexts sometimes a window without it (target: `eos` at once).
pub fn training_sequences(e: &EncodedQa, eos: u32, seq_len: usize, rng: &mut StdRng) -> Vec<(Vec<u32>, Vec<u32>)> {
    let budget = context_budget(e, e.answer.len(), seq_len);
    let (a0, a1) = e.answer_span;
    let n = e.context.len();
    if budget < a1 - a0 {
        return Vec::new();
    }
    if n <= budget {
        return vec![sequence(&e.prefix, &e.context, &e.suffix, &e.answer, eos, seq_len)];
    }
    let lowest = a1.saturating_sub(budget);
    let highest = a0.min(n - budget);
    let start = rng.random_range(lowest..=highest.max(lowest));
    let mut out = vec![sequence(&e.prefix, &e.context[start..start + budget], &e.suffix, &e.answer, eos, seq_len)];
    if rng.random_bool(NEGATIVE_SHARE) {
        let negative_budget = context_budget(e, 0, seq_len);
        // Windows entirely before or after the answer.
        let mut starts = Vec::new();
        if a0 >= negative_budget {
            starts.push(rng.random_range(0..=a0 - negative_budget));
        }
        if n >= a1 + negative_budget {
            starts.push(rng.random_range(a1..=n - negative_budget));
        }
        if let Some(&s) = starts.as_slice().choose(rng) {
            out.push(sequence(&e.prefix, &e.context[s..s + negative_budget], &e.suffix, &[], eos, seq_len));
        }
    }
    out
}

/// Context windows for evaluation: `budget` tokens with half overlap, covering the whole context.
pub fn evaluation_windows(n: usize, budget: usize) -> Vec<(usize, usize)> {
    if n <= budget {
        return vec![(0, n)];
    }
    let stride = (budget / 2).max(1);
    let mut windows = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + budget).min(n);
        windows.push((start, end));
        if end == n {
            break;
        }
        start += stride;
    }
    windows
}

fn log_softmax_at(logits: &[f32], id: u32) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v)) as f64;
    let log_sum = logits.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>().ln() + max;
    logits[id as usize] as f64 - log_sum
}

/// Greedy answer to a question: the best non-empty answer over the context windows (empty if every window says
/// "not here"). Returns the answer tokens and their mean log-probability (including the closing `eos`).
pub fn answer_question<B: Backend>(model: &mut Model<B>, question: &str, context: &str, eos: u32) -> TokenResult<(Vec<u32>, f64)> {
    let seq_len = model.config.seq_len;
    let (prefix, suffix) = prompt_parts(question)?;
    let (ctx, _) = encode_with_offsets(context)?;
    let budget = seq_len.saturating_sub(prefix.len() + suffix.len() + MAX_ANSWER_TOKENS + 1).max(1);
    let mut best: Option<(Vec<u32>, f64)> = None;
    for (start, end) in evaluation_windows(ctx.len(), budget) {
        let mut ids: Vec<u32> = prefix.iter().chain(&ctx[start..end]).chain(&suffix).copied().collect();
        let prompt = ids.len();
        let mut log_prob = 0.0;
        loop {
            let logits = model.next_token_logits(&ids);
            let next = logits.iter().enumerate().fold(0, |b, (i, v)| if *v > logits[b] { i } else { b }) as u32;
            log_prob += log_softmax_at(&logits, next);
            if next == eos || ids.len() - prompt >= MAX_ANSWER_TOKENS {
                break;
            }
            ids.push(next);
        }
        let answer = ids[prompt..].to_vec();
        if answer.is_empty() {
            continue;
        }
        let mean = log_prob / (answer.len() + 1) as f64;
        if best.as_ref().is_none_or(|b| mean > b.1) {
            best = Some((answer, mean));
        }
    }
    Ok(best.unwrap_or((Vec::new(), f64::NEG_INFINITY)))
}

/// SQuAD normalization: lower case, punctuation removed, whitespace collapsed.
pub fn normalize_answer(text: &str) -> Vec<String> {
    text.to_lowercase().chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect::<String>().split_whitespace().map(str::to_string).collect()
}

/// Exact match (0/1) and token F1 between a prediction and the reference answer.
pub fn exact_match_and_f1(prediction: &str, reference: &str) -> (f64, f64) {
    let (p, r) = (normalize_answer(prediction), normalize_answer(reference));
    let exact = if p == r { 1.0 } else { 0.0 };
    if p.is_empty() || r.is_empty() {
        return (exact, exact);
    }
    let mut remaining = r.clone();
    let mut common = 0;
    for token in &p {
        if let Some(i) = remaining.iter().position(|t| t == token) {
            remaining.swap_remove(i);
            common += 1;
        }
    }
    if common == 0 {
        return (exact, 0.0);
    }
    let precision = common as f64 / p.len() as f64;
    let recall = common as f64 / r.len() as f64;
    (exact, 2.0 * precision * recall / (precision + recall))
}

/// Exact match and F1 (in %) over `examples`; prints the first `show` predictions.
pub fn evaluate<B: Backend>(model: &mut Model<B>, examples: &[QaExample], eos: u32, show: usize) -> TokenResult<(f64, f64)> {
    let (mut em, mut f1) = (0.0, 0.0);
    for (i, e) in examples.iter().enumerate() {
        let (answer, _) = answer_question(model, &e.question, &e.context, eos)?;
        let prediction = detokenize(&answer, true)?.trim().to_string();
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

#[derive(Clone, Debug)]
pub struct FinetuneSettings {
    pub data_dir: PathBuf,
    pub train_prefix: String,
    pub test_prefix: String,
    /// Pretrained model to start from.
    pub base: PathBuf,
    /// Where the fine-tuned model is saved.
    pub checkpoint: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub micro_batch: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub eval_every: usize,
    /// Test questions for the exact match / F1 evaluation after every epoch but the last, which uses all of them
    /// (0 = no evaluation, the loss decides alone).
    pub eval_questions: usize,
    pub loss_scaling: bool,
    pub seed: u64,
}

impl FinetuneSettings {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            train_prefix: "germanquad_train".into(),
            test_prefix: "germanquad_test".into(),
            base: PathBuf::from(super::train::CHECKPOINT),
            checkpoint: PathBuf::from("STORAGE/engine_qa"),
            epochs: 3,
            batch_size: 16,
            micro_batch: std::env::var("MICRO_BATCH").ok().and_then(|m| m.parse().ok()).unwrap_or(16),
            learning_rate: 5e-5,
            warmup_steps: 100,
            eval_every: 200,
            eval_questions: 300,
            loss_scaling: false,
            seed: 1,
        }
    }
}

/// Mean answer loss over the test sequences (fixed: the answer window of every question).
fn test_loss<B: Backend>(model: &mut Model<B>, sequences: &[(Vec<u32>, Vec<u32>)], micro: usize) -> f64 {
    let (mut total, mut counted) = (0.0, 0usize);
    for chunk in sequences.chunks(micro) {
        let inputs: Vec<u32> = chunk.iter().flat_map(|s| s.0.iter().copied()).collect();
        let targets: Vec<u32> = chunk.iter().flat_map(|s| s.1.iter().copied()).collect();
        let n = targets.iter().filter(|&&t| t != IGNORE_TARGET).count();
        total += model.forward_backward_targets(&inputs, &targets, false, 1.0) * n as f64;
        counted += n;
    }
    total / counted.max(1) as f64
}

pub fn finetune<B: Backend>(backend: B, settings: &FinetuneSettings) -> Result<(), Box<dyn std::error::Error>> {
    let eos = token_id("<eos>").ok_or("the tokenizer has no <eos> token")?;
    let train = load_squad(&squad_files(&settings.data_dir, &settings.train_prefix)?)?;
    let test = load_squad(&squad_files(&settings.data_dir, &settings.test_prefix)?)?;
    if train.is_empty() || test.is_empty() {
        return Err(format!("no {}* / {}* questions in {}", settings.train_prefix, settings.test_prefix, settings.data_dir.display()).into());
    }
    let (config, params, _, base_state) = load_checkpoint(&settings.base).map_err(|e| format!("{}: {e}", settings.base.display()))?;
    println!("Fine-tuning {} (pretraining step {}, validation loss {:?})", settings.base.display(), base_state.step, base_state.best_val_loss);
    let mut model = Model::from_parameters(backend, config, &params);
    let seq_len = model.config.seq_len;

    let encode_all = |examples: &[QaExample]| -> TokenResult<Vec<EncodedQa>> { Ok(examples.iter().map(encode_example).collect::<TokenResult<Vec<_>>>()?.into_iter().flatten().collect()) };
    let train_encoded = encode_all(&train).map_err(|e| e.to_string())?;
    let test_encoded = encode_all(&test).map_err(|e| e.to_string())?;
    let mut fixed = StdRng::seed_from_u64(7);
    let test_sequences: Vec<_> = test_encoded.iter().filter_map(|e| training_sequences(e, eos, seq_len, &mut fixed).into_iter().next()).collect();
    println!("{} train questions ({} usable), {} test questions ({} for the loss)", train.len(), train_encoded.len(), test.len(), test_sequences.len());
    let mut eval_set = test.clone();
    eval_set.shuffle(&mut StdRng::seed_from_u64(3));
    eval_set.truncate(settings.eval_questions);

    let mut rng = StdRng::seed_from_u64(settings.seed);
    let epoch_sequences = |rng: &mut StdRng| {
        let mut s: Vec<_> = train_encoded.iter().flat_map(|e| training_sequences(e, eos, seq_len, rng)).collect();
        s.shuffle(rng);
        s
    };
    let steps_per_epoch = epoch_sequences(&mut StdRng::seed_from_u64(settings.seed)).len() / settings.batch_size;
    let total_steps = steps_per_epoch * settings.epochs;
    println!("{} epochs x {} steps, batch {}, micro-batch {}, peak learning rate {}", settings.epochs, steps_per_epoch, settings.batch_size, settings.micro_batch, settings.learning_rate);

    let total = model.layout.total;
    let (mut m, mut v) = (model.backend.zeros(total), model.backend.zeros(total));
    let mut state = EngineTrainState { loss_scale: if settings.loss_scaling { 16384.0 } else { 1.0 }, ..EngineTrainState::default() };
    let micro = settings.micro_batch.clamp(1, settings.batch_size);
    let started = Instant::now();
    if let Some(parent) = settings.checkpoint.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut history = std::fs::OpenOptions::new().create(true).append(true).open(super::train::file(&settings.checkpoint, ".history.csv"))?;

    let initial = test_loss(&mut model, &test_sequences, micro);
    println!("before fine-tuning: test answer loss {initial:.4}");
    let (mut loss_sum, mut loss_count, mut skipped, mut clean) = (0.0, 0usize, 0usize, 0usize);
    let mut step = 0;
    for epoch in 1..=settings.epochs {
        let sequences = epoch_sequences(&mut rng);
        for batch in sequences.chunks_exact(settings.batch_size) {
            let lr = learning_rate(step, settings.learning_rate, settings.warmup_steps, total_steps.max(settings.warmup_steps + 1));
            let counted_batch: usize = batch.iter().map(|s| s.1.iter().filter(|&&t| t != IGNORE_TARGET).count()).sum();
            model.zero_grads();
            let mut loss = 0.0;
            for chunk in batch.chunks(micro) {
                let inputs: Vec<u32> = chunk.iter().flat_map(|s| s.0.iter().copied()).collect();
                let targets: Vec<u32> = chunk.iter().flat_map(|s| s.1.iter().copied()).collect();
                let weight = targets.iter().filter(|&&t| t != IGNORE_TARGET).count() as f32 / counted_batch as f32;
                loss += model.forward_backward_targets(&inputs, &targets, true, weight * state.loss_scale) * weight as f64;
            }
            step += 1;
            let norm = model.backend.sum_squares(&model.grads, total).sqrt() / state.loss_scale as f64;
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
                loss_sum += loss;
                loss_count += 1;
                clean += 1;
                if settings.loss_scaling && clean % 1000 == 0 && state.loss_scale < 1048576.0 {
                    state.loss_scale *= 2.0;
                }
            }
            if step % 50 == 0 && loss_count > 0 {
                println!("[{:5.2} h] epoch {epoch} step {step}/{total_steps}  answer loss {:.4}  lr {lr:.2e}  grad norm {norm:.2}  skipped {skipped}", started.elapsed().as_secs_f64() / 3600.0, loss_sum / loss_count as f64);
                (loss_sum, loss_count) = (0.0, 0);
            }
            if step % settings.eval_every == 0 || step == total_steps {
                let loss = test_loss(&mut model, &test_sequences, micro);
                writeln!(history, "{step},{loss:.6},{lr:.6}")?;
                state.step = step;
                if state.best_val_loss.is_none_or(|b| loss < b) {
                    state.best_val_loss = Some(loss);
                    save_checkpoint(&model, &settings.checkpoint, None, &state)?;
                    println!("step {step}: TEST ANSWER LOSS {loss:.4}  (new best, saved {})", settings.checkpoint.display());
                } else {
                    println!("step {step}: TEST ANSWER LOSS {loss:.4}  (best {:.4})", state.best_val_loss.unwrap_or(f64::NAN));
                }
            }
        }
        if settings.eval_questions > 0 {
            // After the last epoch on every test question, before on a fixed sample.
            let questions = if epoch == settings.epochs { &test[..] } else { &eval_set[..] };
            let (em, f1) = evaluate(&mut model, questions, eos, if epoch == settings.epochs { 10 } else { 3 }).map_err(|e| e.to_string())?;
            println!("epoch {epoch}: EXACT MATCH {em:.1} %  F1 {f1:.1} %  ({} test questions)", questions.len());
            writeln!(history, "# epoch {epoch} em {em:.2} f1 {f1:.2}")?;
        }
    }
    println!("done: best test answer loss {:?}, model in {}", state.best_val_loss, settings.checkpoint.display());
    Ok(())
}
