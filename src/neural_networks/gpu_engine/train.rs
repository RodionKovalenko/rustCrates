//! Pretraining with the engine on the token shards of `kaggle/prepare_pretrain.py` (`train_NNNNN.bin` /
//! `val_NNNNN.bin`, little-endian u16 ids).
//!
//! Every shard is cut into non-overlapping windows of `seq_len + 1` tokens, which are taken in a shuffled order
//! (each window exactly once, then the next shard; the order depends on the shard and the pass over the data, so it
//! is reproducible after a resume). AdamW (beta2 0.95, weight decay 0.01) with
//! a linear warmup and a cosine decay to 10 % within every run; gradient clipping at a global norm of 1. With f16
//! GEMMs the loss is scaled (dynamic loss scaling: halved and the step skipped when a gradient overflows, doubled
//! after `LOSS_SCALE_GROWTH_INTERVAL` clean steps). Every `eval_every` steps the validation loss decides whether the
//! checkpoint (parameters, AdamW moments, progress) is replaced.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::{seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};

use super::backend::Backend;
use super::model::{EngineConfig, Model};

pub const CHECKPOINT: &str = "STORAGE/engine_lm";
const INITIAL_LOSS_SCALE: f32 = 16384.0;
const LOSS_SCALE_GROWTH_INTERVAL: usize = 1000;

#[derive(Clone, Debug)]
pub struct EngineTrainSettings {
    pub data_dir: PathBuf,
    /// Path prefix of the checkpoint files (`.params`, `.adam`, `.json`, `.state.json`, `.history.csv`).
    pub checkpoint: PathBuf,
    pub steps: usize,
    pub batch_size: usize,
    pub micro_batch: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub resume: bool,
    pub eval_every: usize,
    pub eval_batches: usize,
    pub log_every: usize,
    pub max_hours: Option<f64>,
    /// Whether the GEMMs run in f16 (then the loss is scaled).
    pub loss_scaling: bool,
    pub config: EngineConfig,
}

impl EngineTrainSettings {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            checkpoint: PathBuf::from(CHECKPOINT),
            steps: 20_000,
            batch_size: 16,
            micro_batch: std::env::var("MICRO_BATCH").ok().and_then(|m| m.parse().ok()).unwrap_or(16),
            learning_rate: 3e-4,
            warmup_steps: 500,
            resume: false,
            eval_every: 500,
            eval_batches: 20,
            log_every: 50,
            max_hours: std::env::var("TRAIN_MAX_HOURS").ok().and_then(|h| h.parse().ok()),
            loss_scaling: false,
            config: EngineConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineTrainState {
    pub step: usize,
    pub best_val_loss: Option<f64>,
    pub shard: usize,
    /// Next position in the shuffled window order of `shard` (None: checkpoint of the earlier random sampling).
    #[serde(default)]
    pub window: Option<usize>,
    pub tokens_seen: u64,
    /// AdamW steps taken (bias correction).
    pub adam_step: u64,
    pub loss_scale: f32,
}

impl Default for EngineTrainState {
    fn default() -> Self {
        Self { step: 0, best_val_loss: None, shard: 0, window: Some(0), tokens_seen: 0, adam_step: 0, loss_scale: INITIAL_LOSS_SCALE }
    }
}

pub fn file(prefix: &Path, suffix: &str) -> PathBuf {
    let mut p = prefix.as_os_str().to_owned();
    p.push(suffix);
    p.into()
}

pub fn shard_files(dir: &Path, prefix: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "bin") && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&format!("{prefix}_"))))
        .collect();
    files.sort();
    Ok(files)
}

pub fn read_shard(path: &Path) -> std::io::Result<Vec<u16>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
}

fn write_f32(path: &Path, values: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    values.iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
    std::fs::write(path, bytes)
}

pub fn read_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Saves parameters, configuration and (optionally) the AdamW moments.
pub fn save_checkpoint<B: Backend>(model: &Model<B>, prefix: &Path, adam: Option<(&B::Buf, &B::Buf)>, state: &EngineTrainState) -> std::io::Result<()> {
    if let Some(parent) = prefix.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_f32(&file(prefix, ".params"), &model.backend.download(&model.params))?;
    std::fs::write(file(prefix, ".json"), serde_json::to_string_pretty(&model.config)?)?;
    if let Some((m, v)) = adam {
        let mut moments = model.backend.download(m);
        moments.extend(model.backend.download(v));
        write_f32(&file(prefix, ".adam"), &moments)?;
    }
    std::fs::write(file(prefix, ".state.json"), serde_json::to_string_pretty(state)?)
}

/// Loads the parameters (and, if present, the AdamW moments and progress) of a checkpoint.
pub fn load_checkpoint(prefix: &Path) -> std::io::Result<(EngineConfig, Vec<f32>, Option<Vec<f32>>, EngineTrainState)> {
    let config: EngineConfig = serde_json::from_str(&std::fs::read_to_string(file(prefix, ".json"))?)?;
    let params = read_f32(&file(prefix, ".params"))?;
    let adam = read_f32(&file(prefix, ".adam")).ok().filter(|m| m.len() == 2 * params.len());
    let state = std::fs::read_to_string(file(prefix, ".state.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    Ok((config, params, adam, state))
}

/// Learning rate at `step` of a run: linear warmup, then cosine decay to 10 % of the peak at `total`.
pub fn learning_rate(step: usize, peak: f64, warmup: usize, total: usize) -> f64 {
    if step < warmup {
        return peak * (step + 1) as f64 / warmup as f64;
    }
    let progress = ((step - warmup) as f64 / (total.saturating_sub(warmup)).max(1) as f64).min(1.0);
    peak * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()))
}

/// Shuffled order of the `count` windows of the shard with index `shard` (counted over all passes, so every pass over
/// the data has a different order).
pub fn window_order(shard: usize, count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..count).collect();
    order.shuffle(&mut rand::rngs::StdRng::seed_from_u64(0x5EED_0000 ^ shard as u64));
    order
}

fn windows(tokens: &[u16], starts: &[usize], len: usize) -> Vec<u32> {
    starts.iter().flat_map(|&s| tokens[s..s + len].iter().map(|&t| t as u32)).collect()
}

/// Mean loss over fixed, non-overlapping validation windows.
pub fn validation_loss<B: Backend>(model: &mut Model<B>, val: &[u16], micro_batch: usize, windows_count: usize) -> f64 {
    let len = model.config.seq_len + 1;
    let n = windows_count.min(val.len() / len);
    assert!(n > 0, "validation shard shorter than one window");
    let (mut total, mut count) = (0.0, 0);
    for chunk in (0..n).collect::<Vec<_>>().chunks(micro_batch.max(1)) {
        let starts: Vec<usize> = chunk.iter().map(|i| i * len).collect();
        total += model.forward_backward(&windows(val, &starts, len), false, 1.0) * starts.len() as f64;
        count += starts.len();
    }
    total / count as f64
}

pub fn pretrain<B: Backend>(backend: B, settings: &EngineTrainSettings) -> Result<(), Box<dyn std::error::Error>> {
    let train_files = shard_files(&settings.data_dir, "train")?;
    let val_files = shard_files(&settings.data_dir, "val")?;
    if train_files.is_empty() || val_files.is_empty() {
        return Err(format!("no train_*.bin / val_*.bin shards in {}", settings.data_dir.display()).into());
    }
    let val = read_shard(&val_files[0])?;

    let resume = settings.resume && file(&settings.checkpoint, ".params").exists();
    let (mut model, mut state, adam) = if resume {
        let (config, params, adam, state) = load_checkpoint(&settings.checkpoint)?;
        println!("Continuing {} at step {} (best validation loss {:?}, AdamW moments {})", settings.checkpoint.display(), state.step, state.best_val_loss, if adam.is_some() { "restored" } else { "new" });
        (Model::from_parameters(backend, config, &params), state, adam)
    } else {
        println!("Creating a new model: {:?}", settings.config);
        (Model::new(backend, settings.config.clone(), 42), EngineTrainState::default(), None)
    };
    if !settings.loss_scaling {
        state.loss_scale = 1.0;
    } else if state.loss_scale <= 1.0 {
        state.loss_scale = INITIAL_LOSS_SCALE;
    }
    let total = model.layout.total;
    let (mut m, mut v) = match &adam {
        Some(moments) => (model.backend.upload(&moments[..total]), model.backend.upload(&moments[total..])),
        None => (model.backend.zeros(total), model.backend.zeros(total)),
    };
    let (params_total, layer_params) = model.layout.parameter_counts();
    println!("{:.1} M parameters ({:.1} M in the transformer layers)", params_total as f64 / 1e6, layer_params as f64 / 1e6);

    let len = model.config.seq_len + 1;
    if state.window.is_none() {
        // A checkpoint of the earlier random sampling: its shard has been used, the next one starts.
        state.shard += 1;
        state.window = Some(0);
        println!("Checkpoint without a window position: continuing with shard {}", state.shard);
    }
    let started = Instant::now();
    let first_step = state.step;
    let last_step = first_step + settings.steps;
    let mut shard_tokens = read_shard(&train_files[state.shard % train_files.len()])?;
    let mut order = window_order(state.shard, shard_tokens.len() / len);
    println!("{} train shards, {} validation tokens; steps {}..{}, batch {} x {} tokens, micro-batch {}", train_files.len(), val.len(), first_step, last_step, settings.batch_size, model.config.seq_len, settings.micro_batch);

    let mut history = std::fs::OpenOptions::new().create(true).append(true).open({
        if let Some(parent) = settings.checkpoint.parent() {
            std::fs::create_dir_all(parent)?;
        }
        file(&settings.checkpoint, ".history.csv")
    })?;
    let (mut loss_sum, mut loss_count, mut window_started) = (0.0, 0usize, Instant::now());
    let (mut norm_sum, mut norm_max, mut skipped) = (0.0f64, 0.0f64, 0usize);
    let mut clean_steps = 0usize;
    let micro = settings.micro_batch.clamp(1, settings.batch_size);

    for step in first_step..last_step {
        let mut position = state.window.unwrap_or(0);
        if position + settings.batch_size > order.len() {
            state.shard += 1;
            shard_tokens = read_shard(&train_files[state.shard % train_files.len()])?;
            order = window_order(state.shard, shard_tokens.len() / len);
            position = 0;
        }
        let starts: Vec<usize> = order[position..position + settings.batch_size].iter().map(|w| w * len).collect();
        state.window = Some(position + settings.batch_size);
        let lr = learning_rate(step - first_step, settings.learning_rate, settings.warmup_steps, settings.steps.max(settings.warmup_steps + 1));

        // Gradients of the batch mean loss, summed over micro-batches (each weighted by its share of the batch).
        model.zero_grads();
        let mut loss = 0.0;
        for chunk in starts.chunks(micro) {
            let weight = chunk.len() as f32 / starts.len() as f32;
            loss += model.forward_backward(&windows(&shard_tokens, chunk, len), true, weight * state.loss_scale) * weight as f64;
        }
        let squares = model.backend.sum_squares(&model.grads, total);
        let norm = squares.sqrt() / state.loss_scale as f64;
        state.step = step + 1;
        state.tokens_seen += (settings.batch_size * model.config.seq_len) as u64;
        if !norm.is_finite() || !loss.is_finite() {
            // Overflow in the scaled f16 gradients (or a diverged step): nothing is applied.
            skipped += 1;
            clean_steps = 0;
            if settings.loss_scaling && state.loss_scale > 1.0 {
                state.loss_scale /= 2.0;
            }
            println!("step {}: skipped (loss {loss}, gradient norm {norm}), loss scale now {}", state.step, state.loss_scale);
        } else {
            let clip = if norm > 1.0 { 1.0 / norm } else { 1.0 };
            state.adam_step += 1;
            model.backend.adamw(&mut model.params, &model.grads, &mut m, &mut v, total, lr as f32, 0.9, 0.95, 1e-8, 0.01, (clip / state.loss_scale as f64) as f32, state.adam_step);
            model.refresh_blocks();
            loss_sum += loss;
            loss_count += 1;
            norm_sum += norm;
            norm_max = norm_max.max(norm);
            clean_steps += 1;
            if settings.loss_scaling && clean_steps % LOSS_SCALE_GROWTH_INTERVAL == 0 && state.loss_scale < 65536.0 * 16.0 {
                state.loss_scale *= 2.0;
            }
        }

        if state.step % settings.log_every == 0 && loss_count > 0 {
            let tokens_per_s = (loss_count * settings.batch_size * model.config.seq_len) as f64 / window_started.elapsed().as_secs_f64();
            println!(
                "[{:6.2} h] step {}  loss {:.4}  lr {:.2e}  grad norm {:.2} (max {:.2})  {:.0} tokens/s  {:.1} M tokens seen{}",
                started.elapsed().as_secs_f64() / 3600.0,
                state.step,
                loss_sum / loss_count as f64,
                lr,
                norm_sum / loss_count as f64,
                norm_max,
                tokens_per_s,
                state.tokens_seen as f64 / 1e6,
                if settings.loss_scaling { format!("  loss scale {}  skipped {}", state.loss_scale, skipped) } else { String::new() }
            );
            (loss_sum, loss_count, window_started, norm_sum, norm_max, skipped) = (0.0, 0, Instant::now(), 0.0, 0.0, 0);
        }

        let out_of_time = settings.max_hours.is_some_and(|h| started.elapsed().as_secs_f64() >= h * 3600.0);
        if state.step % settings.eval_every == 0 || state.step == last_step || out_of_time {
            let val_loss = validation_loss(&mut model, &val, micro, settings.eval_batches * settings.batch_size);
            writeln!(history, "{},{:.6},{:.6}", state.step, val_loss, lr)?;
            if state.best_val_loss.is_none_or(|best| val_loss < best) {
                state.best_val_loss = Some(val_loss);
                save_checkpoint(&model, &settings.checkpoint, Some((&m, &v)), &state)?;
                println!("step {}: VALIDATION LOSS {:.4}  (new best, saved {})", state.step, val_loss, settings.checkpoint.display());
            } else {
                println!("step {}: VALIDATION LOSS {:.4}  (best {:.4}, not saved)", state.step, val_loss, state.best_val_loss.unwrap_or(f64::NAN));
            }
        }
        if out_of_time {
            println!("Time limit of {:.1} h reached at step {}.", settings.max_hours.unwrap_or(0.0), state.step);
            break;
        }
    }
    println!("done: best validation loss {:?}", state.best_val_loss);
    Ok(())
}
