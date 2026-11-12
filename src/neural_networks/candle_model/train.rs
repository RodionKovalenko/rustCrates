//! Pretraining of the candle language model on token shards written by `kaggle/prepare_pretrain.py`
//! (`train_NNNNN.bin` / `val_NNNNN.bin`, little-endian u16 token ids).
//!
//! Random windows of `seq_len + 1` tokens are drawn from one shard after the other; AdamW (beta2 0.95) with linear warmup and a
//! cosine decay to 10 %, gradient clipping at a global norm of 1. Every `eval_every` steps the mean loss on fixed
//! validation windows decides whether the checkpoint is replaced (only a better model is kept). Training continues
//! from the checkpoint with `resume` and stops cleanly after `TRAIN_MAX_HOURS`.

use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use super::model::{Model, ModelConfig};

/// Token positions per chunk of the output head (its logits are chunk x vocab).
pub const HEAD_CHUNK: usize = 512;

/// Default checkpoint (the configuration is stored next to it as `<file>.json`).
pub const CHECKPOINT: &str = "STORAGE/candle_lm.safetensors";

#[derive(Clone, Debug)]
pub struct PretrainSettings {
    pub data_dir: PathBuf,
    pub checkpoint: PathBuf,
    pub steps: usize,
    pub batch_size: usize,
    /// Sequences per forward/backward pass; the gradients of the micro-batches are summed (memory vs. speed).
    pub micro_batch: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub resume: bool,
    pub eval_every: usize,
    pub eval_batches: usize,
    pub log_every: usize,
    pub max_hours: Option<f64>,
    pub config: ModelConfig,
}

impl PretrainSettings {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            checkpoint: PathBuf::from(CHECKPOINT),
            steps: 20_000,
            batch_size: 16,
            micro_batch: std::env::var("MICRO_BATCH").ok().and_then(|m| m.parse().ok()).unwrap_or(8),
            learning_rate: 6e-4,
            warmup_steps: 500,
            resume: false,
            eval_every: 500,
            eval_batches: 20,
            log_every: 50,
            max_hours: std::env::var("TRAIN_MAX_HOURS").ok().and_then(|h| h.parse().ok()),
            config: ModelConfig::default(),
        }
    }
}

/// Progress stored next to the checkpoint (`<file>.state.json`), so a continued run keeps its schedule and baseline.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainState {
    pub step: usize,
    pub best_val_loss: Option<f64>,
    pub shard: usize,
    pub tokens_seen: u64,
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(suffix);
    p.into()
}

/// Token shard files with the given prefix ("train" / "val"), sorted.
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

/// `batch` windows of `len` tokens at the given start offsets, as a (batch, len) u32 tensor.
fn windows_tensor(tokens: &[u16], starts: &[usize], len: usize, device: &Device) -> Result<Tensor> {
    let ids: Vec<u32> = starts.iter().flat_map(|&s| tokens[s..s + len].iter().map(|&t| t as u32)).collect();
    Tensor::from_vec(ids, (starts.len(), len), device)
}

/// Learning rate at `step`: linear warmup, then cosine decay to 10 % of the peak at `total` steps.
pub fn learning_rate(step: usize, peak: f64, warmup: usize, total: usize) -> f64 {
    if step < warmup {
        return peak * (step + 1) as f64 / warmup as f64;
    }
    let progress = ((step - warmup) as f64 / (total.saturating_sub(warmup)).max(1) as f64).min(1.0);
    peak * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()))
}

/// Mean loss over `batches` fixed, non-overlapping validation windows.
pub fn validation_loss(model: &Model, val: &[u16], batch_size: usize, batches: usize, device: &Device) -> Result<f64> {
    let len = model.config.seq_len + 1;
    let available = val.len() / len;
    let n = (batches * batch_size).min(available);
    if n == 0 {
        return Err(candle_core::Error::Msg("validation shard shorter than one window".into()));
    }
    let mut total = 0.0;
    let mut count = 0;
    for chunk in (0..n).collect::<Vec<_>>().chunks(batch_size) {
        let starts: Vec<usize> = chunk.iter().map(|i| i * len).collect();
        let loss = model.loss_no_grad(&windows_tensor(val, &starts, len, device)?, HEAD_CHUNK)?;
        total += loss * starts.len() as f64;
        count += starts.len();
    }
    Ok(total / count as f64)
}

pub fn device() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

pub fn pretrain(settings: &PretrainSettings) -> Result<()> {
    let device = device();
    println!("Device: {}", if device.is_cuda() { "CUDA GPU" } else { "CPU" });

    let train_files = shard_files(&settings.data_dir, "train")?;
    let val_files = shard_files(&settings.data_dir, "val")?;
    if train_files.is_empty() || val_files.is_empty() {
        return Err(candle_core::Error::Msg(format!("no train_*.bin / val_*.bin shards in {} (run kaggle/prepare_pretrain.py)", settings.data_dir.display())));
    }
    let val = read_shard(&val_files[0])?;

    let resume = settings.resume && settings.checkpoint.exists();
    let (model, mut state) = if resume {
        let model = Model::load(&settings.checkpoint, DType::F32, &device)?;
        let state: TrainState = std::fs::read_to_string(sidecar(&settings.checkpoint, ".state.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        println!("Continuing {} at step {} (best validation loss {:?})", settings.checkpoint.display(), state.step, state.best_val_loss);
        (model, state)
    } else {
        println!("Creating a new model: {:?}", settings.config);
        (Model::new(settings.config.clone(), DType::F32, &device, 42)?, TrainState::default())
    };
    let (total_params, layer_params) = model.parameter_counts();
    println!("{:.1} M parameters ({:.1} M in the transformer layers)", total_params as f64 / 1e6, layer_params as f64 / 1e6);

    let len = model.config.seq_len + 1;
    let mut optimizer = AdamW::new(model.vars(), ParamsAdamW { lr: settings.learning_rate, beta2: 0.95, weight_decay: 0.01, ..Default::default() })?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234 + state.step as u64);

    let started = Instant::now();
    let first_step = state.step;
    let last_step = first_step + settings.steps;
    let mut shard_tokens = read_shard(&train_files[state.shard % train_files.len()])?;
    let mut batches_left_in_shard = shard_tokens.len() / (settings.batch_size * len);
    println!(
        "{} train shards, {} validation tokens; steps {}..{}, batch {} x {} tokens",
        train_files.len(),
        val.len(),
        first_step,
        last_step,
        settings.batch_size,
        model.config.seq_len
    );

    let history_path = sidecar(&settings.checkpoint, ".history.csv");
    let mut history = std::fs::OpenOptions::new().create(true).append(true).open(&history_path)?;
    let (mut loss_sum, mut loss_count, mut window_started) = (0.0, 0usize, Instant::now());

    let mut micro_batch = settings.micro_batch.max(1);
    for step in first_step..last_step {
        if batches_left_in_shard == 0 {
            state.shard += 1;
            shard_tokens = read_shard(&train_files[state.shard % train_files.len()])?;
            batches_left_in_shard = shard_tokens.len() / (settings.batch_size * len);
        }
        batches_left_in_shard = batches_left_in_shard.saturating_sub(1);

        let starts: Vec<usize> = (0..settings.batch_size).map(|_| rng.random_range(0..shard_tokens.len() - len)).collect();
        let batch = windows_tensor(&shard_tokens, &starts, len, &device)?;
        let lr = learning_rate(step, settings.learning_rate, settings.warmup_steps, last_step.max(settings.warmup_steps + 1));
        optimizer.set_learning_rate(lr);

        // A micro-batch that does not fit into the GPU memory is halved and the step is repeated.
        let (loss, mut grads) = loop {
            match micro_batched_loss_and_grads(&model, &batch, micro_batch) {
                Ok(result) => break result,
                Err(e) if micro_batch > 1 && is_out_of_memory(&e) => {
                    micro_batch /= 2;
                    println!("Out of GPU memory: continuing with micro-batches of {micro_batch} sequences (MICRO_BATCH={micro_batch})");
                }
                Err(e) => return Err(e),
            }
        };
        clip_gradients(&model, &mut grads, 1.0)?;
        optimizer.step(&grads)?;

        loss_sum += loss;
        loss_count += 1;
        state.step = step + 1;
        state.tokens_seen += (settings.batch_size * model.config.seq_len) as u64;

        if state.step % settings.log_every == 0 {
            let tokens_per_s = (loss_count * settings.batch_size * model.config.seq_len) as f64 / window_started.elapsed().as_secs_f64();
            println!(
                "[{:6.2} h] step {}  loss {:.4}  lr {:.2e}  {:.0} tokens/s  {:.1} M tokens seen",
                started.elapsed().as_secs_f64() / 3600.0,
                state.step,
                loss_sum / loss_count as f64,
                lr,
                tokens_per_s,
                state.tokens_seen as f64 / 1e6
            );
            (loss_sum, loss_count, window_started) = (0.0, 0, Instant::now());
        }

        let out_of_time = settings.max_hours.is_some_and(|h| started.elapsed().as_secs_f64() >= h * 3600.0);
        if state.step % settings.eval_every == 0 || state.step == last_step || out_of_time {
            let val_loss = validation_loss(&model, &val, settings.batch_size, settings.eval_batches, &device)?;
            let improved = state.best_val_loss.is_none_or(|best| val_loss < best);
            writeln!(history, "{},{:.6},{:.6}", state.step, val_loss, lr)?;
            if improved {
                state.best_val_loss = Some(val_loss);
                if let Some(parent) = settings.checkpoint.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                model.save(&settings.checkpoint)?;
                std::fs::write(sidecar(&settings.checkpoint, ".state.json"), serde_json::to_string_pretty(&state).map_err(candle_core::Error::wrap)?)?;
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

fn is_out_of_memory(e: &candle_core::Error) -> bool {
    let text = format!("{e:?}");
    text.contains("OUT_OF_MEMORY") || text.contains("out of memory")
}

/// Loss and gradients of a batch computed in micro-batches of `micro` sequences: each micro-batch contributes with
/// weight (its size / batch size), so the result equals the full-batch mean loss and gradient.
fn micro_batched_loss_and_grads(model: &Model, batch: &Tensor, micro: usize) -> Result<(f64, candle_core::backprop::GradStore)> {
    use super::model::{add_grads, scale_grads};
    let n = batch.dim(0)?;
    let vars = model.vars();
    let mut total: Option<candle_core::backprop::GradStore> = None;
    let mut loss = 0.0;
    for start in (0..n).step_by(micro.max(1)) {
        let len = micro.max(1).min(n - start);
        let (l, mut g) = model.loss_and_grads(&batch.narrow(0, start, len)?, HEAD_CHUNK)?;
        let weight = len as f64 / n as f64;
        loss += l * weight;
        scale_grads(&mut g, &vars, weight)?;
        match total.as_mut() {
            None => total = Some(g),
            Some(t) => add_grads(t, &g, &vars)?,
        }
    }
    Ok((loss, total.expect("non-empty batch")))
}

/// Scales all gradients so that their global L2 norm is at most `max_norm`.
fn clip_gradients(model: &Model, grads: &mut candle_core::backprop::GradStore, max_norm: f64) -> Result<()> {
    let vars = model.vars();
    let mut sum_sq = 0.0f64;
    for var in &vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            sum_sq += g.sqr()?.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        }
    }
    let norm = sum_sq.sqrt();
    if norm > max_norm && norm.is_finite() {
        let scale = max_norm / norm;
        for var in &vars {
            if let Some(g) = grads.remove(var.as_tensor()) {
                grads.insert(var.as_tensor(), (g * scale)?);
            }
        }
    }
    Ok(())
}

/// Greedy continuation of `prompt` (token ids) by `max_new_tokens` tokens; stops at `eos`.
pub fn generate(model: &Model, prompt: &[u32], max_new_tokens: usize, eos: Option<u32>, device: &Device) -> Result<Vec<u32>> {
    let mut ids = prompt.to_vec();
    for _ in 0..max_new_tokens {
        let start = ids.len().saturating_sub(model.config.seq_len);
        let input = Tensor::from_vec(ids[start..].to_vec(), (1, ids.len() - start), device)?;
        let logits = model.forward(&input)?;
        let last = logits.squeeze(0)?.get(ids.len() - start - 1)?;
        let next = last.argmax(0)?.to_scalar::<u32>()?;
        if Some(next) == eos {
            break;
        }
        ids.push(next);
    }
    Ok(ids[prompt.len()..].to_vec())
}
