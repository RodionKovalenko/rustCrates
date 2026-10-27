//! The spectral flow image model for the CLI and the web UI: model info, training with live progress and
//! sampling to PNG. The model itself lives in `spectral_model`.

use std::time::{Instant, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    spectral_model::{
        data::DataSource,
        sample::{sample_grid_to_png_with, DEFAULT_SAMPLE_STEPS},
        train::{default_batch_size, read_checkpoint_summary, train_spectral_flow, Checkpoint, TrainConfig, SPECTRAL_CHECKPOINT},
    },
    training::{
        train_transformer::ModelInfo,
        training_progress::{reset_training_stop, update_training_progress, TrainingProgress},
    },
};

/// Model id of the spectral flow model in the web UI and the training progress.
pub const SPECTRAL_ID: &str = "spectral";

/// Most images one generation request may draw.
pub const MAX_GENERATED_IMAGES: usize = 64;

/// One spectral training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralTrainSettings {
    pub steps: usize,
    /// Image side length (power of two, 32..=512). Ignored when a saved model is continued.
    pub size: usize,
    /// 1 = grayscale, 3 = RGB. Ignored when a saved model is continued.
    pub channels: usize,
    pub data: DataSource,
    /// Images per optimiser step; `None` = what fits in memory for the size (`default_batch_size`).
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Load at most this many images.
    #[serde(default)]
    pub max_images: Option<usize>,
    /// Continue the saved model instead of starting a new one.
    #[serde(default)]
    pub resume: bool,
}

/// Geometry and training state of the saved spectral model.
#[derive(Debug, Clone, Serialize)]
pub struct SpectralSummary {
    pub size: usize,
    pub channels: usize,
    pub step: usize,
    pub data: String,
}

pub fn spectral_model_info() -> ModelInfo {
    let metadata = std::fs::metadata(SPECTRAL_CHECKPOINT).ok();
    let summary = metadata.as_ref().and_then(|_| read_checkpoint_summary(SPECTRAL_CHECKPOINT).ok()).map(|s| SpectralSummary {
        size: s.layout.size,
        channels: s.layout.channels,
        step: s.step,
        data: s.data,
    });
    ModelInfo {
        id: SPECTRAL_ID,
        label: "Spectral flow",
        description: "Image generator in the Fourier domain: MNIST or your own images, grayscale or RGB, 32 to 512 pixels.",
        kind: "image",
        file: SPECTRAL_CHECKPOINT.to_string(),
        trained: summary.is_some(),
        size_bytes: metadata.as_ref().map(|m| m.len()).unwrap_or(0),
        modified: metadata.and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()),
        spectral: summary,
    }
}

/// Trains the spectral model (new or continued); progress is published in `training_progress`.
pub fn train_spectral(settings: SpectralTrainSettings) -> Result<(), String> {
    reset_training_stop();
    update_training_progress(|progress| {
        *progress = TrainingProgress { running: true, model: SPECTRAL_ID.to_string(), message: "Loading model and data ...".to_string(), ..Default::default() };
    });

    let result = run_training(&settings);
    update_training_progress(|progress| {
        progress.running = false;
        if let Err(e) = &result {
            progress.message = format!("Training failed: {e}");
        }
    });
    result
}

fn run_training(settings: &SpectralTrainSettings) -> Result<(), String> {
    let resume = settings.resume && std::path::Path::new(SPECTRAL_CHECKPOINT).exists();
    // A continued model keeps its geometry; the batch size default follows it.
    let size = if resume { read_checkpoint_summary(SPECTRAL_CHECKPOINT).map_err(|e| e.to_string())?.layout.size } else { settings.size };

    let mut cfg = TrainConfig {
        steps: settings.steps,
        size: settings.size,
        channels: settings.channels,
        data: settings.data.clone(),
        max_images: settings.max_images,
        resume,
        log_every: 50.min(settings.steps.max(1)),
        ckpt_every: 1_000,
        // A preview about every 5 % of the run (at least every 2000 steps).
        sample_every: (settings.steps / 20).clamp(100, 2_000),
        ..TrainConfig::default()
    };
    cfg.set_batch_size(settings.batch_size.unwrap_or_else(|| default_batch_size(size)));
    train_spectral_flow(cfg).map_err(|e| e.to_string())
}

/// Generated images as one PNG and how long the sampling took.
pub struct SpectralGeneration {
    pub png: Vec<u8>,
    pub seconds: f64,
    pub seed: u64,
}

/// Samples `count` images (one tiled PNG) from the saved model with `steps` integration steps.
pub fn generate_spectral_png(count: usize, steps: Option<usize>, seed: Option<u64>) -> Result<SpectralGeneration, String> {
    if !std::path::Path::new(SPECTRAL_CHECKPOINT).exists() {
        return Err("The spectral flow model has not been trained yet.".to_string());
    }
    let now = Instant::now();
    let count = count.clamp(1, MAX_GENERATED_IMAGES);
    let steps = steps.unwrap_or(DEFAULT_SAMPLE_STEPS).clamp(4, 512);
    let seed = seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0));

    let ckpt = Checkpoint::load(SPECTRAL_CHECKPOINT).map_err(|e| e.to_string())?;
    let (net, prep) = ckpt.net_and_prep();
    let cols = (count as f64).sqrt().ceil() as usize;
    let rows = count.div_ceil(cols);
    let path = std::env::temp_dir().join(format!("spectral_generated_{}_{seed}.png", std::process::id()));
    let path_str = path.to_string_lossy().to_string();
    sample_grid_to_png_with(&net, &prep, &ckpt.ema, &ckpt.schedule, ckpt.path, rows, cols, steps, seed, &path_str, 0, ckpt.layout).map_err(|e| e.to_string())?;
    let png = std::fs::read(&path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    Ok(SpectralGeneration { png, seconds: now.elapsed().as_secs_f64(), seed })
}
