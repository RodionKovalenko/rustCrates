//! Live state of the running training (written by `transformer_network::train`, read e.g. by the web UI)
//! and the stop request that ends the training after the current batch.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde::Serialize;

/// Losses of one finished epoch (per target token).
#[derive(Clone, Debug, Serialize)]
pub struct EpochStats {
    pub epoch: usize,
    pub training_loss: f64,
    pub validation_loss: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TrainingProgress {
    pub running: bool,
    /// Model being trained (`ModelKind` name).
    pub model: String,
    pub epoch: usize,
    pub num_epochs: usize,
    /// Batches finished in the current epoch.
    pub batch: usize,
    pub num_batches: usize,
    pub training_records: usize,
    pub validation_records: usize,
    /// Loss per target token of the last batch.
    pub batch_loss: Option<f64>,
    pub elapsed_seconds: f64,
    pub history: Vec<EpochStats>,
    pub best_validation_loss: Option<f64>,
    pub best_epoch: Option<usize>,
    /// Last status message ("finished", "stopped", errors, ...).
    pub message: String,
}

static PROGRESS: Lazy<Mutex<TrainingProgress>> = Lazy::new(|| Mutex::new(TrainingProgress::default()));
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Snapshot of the current training state.
pub fn training_progress() -> TrainingProgress {
    PROGRESS.lock().map(|progress| progress.clone()).unwrap_or_default()
}

pub fn update_training_progress(update: impl FnOnce(&mut TrainingProgress)) {
    if let Ok(mut progress) = PROGRESS.lock() {
        update(&mut progress);
    }
}

/// Asks the running training to save the model and stop after the current batch.
pub fn request_training_stop() {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn training_stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

pub fn reset_training_stop() {
    STOP_REQUESTED.store(false, Ordering::SeqCst);
}
