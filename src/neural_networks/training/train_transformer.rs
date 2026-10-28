use std::path::PathBuf;
use std::time::{Instant, UNIX_EPOCH};

use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    database::sled_db::{get_storage_path_transformer_db, SLED_DB_TRANSFORMER_RM_V1, SLED_DB_TRANSFORMER_V1},
    neural_networks::{
        network_components::input::{load_data_xquad_de_as_dataset, qa_input, xquad_de_records, DataTrait, Dataset},
        network_types::{
            neural_network_generic::{get_from_db, print_networt_structure, update_learning_rate, NeuralNetwork, OperationMode},
            transformer::{
                transformer_builder::create_transformer,
                transformer_builder_rm::create_transformer_rm,
                transformer_network::{generate_answer, train, TokenCandidate},
            },
        },
        training::training_progress::{reset_training_stop, update_training_progress, TrainingProgress},
    },
    utils::string::prepare_prompt,
};

/// `num_records` value that selects every batch of the dataset.
pub const ALL_RECORDS: usize = usize::MAX;

/// Peak learning rate of a training run (warmup, then cosine decay). 1e-3 made the loss rise during the warmup.
pub const DEFAULT_LEARNING_RATE: f64 = 3e-4;

/// Seed of the record order: every training run selects the same records and holds out the same 10 % for
/// validation, so the validation examples can be listed and tried out (`qa_examples`).
const RECORD_ORDER_SEED: u64 = 42;

/// The two transformers: same architecture, saved to different model files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Row-major layers (`create_transformer_rm`): batched matmuls with the weights kept on the GPU (`cuda` feature).
    Rm,
    /// Vec layers (`create_transformer`): the reference implementation.
    Vec,
}

impl ModelKind {
    pub const ALL: [ModelKind; 2] = [ModelKind::Rm, ModelKind::Vec];

    pub fn id(self) -> &'static str {
        match self {
            ModelKind::Rm => "rm",
            ModelKind::Vec => "vec",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ModelKind::Rm => "RM transformer",
            ModelKind::Vec => "Vec transformer",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ModelKind::Rm => "Row-major layers: batched matrix multiplications, weights cached on the GPU.",
            ModelKind::Vec => "Reference implementation with Vec<Vec> layers, same architecture.",
        }
    }

    pub fn model_file(self) -> &'static str {
        match self {
            ModelKind::Rm => SLED_DB_TRANSFORMER_RM_V1,
            ModelKind::Vec => SLED_DB_TRANSFORMER_V1,
        }
    }

    pub fn create(self) -> NeuralNetwork {
        match self {
            ModelKind::Rm => create_transformer_rm(OperationMode::TRAINING),
            ModelKind::Vec => create_transformer(OperationMode::TRAINING),
        }
    }
}

/// Saved model (for listing; the model itself is not loaded).
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    /// "text" (transformers) or "image" (spectral flow).
    pub kind: &'static str,
    pub file: String,
    pub trained: bool,
    pub size_bytes: u64,
    /// Last modification as unix seconds.
    pub modified: Option<u64>,
    /// Geometry and steps of the spectral model.
    pub spectral: Option<crate::neural_networks::training::train_spectral::SpectralSummary>,
}

pub fn model_info(kind: ModelKind) -> ModelInfo {
    let path = get_storage_path_transformer_db(kind.model_file());
    let metadata = std::fs::metadata(&path).ok();
    ModelInfo {
        id: kind.id(),
        label: kind.label(),
        description: kind.description(),
        kind: "text",
        file: path.to_string_lossy().to_string(),
        trained: metadata.is_some(),
        size_bytes: metadata.as_ref().map(|m| m.len()).unwrap_or(0),
        modified: metadata.and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()),
        spectral: None,
    }
}

/// Where the complex matrix multiplications run.
pub fn compute_device() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        if crate::neural_networks::utils::gpu_matmul::GPU_MATMUL.lock().map(|gpu| gpu.is_some()).unwrap_or(false) {
            return "GPU (CUDA)";
        }
    }
    "CPU"
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrainSettings {
    pub epochs: usize,
    /// Number of batches taken from the dataset (`ALL_RECORDS` = every record).
    pub num_batches: usize,
    pub batch_size: usize,
    /// Continue training the saved model instead of starting with a new one.
    pub resume: bool,
    /// Peak learning rate.
    pub learning_rate: f64,
}

pub fn train_transformer_from_dataset(num_epochs: usize, num_records: usize, batch_size: usize) -> bool {
    train_model(ModelKind::Vec, TrainSettings { epochs: num_epochs, num_batches: num_records, batch_size, resume: false, learning_rate: DEFAULT_LEARNING_RATE })
}

/// Same training with the row-major transformer (`create_transformer_rm`): with the `cuda` feature its
/// matrix multiplications run on the GPU. Saved to its own model file (`SLED_DB_TRANSFORMER_RM_V1`).
pub fn train_transformer_rm_from_dataset(num_epochs: usize, num_records: usize, batch_size: usize) -> bool {
    train_model(ModelKind::Rm, TrainSettings { epochs: num_epochs, num_batches: num_records, batch_size, resume: false, learning_rate: DEFAULT_LEARNING_RATE })
}

/// Trains `kind` on the XQuAD (de) records selected by `settings`; progress is published in `training_progress`.
pub fn train_model(kind: ModelKind, settings: TrainSettings) -> bool {
    let now = Instant::now();
    reset_training_stop();
    update_training_progress(|progress| {
        *progress = TrainingProgress { running: true, model: kind.id().to_string(), message: "Loading model and data ...".to_string(), ..Default::default() };
    });

    let mut transformer = if settings.resume && model_info(kind).trained {
        match get_from_db(kind.model_file()) {
            Ok(transformer) => {
                println!("Loaded transformer from the database!");
                transformer
            }
            Err(e) => {
                println!("error: {:?}", e);
                let transformer: NeuralNetwork = kind.create();
                println!("Created a new transformer for training.");
                transformer
            }
        }
    } else {
        let transformer: NeuralNetwork = kind.create();
        println!("Created a new transformer for training.");
        transformer
    };

    let learning_rate = settings.learning_rate;
    transformer.learning_rate = learning_rate;
    update_learning_rate(&mut transformer, learning_rate);
    print_networt_structure(&mut transformer);

    let seconds_elapsed = now.elapsed();
    println!("time elapsed in seconds: {:?}", &seconds_elapsed);

    let dataset: Dataset<String, String> = load_data_xquad_de_as_dataset().unwrap();
    let selection = select_record_indices(dataset.input.len(), settings.num_batches, settings.batch_size);
    let dataset_small = records_at(&dataset, &selection);
    println!("selected records: {} (train() keeps 10% of them for validation)", dataset_small.input.len());
    if let Err(e) = save_split(kind, &selection) {
        println!("could not save the training / validation split: {e}");
    }

    update_training_progress(|progress| progress.message = "Training ...".to_string());
    train(&mut transformer, dataset_small, settings.epochs, settings.batch_size);
    let seconds_elapsed_end = now.elapsed();

    println!("time elapsed in seconds: {:?}", seconds_elapsed_end - seconds_elapsed);
    update_training_progress(|progress| {
        progress.running = false;
        progress.elapsed_seconds = seconds_elapsed_end.as_secs_f64();
        if progress.message == "Training ..." {
            progress.message = "Training finished.".to_string();
        }
    });

    println!("done");
    true
}

/// Loads the saved model of `kind`; `Err` if it has not been trained yet.
pub fn load_model(kind: ModelKind) -> Result<NeuralNetwork, String> {
    if !model_info(kind).trained {
        return Err(format!("The {} has not been trained yet.", kind.label()));
    }
    get_from_db(kind.model_file())
}

/// Generated answer, the most likely first tokens and how long the generation took.
#[derive(Debug, Clone, Serialize)]
pub struct Generation {
    pub answer: String,
    pub first_tokens: Vec<TokenCandidate>,
    /// The input as the model read it (escapes decoded, question moved behind the text).
    pub prompt: String,
    pub seconds: f64,
}

/// Generates the answer to free-form input (text and question in any order) with an already loaded model.
pub fn generate_with(network: &mut NeuralNetwork, input: &str, max_new_tokens: usize) -> Generation {
    let now = Instant::now();
    let prompt = prepare_prompt(input);
    let generated = generate_answer(network, &prompt, max_new_tokens);
    Generation { answer: generated.answer, first_tokens: generated.first_tokens, prompt, seconds: now.elapsed().as_secs_f64() }
}

/// Loads the saved model of `kind` and generates the answer to `prompt`.
pub fn generate(kind: ModelKind, prompt: &str, max_new_tokens: usize) -> Result<Generation, String> {
    let mut network = load_model(kind)?;
    Ok(generate_with(&mut network, prompt, max_new_tokens))
}

/// Mixes all records in a fixed order (`RECORD_ORDER_SEED`) and keeps the first `num_batches` batches of
/// `batch_size` records (`ALL_RECORDS` keeps every record). `train()` then splits the selection 90/10 into
/// training and validation (`validation_part`).
pub fn select_records(dataset: &Dataset<String, String>, num_batches: usize, batch_size: usize) -> Dataset<String, String> {
    records_at(dataset, &select_record_indices(dataset.input.len(), num_batches, batch_size))
}

/// Indices of the records `select_records` keeps, in their order.
pub fn select_record_indices(total: usize, num_batches: usize, batch_size: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..total).collect();
    order.shuffle(&mut SmallRng::seed_from_u64(RECORD_ORDER_SEED));
    order.truncate(num_batches.saturating_mul(batch_size).min(total));
    order
}

/// The part of a selection `train()` holds out for validation: the last 10 % (`Dataset::setup_splits`).
pub fn validation_part(selection: &[usize]) -> &[usize] {
    let validation = (selection.len() as f64 * 0.1).round() as usize;
    &selection[selection.len() - validation..]
}

fn records_at(dataset: &Dataset<String, String>, indices: &[usize]) -> Dataset<String, String> {
    let inputs = indices.iter().map(|&i| dataset.input[i].clone()).collect();
    let targets = indices.iter().map(|&i| dataset.target[i].clone()).collect();
    Dataset::new(inputs, targets)
}

// ── Training / validation examples ──────────────────────────────────────────

/// Which records a model was trained and validated on (indices into the XQuAD (de) records).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SplitFile {
    training: Vec<usize>,
    validation: Vec<usize>,
}

fn split_file(kind: ModelKind) -> PathBuf {
    let mut path = get_storage_path_transformer_db(kind.model_file()).into_os_string();
    path.push(".split.json");
    PathBuf::from(path)
}

fn save_split(kind: ModelKind, selection: &[usize]) -> std::io::Result<()> {
    let validation = validation_part(selection);
    let training = &selection[..selection.len() - validation.len()];
    let file = SplitFile { training: training.to_vec(), validation: validation.to_vec() };
    let path = split_file(kind);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_vec(&file)?)
}

/// The split of the last training of `kind`; for a model trained before splits were saved, the split of a
/// training on all records.
fn load_split(kind: ModelKind, total: usize) -> SplitFile {
    if let Some(file) = std::fs::read(split_file(kind)).ok().and_then(|bytes| serde_json::from_slice::<SplitFile>(&bytes).ok()) {
        if file.training.iter().chain(file.validation.iter()).all(|&i| i < total) {
            return file;
        }
    }
    let selection = select_record_indices(total, ALL_RECORDS, 1);
    let validation = validation_part(&selection).to_vec();
    SplitFile { training: selection[..selection.len() - validation.len()].to_vec(), validation }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExampleSplit {
    Training,
    Validation,
}

/// One XQuAD question to try out: `input` is exactly what the model was trained with (text, then question).
#[derive(Debug, Clone, Serialize)]
pub struct QaExample {
    pub index: usize,
    pub question: String,
    pub answer: String,
    pub input: String,
}

/// The validation (never trained on) or training questions of `kind`'s last training.
pub fn qa_examples(kind: ModelKind, split: ExampleSplit) -> Result<Vec<QaExample>, String> {
    let records = xquad_de_records().map_err(|e| e.to_string())?;
    let file = load_split(kind, records.len());
    let indices = match split {
        ExampleSplit::Training => file.training,
        ExampleSplit::Validation => file.validation,
    };
    Ok(indices
        .into_iter()
        .map(|i| {
            let record = &records[i];
            QaExample { index: i, question: record.question.clone(), answer: record.answer.clone(), input: qa_input(&record.context, &record.question) }
        })
        .collect())
}
