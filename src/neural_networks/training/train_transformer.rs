use std::time::{Instant, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{
    database::sled_db::{get_storage_path_transformer_db, SLED_DB_TRANSFORMER_RM_V1, SLED_DB_TRANSFORMER_V1},
    neural_networks::{
        network_components::input::{load_data_xquad_de_as_dataset, DataTrait, Dataset},
        network_types::{
            neural_network_generic::{get_from_db, print_networt_structure, update_learning_rate, NeuralNetwork, OperationMode},
            transformer::{
                transformer_builder::create_transformer,
                transformer_builder_rm::create_transformer_rm,
                transformer_network::{generate_text, train},
            },
        },
        training::training_progress::{reset_training_stop, update_training_progress, TrainingProgress},
    },
};

/// `num_records` value that selects every batch of the dataset.
pub const ALL_RECORDS: usize = usize::MAX;

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
}

pub fn train_transformer_from_dataset(num_epochs: usize, num_records: usize, batch_size: usize) -> bool {
    train_model(ModelKind::Vec, TrainSettings { epochs: num_epochs, num_batches: num_records, batch_size, resume: false })
}

/// Same training with the row-major transformer (`create_transformer_rm`): with the `cuda` feature its
/// matrix multiplications run on the GPU. Saved to its own model file (`SLED_DB_TRANSFORMER_RM_V1`).
pub fn train_transformer_rm_from_dataset(num_epochs: usize, num_records: usize, batch_size: usize) -> bool {
    train_model(ModelKind::Rm, TrainSettings { epochs: num_epochs, num_batches: num_records, batch_size, resume: false })
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

    let learning_rate = 0.001;
    transformer.learning_rate = learning_rate;
    update_learning_rate(&mut transformer, learning_rate);
    print_networt_structure(&mut transformer);

    let seconds_elapsed = now.elapsed();
    println!("time elapsed in seconds: {:?}", &seconds_elapsed);

    let dataset: Dataset<String, String> = load_data_xquad_de_as_dataset().unwrap();
    let dataset_small = select_records(&dataset, settings.num_batches, settings.batch_size);
    println!("selected records: {} (train() keeps 10% of them for validation)", dataset_small.input.len());

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

/// Generated answer and how long the generation took.
#[derive(Debug, Clone, Serialize)]
pub struct Generation {
    pub answer: String,
    pub seconds: f64,
}

/// Generates the answer to `prompt` with an already loaded model.
pub fn generate_with(network: &mut NeuralNetwork, prompt: &str, max_new_tokens: usize) -> Generation {
    let now = Instant::now();
    let answer = generate_text(network, prompt, max_new_tokens);
    Generation { answer, seconds: now.elapsed().as_secs_f64() }
}

/// Loads the saved model of `kind` and generates the answer to `prompt`.
pub fn generate(kind: ModelKind, prompt: &str, max_new_tokens: usize) -> Result<Generation, String> {
    let mut network = load_model(kind)?;
    Ok(generate_with(&mut network, prompt, max_new_tokens))
}

/// Shuffles all records of the dataset and keeps the first `num_batches` batches of `batch_size` records
/// (`ALL_RECORDS` keeps every record). `train()` then splits the selection 90/10 into training and validation.
pub fn select_records(dataset: &Dataset<String, String>, num_batches: usize, batch_size: usize) -> Dataset<String, String> {
    // `Dataset::new` already split the loaded dataset 80/10/10 and `split_into_batches` would only pick from its
    // training part; select from all records instead.
    let mut all_records = dataset.clone();
    all_records.total_training_records_size = 0;
    all_records.total_validation_records_size = 0;

    let mut input_batch: Vec<String> = Vec::new();
    let mut target_batch: Vec<String> = Vec::new();

    for data_batch in all_records.split_into_batches(batch_size).iter().take(num_batches) {
        let (inputs, targets) = data_batch.get_batch(0, batch_size).unwrap();
        input_batch.extend(inputs);
        target_batch.extend(targets);
    }

    Dataset::new(input_batch, target_batch)
}
