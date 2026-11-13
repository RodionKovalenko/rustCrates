use std::env;
use std::io::{self, Write};
use std::sync::Mutex;

use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use neural_networks::neural_networks::network_types::neural_network_generic::NeuralNetwork;
use neural_networks::neural_networks::network_types::transformer::transformer_network::MAX_GENERATED_TOKENS;
use neural_networks::neural_networks::spectral_model::data::DataSource;
use neural_networks::neural_networks::spectral_model::sample::{sample_from_checkpoint, DEFAULT_SAMPLE_STEPS};
use neural_networks::neural_networks::spectral_model::train::SPECTRAL_CHECKPOINT;
use neural_networks::neural_networks::training::qa_datasets::{qa_records_from, split_squad_file, DEFAULT_PART_MB, QA_DATASET_DIR};
use neural_networks::neural_networks::training::train_spectral::{generate_spectral_png, spectral_model_info, train_spectral, SpectralTrainSettings};
use neural_networks::neural_networks::training::train_transformer::{
    compute_device, generate, generate_with, load_model, model_info, qa_examples, train_model, ExampleSplit, ModelKind, TrainSettings, ALL_RECORDS,
    DEFAULT_LEARNING_RATE,
};
use neural_networks::neural_networks::training::training_progress::{request_training_stop, training_progress};
use neural_networks::utils::string::maybe_fix_encoding;
use serde::Deserialize;
use tera::{Context, Tera};

const USAGE: &str = "\
Usage: wav-transformer <command> [arguments]

Language model (RM transformer; Vec transformer with the -vec commands):
  train [epochs] [batches|all] [batch_size] [new|resume] [learning_rate]
                                                       train (default: 100 all 16 new 0.0003; 'resume' continues
                                                       the saved model)
  train-vec [epochs] [batches|all] [batch_size] [new|resume] [learning_rate]
  predict [text and question]                          generate an answer (asks for the input if none is given)
  predict-vec [text]
  split-qa <squad.json> <name> [part_mb]               split a SQuAD-format dataset (e.g. GermanQuAD) into parts of
                                                       part_mb MB (default 10) in datasets/qa/; training uses all files there

GPU language model (build with --features candle, on a GPU --features candle-cuda):
  pretrain <shard folder> [steps] [batch_size] [learning_rate] [new|resume] [window]
                                                       pretrain on token shards of kaggle/prepare_pretrain.py
                                                       (default: 20000 16 0.0006 new 256); checkpoint
                                                       STORAGE/candle_lm.safetensors, only a better model is saved
  lm-generate [max_tokens] <text>                      greedy continuation of a text with the pretrained model

Own GPU engine (hand-written backward pass; on a GPU build with --features gpu-engine, else it runs on the CPU):
  engine-pretrain <shard folder> [steps] [batch_size] [learning_rate] [new|resume] [window] [f16|f32]
                                                       pretrain on token shards of kaggle/prepare_pretrain.py
                                                       (default: 20000 16 0.0003 new 256 f16); checkpoint
                                                       STORAGE/engine_lm.*, only a better model is saved
  engine-bench [micro_batch] [f16|f32] [steps]       training speed of the default model on random tokens
                                                       (no data needed; default: 4 f16 5)

Spectral flow (image model):
  spectral-train [steps] [size] [gray|rgb] [mnist|<image folder>] [new]
                                                       train (default: 100000 32 gray mnist; continues the saved
                                                       model unless 'new', a continued model keeps its size/colour)
  spectral-sample [ckpt] [out.png] [count] [seed] [steps] [scale]

Environment: TRAIN_MAX_HOURS=<hours> stops train / train-vec / pretrain after that time (the best model stays saved).

Web UI:
  server                                               http://localhost:7860 (also started without a command)";

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    {
        // cudarc loads the CUDA DLLs (nvcuda, cublas64_12, ...) at runtime via PATH,
        // so no CUDA_PATH / CUDA_LIBRARY_PATH is needed at build time.
        // Initializes the GPU used by every complex matmul (prints the reason if it falls back to CPU).
        let gpu_available = neural_networks::neural_networks::utils::gpu_matmul::GPU_MATMUL.lock().map(|gpu| gpu.is_some()).unwrap_or(false);
        if gpu_available {
            println!("CUDA initialized successfully 🚀 - all complex matmuls run on the GPU");
        } else {
            println!("CUDA not available. Falling back to CPU");
        }
    }

    #[cfg(not(any(feature = "cuda", feature = "gpu-engine")))]
    println!("Running in CPU mode");

    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("train") => {
            let settings = train_settings(&args)?;
            schedule_training_time_limit();
            train_model(ModelKind::Rm, settings);
        }
        Some("train-vec") => {
            let settings = train_settings(&args)?;
            schedule_training_time_limit();
            train_model(ModelKind::Vec, settings);
        }
        Some("split-qa") => split_qa(&args)?,
        #[cfg(feature = "candle")]
        Some("pretrain") => pretrain_command(&args)?,
        Some("engine-pretrain") => engine_pretrain_command(&args)?,
        Some("engine-bench") => engine_bench_command(&args)?,
        #[cfg(feature = "candle")]
        Some("lm-generate") => lm_generate_command(&args)?,
        Some("predict") => predict(ModelKind::Rm, &args)?,
        Some("predict-vec") => predict(ModelKind::Vec, &args)?,
        // Spectral Flow: see documentation/spectral_transformer.txt
        Some("spectral-train") => {
            train_spectral(spectral_train_settings(&args)?)?;
        }
        // spectral-sample [ckpt] [out.png] [count] [seed] [steps] [scale]
        //
        // `count` samples are laid out in a roughly square grid; each tile
        // is an independent draw, not a variation on one image. The model
        // is unconditional, so which digit appears is decided by the noise.
        // `count 1` gives a single 28x28 digit.
        Some("spectral-sample") => {
            let ckpt = args.get(2).cloned().unwrap_or_else(|| SPECTRAL_CHECKPOINT.to_string());
            let out = args.get(3).cloned().unwrap_or_else(|| "STORAGE/spectral_flow/sample.png".to_string());
            let count: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
            // Default seed: fresh digits on every run; the seed is printed so a sample can be reproduced.
            let seed: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or_else(default_seed);
            // Integration steps of the sampler's ODE solve: 16 is coarse, 64 is usually visibly cleaner.
            let steps: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SAMPLE_STEPS);
            // Pixel upscale; 0 = automatic.
            let scale: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);
            let cols = (count as f64).sqrt().ceil() as usize;
            let rows = count.div_ceil(cols);
            sample_from_checkpoint(&ckpt, rows, cols, steps, seed, &out, scale)?;
            println!("Wrote {out} ({count} sample(s), {rows}x{cols} grid, seed {seed}, {steps} integration steps)");
        }
        Some("server") | None => start_server().await?,
        Some(command) => {
            println!("Unknown command: {}\n\n{}", command, USAGE);
        }
    }

    Ok(())
}

/// `[epochs] [batches|all] [batch_size] [new|resume] [learning_rate]` after the command;
/// default: 100 epochs, all records, batch size 16, new model, `DEFAULT_LEARNING_RATE`.
fn train_settings(args: &[String]) -> Result<TrainSettings, String> {
    let epochs: usize = args.get(2).map(|a| a.parse().map_err(|_| format!("epochs must be a number: {}", a))).transpose()?.unwrap_or(100);
    let num_batches: usize = match args.get(3).map(String::as_str) {
        None | Some("all") => ALL_RECORDS,
        Some(n) => n.parse().map_err(|_| format!("batches must be a number or 'all': {}", n))?,
    };
    let batch_size: usize = args.get(4).map(|a| a.parse().map_err(|_| format!("batch_size must be a number: {}", a))).transpose()?.unwrap_or(16);
    let resume = match args.get(5).map(String::as_str) {
        None | Some("new") => false,
        Some("resume") => true,
        Some(other) => return Err(format!("expected 'new' or 'resume': {}", other)),
    };
    let learning_rate: f64 = args.get(6).map(|a| a.parse().map_err(|_| format!("learning_rate must be a number: {}", a))).transpose()?.unwrap_or(DEFAULT_LEARNING_RATE);
    if !(learning_rate > 0.0 && learning_rate < 1.0) {
        return Err(format!("learning_rate must be between 0 and 1: {}", learning_rate));
    }
    Ok(TrainSettings { epochs, num_batches, batch_size, resume, learning_rate })
}

/// `[steps] [size] [gray|rgb] [mnist|<image folder>] [new]` after `spectral-train`.
fn spectral_train_settings(args: &[String]) -> Result<SpectralTrainSettings, String> {
    let steps: usize = args.get(2).map(|a| a.parse().map_err(|_| format!("steps must be a number: {}", a))).transpose()?.unwrap_or(100_000);
    let size: usize = args.get(3).map(|a| a.parse().map_err(|_| format!("size must be a number: {}", a))).transpose()?.unwrap_or(32);
    if !size.is_power_of_two() || !(32..=512).contains(&size) {
        return Err(format!("size must be 32, 64, 128, 256 or 512: {}", size));
    }
    let channels = match args.get(4).map(String::as_str) {
        None | Some("gray") | Some("grey") => 1,
        Some("rgb") => 3,
        Some(other) => return Err(format!("colour must be 'gray' or 'rgb': {}", other)),
    };
    let data = match args.get(5).map(String::as_str) {
        None | Some("mnist") => DataSource::Mnist,
        Some(folder) => DataSource::Folder(folder.to_string()),
    };
    let resume = args.get(6).map(String::as_str) != Some("new");
    Ok(SpectralTrainSettings { steps, size, channels, data, batch_size: None, max_images: None, resume })
}

/// With `TRAIN_MAX_HOURS` set (e.g. 11 on Kaggle, which ends a session after 12 h), the training stops after the
/// batch that runs when the time is up; the best model stays saved, as after Stop in the web UI.
fn schedule_training_time_limit() {
    let Some(hours) = env::var("TRAIN_MAX_HOURS").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|h| *h > 0.0) else {
        return;
    };
    println!("Training stops after {} h (TRAIN_MAX_HOURS).", hours);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs_f64(hours * 3600.0));
        println!("Time limit of {} h reached: stopping after the current batch.", hours);
        request_training_stop();
    });
}

/// `split-qa <squad.json> <name> [part_mb]`: writes the dataset as `datasets/qa/<name>_001.json`, ... (each a complete SQuAD file).
fn split_qa(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (Some(input), Some(name)) = (args.get(2), args.get(3)) else {
        return Err("usage: split-qa <squad.json> <name> [part_mb]".into());
    };
    let part_mb: usize = args.get(4).map(|a| a.parse().map_err(|_| format!("part_mb must be a number: {}", a))).transpose()?.unwrap_or(DEFAULT_PART_MB);
    let parts = split_squad_file(std::path::Path::new(input), std::path::Path::new(QA_DATASET_DIR), name, part_mb * 1024 * 1024)?;
    let questions = qa_records_from(std::path::Path::new(QA_DATASET_DIR))?.len();
    println!("Wrote {} part(s) to {}/ ({} questions in that folder now).", parts.len(), QA_DATASET_DIR, questions);
    Ok(())
}

/// Generates an answer for the text after the command, or for a text read from the terminal.
/// pretrain <shard folder> [steps] [batch_size] [learning_rate] [new|resume] [window]
#[cfg(feature = "candle")]
fn pretrain_command(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use neural_networks::neural_networks::candle_model::train::{pretrain, PretrainSettings};
    let data_dir = args.get(2).ok_or("usage: pretrain <shard folder> [steps] [batch_size] [learning_rate] [new|resume] [window]")?;
    let mut settings = PretrainSettings::new(data_dir);
    if let Some(steps) = args.get(3) {
        settings.steps = steps.parse()?;
    }
    if let Some(batch) = args.get(4) {
        settings.batch_size = batch.parse()?;
    }
    if let Some(lr) = args.get(5) {
        settings.learning_rate = lr.parse()?;
    }
    settings.resume = args.get(6).is_some_and(|m| m == "resume");
    if let Some(window) = args.get(7) {
        settings.config.window = window.parse()?;
    }
    pretrain(&settings)?;
    Ok(())
}

/// engine-pretrain <shard folder> [steps] [batch_size] [learning_rate] [new|resume] [window] [f16|f32]
fn engine_pretrain_command(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use neural_networks::neural_networks::gpu_engine::train::EngineTrainSettings;
    let data_dir = args.get(2).ok_or("usage: engine-pretrain <shard folder> [steps] [batch_size] [learning_rate] [new|resume] [window] [f16|f32]")?;
    let mut settings = EngineTrainSettings::new(data_dir);
    if let Some(steps) = args.get(3) {
        settings.steps = steps.parse()?;
    }
    if let Some(batch) = args.get(4) {
        settings.batch_size = batch.parse()?;
    }
    if let Some(lr) = args.get(5) {
        settings.learning_rate = lr.parse()?;
    }
    settings.resume = args.get(6).is_some_and(|m| m == "resume");
    if let Some(window) = args.get(7) {
        settings.config.window = window.parse()?;
    }
    let half = args.get(8).is_none_or(|p| p != "f32");
    #[cfg(feature = "gpu-engine")]
    {
        use neural_networks::neural_networks::gpu_engine::cuda::{CudaBackend, Precision};
        let backend = CudaBackend::new(0, if half { Precision::F16 } else { Precision::F32 })?;
        println!("GPU: {} ({} GEMMs)", backend.device_name(), if half { "f16 tensor-core" } else { "f32" });
        settings.loss_scaling = half;
        neural_networks::neural_networks::gpu_engine::train::pretrain(backend, &settings)?;
    }
    #[cfg(not(feature = "gpu-engine"))]
    {
        let _ = half;
        println!("Built without the feature gpu-engine: training on the CPU (f32).");
        neural_networks::neural_networks::gpu_engine::train::pretrain(neural_networks::neural_networks::gpu_engine::cpu::CpuBackend, &settings)?;
    }
    Ok(())
}

/// engine-bench [micro_batch] [f16|f32] [steps]: forward + backward + AdamW of the default model on random tokens.
fn engine_bench_command(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use neural_networks::neural_networks::gpu_engine::{backend::Backend, model::{EngineConfig, Model}};
    let micro: usize = args.get(2).map(|m| m.parse()).transpose()?.unwrap_or(4);
    let half = args.get(3).is_none_or(|p| p != "f32");
    let steps: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5);
    fn run<B: Backend>(backend: B, micro: usize, steps: usize) {
        let config = EngineConfig::default();
        let t = config.seq_len;
        let mut model = Model::new(backend, config.clone(), 1);
        let (mut m, mut v) = (model.backend.zeros(model.layout.total), model.backend.zeros(model.layout.total));
        let mut seed = 7u64;
        let ids: Vec<u32> = (0..micro * (t + 1))
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) % config.vocab_size as u64) as u32
            })
            .collect();
        let mut step = |i: u64| -> f64 {
            model.zero_grads();
            let loss = model.forward_backward(&ids, true, 1.0);
            let total = model.layout.total;
            let finite = model.backend.sum_squares(&model.grads, total).is_finite(); // also waits for the GPU
            model.backend.adamw(&mut model.params, &model.grads, &mut m, &mut v, total, 1e-4, 0.9, 0.95, 1e-8, 0.01, 1.0, i);
            model.refresh_blocks();
            assert!(finite && loss.is_finite(), "non-finite loss or gradient");
            loss
        };
        println!("micro-batch {micro} x {t} tokens, warm-up step ...");
        step(1);
        let started = std::time::Instant::now();
        for i in 0..steps {
            let loss = step(i as u64 + 2);
            println!("  step {}: loss {loss:.4}", i + 1);
        }
        let seconds = started.elapsed().as_secs_f64() / steps as f64;
        println!("{:.3} s per micro-batch = {:.0} tokens/s (a batch of 16 sequences: {:.2} s per step)", seconds, (micro * t) as f64 / seconds, seconds * 16.0 / micro as f64);
    }
    #[cfg(feature = "gpu-engine")]
    {
        use neural_networks::neural_networks::gpu_engine::cuda::{CudaBackend, Precision};
        let backend = CudaBackend::new(0, if half { Precision::F16 } else { Precision::F32 })?;
        println!("GPU: {} ({} GEMMs)", backend.device_name(), if half { "f16 tensor-core" } else { "f32" });
        run(backend, micro, steps);
    }
    #[cfg(not(feature = "gpu-engine"))]
    {
        let _ = half;
        println!("Built without the feature gpu-engine: CPU (f32).");
        run(neural_networks::neural_networks::gpu_engine::cpu::CpuBackend, micro, steps);
    }
    Ok(())
}

/// lm-generate [max_tokens] <text>
#[cfg(feature = "candle")]
fn lm_generate_command(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use neural_networks::neural_networks::{
        candle_model::{
            model::Model,
            train::{device, generate, CHECKPOINT},
        },
        utils::tokenizer::{detokenize, tokenize},
    };
    let (max_tokens, text_start) = match args.get(2).and_then(|a| a.parse::<usize>().ok()) {
        Some(n) => (n, 3),
        None => (100, 2),
    };
    let text = if args.len() > text_start { args[text_start..].join(" ") } else { read_input("Text: ")? };
    let device = device();
    let model = Model::load(std::path::Path::new(CHECKPOINT), candle_core::DType::F32, &device)?;
    let (_, ids) = tokenize(&text).map_err(|e| e.to_string())?;
    let eos = 50279;
    let started = std::time::Instant::now();
    let continuation = generate(&model, &ids, max_tokens, Some(eos), &device)?;
    println!("{}{}", text, detokenize(&continuation, false).map_err(|e| e.to_string())?);
    println!("({} tokens, {:.1} s)", continuation.len(), started.elapsed().as_secs_f64());
    Ok(())
}

fn predict(kind: ModelKind, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = if args.len() > 2 { args[2..].join(" ") } else { read_input("Enter input text for prediction: ")? };
    match generate(kind, &maybe_fix_encoding(&prompt), MAX_GENERATED_TOKENS) {
        Ok(generation) => {
            println!("\n{}: {}\n({:.2} s)", kind.label(), generation.answer, generation.seconds);
            let candidates: Vec<String> = generation.first_tokens.iter().map(|c| format!("{:?} {:.1} %", c.token, 100.0 * c.probability)).collect();
            println!("most likely first tokens: {}", candidates.join(", "));
        }
        Err(message) => println!("{} Train it first: wav-transformer {}", message, if kind == ModelKind::Rm { "train" } else { "train-vec" }),
    }
    Ok(())
}

/// Seed to use when the user did not supply one.
///
/// Defaulting to a fixed seed makes "run it again" produce a byte-identical
/// image, which is astonishing in the bad sense: the obvious way to ask for
/// another sample silently returns the previous one. Defaulting to entropy and
/// *printing* the seed keeps both properties -- a fresh image every run, and
/// exact reproducibility once you know the number to pass back.
fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

fn read_input(prompt: &str) -> Result<String, io::Error> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

// ── Web UI ──────────────────────────────────────────────────────────────────

/// Model loaded for generation, kept until its file changes (e.g. the training saved a new version).
struct LoadedModel {
    kind: ModelKind,
    modified: Option<u64>,
    network: NeuralNetwork,
}

struct AppState {
    loaded_model: Mutex<Option<LoadedModel>>,
}

#[derive(Deserialize)]
struct TrainRequest {
    model: ModelKind,
    epochs: usize,
    /// `None` = all records.
    batches: Option<usize>,
    batch_size: usize,
    #[serde(default)]
    resume: bool,
    learning_rate: Option<f64>,
}

#[derive(Deserialize)]
struct ExamplesQuery {
    model: ModelKind,
    split: ExampleSplit,
}

#[derive(Deserialize)]
struct SpectralGenerateRequest {
    count: usize,
    steps: Option<usize>,
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct GenerateRequest {
    model: ModelKind,
    prompt: String,
    max_tokens: Option<usize>,
}

async fn start_server() -> std::io::Result<()> {
    println!("🚀 Starting Actix-web server at http://localhost:7860");

    let mut tera = Tera::new();
    tera.load_from_glob("templates/**/*").expect("Failed to load templates");

    let app_state = web::Data::new(AppState { loaded_model: Mutex::new(None) });

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(tera.clone()))
            .app_data(app_state.clone())
            .route("/", web::get().to(index))
            .route("/api/status", web::get().to(api_status))
            .route("/api/train", web::post().to(api_train))
            .route("/api/train/stop", web::post().to(api_train_stop))
            .route("/api/generate", web::post().to(api_generate))
            .route("/api/examples", web::get().to(api_examples))
            .route("/api/spectral/train", web::post().to(api_spectral_train))
            .route("/api/spectral/generate", web::post().to(api_spectral_generate))
            .route("/api/spectral/preview", web::get().to(api_spectral_preview))
    })
    .bind(("0.0.0.0", 7860))?
    .run()
    .await
}

fn json_error(status: actix_web::http::StatusCode, message: impl Into<String>) -> HttpResponse {
    HttpResponse::build(status).json(serde_json::json!({ "error": message.into() }))
}

async fn index(tmpl: web::Data<Tera>) -> impl Responder {
    match tmpl.render("index.html", &Context::new()) {
        Ok(body) => HttpResponse::Ok().content_type("text/html; charset=utf-8").body(body),
        Err(err) => {
            eprintln!("Template error: {:?}", err);
            HttpResponse::InternalServerError().body("Template error")
        }
    }
}

async fn api_status() -> impl Responder {
    let mut models: Vec<_> = ModelKind::ALL.iter().map(|&kind| model_info(kind)).collect();
    models.push(spectral_model_info());
    HttpResponse::Ok().json(serde_json::json!({
        "device": compute_device(),
        "models": models,
        "training": training_progress(),
    }))
}

async fn api_train(request: web::Json<TrainRequest>) -> impl Responder {
    if training_progress().running {
        return json_error(actix_web::http::StatusCode::CONFLICT, "A training is already running.");
    }
    if request.epochs == 0 || request.batch_size == 0 || request.batches == Some(0) {
        return json_error(actix_web::http::StatusCode::BAD_REQUEST, "Epochs, batches and batch size must be at least 1.");
    }
    let learning_rate = request.learning_rate.unwrap_or(DEFAULT_LEARNING_RATE);
    if !(learning_rate > 0.0 && learning_rate < 1.0) {
        return json_error(actix_web::http::StatusCode::BAD_REQUEST, "The learning rate must be between 0 and 1.");
    }

    let settings = TrainSettings {
        epochs: request.epochs,
        num_batches: request.batches.unwrap_or(ALL_RECORDS),
        batch_size: request.batch_size,
        resume: request.resume,
        learning_rate,
    };
    let kind = request.model;
    // Training is long, blocking CPU/GPU work: run it on its own thread, not on the server's workers.
    std::thread::spawn(move || {
        train_model(kind, settings);
    });

    HttpResponse::Accepted().json(serde_json::json!({ "started": kind.id() }))
}

/// The validation (or training) questions of the model's last training, to try them out.
async fn api_examples(query: web::Query<ExamplesQuery>) -> impl Responder {
    let (model, split) = (query.model, query.split);
    match web::block(move || qa_examples(model, split)).await {
        Ok(Ok(examples)) => HttpResponse::Ok().json(examples),
        Ok(Err(message)) => json_error(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR, message),
        Err(err) => json_error(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

async fn api_spectral_train(request: web::Json<SpectralTrainSettings>) -> impl Responder {
    if training_progress().running {
        return json_error(actix_web::http::StatusCode::CONFLICT, "A training is already running.");
    }
    let settings = request.into_inner();
    if settings.steps == 0 || settings.batch_size == Some(0) || settings.max_images == Some(0) {
        return json_error(actix_web::http::StatusCode::BAD_REQUEST, "Steps, images per step and the image limit must be at least 1.");
    }
    if !settings.size.is_power_of_two() || !(32..=512).contains(&settings.size) || !(settings.channels == 1 || settings.channels == 3) {
        return json_error(actix_web::http::StatusCode::BAD_REQUEST, "Image size must be 32 to 512 (a power of two), colour grayscale or RGB.");
    }
    if let DataSource::Folder(path) = &settings.data {
        if !std::path::Path::new(path).is_dir() {
            return json_error(actix_web::http::StatusCode::BAD_REQUEST, format!("Image folder not found: {}", path));
        }
    }
    // Loading images and training are long, blocking work: run them on their own thread.
    std::thread::spawn(move || {
        if let Err(e) = train_spectral(settings) {
            eprintln!("[spectral] training failed: {}", e);
        }
    });
    HttpResponse::Accepted().json(serde_json::json!({ "started": "spectral" }))
}

/// The generated images as a PNG; seed and duration in the `X-Seed` / `X-Seconds` headers.
async fn api_spectral_generate(request: web::Json<SpectralGenerateRequest>) -> impl Responder {
    let request = request.into_inner();
    match web::block(move || generate_spectral_png(request.count, request.steps, request.seed)).await {
        Ok(Ok(generation)) => HttpResponse::Ok()
            .content_type("image/png")
            .insert_header(("X-Seed", generation.seed.to_string()))
            .insert_header(("X-Seconds", format!("{:.3}", generation.seconds)))
            .body(generation.png),
        Ok(Err(message)) => json_error(actix_web::http::StatusCode::BAD_REQUEST, message),
        Err(err) => json_error(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Generation failed: {}", err)),
    }
}

/// The latest preview image written by the spectral training (the path comes from the training, not the request).
async fn api_spectral_preview() -> impl Responder {
    let Some(path) = training_progress().sample_image else {
        return json_error(actix_web::http::StatusCode::NOT_FOUND, "No preview yet.");
    };
    match std::fs::read(&path) {
        Ok(png) => HttpResponse::Ok().content_type("image/png").insert_header(("Cache-Control", "no-store")).body(png),
        Err(e) => json_error(actix_web::http::StatusCode::NOT_FOUND, format!("Preview not readable: {}", e)),
    }
}

async fn api_train_stop() -> impl Responder {
    if !training_progress().running {
        return json_error(actix_web::http::StatusCode::CONFLICT, "No training is running.");
    }
    request_training_stop();
    HttpResponse::Ok().json(serde_json::json!({ "stopping": true }))
}

async fn api_generate(request: web::Json<GenerateRequest>, state: web::Data<AppState>) -> impl Responder {
    let prompt = maybe_fix_encoding(request.prompt.trim());
    if prompt.is_empty() {
        return json_error(actix_web::http::StatusCode::BAD_REQUEST, "Please enter a text.");
    }
    let kind = request.model;
    let max_tokens = request.max_tokens.unwrap_or(MAX_GENERATED_TOKENS).clamp(1, 500);

    let result = web::block(move || -> Result<_, String> {
        let modified = model_info(kind).modified;
        let mut loaded = state.loaded_model.lock().map_err(|_| "model lock poisoned".to_string())?;
        let reload = !matches!(&*loaded, Some(model) if model.kind == kind && model.modified == modified);
        if reload {
            *loaded = Some(LoadedModel { kind, modified, network: load_model(kind)? });
        }
        let network = &mut loaded.as_mut().expect("model loaded above").network;
        Ok(generate_with(network, &prompt, max_tokens))
    })
    .await;

    match result {
        Ok(Ok(generation)) => HttpResponse::Ok().json(generation),
        Ok(Err(message)) => json_error(actix_web::http::StatusCode::BAD_REQUEST, message),
        Err(err) => json_error(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Generation failed: {}", err)),
    }
}
