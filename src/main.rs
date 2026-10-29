use std::env;
use std::io::{self, Write};
use std::sync::Mutex;

use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use neural_networks::neural_networks::network_types::neural_network_generic::NeuralNetwork;
use neural_networks::neural_networks::network_types::transformer::transformer_network::MAX_GENERATED_TOKENS;
use neural_networks::neural_networks::spectral_model::data::DataSource;
use neural_networks::neural_networks::spectral_model::sample::{sample_from_checkpoint, DEFAULT_SAMPLE_STEPS};
use neural_networks::neural_networks::spectral_model::train::SPECTRAL_CHECKPOINT;
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

Spectral flow (image model):
  spectral-train [steps] [size] [gray|rgb] [mnist|<image folder>] [new]
                                                       train (default: 100000 32 gray mnist; continues the saved
                                                       model unless 'new', a continued model keeps its size/colour)
  spectral-sample [ckpt] [out.png] [count] [seed] [steps] [scale]

Environment: TRAIN_MAX_HOURS=<hours> stops train / train-vec after that time (the best model stays saved).

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

    #[cfg(not(feature = "cuda"))]
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

/// Generates an answer for the text after the command, or for a text read from the terminal.
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
