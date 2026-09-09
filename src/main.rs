use std::env;
use std::io::{self, Write};
use std::sync::Mutex;

use actix_web::web::Form;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use neural_networks::neural_networks::network_types::transformer::test_transformer::test_train_transformer;
use neural_networks::neural_networks::network_types::transformer::transformer_network::predict_by_text;
use neural_networks::neural_networks::spectral_model::data as spectral_data;
use neural_networks::neural_networks::spectral_model::model::SpectralNet;
use neural_networks::neural_networks::spectral_model::sample::{
    lowpass_reference_png, prior_reference_png, report_band_profile, sample_batch,
    sample_from_checkpoint,
};
use neural_networks::neural_networks::spectral_model::train::Checkpoint as SpectralCheckpoint;
use neural_networks::neural_networks::spectral_model::spectral::SpectralPrep;
use neural_networks::neural_networks::spectral_model::train::{train_spectral_flow, TrainConfig};
use neural_networks::neural_networks::training::train_transformer::train_transformer_from_dataset;
use neural_networks::utils::string::maybe_fix_encoding;
use serde::Deserialize;
use tera::{Context, Tera};

#[derive(Deserialize)]
struct PredictForm {
    prompt: String,
}

#[derive(Deserialize)]
struct TrainForm {
    epochs: usize,
    batch_size: usize,
    num_records: usize,
}

struct AppState {
    training_done: Mutex<bool>,
    training_result: Mutex<Option<String>>,
}

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Test begins");

    #[cfg(feature = "cuda")]
    {
        let _ctx = cust::quick_init().expect("CUDA init failed");
        println!("CUDA initialized successfully 🚀");
    }

    #[cfg(not(feature = "cuda"))]
    println!("Running in CPU mode");

    let args: Vec<String> = env::args().collect();
    println!("Args: {:?}", args);

    if let Some(arg1) = args.get(1) {
        match arg1.as_str() {
            "train" => {
                train_transformer_from_dataset(5000, 4, 2);
            }
            "predict" => {
                let input = read_input("Enter input text for prediction: ")?;
                predict_by_text(&vec![input]);
            }
            "test" => {
                test_train_transformer();
            }
            // Spectral Flow: see documentation/spectral_transformer.txt
            "spectral-warmup" => {
                // Build-order gate 7: bands 0-1, no band schedule. Blurry
                // digit-shaped blobs should appear within minutes.
                let mut cfg = TrainConfig::warmup_gate();
                if let Some(s) = args.get(2).and_then(|s| s.parse::<usize>().ok()) {
                    cfg.steps = s;
                }
                train_spectral_flow(cfg)?;
            }
            "spectral-train" => {
                let mut cfg = TrainConfig::default();
                if let Some(s) = args.get(2).and_then(|s| s.parse::<usize>().ok()) {
                    cfg.steps = s;
                }
                train_spectral_flow(cfg)?;
            }
            "spectral-lowpass" => {
                // Reference ceiling: real digits limited to bands 0..=max_band.
                let max_band = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1usize);
                let data = spectral_data::load_train()?;
                let net = SpectralNet::new(true);
                let prep = SpectralPrep::from_dataset(&data.images);
                let out = format!("STORAGE/spectral_flow/lowpass_band{max_band}.png");
                lowpass_reference_png(&net, &prep, &data.images, max_band, 8, 8, &out)?;
                println!("Wrote {out}");
            }
            "spectral-bench" => {
                neural_networks::neural_networks::spectral_model::bench::run_bench();
            }
            "spectral-diag" => {
                let path = args
                    .get(2)
                    .cloned()
                    .unwrap_or_else(|| "STORAGE/spectral_flow/warmup.bin".to_string());
                let ckpt = SpectralCheckpoint::load(&path)?;
                let net = SpectralNet::new(ckpt.adaptive_roundtrip);
                let prep = SpectralPrep::with_table(ckpt.whitening.clone());
                let data = spectral_data::load_train()?;
                println!("[spectral] checkpoint {path} at step {}", ckpt.step);
                prior_reference_png(&prep, 8, 8, 5, "STORAGE/spectral_flow/prior.png")?;
                report_band_profile(
                    &net, &prep, &ckpt.ema, &ckpt.schedule, ckpt.path, &data.images, 32, 16,
                );
            }
            // spectral-sample [ckpt] [out.png] [count] [seed]
            //
            // `count` samples are laid out in a roughly square grid; each tile
            // is an independent draw, not a variation on one image. The model
            // is unconditional, so which digit appears is decided by the noise.
            // `count 1` gives a single 28x28 digit.
            "spectral-sample" => {
                let ckpt = args
                    .get(2)
                    .cloned()
                    .unwrap_or_else(|| TrainConfig::default().ckpt_path);
                let out = args
                    .get(3)
                    .cloned()
                    .unwrap_or_else(|| "STORAGE/spectral_flow/sample.png".to_string());
                let count: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
                // Default seed 0 makes repeated runs byte-identical, so two
                // checkpoints can be compared on the same noise draws. Pass a
                // seed explicitly to get fresh digits.
                let seed: u64 =
                    args.get(5).and_then(|s| s.parse().ok()).unwrap_or_else(default_seed);
                // Integration steps. This is a genuine quality knob and costs
                // nothing but time: the sampler solves an ODE, and 16 steps is
                // a coarse solve. 64 is usually visibly cleaner.
                let steps: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(64);
                // Pixel upscale; 0 = automatic.
                let scale: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);
                let cols = (count as f64).sqrt().ceil() as usize;
                let rows = count.div_ceil(cols);
                sample_from_checkpoint(&ckpt, rows, cols, steps, seed, &out, scale)?;
                println!(
                    "Wrote {out} ({count} sample(s), {rows}x{cols} grid, seed {seed},                      {steps} integration steps)"
                );
            }
            // spectral-hd [ckpt] [out.png] [seed] [steps]
            //
            // One digit, band-limited-upsampled to 1024x1024 and centred on a
            // 1920x1080 canvas. The upsample is exact ideal interpolation, not
            // a learned model: it enlarges faithfully but invents nothing, so
            // any artefact in the 32x32 sample is enlarged along with it.
            "spectral-hd" => {
                use neural_networks::neural_networks::spectral_model::upscale::{
                    centre_on_hd, spectral_upsample_apodized, write_gray_png,
                    DEFAULT_APODISATION, HD_H, HD_W,
                };
                let ckpt_path = args
                    .get(2)
                    .cloned()
                    .unwrap_or_else(|| TrainConfig::default().ckpt_path);
                let out = args
                    .get(3)
                    .cloned()
                    .unwrap_or_else(|| "STORAGE/spectral_flow/hd.png".to_string());
                let seed: u64 =
                    args.get(4).and_then(|s| s.parse().ok()).unwrap_or_else(default_seed);
                let steps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
                // Apodisation strength; pass 0 for the exact (ringing) upsample.
                let alpha: f32 = args
                    .get(6)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(DEFAULT_APODISATION as f32);

                let ckpt = SpectralCheckpoint::load(&ckpt_path)?;
                let net = SpectralNet::new(ckpt.adaptive_roundtrip);
                let prep = SpectralPrep::with_table(ckpt.whitening.clone());
                let imgs = sample_batch(
                    &net, &prep, &ckpt.ema, &ckpt.schedule, ckpt.path, 1, steps, seed,
                );
                let big = spectral_upsample_apodized(&imgs[0], 32, 1024, alpha as _);
                let canvas = centre_on_hd(&big, 1024);
                write_gray_png(&canvas, HD_W, HD_H, &out)?;
                println!(
                    "Wrote {out} ({HD_W}x{HD_H}, digit at 1024x1024, seed {seed},                      {steps} steps, apodisation {alpha})"
                );
            }
            // spectral-compare [ckpt] [out.png] [seed] [steps]
            //
            // Left: the raw 32x32 sample, nearest-neighbour enlarged so every
            // model pixel is visible as a block. Right: the same sample through
            // the band-limited upscaler. Same data, same size -- so anything
            // visible in one panel and not the other is the interpolation, and
            // any defect present in BOTH came from the generator.
            "spectral-compare" => {
                use neural_networks::neural_networks::spectral_model::upscale::{
                    nearest_upsample, spectral_upsample, write_side_by_side,
                };
                let ckpt_path = args
                    .get(2)
                    .cloned()
                    .unwrap_or_else(|| TrainConfig::default().ckpt_path);
                let out = args
                    .get(3)
                    .cloned()
                    .unwrap_or_else(|| "STORAGE/spectral_flow/compare.png".to_string());
                let seed: u64 =
                    args.get(4).and_then(|s| s.parse().ok()).unwrap_or_else(default_seed);
                let steps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);

                let ckpt = SpectralCheckpoint::load(&ckpt_path)?;
                let net = SpectralNet::new(ckpt.adaptive_roundtrip);
                let prep = SpectralPrep::with_table(ckpt.whitening.clone());
                let imgs = sample_batch(
                    &net, &prep, &ckpt.ema, &ckpt.schedule, ckpt.path, 1, steps, seed,
                );

                const SIDE: usize = 512;
                let blocky = nearest_upsample(&imgs[0], 32, SIDE / 32);
                let smooth = spectral_upsample(&imgs[0], 32, SIDE);
                write_side_by_side(&blocky, &smooth, SIDE, 16, &out)?;
                println!(
                    "Wrote {out} (left: raw 32x32 pixels, right: band-limited upscale,                      seed {seed}, {steps} steps)"
                );
            }
            // spectral-compare3 [ckpt] [out.png] [seed] [steps] [alpha]
            //
            // raw pixels | exact band-limited | apodised. Same sample, same
            // size: the middle panel shows Gibbs ringing, the right shows what
            // tapering the band edge costs and buys.
            "spectral-compare3" => {
                use neural_networks::neural_networks::spectral_model::upscale::{
                    nearest_upsample, spectral_upsample, spectral_upsample_apodized,
                    write_triptych, DEFAULT_APODISATION,
                };
                let ckpt_path = args
                    .get(2)
                    .cloned()
                    .unwrap_or_else(|| TrainConfig::default().ckpt_path);
                let out = args
                    .get(3)
                    .cloned()
                    .unwrap_or_else(|| "STORAGE/spectral_flow/compare3.png".to_string());
                let seed: u64 =
                    args.get(4).and_then(|s| s.parse().ok()).unwrap_or_else(default_seed);
                let steps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
                let alpha: f32 = args
                    .get(6)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(DEFAULT_APODISATION as f32);

                let ckpt = SpectralCheckpoint::load(&ckpt_path)?;
                let net = SpectralNet::new(ckpt.adaptive_roundtrip);
                let prep = SpectralPrep::with_table(ckpt.whitening.clone());
                let imgs = sample_batch(
                    &net, &prep, &ckpt.ema, &ckpt.schedule, ckpt.path, 1, steps, seed,
                );

                // Must be a power of two: the radix-2 FFT that does the upsampling
                // requires it. 384 (= 32 x 12) is not, and fails at runtime.
                const SIDE: usize = 512;
                let raw = nearest_upsample(&imgs[0], 32, SIDE / 32);
                let exact = spectral_upsample(&imgs[0], 32, SIDE);
                let soft = spectral_upsample_apodized(&imgs[0], 32, SIDE, alpha as _);
                write_triptych([&raw, &exact, &soft], SIDE, 12, &out)?;
                println!("Wrote {out} (raw | exact | apodised alpha={alpha}, seed {seed})");
            }
            "server" => start_server().await?,
            _ => println!("Unrecognized argument: {}", arg1),
        }
    } else {
        println!("No arguments provided.");
        start_server().await?;
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

async fn start_server() -> std::io::Result<()> {
    println!("🚀 Starting Actix-web server at http://localhost:7860");

    let mut tera = Tera::new();
    tera.load_from_glob("templates/**/*")
        .expect("Failed to load templates");

    let app_state = web::Data::new(AppState {
        training_done: Mutex::new(true),
        training_result: Mutex::new(None),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(tera.clone()))
            .app_data(app_state.clone())
            .route("/", web::get().to(index))
            .route("/api/train", web::post().to(api_train))
            .route("/api/train_status", web::get().to(api_train_status))
            .route("/api/predict", web::post().to(api_predict))
    })
    .bind(("0.0.0.0", 7860))?
    .run()
    .await
}

async fn index(tmpl: web::Data<Tera>) -> impl Responder {
    let ctx = Context::new();
    match tmpl.render("index.html", &ctx) {
        Ok(body) => HttpResponse::Ok().content_type("text/html").body(body),
        Err(err) => {
            eprintln!("Template error: {:?}", err);
            HttpResponse::InternalServerError().body("Template error")
        }
    }
}

async fn api_train(form: Form<TrainForm>, state: web::Data<AppState>) -> impl Responder {
    let state_clone = state.clone();
    let epochs = form.epochs;
    let num_records = form.num_records;
    let batch_size = form.batch_size;

    actix_web::rt::spawn(async move {
        {
            let mut done = state_clone.training_done.lock().unwrap();
            *done = false;
        }

        let _result: bool = train_transformer_from_dataset(epochs, num_records, batch_size);

        {
            let mut training_result = state_clone.training_result.lock().unwrap();
            *training_result = Some(format!("✅ Training complete with {} epochs.", epochs));
        }

        {
            let mut done = state_clone.training_done.lock().unwrap();
            *done = true;
        }
    });

    HttpResponse::Ok().body("Training started.")
}

async fn api_train_status(state: web::Data<AppState>) -> impl Responder {
    let done = *state.training_done.lock().unwrap();
    let result = state.training_result.lock().unwrap();

    HttpResponse::Ok().json(serde_json::json!({
        "done": done,
        "result": result.clone()
    }))
}

async fn api_predict(form: Form<PredictForm>, tmpl: web::Data<Tera>) -> impl Responder {
    let fixed_prompt = maybe_fix_encoding(&form.prompt);
    println!("fixed prompt: {:?}", &fixed_prompt);

    let output = predict_by_text(&vec![fixed_prompt]);

    let mut ctx = Context::new();
    ctx.insert("infer_result", &output);
    ctx.insert("prompt", &form.prompt);

    match tmpl.render("index.html", &ctx) {
        Ok(body) => HttpResponse::Ok().content_type("text/html").body(body),
        Err(err) => {
            eprintln!("Template error: {:?}", err);
            HttpResponse::InternalServerError().body("Template error")
        }
    }
}
