use std::env;
use std::io::{self, Write};
use std::sync::Mutex;

use actix_web::web::Form;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::Deserialize;
use tera::{Context, Tera};

use neural_networks::neural_networks::{network_types::transformer::transformer_network::predict_by_text, training::train_transformer::train_transformer_from_dataset};
use neural_networks::utils::string::fix_encoding;

#[derive(Deserialize)]
struct PredictForm {
    prompt: String,
}

#[derive(Deserialize)]
struct TrainForm {
    epochs: usize,
}

struct AppState {
    training_done: Mutex<bool>,
    training_result: Mutex<Option<String>>,
}

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Test begins");

    let args: Vec<String> = env::args().collect();
    println!("Args: {:?}", args);

    if let Some(arg1) = args.get(1) {
        match arg1.as_str() {
            "train" => {
                train_transformer_from_dataset(5000);
            }
            "predict" => {
                let input = read_input("Enter input text for prediction: ")?;
                predict_by_text(&vec![input]);
            }
            "server" => start_server().await?,
            _ => println!("Unrecognized argument: {}", arg1),
        }
    } else {
        println!("No arguments provided.");
    }

    Ok(())
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

    let tera = Tera::new("templates/**/*").expect("Failed to load templates");

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

    actix_web::rt::spawn(async move {
        {
            let mut done = state_clone.training_done.lock().unwrap();
            *done = false;
        }

        let _result: bool = train_transformer_from_dataset(epochs);

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
    let fixed_prompt = fix_encoding(&form.prompt);
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
