use std::env;
use std::io::{self, Write};

use neural_networks::neural_networks::{network_types::transformer::transformer_network::predict_by_text, training::train_transformer::train_transformer_from_dataset};

pub enum ARGUMENTS {
    UPHOLD,
    NETWORK,
}

/**
* start with cargo run -- --server
* connect client: cargo run -- --client
tcp::test_connection();
 */

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Test beginns");

    let args: Vec<String> = env::args().collect();
    println!("Args: {:?}", args);

    if let Some(arg1) = args.get(1) {
        match arg1.as_str() {
            "train" => train_transformer_from_dataset(),
            "predict" => {
                let input = read_input("Enter input text for prediction: ")?;
                predict_by_text(&vec![input]);
            }
            _ => println!("Unrecognized argument: {}", arg1),
        }
    } else {
        println!("No arguments provided.");
    }

    Ok(())
}

fn read_input(prompt: &str) -> Result<String, io::Error> {
    print!("{}", prompt);
    io::stdout().flush()?; // Make sure the prompt is displayed before input
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string()) // Remove trailing newline
}
