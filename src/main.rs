use std::env;

use neural_networks::neural_networks::utils::dataset_loader::load_dataset_xquad;

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

    let _args: Vec<String> = env::args().collect();
    //println!("{:?}", args[1]);

    // if args.len() > 1 {
    //     let arg1: &str = &*args[1].clone();
    //
    //     match arg1 {
    //         "uphold" => collect_data_task::update_json_data_from_uphold_api(),
    //         "network" => train_ffn(),
    //         _ => println!(" no argument recognized"),
    //     }
    // }
    // Hugging Face API token (replace with your token)
    // URL to the raw JSON file in the GitHub repository

    load_dataset_xquad()?; 

    Ok(())
}
