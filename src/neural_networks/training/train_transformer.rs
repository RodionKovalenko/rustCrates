use std::time::Instant;

use crate::{
    database::sled_db::{SLED_DB_TRANSFORMER_RM_V1, SLED_DB_TRANSFORMER_V1},
    neural_networks::{
        network_components::input::{load_data_xquad_de_as_dataset, DataTrait, Dataset},
        network_types::{
            neural_network_generic::{get_from_db, print_networt_structure, update_learning_rate, NeuralNetwork, OperationMode},
            transformer::{transformer_builder::create_transformer, transformer_builder_rm::create_transformer_rm, transformer_network::train},
        },
    },
};

/// `num_records` value that selects every batch of the dataset.
pub const ALL_RECORDS: usize = usize::MAX;

pub fn train_transformer_from_dataset(num_epochs: usize, num_records: usize, batch_size: usize) -> bool {
    train_from_dataset(SLED_DB_TRANSFORMER_V1, create_transformer, num_epochs, num_records, batch_size)
}

/// Same training with the row-major transformer (`create_transformer_rm`): with the `cuda` feature its
/// matrix multiplications run on the GPU. Saved to its own model file (`SLED_DB_TRANSFORMER_RM_V1`).
pub fn train_transformer_rm_from_dataset(num_epochs: usize, num_records: usize, batch_size: usize) -> bool {
    train_from_dataset(SLED_DB_TRANSFORMER_RM_V1, create_transformer_rm, num_epochs, num_records, batch_size)
}

fn train_from_dataset(model_file: &str, create_network: fn(OperationMode) -> NeuralNetwork, num_epochs: usize, num_records: usize, batch_size: usize) -> bool {
    let now = Instant::now();
    const REUSE_SAVED_MODEL: bool = false;

    let mut transformer = if REUSE_SAVED_MODEL {
        match get_from_db(model_file) {
            Ok(transformer) => {
                println!("Loaded transformer from the database!");
                transformer
            }
            Err(e) => {
                println!("error: {:?}", e);
                let transformer: NeuralNetwork = create_network(OperationMode::TRAINING);
                println!("Created a new transformer for training.");
                transformer
            }
        }
    } else {
        let transformer: NeuralNetwork = create_network(OperationMode::TRAINING);
        println!("Created a new transformer for training (checkpoint loading disabled).");
        transformer
    };

    let learning_rate = 0.001;
    let num_epochs = num_epochs;
    transformer.learning_rate = learning_rate;
    update_learning_rate(&mut transformer, learning_rate);
    print_networt_structure(&mut transformer);

    let seconds_elapsed = now.elapsed();
    println!("time elapsed in seconds: {:?}", &seconds_elapsed);

    let dataset: Dataset<String, String> = load_data_xquad_de_as_dataset().unwrap();
    let dataset_small = select_records(&dataset, num_records, batch_size);
    println!("selected records: {} (train() keeps 10% of them for validation)", dataset_small.input.len());

    train(&mut transformer, dataset_small, num_epochs, batch_size);
    let seconds_elapsed_end = now.elapsed();

    println!("time elapsed in seconds: {:?}", seconds_elapsed_end - seconds_elapsed);

    println!("done");
    true
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
