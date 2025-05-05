use std::time::Instant;

use crate::{
    database::sled_db::SLED_DB_TRANSFORMER_V1,
    neural_networks::{
        network_components::input::{load_data_xquad_de_as_dataset, DataTrait, Dataset},
        network_types::{
            neural_network_generic::{get_from_db, print_networt_structure, update_learning_rate, NeuralNetwork, OperationMode},
            transformer::{transformer_builder::create_transformer, transformer_network::train},
        },
    },
};

pub fn train_transformer_from_dataset() {
    let now = Instant::now();

    let mut transformer = match get_from_db(SLED_DB_TRANSFORMER_V1) {
        Ok(transformer) => {
            // Successfully loaded transformer from the database
            println!("Loaded transformer from the database!");
            transformer
        }
        Err(e) => {
            println!("error: {:?}", e);
            // Create a new transformer since the database didn't have one
            let transformer: NeuralNetwork = create_transformer(OperationMode::TRAINING);
            println!("Created a new transformer for training.");
            transformer
        }
    };

    let learning_rate = 0.001;
    let num_epochs = 5000;
    transformer.learning_rate = learning_rate;
    update_learning_rate(&mut transformer, learning_rate);
    print_networt_structure(&mut transformer);

    let seconds_elapsed = now.elapsed();
    println!("time elapsed in seconds: {:?}", &seconds_elapsed);

    let dataset = load_data_xquad_de_as_dataset().unwrap();
    let data_batches: Vec<Dataset<String, String>> = dataset.split_into_batches(1);

    let (input_batch, target_batch) = (data_batches[0].get_input(), data_batches[0].get_target());

    println!("input batch: {:?}", input_batch);
    println!("target batch: {:?}", target_batch);
    let dataset_small = Dataset::new(input_batch.clone(), target_batch.clone());

    train(&mut transformer, dataset_small, num_epochs);
    let seconds_elapsed_end = now.elapsed();

    println!("time elapsed in seconds: {:?}", seconds_elapsed_end - seconds_elapsed);
}
