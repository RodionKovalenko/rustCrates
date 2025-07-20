use std::time::Instant;

use crate::{
    database::sled_db::SLED_DB_TRANSFORMER_V1,
    neural_networks::{
        network_components::input::{DataTrait, Dataset},
        network_types::{
            neural_network_generic::{get_from_db, print_networt_structure, update_learning_rate, NeuralNetwork, OperationMode},
            transformer::{transformer_builder::create_transformer, transformer_network::train},
        },
    },
};

pub fn test_train_transformer() {
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
    transformer.learning_rate = learning_rate;
    update_learning_rate(&mut transformer, learning_rate);

    print_networt_structure(&mut transformer);

    let seconds_elapsed = now.elapsed();
    println!("time elapsed in seconds: {:?}", &seconds_elapsed);

    //let input_str1: &str = "Wie geht es dir?";
    let input_str2: &str = "Was ist die Hauptstadt von Deutschland? Kannst du bitte eine kurze Antwort geben?";

    //let target_1: &str = "Mir geht es gut";
    let target_2: &str = "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.";

    let mut input: Vec<String> = Vec::new();
    //input.push(input_str1.to_string());
    input.push(input_str2.to_string());

    let mut target: Vec<String> = Vec::new();
    //target.push(target_1.to_string());
    target.push(target_2.to_string());

    let batch_size = target.len();
    let dataset = Dataset::new(input, target);
    let num_epochs: usize = 5000;

    train(&mut transformer, dataset, num_epochs, batch_size);
    let seconds_elapsed_end = now.elapsed();

    println!("time elapsed in seconds: {:?}", seconds_elapsed_end - seconds_elapsed);
}
