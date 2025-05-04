#[cfg(test)]

mod tests {
    use std::time::Instant;

    use crate::{
        database::sled_db::SLED_DB_TRANSFORMER_V1,
        neural_networks::{
            network_components::input::{load_data_xquad_de_as_dataset, DataTrait, Dataset},
            network_types::{
                neural_network_generic::{get_from_db, print_networt_structure, update_learning_rate, NeuralNetwork, OperationMode},
                transformer::{
                    transformer_builder::create_transformer,
                    transformer_network::{predict_token_by_token, train},
                },
            },
        },
    };

    #[test]
    #[ignore]
    fn test_train_transformer() {
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
        let input_str2: &str = "Was ist die Hauptstadt von Deutschland? Ich möchte es wissen";
        // let input_str3: &str = "Was kommt nach Donnerstag?";
        // let input_str4: &str = "Was macht 2+3 aus?";

        // let input_str5: &str = "Was macht 2+4?";
        // let input_str6: &str = "Was macht 2+5?";
        // let input_str7: &str = "Was macht 2+6?";
        // let input_str8: &str = "Was macht 2+7?";

        //let target1: &str = " Mir geht es gut.";
        let target2: &str = "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.";
        // let target3: &str = "Nach Donnerstag kommt Freitag.";
        // let target4: &str = "2 +3 macht 5";

        // let target5: &str = "2 + 4 macht 6";
        // let target6: &str = "2 + 5 macht 7";
        // let target7: &str = "2 + 6 macht 8";
        // let target8: &str = "2 + 7 macht 9";

        let mut input: Vec<String> = Vec::new();
        //input.push(input_str1.to_string());
        input.push(input_str2.to_string());
        // input.push(input_str3.to_string());
        // input.push(input_str4.to_string());
        // input.push(input_str5.to_string());
        // input.push(input_str6.to_string());
        // input.push(input_str7.to_string());
        // input.push(input_str8.to_string());

        let mut target: Vec<String> = Vec::new();
        //target.push(target1.to_string());
        target.push(target2.to_string());
        // target.push(target3.to_string());
        // target.push(target4.to_string());
        // target.push(target5.to_string());
        // target.push(target6.to_string());
        // target.push(target7.to_string());
        // target.push(target8.to_string());

        let dataset = Dataset::new(input, target);
        let num_epochs: usize = 5000;

        train(&mut transformer, dataset, num_epochs);
        let seconds_elapsed_end = now.elapsed();

        println!("time elapsed in seconds: {:?}", seconds_elapsed_end - seconds_elapsed);
    }

    #[test]
    #[ignore]
    fn test_predict_transformer() {
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

        let seconds_elapsed = now.elapsed();
        println!("time elapsed in seconds: {:?}", &seconds_elapsed);

        // let input_str2: &str = "Context: Die Verteidigung der Panthers gab nur 308 Punkte ab und belegte den sechsten Platz in der Liga,
        //  während sie die NFL mit 24 Interceptions in dieser Kategorie anführte und sich mit vier Pro Bowl-Selektionen rühmen konnte.
        //   Pro Bowl Defensive Tackle Kawann Short führte das Team mit 11 Sacks an, erzwang zudem drei Fumbles und erzielte zwei Fumble Recoverys. Mario Addison,
        // ebenfalls Lineman, addierte 6½ Sacks hinzu. Die Panthers-Line präsentierte auch den erfahrenen Defensive End Jared Allen, einen 5-fachen Pro-Bowler,
        //  der mit 136 Sacks der aktive Anführer in der NFL-Kategorie Karriere-Sacks war, sowie den Defensive End Kony Ealy, der 5 Sacks in nur 9 Starts erzielte.
        //   Nach ihnen wurden zwei der drei Linebacker der Panthers ausgewählt, um im Pro Bowl zu spielen: Thomas Davis und Luke Kuechly.
        //    Davis erzielte 5½ Sacks, vier erzwungene Fumbles und vier Interceptions, während Kuechly das Team bei den Tackles anführte (118),
        //     zwei Fumbles erzwang und vier Pässe abfing. Carolinas Secondarys bestanden aus dem Pro Bowl-Safety Kurt Coleman, der das Team mit einem Karrierehoch
        //      von sieben Interceptions anführte und gleichzeitig 88 Tackles erzielen konnte, und Pro Bowl-Cornerback Josh Norman, der sich während der Saison zur
        //       Shutdown Corner entwickelte und vier Interceptions erzielte, von denen zwei zu Touchdowns für sein Team wurden.
        //        \n <sep> Question: Wie viele Punkte gab die Verteidigung der Panthers ab? ";

        let input_str2: &str = "Context: Die Verteidigung der Panthers gab nur 308 Punkte ab und belegte den sechsten Platz in der Liga
        während sie die NFL mit 24 Interceptions in dieser Kategorie anführte und sich mit vier Pro Bowl-Selektionen rühmen konnte.";

        let mut input: Vec<String> = Vec::new();
        input.push(input_str2.to_string());

        let (_predicted_softmax_targets, all_predicted_tokens) = predict_token_by_token(&mut transformer, &input);

        println!("Input: {:?}", input);
        println!("Predictions token by token: {:?}", all_predicted_tokens);

        let seconds_elapsed_end = now.elapsed();

        println!("time elapsed in seconds: {:?}", seconds_elapsed_end - seconds_elapsed);
    }

    #[test]
    #[ignore]
    fn test_train_transformer_from_dataset() {
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
}
