#[cfg(test)]

mod tests {
    use std::time::Instant;

    use crate::{
        database::sled_db::SLED_DB_TRANSFORMER_V1,
        neural_networks::{
            network_components::input::{DataTrait, Dataset},
            network_types::{
                neural_network_generic::{get_from_db, print_networt_structure, update_learning_rate, NeuralNetwork, OperationMode},
                transformer::{
                    transformer_builder::create_transformer,
                    transformer_network::{predict, train},
                },
            },
            utils::tokenizer::detokenize,
        },
        utils::sampling_methods::{greedy_decoding, top_p_temperature_sampling},
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

        let learning_rate = 0.01;
        transformer.learning_rate = learning_rate;
        update_learning_rate(&mut transformer, learning_rate);
        print_networt_structure(&mut transformer);

        let seconds_elapsed = now.elapsed();
        println!("time elapsed in seconds: {:?}", &seconds_elapsed);

        let input_str1: &str = "Wie geht es dir?";
        //let input_str2: &str = "Was ist die Hauptstadt von Deutschland? Ich möchte es wissen";
        // let input_str3: &str = "Was kommt nach Donnerstag?";
        // let input_str4: &str = "Was macht 2+3 aus?";

        // let input_str5: &str = "Was macht 2+4?";
        // let input_str6: &str = "Was macht 2+5?";
        // let input_str7: &str = "Was macht 2+6?";
        // let input_str8: &str = "Was macht 2+7?";

        let target1: &str = " Mir geht es gut. Und wie geht es dir?";
        // let target2: &str = "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.
        //  Die Großstadt ist mit rund 3,7 Millionen Einwohnern die bevölkerungsreichste und mit 891 Quadratkilometern die flächengrößte
        //  Gemeinde Deutschlands sowie die bevölkerungsreichste Stadt der Europäischen Union.
        //  Berlin zählt zu den ökonomischen Zentren in Europa. Unter den wichtigen Zweigen der städtischen Wirtschaft sind der Tourismus,
        //   die Kreativ- und Kulturwirtschaft, die Biotechnologie und Gesundheitswirtschaft mit Medizintechnik und pharmazeutischer Industrie,
        //    die Informations- und Kommunikationstechnik, die Bau- und Immobilienwirtschaft, die Finanzwirtschaft,
        //     der Handel, die Optoelektronik, die Energietechnik, die Logistik sowie das Messe- und Kongresswesen.
        //     Die Stadt ist ein europäischer Verkehrsknotenpunkt des Straßen-, Schienen- und Luftverkehrs.
        //      Berlin ist ein internationaler Standort für innovative Unternehmensgründer und verzeichnet seit 2010 hohe Zuwachsraten bei der
        //       Zahl der Erwerbstätigen
        //  ";
        //  let target2: &str = "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.
        //  Die Großstadt ist mit rund 3,7 Millionen Einwohnern die bevölkerungsreichste und mit 891 Quadratkilometern die flächengrößte
        //  Gemeinde Deutschlands.";
        // let target3: &str = "Nach Donnerstag kommt Freitag.";
        // let target4: &str = "2 +3 macht 5";

        // let target5: &str = "2 + 4 macht 6";
        // let target6: &str = "2 + 5 macht 7";
        // let target7: &str = "2 + 6 macht 8";
        // let target8: &str = "2 + 7 macht 9";

        let mut input: Vec<String> = Vec::new();
        input.push(input_str1.to_string());
        //input.push(input_str2.to_string());
        // input.push(input_str3.to_string());
        // input.push(input_str4.to_string());
        // input.push(input_str5.to_string());
        // input.push(input_str6.to_string());
        // input.push(input_str7.to_string());
        // input.push(input_str8.to_string());

        let mut target: Vec<String> = Vec::new();
        target.push(target1.to_string());
        //target.push(target2.to_string());
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

        let learning_rate = 0.001;
        print_networt_structure(&mut transformer);
        update_learning_rate(&mut transformer, learning_rate);

        let seconds_elapsed = now.elapsed();
        println!("time elapsed in seconds: {:?}", &seconds_elapsed);

        let input_str2: &str = "Was ist die Hauptstadt von Deutschland? Ich möchte es wissen";
        let target2: &str = "Berlin";
        let mut input: Vec<String> = Vec::new();
        input.push(input_str2.to_string());

        let mut target: Vec<String> = Vec::new();
        target.push(target2.to_string());

        let predicted_softmax_targets: Vec<Vec<Vec<f64>>> = predict(&mut transformer, &input, 0);

        let p = 0.9; // Top-p (Nucleus) threshold
        let temperature = 0.7; // Temperature for controlling randomness

        //println!("predicted softmax targets at 0: {:?}", &predicted_softmax_targets[0][0][0..100]);
        let sampled_tokens = top_p_temperature_sampling(&predicted_softmax_targets, p, temperature);
        println!("Top-p + Temperature Sampling: {:?}", sampled_tokens);

        let predicted_token_batch: Vec<String> = sampled_tokens.iter().map(|token_indices| detokenize(token_indices).unwrap()).collect();
        println!("predicted token tempareture sampling: {:?}", predicted_token_batch);

        let sampled_greedy_tokens = greedy_decoding(&predicted_softmax_targets);
        let predicted_token_batch: Vec<String> = sampled_greedy_tokens.iter().map(|token_indices| detokenize(token_indices).unwrap()).collect();
        println!("predicted token highest index: {:?}", predicted_token_batch);
    }
}
