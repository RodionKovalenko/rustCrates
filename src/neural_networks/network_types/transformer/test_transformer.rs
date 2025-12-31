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

    // let input_str1: &str = "Context: Die Verteidigung der Panthers gab nur 308 Punkte ab und belegte den sechsten Platz in der Liga, während sie die NFL mit 24 Interceptions in dieser Kategorie anführte und sich mit vier Pro Bowl-Selektionen rühmen konnte. Pro Bowl Defensive Tackle Kawann Short führte das Team mit 11 Sacks an, erzwang zudem drei Fumbles und erzielte zwei Fumble Recoverys. Mario Addison, ebenfalls Lineman, addierte 6½ Sacks hinzu. Die Panthers-Line präsentierte auch den erfahrenen Defensive End Jared Allen, einen 5-fachen Pro-Bowler, der mit 136 Sacks der aktive Anführer in der NFL-Kategorie Karriere-Sacks war, sowie den Defensive End Kony Ealy, der 5 Sacks in nur 9 Starts erzielte. Nach ihnen wurden zwei der drei Linebacker der Panthers ausgewählt, um im Pro Bowl zu spielen: Thomas Davis und Luke Kuechly. Davis erzielte 5½ Sacks, vier erzwungene Fumbles und vier Interceptions, während Kuechly das Team bei den Tackles anführte (118), zwei Fumbles erzwang und vier Pässe abfing. Carolinas Secondarys bestanden aus dem Pro Bowl-Safety Kurt Coleman, der das Team mit einem Karrierehoch von sieben Interceptions anführte und gleichzeitig 88 Tackles erzielen konnte, und Pro Bowl-Cornerback Josh Norman, der sich während der Saison zur Shutdown Corner entwickelte und vier Interceptions erzielte, von denen zwei zu Touchdowns für sein Team wurden. <sep> Question: Wie viele Punkte gab die Verteidigung der Panthers ab?";
    // let input_str2: &str = "Context: Die Verteidigung der Panthers gab nur 308 Punkte ab und belegte den sechsten Platz in der Liga, während sie die NFL mit 24 Interceptions in dieser Kategorie anführte und sich mit vier Pro Bowl-Selektionen rühmen konnte. Pro Bowl Defensive Tackle Kawann Short führte das Team mit 11 Sacks an, erzwang zudem drei Fumbles und erzielte zwei Fumble Recoverys. Mario Addison, ebenfalls Lineman, addierte 6½ Sacks hinzu. Die Panthers-Line präsentierte auch den erfahrenen Defensive End Jared Allen, einen 5-fachen Pro-Bowler, der mit 136 Sacks der aktive Anführer in der NFL-Kategorie Karriere-Sacks war, sowie den Defensive End Kony Ealy, der 5 Sacks in nur 9 Starts erzielte. Nach ihnen wurden zwei der drei Linebacker der Panthers ausgewählt, um im Pro Bowl zu spielen: Thomas Davis und Luke Kuechly. Davis erzielte 5½ Sacks, vier erzwungene Fumbles und vier Interceptions, während Kuechly das Team bei den Tackles anführte (118), zwei Fumbles erzwang und vier Pässe abfing. Carolinas Secondarys bestanden aus dem Pro Bowl-Safety Kurt Coleman, der das Team mit einem Karrierehoch von sieben Interceptions anführte und gleichzeitig 88 Tackles erzielen konnte, und Pro Bowl-Cornerback Josh Norman, der sich während der Saison zur Shutdown Corner entwickelte und vier Interceptions erzielte, von denen zwei zu Touchdowns für sein Team wurden. <sep> Question: Wie viele Sacks erzielte Jared Allen in seiner Karriere?";
    // let input_str3: &str = "Context: Die Verteidigung der Panthers gab nur 308 Punkte ab und belegte den sechsten Platz in der Liga, während sie die NFL mit 24 Interceptions in dieser Kategorie anführte und sich mit vier Pro Bowl-Selektionen rühmen konnte. Pro Bowl Defensive Tackle Kawann Short führte das Team mit 11 Sacks an, erzwang zudem drei Fumbles und erzielte zwei Fumble Recoverys. Mario Addison, ebenfalls Lineman, addierte 6½ Sacks hinzu. Die Panthers-Line präsentierte auch den erfahrenen Defensive End Jared Allen, einen 5-fachen Pro-Bowler, der mit 136 Sacks der aktive Anführer in der NFL-Kategorie Karriere-Sacks war, sowie den Defensive End Kony Ealy, der 5 Sacks in nur 9 Starts erzielte. Nach ihnen wurden zwei der drei Linebacker der Panthers ausgewählt, um im Pro Bowl zu spielen: Thomas Davis und Luke Kuechly. Davis erzielte 5½ Sacks, vier erzwungene Fumbles und vier Interceptions, während Kuechly das Team bei den Tackles anführte (118), zwei Fumbles erzwang und vier Pässe abfing. Carolinas Secondarys bestanden aus dem Pro Bowl-Safety Kurt Coleman, der das Team mit einem Karrierehoch von sieben Interceptions anführte und gleichzeitig 88 Tackles erzielen konnte, und Pro Bowl-Cornerback Josh Norman, der sich während der Saison zur Shutdown Corner entwickelte und vier Interceptions erzielte, von denen zwei zu Touchdowns für sein Team wurden. <sep> Question: Wie viele Tackles wurden bei Luke Kuechly registriert?";

    // let target_1: &str = "308";
    // let target_2: &str = "136";
    // let target_3: &str = "118";

    let input_str1: &str = "Was ist die Hauptstadt von Deutschland? Kannst du bitte eine kurze Antwort geben?";
    //Context: Am 28. Februar 2008 unterzeichneten Kibaki und Odinga eine Vereinbarung zur Gründung einer Koalitionsregierung, in welcher Odinga Kenias zweiter Premierminister werden sollte. Gemäß dieser Abmachung sollte der Präsident Kabinettminister aus den Lagern sowohl der PNU als auch der ODM ernennen, in Abhängigkeit davon, wie stark jede der Parteien im Parlament vertreten wäre. Die Vereinbarung legte fest, dass das Kabinett einen Vizepräsidenten und zwei stellvertretende Premierminister enthalten sollte. Nach Debatten wurde sie vom Parlament verabschiedet. Die Koalition sollte bis zum Ende des aktuellen Parlaments andauern oder früher enden, falls eine der beiden Parteien aus der Abmachung aussteigen würde. <sep> Question: Wann unterzeichneten Kibaki und Odinga eine Vereinbarung zur Bildung einer Regierung?
    // let input_str3:
    let input_str2: &str = "Context: Am 28. Februar 2008 unterzeichneten Kibaki und Odinga eine Vereinbarung zur Gründung einer Koalitionsregierung, in welcher Odinga Kenias zweiter Premierminister werden sollte. Gemäß dieser Abmachung sollte der Präsident Kabinettminister aus den Lagern sowohl der PNU als auch der ODM ernennen, in Abhängigkeit davon, wie stark jede der Parteien im Parlament vertreten wäre. Die Vereinbarung legte fest, dass das Kabinett einen Vizepräsidenten und zwei stellvertretende Premierminister enthalten sollte. Nach Debatten wurde sie vom Parlament verabschiedet. Die Koalition sollte bis zum Ende des aktuellen Parlaments andauern oder früher enden, falls eine der beiden Parteien aus der Abmachung aussteigen würde. <sep> Question: Wann unterzeichneten Kibaki und Odinga eine Vereinbarung zur Bildung einer Regierung?";
    // let input_str3: &str = "Was macht 2 + 3 aus?";

    let target_1: &str = "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.";
    let target_2: &str = "National Party";
    // let target_3: &str = "2 + 3 macht 5";

    let mut input: Vec<String> = Vec::new();
    input.push(input_str1.to_string());
    input.push(input_str2.to_string());
    // input.push(input_str3.to_string());

    let mut target: Vec<String> = Vec::new();
    target.push(target_1.to_string());
    target.push(target_2.to_string());
    // target.push(target_3.to_string());

    let _batch_size = target.len();
    let dataset = Dataset::new(input, target);
    let num_epochs: usize = 5000;

    train(&mut transformer, dataset, num_epochs, _batch_size);
    let seconds_elapsed_end = now.elapsed();

    println!("time elapsed in seconds: {:?}", seconds_elapsed_end - seconds_elapsed);
}
