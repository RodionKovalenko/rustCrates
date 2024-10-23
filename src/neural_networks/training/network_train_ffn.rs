use std::collections::HashMap;
use std::time::Instant;
use crate::neural_networks::network_components::input;
use crate::neural_networks::network_components::input::Data;
use crate::neural_networks::network_types::feedforward_network;

use crate::neural_networks::network_types::feedforward_network_generic::FeedforwardNetwork;
use crate::neural_networks::utils::normalization::normalize_max_mean;
use crate::uphold_api::cryptocurrency_api::get_data;
use crate::uphold_api::cryptocurrency_dto::CryptocurrencyDto;

pub fn train_ffn() {
//     let now = Instant::now();
//     let num_iterations = 20000;
//     let mut data_structs = initalize_data_sets();
//     let minibatch_size = 50;
//     let mut layer_weights;

//     let mut feed_net: FeedforwardNetwork<f64> = feedforward_network::initialize_network(
//         &mut data_structs,
//         1,
//         40,
//         1,
//         minibatch_size,
//         0.001,
//     );

//     // Initialize weights here
//     for layer_ind in 0..feed_net.layers.len() {
//         layer_weights = &mut feed_net.layers[layer_ind].input_weights;

//         initialize_weights(10, 20, &mut layer_weights);
//     }

//     train(&mut data_structs, &mut feed_net, num_iterations);

//     println!("time elapsed {}", now.elapsed().as_secs());
// }


// pub fn initalize_data_sets() -> Vec<Data<f64, f64>> {
//     let cryptocurrency_data: Vec<CryptocurrencyDto> = get_data();
//     let mut input_data: Vec<Vec<f64>> = vec![];
//     let mut target_data: Vec<Vec<f64>> = vec![];
//     let mut data_structs: Vec<Data<f64, f64>> = vec![];
//     let mut _currency_to_num_map: HashMap<String, f64> = HashMap::new();
//     let mut input_struct;

//     for i in 0..cryptocurrency_data.len() {
//         let _date = cryptocurrency_data[i].full_date;
//         let currency: Vec<&str> = cryptocurrency_data[i].pair.split('-').collect();
//         let _char_vec: Vec<char> = currency[0].chars().collect();
//         let mut _currency_as_string: String = String::from("");
//         let _currency_as_num: f64;

//         if currency[0] == "ETH" {
//             // if !currency_to_num_map.contains_key(currency[0]) {
//             //     for c in 0..char_vec.len() {
//             //         let int_value = char_vec[c] as i32;
//             //         currency_as_string = format!("{}{}", currency_as_string.clone(), int_value);
//             //     }
//             //     currency_as_num = currency_as_string.parse::<f64>().unwrap();
//             //     currency_to_num_map.insert(String::from(currency[0]),
//             //                                currency_as_num);
//             // } else {
//             //     currency_as_num = *currency_to_num_map.get(currency[0].clone()).unwrap();
//             // }
//             //
//             // let date_number: f64 = (date.year() as f64 + (date.month() as f64 * 12.0)
//             //     + (date.day() as f64 * 30.0)
//             //     + (date.hour() as f64 * 60.0) + (date.minute() as f64)) as f64;
//             // input_data.push(vec![
//             //     date_number,
//             // ]);
//             //
//             // target_data.push(vec![cryptocurrency_data[i].bid as f64]);

//         }
//     }

//     input_data.push(vec![
//         0.05,
//         0.1
//     ]);
//     input_data.push(vec![
//         1.0,
//         0.5
//     ]);
//     input_data.push(vec![
//         0.3,
//         1.0
//     ]);
//     input_data.push(vec![
//         0.48,
//         0.33
//     ]);
//     input_data.push(vec![
//         0.87,
//         0.1
//     ]);
//     input_data.push(vec![
//         0.23,
//         1.0
//     ]);
//     target_data.push(vec![1.0]);
//     target_data.push(vec![0.5]);
//     target_data.push(vec![0.2]);
//     target_data.push(vec![0.4]);
//     target_data.push(vec![0.8]);
//     target_data.push(vec![0.1]);

//     // let mean_input = get_mean_2d(&input_data);
//     // let variance_input = get_variance_2d(&input_data, mean_input);
//     //
//     // let mean_target = get_mean_2d(&target_data);
//     // let variance_target = get_variance_2d(&target_data, mean_target);

//     let normalized_input_data: Vec<Vec<f64>> = normalize_max_mean(&input_data);
//     let normalized_target_data: Vec<Vec<f64>> = normalize_max_mean(&target_data);

//     println!("normalized input: {:?}", normalized_input_data);
//     println!("");
//     println!("normalized targets: {:?}", normalized_target_data);

//     for i in 0..input_data.len() {
//         input_struct = input::Data {
//             input: vec![normalized_input_data[i].clone()],
//             target: vec![normalized_target_data[i].clone()],
//         };
//         data_structs.push(input_struct);
//     }

//     println!("Data structs : {}", data_structs.len());

//     data_structs
}