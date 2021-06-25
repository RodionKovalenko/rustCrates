extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate neural_networks;

use std::env;

#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]
use neural_networks::network_types::*;
use neural_networks::network_components::*;

pub mod uphold_api;

#[allow(unused_imports)]
use uphold_api::*;
use uphold_api::cryptocurrency_dto::*;
#[allow(unused_imports)]
use neural_networks::utils::normalization::*;
#[allow(unused_imports)]
use std::time::Instant;
use crate::uphold_api::cryptocurrency_api::get_data;
use chrono::{Datelike, Timelike, DurationRound};
use std::collections::HashMap;
use neural_networks::network_components::input::Data;
#[allow(unused_imports)]
use neural_networks::network_types::feedforward_network::train;
#[allow(unused_imports)]
use neural_networks::network_types::feedforward_network_generic::FeedforwardNetwork;
use rand::Rng;

pub enum ARGUMENTS {
    UPHOLD,
    NETWORK,
}

/**
* start with cargo run -- --server
* connect client: cargo run -- --client
tcp::test_connection();
 */
fn main() {
    println!("Test beginns");

    let args: Vec<String> = env::args().collect();
    println!("{:?}", args[1]);

    if args.len() > 1 {
        let arg1: &str = &*args[1].clone();

        match arg1 {
            "uphold" => collect_data_task::update_json_data_from_uphold_api(),
            "network" => start_neural_network(),
            _ => println!(" no argument recognized"),
        }
    } else {
        // default
        collect_data_task::update_json_data_from_uphold_api();
    }
}

pub fn start_neural_network() {
    let now = Instant::now();
    let num_iterations = 20000;
    let mut data_structs = initalize_data_sets();
    let minibatch_size = 50;

    let mut feed_net: FeedforwardNetwork<f64> = feedforward_network::initialize_network(
        &mut data_structs,
        1,
        40,
        1,
        minibatch_size,
        0.001,
    );

    train(&mut data_structs, &mut feed_net, minibatch_size, num_iterations);

    println!("time elapsed {}", now.elapsed().as_secs());
}

pub fn initalize_data_sets() -> Vec<Data<f64>> {
    let cryptocurrency_data: Vec<CryptocurrencyDto> = get_data();
    let mut input_data: Vec<Vec<f64>> = vec![];
    let mut target_data: Vec<Vec<f64>> = vec![];
    let mut data_structs: Vec<Data<f64>> = vec![];
    let mut currency_to_num_map: HashMap<String, f64> = HashMap::new();
    let mut input_struct;
    let mut rng = rand::thread_rng();

    for i in 0..cryptocurrency_data.len() {
        let date = cryptocurrency_data[i].full_date;
        let currency: Vec<&str> = cryptocurrency_data[i].pair.split('-').collect();
        let char_vec: Vec<char> = currency[0].chars().collect();
        let mut currency_as_string: String = String::from("");
        let currency_as_num;

        if currency[0] == "ETH" {
            if !currency_to_num_map.contains_key(currency[0]) {
                for c in 0..char_vec.len() {
                    let int_value = char_vec[c] as i32;
                    currency_as_string = format!("{}{}", currency_as_string.clone(), int_value);
                }
                currency_as_num = currency_as_string.parse::<f64>().unwrap();
                currency_to_num_map.insert(String::from(currency[0]),
                                           currency_as_num);
            } else {
                currency_as_num = *currency_to_num_map.get(currency[0].clone()).unwrap();
            }

            let date_number: f64 = (date.year() as f64 + (date.month() as f64 * 12.0)
                + (date.day() as f64 * 30.0)
                + (date.hour() as f64 * 60.0) + (date.minute() as f64)) as f64;
            input_data.push(vec![
                date_number,
            ]);

            target_data.push(vec![cryptocurrency_data[i].bid as f64]);

            // input_data.push(vec![
            //     rng.gen_range(0.0, 1.0) as f64,
            //     rng.gen_range(0.0, 1.0) as f64,
            //     rng.gen_range(0.0, 1.0) as f64,
            //     rng.gen_range(0.0, 1.0) as f64,
            //     rng.gen_range(0.0, 1.0) as f64,
            //     rng.gen_range(0.0, 1.0) as f64,
            // ]);
            // target_data.push(vec![rng.gen_range(0.0,1.0) as f64]);
        }
    }

    // input_data.push(vec![
    //     0.05,
    //     0.1
    // ]);
    // input_data.push(vec![
    //     1.0,
    //     0.5
    // ]);
    // input_data.push(vec![
    //     0.3,
    //     1.0
    // ]);
    // input_data.push(vec![
    //     0.48,
    //     0.33
    // ]);
    // input_data.push(vec![
    //     0.87,
    //     0.1
    // ]);
    // input_data.push(vec![
    //     0.23,
    //     1.0
    // ]);
    // target_data.push(vec![1.0]);
    // target_data.push(vec![0.5]);
    // target_data.push(vec![0.2]);
    // target_data.push(vec![0.4]);
    // target_data.push(vec![0.8]);
    // target_data.push(vec![0.1]);

    // let mean_input = get_mean_2d(&input_data);
    // let variance_input = get_variance_2d(&input_data, mean_input);
    //
    // let mean_target = get_mean_2d(&target_data);
    // let variance_target = get_variance_2d(&target_data, mean_target);

    let normalized_input_data: Vec<Vec<f64>> = normalize_max_mean(&input_data);
    let normalized_target_data: Vec<Vec<f64>> = normalize_max_mean(&target_data);

    println!("normalized input: {:?}", normalized_input_data);
    println!("");
    println!("normalized targets: {:?}", normalized_target_data);

    for i in 0..input_data.len() {
        input_struct = input::Data {
            input: vec![normalized_input_data[i].clone()],
            target: vec![normalized_target_data[i].clone()],
        };
        data_structs.push(input_struct);
    }

    println!("Data structs : {}", data_structs.len());

    data_structs
}