use chrono::NaiveDate;
use plotters::prelude::*;
use rand::seq::IndexedRandom;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::f64;
use std::io::Result;
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::uphold_api::cryptocurrency_dto::CryptocurrencyDto;
use crate::uphold_api::cryptocurrency_dto::DatePriceDto;

use super::stock_apis::get_data_from_yahoo;
use super::stock_apis::DATASET_DIR;

#[derive(Debug, Serialize, Deserialize)]
pub struct MonteCarloParams {
    pub num_simulations: i32,
    pub year_to_predict: i32,
}

fn load_historical_data(crypto_name: &str) -> Result<(Vec<String>, Vec<f64>)> {
    let filename = format!("{}_historical_data.json", crypto_name);
    let full_path = Path::new(DATASET_DIR).join(filename);

    let data_array: Vec<CryptocurrencyDto> = get_data_from_yahoo(full_path.to_str().unwrap(), crypto_name);
    let mut dates = vec![];
    let mut prices: Vec<f64> = vec![];

    println!("data_array: {:?}", data_array.len());
    println!("stock item: {:?}", data_array[0]);

    let mut filtered_cryptocurrencies: Vec<CryptocurrencyDto> = vec![];

    for crypto_currency_dto in data_array {
        if crypto_currency_dto.pair == crypto_name {
            filtered_cryptocurrencies.push(crypto_currency_dto);
        }
    }

    let data_price_dtos = interpolate_missing_dates(filtered_cryptocurrencies);

    //println!("crypto dtos: {:?}", data_price_dtos);

    for crypto_currency_dto in data_price_dtos {
        // Handling the formatting of full_date

        match crypto_currency_dto.full_date.format("%Y-%m-%d").to_string() {
            formatted_date => {
                dates.push(formatted_date);
                prices.push(crypto_currency_dto.price);
            }
        }
    }

    Ok((dates, prices))
}

pub fn interpolate_missing_dates(data: Vec<CryptocurrencyDto>) -> Vec<DatePriceDto> {
    use chrono::Duration; // Required for adding days to dates

    let mut data_map: HashMap<NaiveDate, f64> = HashMap::new();

    // Step 1: Load the data into a map for easy access
    for entry in data {
        let price = ((entry.ask + entry.bid) / 2.0) as f64;

        // Handling NaiveDate
        let date = entry.full_date.date_naive();
        data_map.insert(date, price);
    }

    // Step 2: Identify the missing dates and interpolate the values
    let mut result = Vec::new();
    let min_date = *data_map.keys().min().unwrap();
    let max_date = *data_map.keys().max().unwrap();

    let previous_date = min_date;
    let mut previous_price = data_map[&previous_date];

    // Step 3: Loop over all dates in the range from min_date to max_date
    let mut current_date = min_date;
    while current_date <= max_date {
        if let Some(&price) = data_map.get(&current_date) {
            // If the price is already available, just add it
            result.push(DatePriceDto { full_date: current_date, price });
            previous_price = price;
        } else {
            // Interpolate the missing value (linear interpolation)
            let mut next_date = current_date + Duration::days(1); // Start looking for the next available date
            while !data_map.contains_key(&next_date) && next_date <= max_date {
                next_date = next_date + Duration::days(1);
            }

            if let Some(&next_price) = data_map.get(&next_date) {
                // Perform linear interpolation
                let days_gap = (next_date - current_date).num_days() as f64;
                let interpolated_price = previous_price + (next_price - previous_price) / (days_gap + 1.0) * (1.0);

                data_map.insert(next_date, interpolated_price);

                // Add the interpolated date and price
                result.push(DatePriceDto { full_date: current_date, price: interpolated_price });
            }
        }

        // Move to the next day
        current_date = current_date + Duration::days(1);
    }

    //println!("result map : {:?}", &result);
    result
}

fn mean(data: &[f64]) -> f64 {
    data.iter().copied().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    let data_mean = mean(data);
    let variance = data.iter().map(|v| (v - data_mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

pub fn simulate(monte_carlo_params: MonteCarloParams, crypto_name: &str) {
    let (_dates, historical_prices) = match load_historical_data(crypto_name) {
        Ok((dates, prices)) => (dates, prices),
        Err(e) => {
            eprintln!("Error loading historical data: {}", e);
            return;
        }
    };

    let years_to_predict = monte_carlo_params.year_to_predict as usize;
    let n: usize = years_to_predict * 365; // crypto runs 365 days
    let m = monte_carlo_params.num_simulations as usize;
    let dt = 1.0 / 365.0;

    let historical_returns: Vec<f64> = historical_prices.windows(2).map(|pair| if pair[0] > 0.0 { (pair[1] / pair[0]).ln() } else { 0.0 }).collect();

    let s0 = *historical_prices.last().unwrap();
    let mu = mean(&historical_returns);
    let sigma = std_dev(&historical_returns);

    let seed = [0; 32];
    let mut rng = StdRng::from_seed(seed);

    let mut simulated_paths = vec![vec![0.0; n]; m];
    for path in simulated_paths.iter_mut() {
        path[0] = s0;
    }

    for i in 1..n {
        let sampled_returns: Vec<f64> = (0..m).map(|_| *historical_returns.choose(&mut rng).unwrap()).collect();

        for j in 0..m {
            let prev_price = simulated_paths[j][i - 1];
            let z = sampled_returns[j];
            let next_price = prev_price * ((mu - 0.5 * sigma.powi(2)) * dt + sigma * dt.sqrt() * z).exp();
            simulated_paths[j][i] = next_price.max(0.0);
        }
    }

    // Plotting using plotters as in your original code...
    let root = BitMapBackend::new("simulated_paths.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_value = simulated_paths.iter().flatten().copied().fold(f64::INFINITY, f64::min);
    let max_value = simulated_paths.iter().flatten().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Monte Carlo Simulation of Stock Prices", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..n as u32, min_value..max_value)
        .unwrap();

    chart.configure_mesh().x_desc("Time (days)").y_desc("Price").draw().unwrap();

    for path in &simulated_paths {
        let series: Vec<(u32, f64)> = path.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        chart.draw_series(LineSeries::new(series, &BLUE.mix(0.5))).unwrap();
    }

    // Analyze the final prices
    for i in 28..30 {
        let day_n = n - i;
        let final_prices: Vec<f64> = simulated_paths.iter().map(|p| p[day_n]).collect();

        let mean_price = mean(&final_prices);
        let std_dev_price = std_dev(&final_prices);

        let mut sorted_prices = final_prices.clone();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p5_price = sorted_prices[(0.05 * sorted_prices.len() as f64) as usize];
        let p95_price = sorted_prices[(0.95 * sorted_prices.len() as f64) as usize];

        println!("Day {}: Mean = {:.2}, Std Dev = {:.2}, 95% CI = ({:.2}, {:.2})", day_n, mean_price, std_dev_price, p5_price, p95_price);
    }
}
