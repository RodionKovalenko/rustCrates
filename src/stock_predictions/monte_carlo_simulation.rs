use chrono::NaiveDate;
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use rand::seq::SliceRandom;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::f64;
use std::io::Result;
use std::path::Path;

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
            result.push(DatePriceDto {
                full_date: current_date,
                price,
            });
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
                let interpolated_price =
                    previous_price + (next_price - previous_price) / (days_gap + 1.0) * (1.0);

                data_map.insert(next_date, interpolated_price);

                // Add the interpolated date and price
                result.push(DatePriceDto {
                    full_date: current_date,
                    price: interpolated_price,
                });
            }
        }

        // Move to the next day
        current_date = current_date + Duration::days(1);
    }

    //println!("result map : {:?}", &result);
    result
}

pub fn simulate(monte_carlo_params: MonteCarloParams, crypto_name: &str) {
    let (_dates, historical_prices) = match load_historical_data(crypto_name) {
        Ok((dates, prices)) => (dates, prices),
        Err(e) => {
            eprintln!("Error loading historical data: {}", e);
            return;
        }
    };

    println!("historical prices len: {}", historical_prices.len());

    let years_to_predict = monte_carlo_params.year_to_predict as usize; // Simulating for n years
    let n: usize = (years_to_predict * 252) as usize; // 252 trading days per year or 365 days for cryptos
    let m = monte_carlo_params.num_simulations as usize; // Number of simulations (Monte Carlo paths)
    let dt = 1.0 / 365.0 as f64; // Daily time increment

    // Step 1: Calculate daily log returns
    let historical_returns: Vec<f64> = historical_prices
        .windows(2)
        .map(|pair| {
            if pair[0] > 0.0 {
                (pair[1] / pair[0]).ln()
            } else {
                0.0 // Avoid division by zero if price is zero (though unlikely in real data)
            }
        })
        .collect();

    println!("historical prices : {:?}", &historical_prices.len());

    // Monte Carlo Simulation Parameters
    let s0 = *historical_prices.last().unwrap(); // Last known price

    // Step 2: Simulate future paths
    let mut simulated_paths = Array2::<f64>::zeros((m, n));
    simulated_paths.column_mut(0).fill(s0);

    let mut rng = rand::thread_rng();
    let historical_data_nd = Array1::from(historical_returns.clone());
    let mu = historical_data_nd.mean().unwrap();
    let sigma = historical_data_nd.std(1.0);

    println!("mu : {}", &mu);
    println!("var: {}", &sigma);
    println!("dt: {}", &dt);

    assert!(
        historical_returns.iter().all(|&x| x > -1.0),
        "Log returns have invalid values."
    );

    for i in 1..n {
        let sampled_returns: Vec<f64> = (0..m)
            .map(|_| *historical_returns.choose(&mut rng).unwrap())
            .collect();

        for j in 0..m {
            let prev_price = simulated_paths[[j, i - 1]];
            let z = sampled_returns[j];
            let sampled_price = prev_price * ((mu - 0.5 * sigma.powf(2.0)) * dt + sigma * dt.sqrt() * z).exp();

            if sampled_price <= 0.0 {
                panic!("simulated prices cannot be null {}", sampled_price);
            }
            simulated_paths[[j, i]] = sampled_price;
        }
    }

    // println!("simulated paths: {:?}", &simulated_paths);

    // Step 3: Plot the simulated paths
    let root = BitMapBackend::new("simulated_paths.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_value = simulated_paths
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_value = simulated_paths
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("min value {}", &min_value);
    println!("max_value {}", &max_value);

    let mut chart = ChartBuilder::on(&root)
        .caption("Monte Carlo Simulation of Stock Prices", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..n as u32, min_value..max_value)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Time (days)")
        .y_desc("Stock Price")
        .draw()
        .unwrap();

    for i in 0..m {
        let path: Vec<(u32, f64)> = (0..n)
            .map(|j| (j as u32, simulated_paths[[i, j]]))
            .collect();
        chart
            .draw_series(LineSeries::new(path, &BLUE.mix(0.5)))
            .unwrap();
    }

    // Step 4: Analyze the final prices

    for i in 28..30 {
        let day_n = n - i;
        let final_prices: Array1<f64> = simulated_paths.column(day_n).to_owned();

        println!("day {}", &day_n);

        let mean_price = final_prices.mean().unwrap();
        let std_dev_price = final_prices.std(1.0);
        let mut final_prices_vec = final_prices.to_vec();
        final_prices_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p5_price = final_prices_vec[(0.05 * final_prices_vec.len() as f64) as usize];
        let p95_price = final_prices_vec[(0.95 * final_prices_vec.len() as f64) as usize];

        println!(
            "Predicted stock price after 1 year (mean): ${:.2}",
            mean_price
        );
        println!("Standard deviation of predictions: ${:.2}", std_dev_price);
        println!(
            "95% confidence interval: (${:.2}, ${:.2})",
            p5_price, p95_price
        );
    }
}
