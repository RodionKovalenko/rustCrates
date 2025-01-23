use std::{io::Read, path::Path};

use crate::{neural_networks::utils::file::get_or_create_file, uphold_api::{cryptocurrency_dto::CryptocurrencyDto, file_utils}};
use chrono::{DateTime, Datelike, Local, Timelike};
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use time::{Date, Month, OffsetDateTime, Time};
use yahoo_finance_api::YahooConnector;

pub const DATASET_DIR: &str = "datasets";

#[derive(Clone, Serialize, Deserialize)]
struct Quote {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64
}

impl CryptocurrencyDto {
    fn from(quote: Quote, stock_ticker_sym: &str) -> Self {
        // Parse the timestamp string to DateTime
        let timestamp:DateTime<chrono::Utc> = DateTime::from_timestamp(quote.timestamp.parse::<i64>().unwrap(), 0).unwrap();
        let full_date: DateTime<Local> =  timestamp.with_timezone(&Local);

        CryptocurrencyDto {
            full_date,
            day: full_date.day() as u8,
            month: full_date.month() as u8,
            year: full_date.year() as u16,
            hour: full_date.hour() as u8,
            minute: full_date.minute() as u8,
            currency: "EUR".to_string(), // or set to a dynamic value if necessary
            pair: stock_ticker_sym.to_string(), // or set to a dynamic value if necessary
            ask: quote.open as f32, // Assuming you want to use the 'open' price as 'ask'
            bid: quote.close as f32, // Assuming you want to use the 'close' price as 'bid'
        }
    }
}

#[tokio::main]
pub async fn fetch_and_save_historical_data(stock_name: &str) {
    let filename = format!("{}_historical_data.json", stock_name);
    let full_path = Path::new(DATASET_DIR).join(filename);

    println!("full path : {:?}", &full_path);
    
    let data_file = get_or_create_file(full_path.to_str().unwrap(), false);
    // Create a Yahoo Finance connector
    let provider = YahooConnector::new().expect("Failed to create YahooConnector");

    let date = Date::from_calendar_date(2021, Month::January, 1).expect("Invalid date");
    let time = Time::from_hms(0, 0, 0).expect("Invalid time");
    let start_date = OffsetDateTime::new_utc(date, time);

    let current_date = OffsetDateTime::now_utc().date();
    let time = Time::from_hms(0, 0, 0).expect("Invalid time");
    let end_date = OffsetDateTime::new_utc(current_date, time);

    // Fetch historical data for BTC-EUR (Bitcoin to Euro)
    let response = provider
        .get_quote_history(&stock_name, start_date, end_date)
        .await;

    match response {
        Ok(quotes) => {
            // Collect the quotes into a Vec of Quote structs
            let quote_data: Vec<Quote> = quotes
                .quotes()
                .unwrap()
                .into_iter()
                .map(|quote| Quote {
                    timestamp: quote.timestamp.to_string(),
                    open: quote.open,
                    high: quote.high,
                    low: quote.low,
                    close: quote.close,
                    volume: quote.volume,
                })
                .collect();

            // Write the collected data to the JSON file
            to_writer(data_file, &quote_data).unwrap();
            println!("Data written to BTC-EUR_historical_data.json");
        }
        Err(e) => eprintln!("Error fetching data: {:?}", e),
    }
}


pub fn get_data_from_yahoo(filename: &str, stock_ticker_sym: &str) -> Vec<CryptocurrencyDto> {
    let mut file = file_utils::get_or_create_file(&filename, false);
    let mut data = String::new();
    let mut data_crypto_dtos: Vec<CryptocurrencyDto> = vec![];

    file.read_to_string(&mut data).expect("Unable to open");

    if !data.is_empty() {
        let data_quotes: Vec<Quote> = serde_json::from_str(&data).expect("JSON was not well-formatted");

        for quote_dto in data_quotes.iter() {
            // Convert &Quote to CryptocurrencyDto using .into()
            data_crypto_dtos.push(CryptocurrencyDto::from((*quote_dto).clone(), stock_ticker_sym));
        }
    }


    data_crypto_dtos
}
