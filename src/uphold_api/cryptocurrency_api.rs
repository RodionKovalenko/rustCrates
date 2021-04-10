extern crate reqwest;

use error_chain::*;
use std::io::Read;
use std::io::prelude::*;
use chrono::{DateTime, Timelike, Local, Datelike};
use crate::uphold_api::file_utils;
use crate::uphold_api::cryptocurrency_dto::CryptocurrencyDto;
use file_utils::get_or_create_file;

error_chain! {
    foreign_links {
        Io(std::io::Error);
        HttpRequest(reqwest::Error);
    }
}

const ZRX_EUR: &str = "ZRX-EUR";
const BAL_EUR: &str = "BAL-EUR";
const BTC_EUR: &str = "BTC-EUR";
const BCH_EUR: &str = "BCH-EUR";
const BTG_EUR: &str = "BTG-EUR";
const BTC0_EUR: &str = "BTC0-EUR";
const DASH_EUR: &str = "DASH-EUR";
const LINK_EUR: &str = "LINK-EUR";
const ADA_EUR: &str = "ADA-EUR";
const ATOM_EUR: &str = "ATOM-EUR";
const DGB_EUR: &str = "DGB-EUR";
const DOGE_EUR: &str = "DOGE-EUR";
const ETH_EUR: &str = "ETH-EUR";
const IOTA_EUR: &str = "IOTA-EUR";
const NEM_EUR: &str = "XEM-EUR";
const DOT_EUR: &str = "DOT-EUR";
const XLM_EUR: &str = "XLM-EUR";
const TRX_EUR: &str = "TRX-EUR";
const XRP_EUR: &str = "XRP-EUR";
const ZIL_EUR: &str = "ZIL-EUR";

pub const FILE_NAME: &str = "cryptocurrency_rates_history";
const FILE_FORMAT: &str = "json";

static CRYPTOCURRENCIES: [&str; 20] = [ZRX_EUR, BAL_EUR, BTC_EUR, BCH_EUR, BTG_EUR, DASH_EUR, LINK_EUR, ADA_EUR,
    BTC0_EUR, ATOM_EUR, DGB_EUR, DOGE_EUR, ETH_EUR, IOTA_EUR, NEM_EUR, DOT_EUR, XLM_EUR, TRX_EUR, XRP_EUR, ZIL_EUR];


// let mut res = reqwest::get("https://api.uphold.com/v0/ticker/BAL-EUR")?;
// let mut body = String::new();
// res.read_to_string(&mut body)?;

// println!("Status: {}", res.status());
// println!("Headers:\n{:#?}", res.headers());
// println!("Body:\n{}", body);
// println!("Ask: {}", body);
// println!("Bid: {}", body);

#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]
pub fn update_currency_prices_from_uphold_web_api() -> Result<()> {
    let mut data_array = self::get_data();
    let full_file_name = format!("{}.{}", FILE_NAME, FILE_FORMAT);

    println!("total size of records before: {}", data_array.len());

    // create new file to put all the data records
    let mut file = get_or_create_file(&full_file_name, true);

    let current_date: DateTime<Local> = Local::now();
    let day: u8 = current_date.day() as u8;
    let month: u8 = current_date.month() as u8;
    let year: u16 = current_date.year() as u16;
    let hour: u8 = current_date.hour() as u8;
    let minutes: u8 = current_date.minute() as u8;

    let mut request_url;
    let mut res;
    let mut body;

    for pair in &self::CRYPTOCURRENCIES {
        // println!("Pair {:?} ", &pair);
        request_url = format!("https://api.uphold.com/v0/ticker/{}", &pair);

        res = reqwest::get(&request_url)?;
        body = String::new();
        res.read_to_string(&mut body)?;

        let json: serde_json::Value =
            serde_json::from_str(&mut body).expect("JSON was not well-formatted");

        let data_record = CryptocurrencyDto {
            full_date: current_date,
            day: day,
            month: month,
            year: year,
            hour: hour,
            minute: minutes,
            currency: String::from(json.get("currency").unwrap().to_string().replace("\"", "")).parse().unwrap(),
            pair: String::from(pair.to_string()),
            ask: String::from(json.get("ask").unwrap().to_string().replace("\"", "")).parse().unwrap(),
            bid: String::from(json.get("bid").unwrap().to_string().replace("\"", "")).parse().unwrap(),
        };

        data_array.push(data_record);
    }

    let data_json = String::from(format!("{}", serde_json::to_string(&data_array).unwrap()));
    file.write_all(data_json.as_bytes())?;
    println!("total size of records after: {}", data_array.len());

    Ok(())
}

/**
   opens existing data to cryptocurrencies and parse json format to DataStructure struct in rust
*/
pub fn get_data() -> Vec<CryptocurrencyDto> {
    let full_file_name = format!("{}.{}", FILE_NAME, FILE_FORMAT);
    let mut file = file_utils::get_or_create_file(&full_file_name, false);
    let mut data = String::new();
    let mut json: Vec<CryptocurrencyDto> = vec![];

    file.read_to_string(&mut data).expect("Unable to open");

    if !data.is_empty() {
        json = serde_json::from_str(&data).expect("JSON was not well-formatted");
    }

    json
}