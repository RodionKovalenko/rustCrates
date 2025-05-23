extern crate reqwest;

use crate::uphold_api::cryptocurrency_dto::CryptocurrencyDto;
use crate::uphold_api::file_utils;
use chrono::{DateTime, Datelike, Local, Timelike};
use file_utils::get_or_create_file;
use reqwest::Client;
use std::io::prelude::*;
use std::io::Read;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP Request Error: {0}")]
    HttpRequest(#[from] reqwest::Error),
}

pub const AAVE_EUR: &str = "AAVE-EUR";
pub const ADA_EUR: &str = "ADA-EUR";
pub const ATOM_EUR: &str = "ATOM-EUR";
pub const BAL_EUR: &str = "BAL-EUR";
pub const BTC_EUR: &str = "BTC-EUR";
pub const BCH_EUR: &str = "BCH-EUR";
pub const BTG_EUR: &str = "BTG-EUR";
pub const BTC0_EUR: &str = "BTC0-EUR";
pub const COMP_EUR: &str = "COMP-EUR";
pub const DASH_EUR: &str = "DASH-EUR";
pub const DCR_EUR: &str = "DCR-EUR";
pub const EOS_EUR: &str = "EOS-EUR";
pub const ENJ_EUR: &str = "ENJ-EUR";
pub const ETH_EUR: &str = "ETH-EUR";
pub const FIL_EUR: &str = "FIL-EUR";
pub const FLOW_EUR: &str = "FLOW-EUR";
pub const HBAR_EUR: &str = "HBAR-EUR";
pub const HNT_EUR: &str = "HNT-EUR";
pub const IOTA_EUR: &str = "IOTA-EUR";
pub const LTC_EUR: &str = "LTC-EUR";
pub const MKR_EUR: &str = "MKR-EUR";
pub const NEO_EUR: &str = "NEO-EUR";
pub const NANO_EUR: &str = "NANO-EUR";
pub const MATIC_EUR: &str = "MATIC-EUR";
pub const REN_EUR: &str = "REN-EUR";
pub const SRM_EUR: &str = "SRM-EUR";
pub const SOL_EUR: &str = "SOL-EUR";
pub const SNX_EUR: &str = "SNX-EUR";
pub const XTZ_EUR: &str = "XTZ-EUR";
pub const GRT_EUR: &str = "GRT-EUR";
pub const THETA_EUR: &str = "THETA-EUR";
pub const UMA_EUR: &str = "UMA-EUR";
pub const UNI_EUR: &str = "UNI-EUR";
pub const VET_EUR: &str = "VET-EUR";
pub const WBTC_EUR: &str = "WBTC-EUR";
pub const DGB_EUR: &str = "DGB-EUR";
pub const DOGE_EUR: &str = "DOGE-EUR";
pub const DOT_EUR: &str = "DOT-EUR";
pub const LINK_EUR: &str = "LINK-EUR";
pub const NEM_EUR: &str = "XEM-EUR";
pub const TRX_EUR: &str = "TRX-EUR";
pub const XLM_EUR: &str = "XLM-EUR";
pub const XCH_EUR: &str = "XCH-EUR";
pub const XRP_EUR: &str = "XRP-EUR";
pub const ZIL_EUR: &str = "ZIL-EUR";
pub const ZRX_EUR: &str = "ZRX-EUR";

pub const FILE_NAME: &str = "cryptocurrency_rates_history";
const FILE_FORMAT: &str = "json";

static CRYPTOCURRENCIES: [&str; 46] = [
    AAVE_EUR, ADA_EUR, ATOM_EUR, BAL_EUR, BTC_EUR, BCH_EUR, BTG_EUR, BTC0_EUR, COMP_EUR, DASH_EUR, DCR_EUR, EOS_EUR, ENJ_EUR, ETH_EUR, FIL_EUR, FLOW_EUR, HBAR_EUR, HNT_EUR, IOTA_EUR, LTC_EUR, MKR_EUR, NEO_EUR, NANO_EUR, MATIC_EUR, REN_EUR, SRM_EUR, SOL_EUR, SNX_EUR, XTZ_EUR, GRT_EUR, THETA_EUR,
    UMA_EUR, UNI_EUR, VET_EUR, WBTC_EUR, DGB_EUR, DOGE_EUR, DOT_EUR, LINK_EUR, NEM_EUR, TRX_EUR, XLM_EUR, XCH_EUR, XRP_EUR, ZIL_EUR, ZRX_EUR,
];

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
pub async fn update_currency_prices_from_uphold_web_api() -> Result<(), MyError> {
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
    let client = Client::new();

    for pair in &self::CRYPTOCURRENCIES {
        // println!("Pair {:?} ", &pair);
        request_url = format!("https://api.uphold.com/v0/ticker/{}", &pair);

        res = client.get(&request_url).send().await?;
        body = res.text().await?;

        let json: serde_json::Value = serde_json::from_str(&mut body).expect("JSON was not well-formatted");

        let data_record = CryptocurrencyDto {
            full_date: current_date,
            day,
            month,
            year,
            hour,
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
    let mut crypto_dtos: Vec<CryptocurrencyDto> = vec![];

    file.read_to_string(&mut data).expect("Unable to open");

    if !data.is_empty() {
        crypto_dtos = serde_json::from_str(&data).expect("JSON was not well-formatted");
    }

    crypto_dtos
}
