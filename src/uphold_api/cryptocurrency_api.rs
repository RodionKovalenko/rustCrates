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

const AAVE_EUR: &str = "AAVE-EUR";
const ADA_EUR: &str = "ADA-EUR";
const ATOM_EUR: &str = "ATOM-EUR";
const BAL_EUR: &str = "BAL-EUR";
const BTC_EUR: &str = "BTC-EUR";
const BCH_EUR: &str = "BCH-EUR";
const BTG_EUR: &str = "BTG-EUR";
const BTC0_EUR: &str = "BTC0-EUR";
const COMP_EUR: &str = "COMP-EUR";
const DASH_EUR: &str = "DASH-EUR";
const DCR_EUR: &str = "DCR-EUR";
const EOS_EUR: &str = "EOS-EUR";
const ENJ_EUR: &str = "ENJ-EUR";
const ETH_EUR: &str = "ETH-EUR";
const FIL_EUR: &str = "FIL-EUR";
const FLOW_EUR: &str = "FLOW-EUR";
const HBAR_EUR: &str = "HBAR-EUR";
const HNT_EUR: &str = "HNT-EUR";
const IOTA_EUR: &str = "IOTA-EUR";
const LTC_EUR: &str = "LTC-EUR";
const MKR_EUR: &str = "MKR-EUR";
const NEO_EUR: &str = "NEO-EUR";
const NANO_EUR: &str = "NANO-EUR";
const MATIC_EUR: &str = "MATIC-EUR";
const REN_EUR: &str = "REN-EUR";
const SRM_EUR: &str = "SRM-EUR";
const SOL_EUR: &str = "SOL-EUR";
const SNX_EUR: &str = "SNX-EUR";
const XTZ_EUR: &str = "XTZ-EUR";
const GRT_EUR: &str = "GRT-EUR";
const THETA_EUR: &str = "THETA-EUR";
const UMA_EUR: &str = "UMA-EUR";
const UNI_EUR: &str = "UNI-EUR";
const VET_EUR: &str = "VET-EUR";
const WBTC_EUR: &str = "WBTC-EUR";
const DGB_EUR: &str = "DGB-EUR";
const DOGE_EUR: &str = "DOGE-EUR";
const DOT_EUR: &str = "DOT-EUR";
const LINK_EUR: &str = "LINK-EUR";
const NEM_EUR: &str = "XEM-EUR";
const TRX_EUR: &str = "TRX-EUR";
const XLM_EUR: &str = "XLM-EUR";
const XCH_EUR: &str = "XCH-EUR";
const XRP_EUR: &str = "XRP-EUR";
const ZIL_EUR: &str = "ZIL-EUR";
const ZRX_EUR: &str = "ZRX-EUR";

pub const FILE_NAME: &str = "cryptocurrency_rates_history";
const FILE_FORMAT: &str = "json";

static CRYPTOCURRENCIES: [&str; 46] = [
    AAVE_EUR, ADA_EUR, ATOM_EUR, BAL_EUR, BTC_EUR, BCH_EUR, BTG_EUR, BTC0_EUR, COMP_EUR, DASH_EUR,
    DCR_EUR, EOS_EUR, ENJ_EUR, ETH_EUR, FIL_EUR, FLOW_EUR, HBAR_EUR, HNT_EUR, IOTA_EUR, LTC_EUR,
    MKR_EUR, NEO_EUR, NANO_EUR, MATIC_EUR, REN_EUR, SRM_EUR, SOL_EUR, SNX_EUR, XTZ_EUR, GRT_EUR,
    THETA_EUR, UMA_EUR, UNI_EUR, VET_EUR, WBTC_EUR, DGB_EUR, DOGE_EUR, DOT_EUR, LINK_EUR, NEM_EUR,
    TRX_EUR, XLM_EUR, XCH_EUR, XRP_EUR, ZIL_EUR, ZRX_EUR,
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
    let mut json: Vec<CryptocurrencyDto> = vec![];

    file.read_to_string(&mut data).expect("Unable to open");

    if !data.is_empty() {
        json = serde_json::from_str(&data).expect("JSON was not well-formatted");
    }

    json
}
