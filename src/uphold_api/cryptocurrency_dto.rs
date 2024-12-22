use chrono::{DateTime, Local, NaiveDate};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CryptocurrencyDto {
    pub full_date: DateTime<Local>,
    pub day: u8,
    pub month: u8,
    pub year: u16,
    pub hour: u8,
    pub minute: u8,
    pub currency: String,
    pub pair: String,
    pub ask: f32,
    pub bid: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatePriceDto {
    pub full_date: NaiveDate,
    pub price: f64
}