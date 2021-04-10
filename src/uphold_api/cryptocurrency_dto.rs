use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
#[derive(Serialize)]
#[derive(Deserialize)]
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