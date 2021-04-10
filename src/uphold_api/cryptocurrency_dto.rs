use chrono::{DateTime, Local};

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