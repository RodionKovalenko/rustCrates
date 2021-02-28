use std::thread;
use std::time::Duration;
use std::time::Instant;
#[path = "cryptocurrency_api.rs"]
mod cryptocurrency_api;

// 30 min = 1800 seconds
const COLLECT_PERIOD_IN_SECONDS: u64 = 10;
const BACKUP_PERIOD_IN_SECONDS: u64 = 20;

pub fn update_json_data_from_uphold_api() {
    let now = Instant::now();
    let time = now.elapsed();
    println!("{:?}", time);

    loop {
        println!("time elapsed {:?}", time);

        cryptocurrency_api::update_currency_prices_from_uphold_web_api().expect("cannot update json file");
        thread::sleep(Duration::from_secs(COLLECT_PERIOD_IN_SECONDS));
    }
}