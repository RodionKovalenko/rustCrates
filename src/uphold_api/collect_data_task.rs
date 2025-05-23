use std::thread;
use std::time::Duration;
use std::time::Instant;
use crate::uphold_api::{crypto_predictions, cryptocurrency_api, file_utils};

// 30 min = 1800 seconds
const COLLECT_PERIOD_IN_SECONDS: u64 = 1800;
// every 3 hours make backup
const BACKUP_PERIOD_IN_SECONDS: u64 = 10800;
// 24 hours
const CRYPTO_PREDICTIONS_IN_SECONDS: u64 = 86400;

#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]
pub fn update_json_data_from_uphold_api() {
    let now = Instant::now();
    let mut seconds_elapsed;

    loop {
        seconds_elapsed = now.elapsed().as_secs();

        if seconds_elapsed % COLLECT_PERIOD_IN_SECONDS == 0 {
            thread::spawn(move || {
                seconds_elapsed = now.elapsed().as_secs();

                println!("");
                println!("this is thread number 1__________________start");
                println!("time elapsed {:?}", seconds_elapsed);

                let runtime = tokio::runtime::Runtime::new().unwrap();

                // Block on the async function and get its result
                let result = runtime.block_on(cryptocurrency_api::update_currency_prices_from_uphold_web_api()).unwrap();

                println!("this is thread number 1__________________finish");
                println!("");
            });
        }

        seconds_elapsed = now.elapsed().as_secs();

        if seconds_elapsed % BACKUP_PERIOD_IN_SECONDS == 0 {
            thread::spawn(move || {
                seconds_elapsed = now.elapsed().as_secs();

                println!("");
                println!("this is thread number 2__________________start");
                println!("time elapsed {:?}", seconds_elapsed);

                file_utils::make_backup_file(cryptocurrency_api::FILE_NAME);

                println!("this is thread number 2__________________finish");
                println!("");
            });
        }

        seconds_elapsed = now.elapsed().as_secs();

        if seconds_elapsed % CRYPTO_PREDICTIONS_IN_SECONDS == 0 {
            thread::spawn(move || {
                seconds_elapsed = now.elapsed().as_secs();

                println!("");
                println!("this is thread predictions number 3__________________start");
                crypto_predictions::make_prediction_daily();
                println!("");
                println!("this is thread predictions number finish");
            });
        }

        thread::sleep(Duration::from_secs(COLLECT_PERIOD_IN_SECONDS));
    }
}