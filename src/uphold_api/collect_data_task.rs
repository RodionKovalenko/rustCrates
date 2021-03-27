use std::thread;
use std::time::Duration;
use std::time::Instant;
#[path = "cryptocurrency_api.rs"]
mod cryptocurrency_api;
#[path = "file_utils.rs"]
mod file_utils;

// 30 min = 1800 seconds
const COLLECT_PERIOD_IN_SECONDS: u64 = 1800;
// every 3 hours make backup
const BACKUP_PERIOD_IN_SECONDS: u64 = 10800;

pub fn update_json_data_from_uphold_api() {
    let mut now = Instant::now();
    let mut seconds_elapsed = now.elapsed().as_secs();
    let mut thread1_uphold_data_collect;
    let mut thread2_backup;

    loop {
        seconds_elapsed = now.elapsed().as_secs();

        if seconds_elapsed % COLLECT_PERIOD_IN_SECONDS == 0 {
            thread1_uphold_data_collect = thread::spawn(move || {
            
                seconds_elapsed = now.elapsed().as_secs();

                println!("");
                println!("this is thread number 1__________________start");
                println!("time elapsed {:?}", seconds_elapsed);

                cryptocurrency_api::update_currency_prices_from_uphold_web_api().expect("cannot update json file");

                println!("this is thread number 1__________________finish");
                println!("");
            });
        }

        if seconds_elapsed % BACKUP_PERIOD_IN_SECONDS == 0 {
            thread2_backup = thread::spawn(move || {
                seconds_elapsed = now.elapsed().as_secs();

                println!("");
                println!("this is thread number 2__________________start");
                println!("time elapsed {:?}", seconds_elapsed);

                file_utils::make_backup_file(cryptocurrency_api::FILE_NAME);

                println!("this is thread number 2__________________finish");
                println!("");
            });
        }

        thread::sleep(Duration::from_secs(COLLECT_PERIOD_IN_SECONDS));
    }
}