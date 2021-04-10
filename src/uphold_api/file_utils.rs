use std::fs::File;
use std::fs::OpenOptions;
use std::path::Path;
use std::io::Write;
use std::io::Read;
use chrono::{DateTime, Timelike, Local, Datelike};

const BACKUP_DIR: &str = "backup";
const FILE_FORMAT_BACK_UP: &str = "json";

pub fn get_or_create_file(filename: &str, create: bool) -> File {
    let path =  Path::new(&filename);

    if !path.exists() || create {
       create_new_file(filename);
    }

    let f = OpenOptions::new()
        .write(true)
        .read(true)
        .open(&filename)
        .unwrap();

        f
}

pub fn create_new_file(filename: &str) {
    File::create(&filename).expect("cannot create file");
}

pub fn make_backup_file(filename: &str) {
    let full_file_name = format!("{}.{}", &filename, FILE_FORMAT_BACK_UP);
    let path_to_file = Path::new(&full_file_name);

    let current_date: DateTime<Local> = Local::now();
    let day: u8 = current_date.day() as u8;
    let month: u8 = current_date.month() as u8;
    let year: u16 = current_date.year() as u16;
    let hour: u8 = current_date.hour() as u8;
    let minutes: u8 = current_date.minute() as u8;

    let backup_time_stamp = format!("{}-{}-{}-{}-{}", year, month, day, hour, minutes);
    let backup_time_dir = format!("{}/{}_{}", BACKUP_DIR, &filename, backup_time_stamp);
    let backup_file_name = format!("{}.{}", &backup_time_dir, FILE_FORMAT_BACK_UP);
    let path_to_backup = Path::new(&backup_file_name);

    if !path_to_backup.exists() {
        std::fs::create_dir_all(BACKUP_DIR).expect("Could not create directory");
    }

    let mut backup_file: File = get_or_create_file(&backup_file_name, true);

    if path_to_file.exists() {
        let mut file: File = get_or_create_file(&full_file_name, false);
        let mut data = String::new();

        file.read_to_string(&mut data).expect("Unable to open");

        if !data.is_empty() {
            // create new backup file to put in data in
            println!("{:?}", backup_file_name);
            backup_file.write_all(data.as_bytes()).expect("Could not write to file");
        }
    }
}