use std::path::{Path, PathBuf};

use lazy_static::lazy_static;
use sled::Db;

pub const STORAGE_DIR: &str = "STORAGE";

pub const SLED_DB_TOKENIZER: &str = "SLED_EMBEDDING";
pub const SLED_DB_REVERSE_TOKENIZER: &str = "SLED_REVERSE_TOKENIZER";
pub const SLED_DB_TRANSFORMER: &str = "31032025092609280930";

pub fn get_storage_path(path: &str) -> PathBuf {
    let path = Path::new(STORAGE_DIR).join(path);
    path   
}

lazy_static! {
    static ref DB: Db = sled::open(get_storage_path(SLED_DB_TOKENIZER)).expect("failed to open database in the embedding layer");
}

pub fn get_db() -> &'static Db {
    &DB
}