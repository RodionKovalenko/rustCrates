use std::path::{Path, PathBuf};

use lazy_static::lazy_static;
use sled::Db;

pub const STORAGE_DIR: &str = "STORAGE";

pub const SLED_DB_EMBEDDING: &str = "SLED_EMBEDDING";
pub const SLED_DB_TRANSFORMER_V1: &str = "transformer_01.04.2025";

pub fn get_storage_path_embedding_db(filename: &str) -> PathBuf {
    let path = Path::new(STORAGE_DIR).join(filename);
    path
}

pub fn get_storage_path_transformer_db(filename: &str) -> PathBuf {
    // STORAGE/SLED_TRANSFORMER
    let path = Path::new(STORAGE_DIR).join(filename);
    println!("path in transformer: {:?}", path.to_str());
    path
}


lazy_static! {
    static ref DB_EMBEDDING: Db = sled::open(get_storage_path_embedding_db(SLED_DB_EMBEDDING)).expect("Failed to open SLED_EMBEDDING database");
}

pub fn get_db_embedding() -> &'static Db {
    &DB_EMBEDDING
}