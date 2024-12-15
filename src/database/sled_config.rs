use std::path::{Path, PathBuf};

pub const STORAGE_DIR: &str = "STORAGE";

pub const SLED_DB_TOKENIZER: &str = "SLED_TOKENIZER";
pub const SLED_DB_REVERSE_TOKENIZER: &str = "SLED_REVERSE_TOKENIZER";

pub const SLED_DB_TRANSFORMER: &str = "SLED_TRANSFORMER";

pub fn get_storage_path(path: &str) -> PathBuf {
    Path::new(STORAGE_DIR).join(path)
}
