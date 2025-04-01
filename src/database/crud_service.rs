use sled::{self, Db, IVec};
use std::convert::TryInto;

use super::sled_db::{get_storage_path_embedding_db, SLED_DB_EMBEDDING};

pub fn insert_token(token: &str) -> Result<(), sled::Error> {
    // Example data to store: Text as key and number as value
    let key = token.as_bytes(); // The text key, converted to bytes
    let mut v: i64 = 1;

    let existing_value = get_value(token).unwrap();
    // Open a Sled database (it will create a new one if it doesn't exist)
    let db: Db = sled::open(get_storage_path_embedding_db(SLED_DB_EMBEDDING))?;

    if existing_value == 0 {
        let iter = db.iter().rev(); // Reverse iteration

        for entry in iter {
            let (_k, val) = entry?;
            v = convert_bytes_to_i64(val).unwrap();
            v += 1;
            break; // Only process the first (most recent) entry
        }

        let value: [u8; 8] = v.to_be_bytes(); // The number value, converted to bytes (using Big-Endian)

        // Convert the fixed-size array [u8; 4] into a slice and then to IVec
        let value_vec = sled::IVec::from(&value[..]); // Convert to a slice first

        // Insert the key-value pair into the database
        db.insert(key, value_vec)?;
    }

    // Explicitly drop the database (although it would be dropped automatically)
    drop(db);

    Ok(())
}

pub fn get_value(token: &str) -> Result<i64, sled::Error> {
    // Open a Sled database (it will create a new one if it doesn't exist)
    let db = sled::open(get_storage_path_embedding_db(SLED_DB_EMBEDDING))?;

    let key = token.as_bytes();
    let mut number: i64 = 0;

    // Retrieve the value back from the database using the key
    if let Some(retrieved_value) = db.get(key)? {
        // Ensure the length is correct (i.e., 4 bytes for an i32)
        number = convert_bytes_to_i64(retrieved_value).unwrap();
    }

    // Explicitly drop the database (although it would be dropped automatically)
    drop(db);

    Ok(number)
    // Close the database (it will be automatically flushed when it goes out of scope)
}

pub fn convert_bytes_to_i64(bytes: IVec) -> Result<i64, sled::Error> {
    if bytes.len() == 8 {
        // Try to convert the bytes into an i64 in big-endian format
        let number = i64::from_be_bytes(bytes.as_ref().try_into().map_err(|_| {
            sled::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to convert slice",
            ))
        })?);
        Ok(number)
    } else {
        Err(sled::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Expected 8 bytes for i64 conversion",
        )))
    }
}
