use crate::neural_networks::network_types::feedforward_network;
use crate::neural_networks::network_types::feedforward_network_generic::FeedforwardNetwork;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{fs, io};

pub fn create_new_file(filename: &str) {
    File::create(&filename).expect("cannot create file");
}

pub fn get_or_create_file(filename: &str, create: bool) -> File {
    let path = Path::new(&filename);

    if !path.exists() || create {
        create_new_file(filename);
    }

    OpenOptions::new()
        .write(true)
        .read(true)
        .open(&filename)
        .unwrap()
}

pub fn get_files_in_directory(path: &str) -> io::Result<Vec<PathBuf>> {
    let entries = fs::read_dir(path)?;
    let all: Vec<PathBuf> = entries
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    Ok(all)
}

pub fn serialize<const M: usize, const N: usize>(feed_net: &FeedforwardNetwork<M, N>) {
    let mut network = feed_net.clone();
    let mut file = get_or_create_file(&feedforward_network::FILE_NAME, true);

    let data_json = String::from(format!("{}", serde_json::to_string(&network).unwrap()));

    match file.write_all(data_json.as_bytes()) {
        Err(e) => println!("{:?}", e),
        _ => (),
    }
}

pub fn serialize_generic<T: Sized + Serialize>(array: &T, filename: &str) {
    let cloned_array = array;
    let mut file = get_or_create_file(filename, true);

    let data_json = String::from(format!("{}", serde_json::to_string(&cloned_array).unwrap()));

    match file.write_all(data_json.as_bytes()) {
        Err(e) => println!("{:?}", e),
        _ => (),
    }
}

pub fn deserialize<const M: usize, const N: usize>(network: FeedforwardNetwork<M, N>) -> FeedforwardNetwork<M, N> {
    let mut file = get_or_create_file(&feedforward_network::FILE_NAME, false);
    let mut data = String::new();
    let mut save_network: FeedforwardNetwork<M, N> = network;

    file.read_to_string(&mut data).expect("Unable to open");

    if !data.is_empty() {
        save_network = serde_json::from_str(&data).expect("JSON was not well-formatted");
    }

    save_network
}
