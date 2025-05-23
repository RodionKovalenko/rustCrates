use crate::neural_networks::network_types::neural_network_generic::FILE_NAME;
use crate::neural_networks::network_types::neural_network_generic::NeuralNetwork;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{fs, io};

pub fn create_new_file(filename: &str) {
    File::create(&filename).expect("cannot create file");
}

pub fn get_or_create_file(filename: &str, create: bool) -> File {
    let path = Path::new(&filename);

    let dir = path.parent().unwrap();
    if !dir.exists() {
        create_dir_all(dir).expect("Failed to create directory");
    }

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

pub fn serialize(feed_net: &NeuralNetwork) {
    let network = feed_net.clone();
    let mut file = get_or_create_file(&FILE_NAME, true);

    let data_json = String::from(format!("{}", serde_json::to_string(&network).unwrap()));

    match file.write_all(data_json.as_bytes()) {
        Err(e) => println!("{:?}", e),
        _ => (),
    }
}

pub fn serialize_bin(binary_data: &Vec<u8>, filepath: &str) -> io::Result<()> {
    let mut file: File = get_or_create_file( filepath, false);
    file.write_all(&binary_data).expect("Writing binary file has not succeeded");

    Ok(())
}

pub fn derialize_bin<T>(filepath: &str) ->io::Result<T> 
where T: DeserializeOwned {
    let mut file: File = get_or_create_file( filepath, false);

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
   
    let my_struct: T = bincode::deserialize(&buffer).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, format!("Deserialization failed: {}", e))
    })?;

    // Return the deserialized struct
    Ok(my_struct)
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

pub fn deserialize(network: NeuralNetwork) -> NeuralNetwork {
    let mut file = get_or_create_file(&FILE_NAME, false);
    let mut data = String::new();
    let mut save_network: NeuralNetwork = network;

    file.read_to_string(&mut data).expect("Unable to open");

    if !data.is_empty() {
        save_network = serde_json::from_str(&data).expect("JSON was not well-formatted");
    }

    save_network
}

pub fn remove_file(filepath: &str) {
    if Path::new(filepath).exists() {
        fs::remove_file(filepath).expect("Failed to remove file");
        println!("File removed successfully");
    } else {
        eprintln!("File does not exist, cannot remove.");
    }
}