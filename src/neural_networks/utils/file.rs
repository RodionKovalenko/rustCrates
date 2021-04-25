use crate::network_types::feedforward_network_generic::FeedforwardNetwork;
use crate::network_types::feedforward_network;
use std::path::Path;
use std::fs::{File, OpenOptions};
use std::io::{Write, Read};

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

pub fn serialize(feed_net: &FeedforwardNetwork<f64>) {
    let mut network = feed_net.clone();
    let mut file = get_or_create_file(&feedforward_network::FILE_NAME, true);

    for l in 0..network.layers.len() {
        network.layers[l].input_data = vec![];
    }

    let data_json = String::from(format!("{}", serde_json::to_string(&network).unwrap()));

    match file.write_all(data_json.as_bytes()) {
        Err(e) => println!("{:?}", e),
        _ => ()
    }
}

pub fn deserialize(network: FeedforwardNetwork<f64>) -> FeedforwardNetwork<f64> {
    let mut file = get_or_create_file(&feedforward_network::FILE_NAME, false);
    let mut data = String::new();
    let mut save_network: FeedforwardNetwork<f64> = network;
    file.read_to_string(&mut data).expect("Unable to open");

    if !data.is_empty() {
        save_network = serde_json::from_str(&data).expect("JSON was not well-formatted");
    }

    save_network
}