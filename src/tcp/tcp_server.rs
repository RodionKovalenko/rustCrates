use crate::tcp::tcp_stream_actions::{send_msg, read_tcp_stream};
use std::net::{TcpListener, TcpStream};
use std::io::{Error};
use std::{thread};
use rand::Rng;

// Handles a single client
fn handle_client(stream: TcpStream) -> Result<(), Error> {
    println!("Incoming connection from: {}", stream.peer_addr()?);
    let mut server_stream = stream.try_clone().expect("cannot clone stream");
    loop {
        read_tcp_stream(&mut server_stream, String::from("Message from client"));

        let rand_number: u32 = rand::rng().random::<u32>();
        let message = format!("{}{}", "hi client ", rand_number);

        send_msg(&mut server_stream, message).unwrap_or_else(|err| println!("{:?}", err));
    }
}

pub fn start_server(server_address_with_port: String) {
    println!("Server is starting ");

    let listener = TcpListener::bind(server_address_with_port).expect("Could not bind");
    for stream in listener.incoming() {
        match stream {
            Err(e) => eprintln!("failed: {}", e),
            Ok(stream) => {
                thread::spawn(move || {
                    handle_client(stream).unwrap_or_else(|error|
                        eprintln!("{:?}", error));
                });
            }
        }
    }
}