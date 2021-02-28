use crate::tcp::tcp_stream_actions::{send_msg, read_tcp_stream};
use std::net::TcpStream;
use std::time::Duration;
use std::io::stdin;

/**
connect client to given ip address and port
*/
pub fn start_client(server_address_with_port: String) {
    println!("Client connecting to the server with address: {}", server_address_with_port);

    let mut stream = TcpStream::connect(server_address_with_port).expect("Could not connect to server");

    // set time out of 1200 sec = 20 minutes
    stream.set_read_timeout(Some(Duration::from_secs(1200))).expect("Could not set a read timeout");

    loop {
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read from stdin");

        send_msg(&mut stream, input).unwrap_or_else(|err| println!("{:?}", err));
        read_tcp_stream(&mut stream, String::from("Response from server"));
    }
}