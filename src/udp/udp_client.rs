use std::net::UdpSocket;
use std::{str,io};
pub fn start_client(server_address_with_port: String) {
    let socket = UdpSocket::bind(server_address_with_port.clone())
        .expect("Could not bind client socket");

    socket.connect(server_address_with_port.clone())
        .expect("Could not connect to server");

    loop {
        let mut input = String::new();
        let mut buffer = [0u8; 1500];

        io::stdin().read_line(&mut input)
            .expect("Failed to read from stdin");
        socket.send(input.as_bytes())
            .expect("Failed to write to server");
        socket.recv_from(&mut buffer)
            .expect("Could not read into buffer");
        print!("{}", str::from_utf8(&buffer)
            .expect("Could not write buffer as string"));
    }
}