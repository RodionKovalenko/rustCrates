use std::{env};
use crate::tcp::{tcp_client, tcp_server};

pub mod udp_client;
pub mod udp_server;

pub fn test_connection() {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Please provide --client or --server as argument");
        std::process::exit(1);
    }

    let arguments = String::from(args[1].to_string());
    println!("arguments called {}", arguments);

    if args[1] == "server" {
        udp_server::start_server(String::from("127.0.0.1:8888"))
    } else  if args[1] == "client" {
        udp_client::start_client(String::from("192.168.2.111:8888"))
    }
}