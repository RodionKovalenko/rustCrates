use std::net::TcpStream;
use std::io::{Write, BufReader, BufRead};
use std::str;

/**
send msg to client or server using TcpStream
*/
pub fn send_msg(stream: &mut TcpStream, message: String) -> Result<(), std::io::Error> {
    let message_with_break_line = message + "\n";
    // send message as utf8 byte array
    stream.write_all(message_with_break_line.as_bytes()).expect("Failed to write to server");
    Ok(())
}

/**
read message from TcpStream sent by server or client and display them on console
*/
pub fn read_tcp_stream(stream: &mut TcpStream, message: String) -> String {
    let mut buffer: Vec<u8> = Vec::new();
    let mut reader = BufReader::new(stream);

    reader.read_until(b'\n', &mut buffer).expect("Could not read into buffer");

    // decode utf8 array as readable string  
    let input = str::from_utf8(&buffer).expect("Could not write buffer as string");

    if input == "" {
        eprintln!("Empty response from server");
    }

    print!("{}: {} ", message, input);
    println!("buffer {:?}", &buffer);
    String::from(input)
}