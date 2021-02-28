use std::fs::File;
use std::fs::OpenOptions;
use std::path::Path;

pub fn get_or_create_file(filename: &str, create: bool) -> File {
    let path =  Path::new(&filename);

    if !path.exists() || create {
       create_new_file(filename);
    }

    let f = OpenOptions::new()
        .write(true)
        .read(true)
        .open(&filename)
        .unwrap();
   f
}

pub fn create_new_file(filename: &str) {
    File::create(&filename).expect("cannot create file");
}