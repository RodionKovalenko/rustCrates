use std::fs::File;
use std::path::Path;
use image::{GenericImageView, Pixel, Rgba};
use crate::neural_networks::utils::file::get_files_in_directory;

pub fn get_pixels_from_images(directory: &str) -> Vec<Vec<Vec<Vec<i32>>>>{
    let mut image_pixel_data: Vec<Vec<Vec<Vec<i32>>>> = vec![];

    match get_files_in_directory(directory) {
        Ok(files) => {
            for file in files {
                if file.is_file() {
                    println!("{} is a file", file.display());

                    image_pixel_data.push(get_pixels_from_image(&file.display().to_string().as_str()));
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    image_pixel_data
}

pub fn get_pixels_from_image(imagepath: &str) -> Vec<Vec<Vec<i32>>> {
    let mut img = image::open(&Path::new(imagepath)).unwrap();
    let img_width = img.width();
    let img_height = img.height();

    let mut pixel_vec: Vec<Vec<Vec<i32>>> = Vec::new();

    let mut i = 0;

    for i in 0..img_width {
        let mut vec_x = Vec::new();
        pixel_vec.push(vec_x);

        for j in 0..img_height {
            let mut vec_y: Vec<i32> = Vec::new();
            pixel_vec[i.clone() as usize].push(vec_y);
        }
    }

    println!("image width x height: {} x {}", img_width, img_height);

    for p in img.pixels() {
        let rgba: Rgba<u8> = p.2.to_rgba();
        let x = p.0 as usize;
        let y = p.1 as usize;

       // println!(" x, y:  {}, {}", x, y);

        let r = rgba[0].clone() as i32;
        let g = rgba[1].clone() as i32;
        let b = rgba[2].clone() as i32;
        let a = rgba[3].clone() as i32;

        pixel_vec[x.clone()][y.clone()] = vec![r, g, b, a];
    }

    pixel_vec
}