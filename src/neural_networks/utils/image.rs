use std::path::Path;
use image::{GenericImageView, ImageBuffer, Pixel, Rgba, RgbaImage};
use num_traits::abs;
use crate::neural_networks::utils::file::{get_files_in_directory};

pub fn get_pixels_from_images(directory: &str) -> Vec<Vec<Vec<Vec<i32>>>> {
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

    for i in 0..img_width {
        pixel_vec.push(Vec::new());

        for _j in 0..img_height {
            pixel_vec[i.clone() as usize].push(Vec::new());
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

pub fn save_as_grey_scale(original_image_path: &str, imagepath: &str) -> Vec<Vec<f64>> {
    let img = image::open(&Path::new(original_image_path)).unwrap();
    let img_width: u32 = img.width().clone();
    let img_height: u32 = img.height().clone();

    let mut buffer: RgbaImage = ImageBuffer::new(img_width, img_height);

    let mut pixel_vec: Vec<Vec<f64>> = Vec::new();

    for _i in 0..img_height {
        pixel_vec.push(vec![0.0; img_width.clone() as usize]);
    }

    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        let rgba = img.get_pixel(x, y);

        let r = rgba[0].clone() as i32;
        let g = rgba[1].clone() as i32;
        let b = rgba[2].clone() as i32;
        let a = rgba[3].clone() as i32;
        let mean: u8 = ((r + g + b + a) / 4) as u8;

        *pixel = Rgba([mean.clone(), mean.clone(), mean.clone(), mean.clone()]);

        pixel_vec[y.clone() as usize][x.clone() as usize] = mean.clone() as f64;
    }

    //serialize_generic(&pixel_vec, "test_array.txt");
    // println!("pixels: {:?}", pixel_vec);

    buffer.save(imagepath).unwrap();

    pixel_vec
}

pub fn get_pixels_as_rgba(original_image_path: &str) -> Vec<Vec<f64>> {
    let img = image::open(&Path::new(original_image_path)).unwrap();
    let img_width: u32 = img.width().clone();
    let img_height: u32 = img.height().clone();

    let mut buffer: RgbaImage = ImageBuffer::new(img_width, img_height);
    let mut pixel_vec: Vec<Vec<f64>> = Vec::new();

    let mut r: i32;
    let mut g: i32;
    let mut b: i32;
    let mut a: i32;

    for _i in 0..img_height {
        pixel_vec.push(vec![0.0; img_width.clone() as usize]);
    }

    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        let rgba = img.get_pixel(x, y);

        r = rgba[0].clone() as i32;
        g = rgba[1].clone() as i32;
        b = rgba[2].clone() as i32;
        a = rgba[3].clone() as i32;;

        pixel_vec[y.clone() as usize][x.clone() as usize] = ((a << 24) + (r << 16) + (g << 8) + b) as f64;
    }

    pixel_vec
}

pub fn save_image_from_pixels(image_data: &Vec<Vec<f64>>, image_path: &str) {
    let img_width: u32 = image_data[0].len() as u32;
    let img_height: u32 = image_data.len() as u32;

    let mut buffer: RgbaImage = ImageBuffer::new(img_width, img_height);
    let mut pixel_vec: Vec<Vec<f64>> = Vec::new();
    let mut r: u8;
    let mut g: u8;
    let mut b: u8;
    let mut a: u8;
    let mut v;
    let mut mean: Vec<f32> = Vec::new();

    for _i in 0..img_height {
        mean.push(image_data[_i as usize].iter().sum::<f64>() as f32 / image_data[0].len() as f32);

        pixel_vec.push(vec![0.0; img_width.clone() as usize]);
    }

    let mut brightness_factor: f32 = 1.0;

    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        v = image_data[y.clone() as usize][x.clone() as usize].clone() as i32;

        a = (((v.clone() >> 24) & 0xff) as f32 * (brightness_factor.clone())).clamp(0.0, 255.0) as u8;
        r = (((v.clone() >> 16) & 0xff) as f32 * (brightness_factor.clone())).clamp(0.0, 255.0) as u8;
        g = (((v.clone() >> 8) & 0xff) as f32 * (brightness_factor.clone())).clamp(0.0, 255.0) as u8;
        b = (((v.clone() >> 0) & 0xff) as f32 * (brightness_factor.clone())).clamp(0.0, 255.0) as u8;

        *pixel = Rgba([r.clone(), g.clone(), b.clone(), a.clone()]);
    }

    buffer.save(image_path).unwrap();
}