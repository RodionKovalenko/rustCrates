use std::path::Path;
use image::{GenericImageView, ImageBuffer, Pixel, Rgba, RgbaImage};
use num_traits::Num;
use crate::neural_networks::utils::file::get_files_in_directory;

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
    let img = image::open(&Path::new(imagepath)).unwrap();
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

pub fn get_pixels_as_rgba(original_image_path: &str) -> Vec<Vec<Rgba<u8>>> {
    let img = image::open(&Path::new(original_image_path)).unwrap();
    let img_width: u32 = img.width().clone();
    let img_height: u32 = img.height().clone();

    let mut buffer: RgbaImage = ImageBuffer::new(img_width, img_height);
    let mut pixel_vec: Vec<Vec<Rgba<u8>>> = vec![vec![Rgba([0, 0, 0, 0]); img_width.clone() as usize]; img_height.clone() as usize];

    for (x, y, _pixel) in buffer.enumerate_pixels_mut() {
        let rgba = img.get_pixel(x, y);

        pixel_vec[y.clone() as usize][x.clone() as usize] = rgba;
    }

    pixel_vec
}

pub fn get_pixel_separate_rgba(original_image_path: &str) -> Vec<Vec<Vec<f64>>> {
    let pixels_rgba: Vec<Vec<Rgba<u8>>> = get_pixels_as_rgba(original_image_path);

    let w = pixels_rgba[0].len();
    let h = pixels_rgba.len();
    let mut r_arr: Vec<Vec<f64>> = vec![vec![0.0; w]; h];
    let mut g_arr: Vec<Vec<f64>> = vec![vec![0.0; w]; h];
    let mut b_arr: Vec<Vec<f64>> = vec![vec![0.0; w]; h];
    let mut a_arr: Vec<Vec<f64>> = vec![vec![0.0; w]; h];
    let mut pixels: Vec<Vec<f64>> = vec![vec![0.0; w]; h];
    let mut is_a_present: bool = false;

    for i in 0..pixels_rgba.len() {
        for j in 0..pixels_rgba[i].len() {
            let r:i32 = (pixels_rgba[i][j][0].clone() as i32) << 16;
            let g:i32 = (pixels_rgba[i][j][1].clone() as i32) << 8;
            let b: i32 = pixels_rgba[i][j][2].clone() as i32;
            let a:i32 = (pixels_rgba[i][j][3].clone() as i32) << 24;

            if a > 0 {
                is_a_present = true;
            }

            r_arr[i][j] = r as f64;
            g_arr[i][j] = g as f64;
            b_arr[i][j] = b as f64;
            a_arr[i][j] = a as f64;
            pixels[i][j] = (a + r  + g + b) as f64;
        }
    }

    if is_a_present {
        return vec![r_arr, g_arr, b_arr, a_arr, pixels];
    }

    vec![r_arr, g_arr, b_arr,pixels]
}

pub fn get_pixels(original_image_path: &str) -> Vec<Vec<f64>> {
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

    for (x, y, _pixel) in buffer.enumerate_pixels_mut() {
        let rgba = img.get_pixel(x, y);

        r = rgba[0].clone() as i32;
        g = rgba[1].clone() as i32;
        b = rgba[2].clone() as i32;
        a = rgba[3].clone() as i32;

        pixel_vec[y.clone() as usize][x.clone() as usize] = ((a << 24) + (r << 16) + (g << 8) + b) as f64;
    }

    pixel_vec
}

pub fn save_image_from_pixels<T>(image_data: &Vec<Vec<T>>, image_path: &str)
    where
        T: Num + Into<f64> + Copy,
{
    let img_width: u32 = image_data[0].len() as u32;
    let img_height: u32 = image_data.len() as u32;

    let mut buffer: RgbaImage = ImageBuffer::new(img_width, img_height);

    let brightness_factor: f32 = 1.0;

    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        let v_f64: f64 = image_data[y.clone() as usize][x.clone() as usize].into();
        // Round to nearest integer and then try to convert to i32
        let v_i32 = v_f64.round() as i32;

        let a = (((v_i32.clone() >> 24) & 0xff) as f32 * (brightness_factor.clone())) as u8;
        let r = (((v_i32.clone() >> 16) & 0xff) as f32 * (brightness_factor.clone())) as u8;
        let g = (((v_i32.clone() >> 8) & 0xff) as f32 * (brightness_factor.clone())) as u8;
        let b = (((v_i32.clone() >> 0) & 0xff) as f32 * (brightness_factor.clone())) as u8;

        *pixel = Rgba([r.clone(), g.clone(), b.clone(), a.clone()]);
    }

    buffer.save(image_path).unwrap();
}