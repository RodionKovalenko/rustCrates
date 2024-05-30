use num_complex::Complex;
use crate::neural_networks::utils::image::{get_pixel_separate_rgba, save_image_from_pixels};
use crate::uphold_api::file_utils::remove_dir_contents;
use crate::utils::data_converter::{convert_c_to_f64_3d, convert_to_c_array_f64_3d};
use crate::wavelet_transform::cwt::cwt;
use crate::wavelet_transform::cwt_complex::CWTComplex;
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;
use crate::wavelet_transform::dwt::{get_ll_hl_lh_hh, transform_2_d};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
use crate::wavelet_transform::modes::WaveletMode;

pub fn decompose_in_wavelets() {
    let pixels: Vec<Vec<Vec<f64>>> = get_pixel_separate_rgba("training_data/1.jpg");

    let wavelet_type = DiscreteWaletetType::DB2;
    let cw_type = ContinuousWaletetType::CMOR;
    let scales: Vec<f64> = vec![1.0, 15.0, 30.0, 50.0];
    let wavelet_mode = WaveletMode::SYMMETRIC;
    const MIN_HEIGHT: usize = 100;
    const MIN_WIDTH: usize= 100;

    let mut wavelet = CWTComplex {
        scales,
        cw_type,
        sampling_period: 1.0,
        m: 1.0,
        fb: 1.5,
        fc: 1.0,
        frequencies: vec![0.0],
    };

    let mut dw_transformed: Vec<Vec<f64>>;
    let dec_levels = 5;
    let mut decomposed_levels: Vec<Vec<Vec<f64>>> = Vec::new();

    remove_dir_contents("tests").unwrap_or_else(|why| {
        println!("! {:?}", why.kind());
    });

    println!("Transform : ==================================================================");
    for p in 0..pixels.len() {
        // encode with wavelet transform
        let mut pixel_rgba: Vec<Vec<f64>> = pixels[p].clone();

        for _i in 0..dec_levels.clone() {
            dw_transformed = transform_2_d(&pixel_rgba, &wavelet_type, &wavelet_mode);

            if dw_transformed.len() < MIN_HEIGHT || dw_transformed[0].len() < MIN_WIDTH {
                break;
            }
            //  save as images
            let ll_lh_hl_hh: Vec<Vec<Vec<f64>>> = get_ll_hl_lh_hh(&dw_transformed);

            pixel_rgba = ll_lh_hl_hh[0].clone();
        }
        decomposed_levels.push(pixel_rgba);
    }

    for p in 0..decomposed_levels.len() {
        let wavelet_pixels: Vec<Vec<f64>> = decomposed_levels[p].clone();
        let (transformed, _frequencies) = cwt(&wavelet_pixels, &mut wavelet).unwrap();
        let cwt_transformed: Vec<Vec<Vec<Complex<f64>>>> = convert_to_c_array_f64_3d(transformed);

        let cwt_pixels: Vec<Vec<Vec<f64>>> = convert_c_to_f64_3d(&cwt_transformed);

        for i in 0..cwt_pixels.len() {
            let file_name = String::from(format!("{}_cwt_{}_{}_{}.jpg", "tests/cwt_", p.clone(), p.clone(), i));
            save_image_from_pixels(&cwt_pixels[i], &file_name);
        }

        println!("level: {}, length: height: {}, width: {}\n", (&dec_levels), wavelet_pixels.len(), wavelet_pixels[1].len());
    }

    println!("==================================================================");
}