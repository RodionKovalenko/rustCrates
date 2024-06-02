use num_complex::Complex;
use crate::neural_networks::utils::image::{get_pixel_separate_rgba, save_image_from_pixels};
use crate::uphold_api::file_utils::remove_dir_contents;
use crate::utils::data_converter::{convert_c_to_f64_3d, convert_to_c_array_f64_3d};
use crate::wavelet_transform::cwt::cwt;
use crate::wavelet_transform::cwt_complex::CWTComplex;
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;
use crate::wavelet_transform::dwt::{get_ll_hl_lh_hh, transform_2_df64};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
use crate::wavelet_transform::modes::WaveletMode;

pub fn decompose_in_wavelet_2d_default(image_path: &str) -> Vec<Vec<Vec<Complex<f64>>>> {
    let wavelet_type = DiscreteWaletetType::DB1;
    let wavelet_mode = WaveletMode::SYMMETRIC;

    let cw_type = ContinuousWaletetType::SHAN;
    let scales: Vec<f64> = vec![2.0, 4.0, 8.0, 16.0];
    let min_height: usize = 1250;
    let min_width: usize = 1250;

    let decomposition_level: i32 = 1;

    let mut cwt_complex_wavelet = CWTComplex {
        scales,
        cw_type,
        sampling_period: 1.0,
        m: 1.0,
        fb: 1.5,
        fc: 1.0,
        frequencies: vec![0.0],
    };

    return decompose_in_wavelets(&image_path,
                                 &wavelet_type,
                                 &wavelet_mode,
                                 &mut cwt_complex_wavelet,
                                 &min_height,
                                 &min_width,
                                 &decomposition_level);
}

pub fn decompose_in_wavelets(image_path: &str,
                             dw_type: &DiscreteWaletetType,
                             dw_mode: &WaveletMode,
                             cwt_complex_wavelet: &mut CWTComplex,
                             min_height: &usize,
                             min_width: &usize,
                             dec_levels: &i32) -> Vec<Vec<Vec<Complex<f64>>>> {
    let pixels: Vec<Vec<Vec<f64>>> = get_pixel_separate_rgba(image_path);

    let mut dw_transformed: Vec<Vec<f64>>;
    let mut decomposed_levels: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut cwt_transformed: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

    for p in 0..pixels.len() {
        // encode with wavelet transform
        let mut pixel_rgba: Vec<Vec<f64>> = pixels[p].clone();

        for _i in 0..dec_levels.clone() {
            dw_transformed = transform_2_df64(&pixel_rgba, &dw_type, &dw_mode);

            if dw_transformed.len() < min_height.clone() || dw_transformed[0].len() < min_width.clone() {
                break;
            }
            //  save as images
            let ll_lh_hl_hh: Vec<Vec<Vec<f64>>> = get_ll_hl_lh_hh(&dw_transformed);

            pixel_rgba = ll_lh_hl_hh[0].clone();
        }
        decomposed_levels.push(pixel_rgba);
    }

    remove_dir_contents("tests").unwrap_or_else(|why| {
        println!("! {:?}", why.kind());
    });

    for p in 0..decomposed_levels.len() {
        let wavelet_pixels: Vec<Vec<f64>> = decomposed_levels[p].clone();
        let (transformed, _frequencies) = cwt(&wavelet_pixels, cwt_complex_wavelet).unwrap();
        let cwt: Vec<Vec<Vec<Complex<f64>>>> = convert_to_c_array_f64_3d(transformed);


        let cwt_pixels: Vec<Vec<Vec<f64>>> = convert_c_to_f64_3d(&cwt);

        let file_name = String::from(format!("{}_cwt_{}_{}_dwt{}.jpg", "tests/cwt_", p.clone(), p.clone(), p));
        save_image_from_pixels(&wavelet_pixels, &file_name);

        for i in 0..cwt_pixels.len() {
            let file_name = String::from(format!("{}_cwt_{}_{}_{}.jpg", "tests/cwt_", p.clone(), p.clone(), i));
            save_image_from_pixels(&cwt_pixels[i], &file_name);
        }

        for i in 0..cwt.len() {
            cwt_transformed.push(cwt[i].clone());
        }

        println!("level: {}, length: height: {}, width: {}\n", (&dec_levels), wavelet_pixels.len(), wavelet_pixels[1].len());
    }


    cwt_transformed
}