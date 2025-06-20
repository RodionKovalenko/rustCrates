use crate::neural_networks::utils::image::get_pixel_separate_rgba;
use crate::utils::data_converter::{convert_to_c_array_f64_3d, convert_to_f64_1d, convert_to_f64_2d, convert_to_f64_3d};
use crate::utils::normalization::normalize_f64;
use crate::utils::num_trait::ArrayType;
use crate::wavelet_transform::cwt::cwt;
use crate::wavelet_transform::cwt_complex::CWTComplex;
use crate::wavelet_transform::cwt_types::ContinuousWaletetType;
use crate::wavelet_transform::dwt::{dwt_1d, dwt_2d_full, get_ll_hl_lh_hh};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
use crate::wavelet_transform::modes::WaveletMode;
use num_complex::Complex;

pub const DECOMPOSITION_LEVELS: u32 = 5;

pub fn get_pixels_rgba(image_path: &str) -> Vec<Vec<Vec<f64>>> {
    get_pixel_separate_rgba(image_path)
}

pub fn apply_wavelet_positional_emb(input: &Vec<f64>, dw_type: DiscreteWaletetType, dw_mode: WaveletMode, decomposition_levels: usize, position: usize, seq_len: usize, emb_len: usize) -> Vec<f64> {
    // encode with wavelet transform
    let mut input_decomposed: Vec<f64> = vec![0.0; input.len()];

    let position_ratio = position as f64 / seq_len as f64;
    let index = (position_ratio * emb_len as f64).floor() as usize;
    let index = index.min(emb_len - 1); // clamp to valid range

    input_decomposed[index] = position_ratio;

    for _i in 0..decomposition_levels {
        input_decomposed = dwt_1d(&input_decomposed, &dw_type, &dw_mode);
    }

    if input_decomposed.len() < emb_len {
        input_decomposed.resize(emb_len, 0.0);
    } else {
        input_decomposed.truncate(emb_len);
    }

    normalize_f64(&input_decomposed)
}

pub fn decompose_in_wavelet_2d_default<T: ArrayType>(input: &T) -> Vec<Vec<Vec<Complex<f64>>>> {
    let wavelet_type = DiscreteWaletetType::DB1;
    let wavelet_mode = WaveletMode::SYMMETRIC;

    let cw_type = ContinuousWaletetType::CMOR;
    let scales: Vec<f64> = vec![2.0, 8.0, 16.0, 32.0];
    let min_height: usize = 28;
    let min_width: usize = 28;

    let decomposition_level: i32 = DECOMPOSITION_LEVELS as i32;

    let mut cwt_complex_wavelet = CWTComplex {
        scales,
        cw_type,
        sampling_period: 1.0,
        m: 1.0,
        fb: 1.5,
        fc: 1.0,
        frequencies: vec![0.0],
    };

    return decompose_in_wavelets(input, &wavelet_type, &wavelet_mode, &mut cwt_complex_wavelet, &min_height, &min_width, &decomposition_level);
}

/**
 * return wavelel transformed pixels in 4 D <complex<f64>>
 */
pub fn decompose_in_wavelets<T: ArrayType>(input_data: &T, dw_type: &DiscreteWaletetType, dw_mode: &WaveletMode, cwt_complex_wavelet: &mut CWTComplex, min_height: &usize, min_width: &usize, dec_levels: &i32) -> Vec<Vec<Vec<Complex<f64>>>> {
    let num_dim = input_data.dimension();
    let mut pixels: Vec<Vec<Vec<f64>>> = Vec::new();

    match num_dim {
        1 => {
            let mut two_d_vec = Vec::new();
            two_d_vec.push(convert_to_f64_1d(input_data));

            pixels.push(two_d_vec);
        }
        2 => {
            pixels.push(convert_to_f64_2d(input_data));
        }
        3 => pixels = convert_to_f64_3d(input_data),
        _ => (),
    };

    let mut dw_transformed: Vec<Vec<f64>>;
    let mut decomposed_levels: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut cwt_transformed: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

    for p in 0..pixels.len() {
        // encode with wavelet transform
        let mut pixel_rgba: Vec<Vec<f64>> = pixels[p].clone();

        for _i in 0..dec_levels.clone() {
            dw_transformed = dwt_2d_full(&pixel_rgba, &dw_type, &dw_mode);

            //println!("dw transform : {:?}, {:?}", dw_transformed.len(), dw_transformed[0].len());

            if (input_data.dimension() != 1 && dw_transformed.len() < min_height.clone()) || dw_transformed[0].len() < min_width.clone() {
                break;
            }
            //  save as images
            let ll_lh_hl_hh: Vec<Vec<Vec<f64>>> = get_ll_hl_lh_hh(&dw_transformed);

            //println!("ll lh hl hh transform len: {}, {}, {}", ll_lh_hl_hh.len(), ll_lh_hl_hh[0].len(), ll_lh_hl_hh[0][0].len());
            pixel_rgba = ll_lh_hl_hh[0].clone();
        }
        decomposed_levels.push(pixel_rgba);
    }

    //println!("condensed wavelet: {}, {:?}, {}", decomposed_levels.len(), decomposed_levels[0].len(), decomposed_levels[0][0].len());

    // remove_dir_contents("tests").unwrap_or_else(|why| {
    //     println!("! {:?}", why.kind());
    // });

    for p in 0..decomposed_levels.len() {
        let wavelet_pixels: Vec<Vec<f64>> = decomposed_levels[p].clone();
        let (transformed, _frequencies) = cwt(&wavelet_pixels, cwt_complex_wavelet).unwrap();
        let cwt: Vec<Vec<Vec<Complex<f64>>>> = convert_to_c_array_f64_3d(transformed);

        //let cwt_pixels: Vec<Vec<Vec<f64>>> = convert_c_to_f64_3d(&cwt);
        // for i in 0..cwt_pixels.len() {
        //     let file_name = String::from(format!("{}_cwt_{}_{}_{}.jpg", "tests/cwt_", p.clone(), p.clone(), i));
        //     save_image_from_pixels(&cwt_pixels[i], &file_name);
        // }

        for i in 0..cwt.len() {
            cwt_transformed.push(cwt[i].clone());
        }
    }

    cwt_transformed
}
