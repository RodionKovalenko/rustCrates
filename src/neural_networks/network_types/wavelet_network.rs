use crate::neural_networks::utils::image::{get_pixel_separate_rgba, save_image_from_pixels};
use crate::uphold_api::file_utils::remove_dir_contents;
use crate::wavelet_transform::dwt::{get_ll_hl_lh_hh, inverse_transform_2_d, transform_2_d};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
use crate::wavelet_transform::modes::WaveletMode;

pub fn test_decomposition_dwt() {
    let pixels: Vec<Vec<Vec<f64>>> = get_pixel_separate_rgba("training_data/1.jpg");

    let wavelet_type = DiscreteWaletetType::BIOR13;
    let wavelet_mode = WaveletMode::SYMMETRIC;

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

        for i in 0..dec_levels.clone() {
            dw_transformed = transform_2_d(&pixel_rgba, &wavelet_type, &wavelet_mode);

            //  save as images
            decomposed_levels.push(dw_transformed.clone());
            let ll_lh_hl_hh: Vec<Vec<Vec<f64>>> = get_ll_hl_lh_hh(&dw_transformed);

            pixel_rgba = ll_lh_hl_hh[0].clone();

            for (ind, vec) in ll_lh_hl_hh.iter().enumerate() {
                if ind != 0 {
                    continue;
                }
                println!("level: {}, length: height: {}, width: {}\n", (&i + 1), vec.len(), vec[1].len());

                let file_name = String::from(format!("{}_decomp_level_{}_{}_{}.jpg", "tests/dwt_", p.clone(), i.clone(), ind));
                save_image_from_pixels(&vec, &file_name);
            }
        }
    }

    // decode into original data
    for i in (0..dec_levels.clone() * pixels.len()).rev() {
        let level = i as u32;

       let mut decomposed_wavelet = decomposed_levels.get(i.clone()).unwrap().to_vec();

        println!("inverse level: before: {}, inverse transform: length: HEIGHT: {}, width: {}\n", i, decomposed_wavelet.len(), decomposed_wavelet[1].len());
        let inverse_transformed = inverse_transform_2_d(&decomposed_wavelet, &wavelet_type, &wavelet_mode, level);
        println!("inverse level: after: {}, inverse transform: length: HEIGHT: {}, width: {}\n", i, inverse_transformed.len(), inverse_transformed[1].len());
        // println!("inverse transformed: {:?} \n", &inverse_transformed);
        let file_name = String::from(format!("{}_restored_level_{}.jpg", "tests/dwt_", i.clone()));
        save_image_from_pixels(&inverse_transformed, &file_name);
    }
    println!("==================================================================");
}