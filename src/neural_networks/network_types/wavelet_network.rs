use crate::neural_networks::utils::image::{get_pixels_as_rgba, save_image_from_pixels};
use crate::wavelet_transform::dwt::{get_ll_lh_hl_hh, inverse_transform_2_d, transform_2_d};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
use crate::wavelet_transform::modes::WaveletMode;

pub fn test_decomposition() {
    let pixels : Vec<Vec<f64>>= get_pixels_as_rgba("training_data/1.jpg");
    // let mut n: Vec<Vec<f64>> = Vec::new();
    let mut n = pixels;

    // println!("n : {:?}", &n);
    println!("Transform : ==================================================================");
    let mut dw_transformed:Vec<Vec<f64>>= Vec::new();
    let mut inverse_transformed = Vec::new();
    let mut ll_lh_hl_hh: Vec<Vec<Vec<f64>>> = Vec::new();

    println!("length: height: {}, width: {}", n.len(), n[1].len());
    let dec_levels = 3;

    for i in 0..dec_levels.clone() {
        // println!("before level: {}, length: height: {}, width: {}\n", &i, dw_transformed.len(), dw_transformed[1].len());
        dw_transformed = transform_2_d(&n, &DiscreteWaletetType::DB3, &WaveletMode::SYMMETRIC);
        println!("after level: {}, length: height: {}, width: {}\n", &i, dw_transformed.len(), dw_transformed[1].len());
        // n = dw_transformed.clone();
        // println!("transformed: {:?}\n", &dw_transformed);

        //  save as images
        ll_lh_hl_hh = get_ll_lh_hl_hh(&dw_transformed);
        n = ll_lh_hl_hh[0].clone();
        let mut count = 1;
        for vec in ll_lh_hl_hh {
            let file_name = String::from(format!("{}_decomp_level_{}_{}.jpg", "tests/dwt_", i.clone(), count.clone()));
            save_image_from_pixels(&vec, &file_name);

            count +=1;
        }
    }

    inverse_transformed = dw_transformed.clone();
    for i in (0..dec_levels.clone()).rev() {
        let level = i as u32;
        println!("inverse level: before: {}, inverse transform: length: height: {}, width: {}\n", i, inverse_transformed.len(), inverse_transformed[1].len());
        inverse_transformed = inverse_transform_2_d(&inverse_transformed, &DiscreteWaletetType::DB3, &WaveletMode::SYMMETRIC, level);
        println!("inverse level: after: {}, inverse transform: length: height: {}, width: {}\n", i, inverse_transformed.len(), inverse_transformed[1].len());
        // println!("inverse transformed: {:?} \n", &inverse_transformed);

        let mut count = 1;
        ll_lh_hl_hh = get_ll_lh_hl_hh(&inverse_transformed);
        for vec in ll_lh_hl_hh {
            let file_name = String::from(format!("{}_restored_level_{}_{}.jpg", "tests/dwt_", i.clone(), count.clone()));
            save_image_from_pixels(&vec, &file_name);

            count +=1;
        }
    }
    println!("==================================================================");
}