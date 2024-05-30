// use crate::neural_networks::utils::image::{get_pixel_separate_rgba, get_pixels_as_rgba, save_image_from_pixels};
// use crate::wavelet_transform::dwt::{get_ll_lh_hl_hh, inverse_transform_2_d, transform_2_d};
// use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
// use crate::wavelet_transform::modes::WaveletMode;
//
// pub fn test_decomposition_cwt() {
//     let pixels: Vec<Vec<Vec<u8>>> = get_pixel_separate_rgba("training_data/1.jpg");
//     let mut count = 1;
//     let mut dw_transformed: Vec<Vec<f64>>;
//
//     let wavelet_type = DiscreteWaletetType::COIF4;
//     let wavelet_mode = WaveletMode::SYMMETRIC;
//
//     let mut dw_transformed: Vec<Vec<f64>>;
//     let mut inverse_transformed: Vec<Vec<f64>>;
//     let mut ll_lh_hl_hh: Vec<Vec<Vec<u8>>>;
//
//     let dec_levels = 6;
//     let mut decomposed_levels: Vec<Vec<Vec<f64>>> = Vec::new();
//
//     println!("Transform : ==================================================================");
//
//     for p in 0..pixels.len() {
//         // encode with wavelet transform
//         let mut pixel_rgba: Vec<Vec<u8>> = pixels[p].clone();
//
//         for i in 0..dec_levels.clone() {
//             dw_transformed = transform_2_d(&pixel_rgba, &wavelet_type, &wavelet_mode);
//
//             let file_name = String::from(format!("{}_decomp_level_{}_{}.jpg", "tests/dwt_", i.clone(), count.clone()));
//             save_image_from_pixels(&dw_transformed, &file_name);
//
//             //  save as images
//             decomposed_levels.push(dw_transformed.clone());
//             ll_lh_hl_hh = get_ll_lh_hl_hh(&dw_transformed);
//
//             pixel_rgba = ll_lh_hl_hh[0].clone();
//             let mut count = 1;
//             for vec in ll_lh_hl_hh {
//                 let file_name = String::from(format!("{}_decomp_level_{}_{}.jpg", "tests/dwt_", i.clone(), count.clone()));
//                 save_image_from_pixels(&vec, &file_name);
//
//                 count += 1;
//             }
//         }
//     }
//
//
//
//     // // decode into original data
//     // for i in (0..dec_levels.clone()).rev() {
//     //     let level = i as u32;
//     //
//     //     inverse_transformed = decomposed_levels.get(i.clone()).unwrap().to_vec();
//     //
//     //     println!("inverse level: before: {}, inverse transform: length: HEIGHT: {}, width: {}\n", i, inverse_transformed.len(), inverse_transformed[1].len());
//     //     inverse_transformed = inverse_transform_2_d(&inverse_transformed, &wavelet_type, &wavelet_mode, level);
//     //     println!("inverse level: after: {}, inverse transform: length: HEIGHT: {}, width: {}\n", i, inverse_transformed.len(), inverse_transformed[1].len());
//     //     // println!("inverse transformed: {:?} \n", &inverse_transformed);
//     //     let file_name = String::from(format!("{}_restored_level_{}.jpg", "tests/dwt_", i.clone()));
//     //     save_image_from_pixels(&inverse_transformed, &file_name);
//     // }
//     // println!("==================================================================");
// }