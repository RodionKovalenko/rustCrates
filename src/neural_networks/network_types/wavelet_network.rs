pub fn test_decomposition_cwt() {
    // let pixels: Vec<Vec<f64>> = get_pixels_as_rgba("training_data/1.jpg");
    // // let mut n: Vec<Vec<f64>> = Vec::new();
    // let mut n = pixels;
    //
    // // println!("n : {:?}", &n);
    // println!("Transform : ==================================================================");
    // let mut dw_transformed: Vec<Vec<f64>> = Vec::new();
    // let mut ll_lh_hl_hh: Vec<Vec<Vec<f64>>> = Vec::new();
    //
    // println!("length: height: {}, width: {}", n.len(), n[1].len());

    // let wavelet_type = ContinuousWaletetType::MEXH;
    // let scales: Vec<f64> = (1..6).map(|x| x as f64).collect();

    // encode with wavelet transform
    // for p in pixels.iter() {
    //     dw_transformed = cwt_1d(p, &scales, &wavelet_type, &1.0);
    //
    //     ll_lh_hl_hh = get_ll_lh_hl_hh(&dw_transformed);
    //
    //     n = ll_lh_hl_hh[0].clone();
    //     let mut count = 1;
    //     // for vec in ll_lh_hl_hh {
    //     //     let file_name = String::from(format!("{}_decomp_level_{}_{}.jpg", "tests/dwt_", i.clone(), count.clone()));
    //     //     save_image_from_pixels(&vec, &file_name);
    //     //
    //     //     count += 1;
    //     // }
    // }
}