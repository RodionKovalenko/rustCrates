#[cfg(test)]
mod test_wavelet_compression {

    use crate::{
        neural_networks::utils::{derivative::test_gradient_error_2d_f64, matrix::transpose, random_arrays::generate_random_f64_2d},
        wavelet_transform::{
            dwt::{combine_ll_hh, combine_ll_lh_hl_hh, get_ll_hh, get_ll_hl_lh_hh, transform_2_d_partial_f64, transform_2_df64},
            dwt_types::DiscreteWaletetType,
            modes::WaveletMode,
        },
    };

    #[test]
    fn test_wavelet_full() {
        let seq_len = 6;
        let dim = 6;

        let matrix_1: Vec<Vec<f64>> = generate_random_f64_2d(seq_len, dim);

        let wav_trans_1: Vec<Vec<f64>> = transform_2_df64(&matrix_1, &DiscreteWaletetType::DB1, &WaveletMode::SYMMETRIC);

        println!("\n matrix 1: dim: {} {}", matrix_1.len(), matrix_1[0].len());
        println!("\n wav_trans_1: dim: {} {}", wav_trans_1.len(), wav_trans_1[0].len());

        println!("\n wav trans: {:?}", wav_trans_1);

        let wav_hh_hl_lh_hh: Vec<Vec<Vec<f64>>> = get_ll_hl_lh_hh(&wav_trans_1);

        println!("\n wav_hh_hl_lh_hh: {:?}", wav_hh_hl_lh_hh);

        let combined_wav: Vec<Vec<f64>> = combine_ll_lh_hl_hh(&wav_hh_hl_lh_hh);

        println!("\n combined wav_hh_hl_lh_hh: {:?}", combined_wav);

        test_gradient_error_2d_f64(&wav_trans_1, &combined_wav, 1e-6);
    }

    #[test]
    fn test_wavelet_row_partial() {
        let seq_len = 453;
        let dim = 16;

        let matrix_1: Vec<Vec<f64>> = transpose(&generate_random_f64_2d(seq_len, dim));

        let wav_trans_1: Vec<Vec<f64>> = transform_2_d_partial_f64(&matrix_1, &DiscreteWaletetType::DB1, &WaveletMode::SYMMETRIC);

        println!("\n matrix 1: dim: {} {}", matrix_1.len(), matrix_1[0].len());
        println!("\n wav_trans_1: dim: {} {}", wav_trans_1.len(), wav_trans_1[0].len());

        let wav_hh_ll: Vec<Vec<Vec<f64>>> = get_ll_hh(&wav_trans_1);
        let combined_wav: Vec<Vec<f64>> = combine_ll_hh(&wav_hh_ll);
        let wav_ll = &wav_hh_ll[0];

        println!("print wav trends: {} {}", wav_ll.len(), wav_ll[0].len());

        println!("\n combined wav_hh_ll: {:?} {}", combined_wav.len(), combined_wav[0].len());
        test_gradient_error_2d_f64(&wav_trans_1, &combined_wav, 1e-6);
    }
}
