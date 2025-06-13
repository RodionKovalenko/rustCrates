#[cfg(test)]
mod test_wavelet_compression {

    use num::Complex;

    use crate::{
        neural_networks::utils::{
            derivative::{test_gradient_error_2d, test_gradient_error_2d_f64},
            matrix::transpose,
            random_arrays::{generate_random_complex_2d, generate_random_f64_2d},
        },
        wavelet_transform::{
            dwt::{combine_ll_hh, combine_ll_lh_hl_hh, dwt_2d_full, dwt_2d_partial, get_ll_hh, get_ll_hl_lh_hh, inverse_dwt_2d_partial},
            dwt_types::DiscreteWaletetType,
            modes::WaveletMode,
        },
    };

    #[test]
    fn test_wavelet_full() {
        let seq_len = 6;
        let dim = 6;

        let matrix_1: Vec<Vec<f64>> = generate_random_f64_2d(seq_len, dim);

        let wav_trans_1: Vec<Vec<f64>> = dwt_2d_full(&matrix_1, &DiscreteWaletetType::DB1, &WaveletMode::SYMMETRIC);

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
        let seq_len = 8;
        let dim = 8;

        let matrix_1: Vec<Vec<f64>> = transpose(&generate_random_f64_2d(seq_len, dim));

        let wav_trans_1: Vec<Vec<f64>> = dwt_2d_partial(&matrix_1, &DiscreteWaletetType::DB1, &WaveletMode::SYMMETRIC);

        println!("\n matrix 1: dim: {} {}", matrix_1.len(), matrix_1[0].len());
        println!("\n wav_trans_1: dim: {} {}", wav_trans_1.len(), wav_trans_1[0].len());

        let wav_hh_ll: Vec<Vec<Vec<f64>>> = get_ll_hh(&wav_trans_1);
        let combined_wav: Vec<Vec<f64>> = combine_ll_hh(&wav_hh_ll);
        let wav_ll = &wav_hh_ll[0];

        println!("print wav trends: {} {}", wav_ll.len(), wav_ll[0].len());

        println!("\n combined wav_hh_ll: {:?} {}", combined_wav.len(), combined_wav[0].len());
        test_gradient_error_2d_f64(&wav_trans_1, &combined_wav, 1e-6);
    }

    #[test]
    fn test_complex_wavelet_row_partial() {
        let seq_len = 450;
        let dim = 16;

        let matrix_1: Vec<Vec<Complex<f64>>> = transpose(&generate_random_complex_2d(seq_len, dim));

        let wav_trans_1: Vec<Vec<Complex<f64>>> = dwt_2d_partial(&matrix_1, &DiscreteWaletetType::DB1, &WaveletMode::SYMMETRIC);

        println!("\n matrix 1: dim: {} {}", matrix_1.len(), matrix_1[0].len());
        println!("\n wav_trans_1: dim: {} {}", wav_trans_1.len(), wav_trans_1[0].len());

        let wav_hh_ll: Vec<Vec<Vec<Complex<f64>>>> = get_ll_hh(&wav_trans_1);
        let combined_wav: Vec<Vec<Complex<f64>>> = combine_ll_hh(&wav_hh_ll);
        let wav_ll = &wav_hh_ll[0];

        // println!("\n original matrix: {:?}", matrix_1);
        // println!("\n wavelet complex: {:?}", wav_trans_1);

        println!("print wav trends: {} {}", wav_ll.len(), wav_ll[0].len());

        println!("\n combined wav_hh_ll: {:?} {}", combined_wav.len(), combined_wav[0].len());
        test_gradient_error_2d(&wav_trans_1, &combined_wav, 1e-6);

        let matrix_restored = inverse_dwt_2d_partial(&wav_trans_1, &DiscreteWaletetType::DB1, &WaveletMode::SYMMETRIC, 0);

        // println!("matrix: {:?}", matrix_restored);
        test_gradient_error_2d(&matrix_1, &matrix_restored, 1e-6);
    }
}
