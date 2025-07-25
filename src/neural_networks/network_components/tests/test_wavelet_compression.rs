#[cfg(test)]
mod test_wavelet_compression {

    use num::Complex;

    use crate::{
        neural_networks::utils::{
            derivative::{test_gradient_error_2d, test_gradient_error_2d_f64},
            matrix::{average_matrix_by_scalar, transpose},
            random_arrays::{generate_random_complex_2d, generate_random_f64_2d},
        },
        wavelet_transform::{
            dwt::{combine_ll_hh, combine_ll_lh_hl_hh, dwt_2d_full, dwt_2d_partial, get_ll_hh, get_ll_hl_lh_hh, inverse_dwt_2d_partial},
            dwt_types::DiscreteWaveletType,
            modes::WaveletMode,
        },
    };

    #[test]
    fn test_wavelet_full() {
        let seq_len = 6;
        let dim = 6;

        let matrix_1: Vec<Vec<f64>> = generate_random_f64_2d(seq_len, dim);

        let wav_trans_1: Vec<Vec<f64>> = dwt_2d_full(&matrix_1, &DiscreteWaveletType::DB1, &WaveletMode::SYMMETRIC);

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

        let wav_trans_1: Vec<Vec<f64>> = dwt_2d_partial(&matrix_1, &DiscreteWaveletType::DB1, &WaveletMode::SYMMETRIC);

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

        let wav_trans_1: Vec<Vec<Complex<f64>>> = dwt_2d_partial(&matrix_1, &DiscreteWaveletType::DB1, &WaveletMode::SYMMETRIC);

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

        let matrix_restored = inverse_dwt_2d_partial(&wav_trans_1, &DiscreteWaveletType::DB1, &WaveletMode::SYMMETRIC, 0);

        // println!("matrix: {:?}", matrix_restored);
        test_gradient_error_2d(&matrix_1, &matrix_restored, 1e-6);
    }

    #[test]
    fn test_complex_multilevel_wavelet_partial() {
        let seq_len = 20;
        let dim = 4;
        let compr_levels = 10;
        let wavelet_type = &DiscreteWaveletType::DB4;

        let matrix_1: Vec<Vec<Complex<f64>>> = transpose(&generate_random_complex_2d(seq_len, dim));

        let mut wav_trans_1 = matrix_1.clone();
        for _i in 0..compr_levels {
            wav_trans_1 = dwt_2d_partial(&wav_trans_1, wavelet_type, &WaveletMode::SYMMETRIC);
            println!("\n wav_trans at {}: dim: {} {}", _i, wav_trans_1.len(), wav_trans_1[0].len());
        }

        let mut matrix_restored = wav_trans_1.clone();
        for _i in 0..compr_levels {
            matrix_restored = inverse_dwt_2d_partial(&matrix_restored, wavelet_type, &WaveletMode::SYMMETRIC, 0);
        }

        println!("original matrix: {:?}", matrix_1);
        println!("\n matrix 1: dim: {} {}", matrix_1.len(), matrix_1[0].len());

        println!("restored original matrix: {:?}", matrix_restored);
        println!("restored original matrix dim: {} {}", matrix_restored.len(), matrix_restored[0].len());

        test_gradient_error_2d(&matrix_1, &matrix_restored, 1e-6);
    }

    #[test]
    fn test_complex_multilevel_compressed_wavelet_partial() {
        let seq_len = 75;
        let dim = 4;
        let compr_levels = 10;
        let wavelet_type = &DiscreteWaveletType::DB4;

        let matrix_1: Vec<Vec<Complex<f64>>> = transpose(&generate_random_complex_2d(seq_len, dim));
        let mut details_coeffs: Vec<Vec<Vec<Complex<f64>>>> = vec![];

        let mut wav_trans_1: Vec<Vec<Complex<f64>>> = matrix_1.clone();
        for _i in 0..compr_levels {
            wav_trans_1 = dwt_2d_partial(&wav_trans_1, wavelet_type, &WaveletMode::SYMMETRIC);

            let ll_hh: Vec<Vec<Vec<Complex<f64>>>> = get_ll_hh(&wav_trans_1);
            wav_trans_1 = ll_hh[0].clone();
            details_coeffs.push(ll_hh[1].clone());
            println!("details coeff at index: {}, {} {}", _i, ll_hh[1].len(), ll_hh[1][0].len());
            println!("\n wav_trans at {}: dim: {} {}", _i, wav_trans_1.len(), wav_trans_1[0].len());
        }

        let mut matrix_restored: Vec<Vec<Complex<f64>>> = wav_trans_1.clone();
        for _i in (0..compr_levels).rev() {
            println!("index i in decompression: {}", _i);

            println!("matrix restored before extend: {} {}", matrix_restored.len(), matrix_restored[0].len());
            for j in 0..matrix_restored.len() {
                if matrix_restored[j].len() != details_coeffs[_i][j].len() {
                    let min_len = matrix_restored[j].len().min(details_coeffs[_i][j].len());
                    matrix_restored[j].truncate(min_len);
                    details_coeffs[_i][j].truncate(min_len);
                }
                matrix_restored[j].extend_from_slice(&details_coeffs[_i][j]);
            }
            println!("matrix restored after extend: {} {}", matrix_restored.len(), matrix_restored[0].len());
            matrix_restored = inverse_dwt_2d_partial(&matrix_restored, wavelet_type, &WaveletMode::SYMMETRIC, 0);
            println!("matrix restored after dwt inv.: {} {}", matrix_restored.len(), matrix_restored[0].len());
        }

        // println!("\n original matrix: {:?}", matrix_1);
        println!("\n matrix 1: dim: {} {}", matrix_1.len(), matrix_1[0].len());

        // println!("\n restored original matrix: {:?}", matrix_restored);
        println!("\n restored original matrix dim: {} {}", matrix_restored.len(), matrix_restored[0].len());

        test_gradient_error_2d(&matrix_1, &matrix_restored, 1e-6);
    }

    #[test]
    fn test_complex_batch_averaging() {
        let matrix_1: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.5, 0.5), Complex::new(0.1, 0.2), Complex::new(0.2, 0.6), Complex::new(0.3, 0.9), Complex::new(0.4, 0.5)]];

        let averaging = average_matrix_by_scalar(&matrix_1, 2.0);

        println!("\n original matrix: {:?}", matrix_1);
        println!("\n batch size: {}", 2.0);

        println!("\n averaging: dim: {} {}", averaging.len(), averaging[0].len());
        println!("\n averaging: {:?}", averaging);
    }
}
