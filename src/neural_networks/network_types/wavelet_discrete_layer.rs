use core::fmt::Debug;
use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::{
        network_components::{gradient_struct::Gradient, layer::LayerEnum, layer_input_struct::LayerInput, layer_output_struct::LayerOutput, norm_layer::NormalNormLayer},
        utils::matrix::{add_matrix_2d_c, add_matrix_3d, transpose},
    },
    utils::array::unzip5,
    wavelet_transform::{
        dwt::{dwt_1d, dwt_2d_full, dwt_2d_partial, get_ll_hh, get_ll_hh_1d, get_ll_hl_lh_hh, grad_dwt_2d, grad_dwt_2d_partial},
        dwt_types::DiscreteWaveletType,
        modes::WaveletMode,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteWaveletLayer {
    pub wavelet: DiscreteWaveletType,
    pub wavelet_mode: WaveletMode,
    pub is_full_mode: bool,
    pub wavelet_size: usize,
    pub compression_levels: usize,
    pub norm_layer: Option<LayerEnum>,
    pub add_details: bool,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,

    #[serde(skip)]
    pub input_only_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub previous_gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub trend_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub details_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub compression_dims: Option<Vec<Vec<usize>>>,
    #[serde(skip)]
    pub compressed_padding_mask_b: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub target_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub target_batch_ids: Option<Vec<Vec<u32>>>,
}

impl DiscreteWaveletLayer {
    pub fn new() -> Self {
        let _norm_layer = Some(LayerEnum::Norm(Box::new(NormalNormLayer::new(16, 1e-10, 0.001))));

        Self {
            input_batch: None,
            trend_input_batch: None,
            input_only_batch: None,
            previous_gradient_input_batch: None,
            gradient: None,
            output_batch: None,
            time_step: 0,
            wavelet: DiscreteWaveletType::DB6,
            wavelet_size: 32,
            compression_levels: 8,
            wavelet_mode: WaveletMode::ZERO,
            add_details: true,
            is_full_mode: false,
            details_batch: None,
            compression_dims: None,
            compressed_padding_mask_b: None,
            padding_mask_batch: None,
            target_batch: None,
            target_batch_ids: None,
            norm_layer: _norm_layer,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();
        let target_batch_ids: Vec<Vec<u32>> = layer_input.get_target_batch_ids();
        let forward_only = layer_input.get_forward_only();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
        let time_step = layer_input.get_time_step();

        let results: Vec<_> = input_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, input)| {
                if self.is_full_mode {
                    let dwt_full = dwt_2d_full(input, &self.wavelet, &self.wavelet_mode);
                    let ll_hl_lh_hh = get_ll_hl_lh_hh(&dwt_full);
                    (ll_hl_lh_hh[0].to_vec(), vec![], vec![], vec![], vec![]) // No detail saved in full mode
                } else {
                    let padding_mask = &padding_mask_batch[batch_ind];
                    let mut target_ids: &Vec<u32> = &vec![];

                    if !target_batch_ids.is_empty() {
                        target_ids = &target_batch_ids[batch_ind];
                    }

                    let mut trend: Vec<Vec<Complex<f64>>> = input.clone();
                    let mut details: Vec<Vec<Complex<f64>>> = vec![];
                    let mut input_only_separated = input.clone();
                    let mut target_emb: Vec<Vec<Complex<f64>>> = vec![];
                    let mut comp_pad_mask_b: Vec<u32> = padding_mask.clone();
                    let mut compression_dims: Vec<usize> = vec![];

                    if !forward_only || (forward_only && time_step == 0) {
                        let (mut input_only, target, pad_inp_mask) = self.separate_input_target(input, target_ids, padding_mask);

                        if input_only.is_empty() {
                            input_only = input.clone();
                        }
                        let (new_trend, _new_details, compression_dim) = self.compress_partial(&input_only);

                        compression_dims = compression_dim;
                        comp_pad_mask_b = self.compress_padding_mask(&pad_inp_mask);
                        input_only_separated = input_only;

                        trend = new_trend;
                        details = _new_details;
                        target_emb = target;
                    }

                    if !target_emb.is_empty() {
                        // [input + padding + target]
                        trend.extend_from_slice(&target_emb);
                        comp_pad_mask_b.extend_from_slice(&vec![1; target_emb.len()]);
                    }
                    //assert_eq!(comp_pad_mask_b.len(), trend.len());

                    // println!("trend final compressed: {:?}", trend.len());

                    (trend, details, input_only_separated, compression_dims, comp_pad_mask_b)
                }
            })
            .collect();

        let (trend_batch, details_batch, input_only, compression_dims, comp_pad_mask_b) = unzip5(results);

        let mut layer_input = layer_input.clone();
        layer_input.set_input_batch(trend_batch.clone());
        layer_input.set_input_batch_before(trend_batch.clone());

        let mut trend_batch: Vec<Vec<Vec<Complex<f64>>>> = trend_batch.clone();

        // Apply the RMS normalization layer
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let rms_output = rms_norm_layer.forward(&layer_input);
                    trend_batch = rms_output.get_output_batch();
                    //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
                }
                LayerEnum::Norm(norm_layer) => {
                    let layer_output = norm_layer.forward(&layer_input);
                    trend_batch = layer_output.get_output_batch();

                    //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
                }
                _ => {}
            }
        }

        self.input_batch = Some(input_batch.clone());
        self.input_only_batch = Some(input_only);
        self.trend_input_batch = Some(trend_batch.clone());
        self.details_batch = Some(details_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.output_batch = Some(trend_batch.clone());
        self.target_batch_ids = Some(target_batch_ids);
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.compressed_padding_mask_b = Some(comp_pad_mask_b.clone());
        self.compression_dims = Some(compression_dims);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(trend_batch.clone());
        layer_output.set_padding_mask_batch(padding_mask_batch.clone());

        if trend_batch[0].len() != padding_mask_batch[0].len() {
            layer_output.set_padding_mask_batch(comp_pad_mask_b.clone());
        }

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found");
        let target_batch_ids = self.target_batch_ids.as_ref().expect("no target_batch_ids found");
        let input_only_batch = self.input_only_batch.as_ref().expect("no input only batch found");
        let mut grad_output_batch = previous_gradient.get_gradient_input_batch();
        let compression_dims = self.compression_dims.as_ref().expect("no compression dwt found");

        assert_eq!(input_batch.len(), grad_output_batch.len(), "Input and gradient batch size mismatch");

        let mut output_gradient_norm = vec![];

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(grad_output_batch.clone());

        //Apply RMSNorm backpropagation if it's present
        if let Some(norm_layer) = &mut self.norm_layer {
            match norm_layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    gradient = rms_norm_layer.backward(&grad_output_batch);
                    grad_output_batch = gradient.get_gradient_input_batch();
                    output_gradient_norm = grad_output_batch.clone();
                    // println!("FFN, gradient from RMS Norm backward: {}, {}, {}", output_gradients.len(), output_gradients[0].len(), output_gradients[0][0].len());
                }
                LayerEnum::Norm(norm_layer) => {
                    gradient = norm_layer.backward(&gradient);
                    grad_output_batch = gradient.get_gradient_input_batch();
                    output_gradient_norm = grad_output_batch.clone();
                    //println!("FFN, gradient from Norm backward: {}, {}, {}", output_gradients.len(), output_gradients[0].len(), output_gradients[0][0].len());
                }
                _ => {}
            }
        }

        let mut grad_input_batch: Vec<Vec<Vec<Complex<f64>>>> = grad_output_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, grad_output)| {
                if self.is_full_mode {
                    grad_dwt_2d(grad_output, &self.wavelet, &self.wavelet_mode)
                } else {
                    let input_only = &input_only_batch[batch_ind];
                    let target_ids = &target_batch_ids[batch_ind];
                    let compression_dim = &compression_dims[batch_ind];
                    let target_len = target_ids.len();
                    // let target_len = 0;

                    let mut gradient_decompr = self.decompress_partial(&grad_output, target_len, compression_dim);

                    if input_only.len() != gradient_decompr.len() {
                        gradient_decompr = self.align_gradient_to_input(input_only, &gradient_decompr);
                        //gradient_decompr = self.align_gradient_rows_complex(input_only, &gradient_decompr, &grad_output);
                    }

                    for i in (grad_output.len() - target_ids.len())..grad_output.len() {
                        gradient_decompr.push(grad_output[i].clone());
                    }

                    gradient_decompr
                }
            })
            .collect();

        if !output_gradient_norm.is_empty() {
            grad_input_batch = add_matrix_3d(&grad_input_batch, &grad_input_batch);
        }

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            grad_input_batch = add_matrix_3d(&grad_input_batch, &previous_gradient.get_gradient_input_batch());
        }

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(grad_input_batch.clone());

        self.previous_gradient_input_batch = Some(grad_input_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn compress_partial(&self, input: &[Vec<Complex<f64>>]) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<usize>) {
        let mut wav_out = input.to_vec();
        let mut details = Vec::new();
        let mut compression_dim: Vec<usize> = vec![];

        //  println!("compression_________________________________________________");
        for _i in 0..self.compression_levels {
            //println!("trend dim: {} {}", trend.len(), trend[0].len());
            let dwt_partial = dwt_2d_partial(&transpose(&wav_out), &self.wavelet, &self.wavelet_mode);
            let ll_hh = get_ll_hh(&dwt_partial);

            let trend = transpose(&ll_hh[0]);
            details = transpose(&ll_hh[1]);

            if self.add_details {
                wav_out = add_matrix_2d_c(&trend, &details);
            } else {
                wav_out = trend.clone();
            }

            compression_dim.push(trend.len());
        }

        //println!("trend dim: {} {}", trend.len(), trend[0].len());
        (wav_out, details, compression_dim)
    }

    pub fn decompress_partial(&self, grad_output: &Vec<Vec<Complex<f64>>>, target_len: usize, _compression_dim: &Vec<usize>) -> Vec<Vec<Complex<f64>>> {
        let seq_len = grad_output.len();
        let gradient_without_target = grad_output[..seq_len.saturating_sub(target_len)].to_vec();
        let mut gradient_transp = transpose(&gradient_without_target);

        for _l in (0.._compression_dim.len()).rev() {
            for i in 0..gradient_transp.len() {
                let detail_extension: Vec<Complex<f64>>;

                if self.add_details {
                    detail_extension = gradient_transp[i].to_vec();
                } else {
                    detail_extension = vec![Complex::new(0.0, 0.0); _compression_dim[_l]];
                }

                gradient_transp[i].extend_from_slice(&detail_extension);
            }

            gradient_transp = grad_dwt_2d_partial(&gradient_transp, &self.wavelet, &self.wavelet_mode);

            // println!("grad restored at index {}: {} {}", _compression_dim[_l], gradient_transp.len(), gradient_transp[0].len());
        }

        let gradient_transposed = transpose(&gradient_transp);

        gradient_transposed
    }
    pub fn align_vectors(&self, a: &mut Vec<Complex<f64>>, b: &mut Vec<Complex<f64>>) {
        let len_a = a.len();
        let len_b = b.len();

        if len_a < len_b {
            a.extend(std::iter::repeat(Complex::new(0.0, 0.0)).take(len_b - len_a));
        }
    }
    pub fn truncate_vectors(&self, a: &mut Vec<Complex<f64>>, b_len: usize) {
        let min_len = a.len().min(b_len);
        a.truncate(min_len);
    }
    pub fn compress_padding_mask(&self, padding_mask: &Vec<u32>) -> Vec<u32> {
        let input_f64: Vec<f64> = padding_mask.iter().map(|v| *v as f64).collect();
        let mut dwt_partial: Vec<f64> = input_f64.clone();

        for _ in 0..self.compression_levels {
            let dwt = dwt_1d(&dwt_partial, &self.wavelet, &self.wavelet_mode);
            let wav_hh_ll: Vec<Vec<f64>> = get_ll_hh_1d(&dwt);
            dwt_partial = wav_hh_ll[0].clone(); // Approximation (LL)

            // if dwt_partial.len() % self.wavelet_size != 0 {
            //     // dwt_partial.resize(dwt_partial.len() + (self.wavelet_size - (dwt_partial.len() % self.wavelet_size)), 1.0);
            // }
        }

        // Step 4: Threshold the approximation to get new mask

        //  println!("original dwt: {:?}", dwt_partial);
        let compressed_mask: Vec<u32> = vec![1; dwt_partial.len()];

        // println!("original padding_mask: {:?}", padding_mask);
        // println!("compressed mask: {:?}", compressed_mask);
        compressed_mask
    }
    pub fn align_gradient_rows_complex(&self, input: &Vec<Vec<Complex<f64>>>, gradient_decompr: &Vec<Vec<Complex<f64>>>, gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let input_rows = input.len();
        let cols = gradient_decompr[0].len();

        let mut result = Vec::with_capacity(gradient.len());

        // assert_eq!(gradient.len(), input.len());

        // Copy the first (input_rows - 1) rows
        for i in 0..(input_rows - 1) {
            if i < gradient_decompr.len() - 1 {
                result.push(gradient_decompr[i].clone());
            }
        }

        // Sum remaining rows into the last row
        let mut last_row = vec![Complex::new(0.0, 0.0); cols];

        for i in (input_rows - 1)..gradient_decompr.len() {
            for j in 0..cols {
                last_row[j] += gradient_decompr[i][j];
            }
        }
        result.push(last_row);

        result
    }
    pub fn separate_input_target(&self, input: &Vec<Vec<Complex<f64>>>, target_ids: &Vec<u32>, padding_mask: &Vec<u32>) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<u32>) {
        let mut input_without_target: Vec<Vec<Complex<f64>>> = vec![];
        let mut target: Vec<Vec<Complex<f64>>> = vec![];
        let mut padding_input_mask: Vec<u32> = vec![];

        let total_len = input.len();
        let target_len = target_ids.len();
        let padding_len = padding_mask.iter().filter(|&&x| x == 0).count();

        assert_eq!(padding_mask.len(), total_len);
        assert_eq!(target_len + padding_len <= total_len, true);

        // Heuristic: if target comes after padding (input + padding + target)
        // the last `target_len` items in the input (excluding padding) are target
        let non_padded_indices: Vec<usize> = padding_mask.iter().enumerate().filter_map(|(i, &m)| if m == 1 { Some(i) } else { None }).collect();

        // If the last `target_len` non-padded elements are at the end, it's input + padding + target
        let expected_target_indices: Vec<usize> = non_padded_indices.iter().rev().take(target_len).cloned().collect();

        let mut is_input_padding_target = true;
        for &idx in &expected_target_indices {
            if idx >= total_len - target_len {
                continue;
            } else {
                is_input_padding_target = false;
                break;
            }
        }

        if is_input_padding_target {
            // Case: input + padding + target
            for i in 0..total_len {
                if padding_mask[i] == 1 {
                    if non_padded_indices[non_padded_indices.len() - target_len..].contains(&i) {
                        target.push(input[i].clone());
                    } else {
                        input_without_target.push(input[i].clone());
                        padding_input_mask.push(1);
                    }
                } else {
                    input_without_target.push(input[i].clone());
                    padding_input_mask.push(0);
                }
            }
        } else {
            // Case: input + target + padding
            let input_len = total_len - target_len - padding_len;
            for i in 0..input_len {
                input_without_target.push(input[i].clone());
                padding_input_mask.push(1);
            }
            for i in input_len..input_len + padding_len {
                input_without_target.push(input[i].clone());
                padding_input_mask.push(0);
            }
            for i in input_len + padding_len..total_len {
                target.push(input[i].clone());
            }
        }

        assert_eq!(input_without_target.len() + target.len(), input.len());
        assert_eq!(padding_input_mask.len(), input.len() - target.len());

        (input_without_target, target, padding_input_mask)
    }
    pub fn align_gradient_to_input(&self, input: &Vec<Vec<Complex<f64>>>, gradient_decompr: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let input_rows = input.len();
        let input_cols = input[0].len();

        let mut result = vec![vec![Complex::new(0.0, 0.0); input_cols]; input_rows];

        // println!("gradient restored dim: {} {}", gradient_decompr.len(), gradient_decompr[0].len());

        // Copy the first (input_rows - 1) rows
        for i in 0..input.len() {
            for j in 0..input[i].len() {
                result[i][j] = gradient_decompr[i][j];
            }
        }

        // if gradient_decompr.len() != input.len() {
        //     let len_diff: usize = gradient_decompr.len() - input.len();

        //     for i in (input.len() - 7)..input.len() {
        //         for j in 0..input[i].len() {
        //             result[i][j] += gradient_decompr[i + len_diff][j];
        //         }
        //     }
        // }

        // println!("grad aligned to input: {} {}", result.len(), result[0].len());

        result
    }

    pub fn update_parameters(&mut self) {
         // Apply RMSNorm backpropagation if it's present
        if let Some(layer_enum) = &mut self.norm_layer {
            match layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    rms_norm_layer.update_parameters();
                }
                LayerEnum::Norm(norm_layer) => {
                    norm_layer.update_parameters();
                }
                _ => {}
            }
        }
    }
}
