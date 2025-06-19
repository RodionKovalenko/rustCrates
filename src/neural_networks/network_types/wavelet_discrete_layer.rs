use core::fmt::Debug;
use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
        utils::matrix::transpose,
    },
    wavelet_transform::{
        dwt::{dwt_1d, dwt_2d_full, dwt_2d_partial, get_ll_hh, get_ll_hh_1d, get_ll_hl_lh_hh, grad_dwt_2d, grad_dwt_2d_partial},
        dwt_types::DiscreteWaletetType,
        modes::WaveletMode,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteWaveletLayer {
    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    trend_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    input_only_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub previous_gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    gradient: Option<Gradient>,
    time_step: usize,
    #[serde(skip)]
    output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub wavelet: DiscreteWaletetType,
    pub wavelet_mode: WaveletMode,
    pub is_full_mode: bool,
    #[serde(skip)]
    pub approx_details: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub compression_levels: Option<Vec<i32>>,
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
        Self {
            input_batch: None,
            trend_input_batch: None,
            input_only_batch: None,
            previous_gradient_input_batch: None,
            gradient: None,
            output_batch: None,
            time_step: 0,
            wavelet: DiscreteWaletetType::DB1,
            wavelet_mode: WaveletMode::SYMMETRIC,
            is_full_mode: false,
            approx_details: None,
            compression_levels: None,
            compressed_padding_mask_b: None,
            padding_mask_batch: None,
            target_batch: None,
            target_batch_ids: None,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();
        let target_batch_ids: Vec<Vec<u32>> = layer_input.get_target_batch_ids();
        let forward_only = layer_input.get_forward_only();
        let padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
        let mut _min_compressed_dim = 16;
        let time_step = layer_input.get_time_step();

        if !forward_only {
            _min_compressed_dim = target_batch_ids[0].len();
        }

        let results: Vec<_> = input_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, input)| {
                if self.is_full_mode {
                    let dwt_full = dwt_2d_full(input, &self.wavelet, &self.wavelet_mode);
                    let ll_hl_lh_hh = get_ll_hl_lh_hh(&dwt_full);
                    (ll_hl_lh_hh[0].to_vec(), vec![], vec![], 0) // No detail saved in full mode
                } else {
                    let padding_mask = &padding_mask_batch[batch_ind];
                    let mut target_ids: &Vec<u32> = &vec![];

                    if !target_batch_ids.is_empty() {
                        target_ids = &target_batch_ids[batch_ind];
                    }

                    let mut trend: Vec<Vec<Complex<f64>>> = input.clone();
                    let mut input_only_separated = input.clone();
                    let mut target_emb: Vec<Vec<Complex<f64>>> = vec![];
                    let mut comp_pad_mask_b: Vec<u32> = padding_mask.clone();
                    let mut compress_levels: i32 = 0;

                    if !forward_only || (forward_only && time_step == 0) {
                        let (input_only, target, pad_inp_mask) = self.separate_input_target(input, target_ids, padding_mask);

                        // println!("input only dim: {:?} {}", input_only.len(), input_only[0].len());
                        // println!("input dim: {:?} {}", input.len(), input[0].len());
                        // println!("target ids dim: {:?}", target_ids.len());

                        let (new_trend, _new_details) = self.compress_partial(&input_only);

                        comp_pad_mask_b = self.compress_padding_mask(&pad_inp_mask);
                        input_only_separated = input_only;

                        trend = new_trend;
                        target_emb = target;
                        compress_levels += 1;
                    }

                    // println!("trend before extend slice: {} {}", trend.len(), trend[0].len());
                    // println!("padding mask before extend slice: {:?}", comp_pad_mask_b);

                    if !target_emb.is_empty() {
                        // [input + padding + target]
                        trend.extend_from_slice(&target_emb);
                        comp_pad_mask_b.extend_from_slice(&vec![1; target_emb.len()]);
                    }

                    // println!("trend after extend slice: {} {}", trend.len(), trend[0].len());
                    // println!("padding mask after extend slice: {:?}", comp_pad_mask_b);

                    (trend, input_only_separated, comp_pad_mask_b, compress_levels)
                }
            })
            .collect();

        let (trend_batch, input_only, comp_pad_mask_b, compress_levels) = self.unzip4(results);

        self.input_batch = Some(input_batch.clone());
        self.input_only_batch = Some(input_only);
        self.trend_input_batch = Some(trend_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.output_batch = Some(trend_batch.clone());
        self.compression_levels = Some(compress_levels);
        self.target_batch_ids = Some(target_batch_ids);
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.compressed_padding_mask_b = Some(comp_pad_mask_b.clone());

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
        let grad_output_batch = previous_gradient.get_gradient_input_batch();

        assert_eq!(input_batch.len(), grad_output_batch.len(), "Input and gradient batch size mismatch");

        let grad_input_batch: Vec<Vec<Vec<Complex<f64>>>> = grad_output_batch
            .par_iter()
            .enumerate()
            .map(|(batch_ind, grad_output)| {
                if self.is_full_mode {
                    grad_dwt_2d(grad_output, &self.wavelet, &self.wavelet_mode)
                } else {
                    let input_only = &input_only_batch[batch_ind];
                    let target_ids = &target_batch_ids[batch_ind];

                    let mut gradient_decompr = self.decompress_partial(&grad_output, target_ids.len());

                    // println!("gradient decomp dim: {} {}", gradient_decompr.len(), gradient_decompr[0].len());
                    // println!("input_only dim: {} {}", input_only.len(), input_only[0].len());
                    // println!("gradient output dim: {} {}", grad_output.len(), grad_output[0].len());

                    if input_only.len() != gradient_decompr.len() {
                        gradient_decompr = self.align_gradient_rows_complex(input_only, &gradient_decompr, grad_output);
                    }

                    for i in (grad_output.len() - target_ids.len())..grad_output.len() {
                        gradient_decompr.push(grad_output[i].clone());
                    }

                    gradient_decompr
                }
            })
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(grad_input_batch.clone());

        self.previous_gradient_input_batch = Some(grad_input_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }
    pub fn unzip3<A, B, C>(&self, v: Vec<(A, B, C)>) -> (Vec<A>, Vec<B>, Vec<C>) {
        let mut va = Vec::with_capacity(v.len());
        let mut vb = Vec::with_capacity(v.len());
        let mut vc = Vec::with_capacity(v.len());
        for (a, b, c) in v {
            va.push(a);
            vb.push(b);
            vc.push(c);
        }
        (va, vb, vc)
    }
    pub fn unzip4<A, B, C, D>(&self, v: Vec<(A, B, C, D)>) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>) {
        let mut va = Vec::with_capacity(v.len());
        let mut vb = Vec::with_capacity(v.len());
        let mut vc = Vec::with_capacity(v.len());
        let mut vd = Vec::with_capacity(v.len());
        for (a, b, c, d) in v {
            va.push(a);
            vb.push(b);
            vc.push(c);
            vd.push(d)
        }
        (va, vb, vc, vd)
    }
    pub fn separate_input_target(&self, input: &Vec<Vec<Complex<f64>>>, target_ids: &Vec<u32>, padding_mask: &Vec<u32>) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<u32>) {
        let mut input_without_target: Vec<Vec<Complex<f64>>> = vec![];
        let mut target: Vec<Vec<Complex<f64>>> = vec![];
        let padding_len = padding_mask.iter().filter(|x| **x == 0).count();
        let input_end = input.len() - target_ids.len() - padding_len;
        let mut padding_input_mask: Vec<u32> = vec![1; input_end];

        // [input + padding + target]
        // println!("\n input size: {}", input.len());
        // println!("\n input end: {}", input_end);
        // println!("\n padding size: {}", padding_len);
        // println!("\n target size: {}", target_ids.len());

        // only input
        for i in 0..input_end {
            input_without_target.push(input[i].clone());
        }
        // padding added to input
        let mut pad_ind_end = 0;

        for i in 0..padding_mask.len() {
            if padding_mask[i] == 0 {
                input_without_target.push(input[i].clone());
                padding_input_mask.push(0);

                pad_ind_end = i;
            }
        }

        if pad_ind_end == 0 {
            pad_ind_end = input_end;
        }

        for i in pad_ind_end..padding_mask.len() {
            if padding_mask[i] == 1 {
                target.push(input[i].clone());
            }
        }

        // println!("input without target: {:?}", input_without_target);
        // println!("target only: {:?}", target);
        // println!("padding input mask: {:?}", padding_input_mask);

        assert_eq!(target.len() + input_without_target.len(), input.len());
        assert_eq!(padding_input_mask.len(), input.len() - target.len());

        (input_without_target, target, padding_input_mask)
    }
    pub fn compress_partial(&self, input: &Vec<Vec<Complex<f64>>>) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>) {
        let dwt_partial: Vec<Vec<Complex<f64>>> = dwt_2d_partial(&transpose(&input), &self.wavelet, &self.wavelet_mode);
        let wav_hh_ll: Vec<Vec<Vec<Complex<f64>>>> = get_ll_hh(&dwt_partial);
        let detail: Vec<Vec<Complex<f64>>> = wav_hh_ll[1].clone(); // Save HH detail
        let approx: Vec<Vec<Complex<f64>>> = transpose(&wav_hh_ll[0]); // Approximation (LL)
        (approx, detail)
    }
    pub fn decompress_partial(&self, grad_output: &Vec<Vec<Complex<f64>>>, target_len: usize) -> Vec<Vec<Complex<f64>>> {
        let mut gradient_comp = vec![];

        for i in 0..(grad_output.len() - target_len) {
            gradient_comp.push(grad_output[i].clone());
        }

        let gradient_partial = grad_dwt_2d_partial(&transpose(&gradient_comp), &self.wavelet, &self.wavelet_mode);
        let gradient_transposed = transpose(&gradient_partial);

        gradient_transposed
    }
    pub fn compress_padding_mask(&self, padding_mask: &Vec<u32>) -> Vec<u32> {
        let input_f64: Vec<f64> = padding_mask.iter().map(|v| *v as f64).collect();
        let dwt_partial: Vec<f64> = dwt_1d(&input_f64, &self.wavelet, &self.wavelet_mode);

        let wav_hh_ll: Vec<Vec<f64>> = get_ll_hh_1d(&dwt_partial);
        let approx: Vec<f64> = wav_hh_ll[0].clone(); // Approximation (LL)

        // Step 4: Threshold the approximation to get new mask
        let compressed_mask: Vec<u32> = approx.iter().map(|&val| if val > 0.99 { 1 } else { 0 }).collect();

        // println!("padding_mask: {:?}", input_f64);
        // println!("paddimg mask dim: {:?}", input_f64.len());

        // println!("compressed_mask: {:?}", compressed_mask);
        // println!("compressed_mask dim: {:?}", compressed_mask.len());

        compressed_mask
    }
    pub fn align_gradient_rows_complex(&self, input: &Vec<Vec<Complex<f64>>>, gradient_decompr: &Vec<Vec<Complex<f64>>>, gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let input_rows = input.len();
        let cols = gradient_decompr[0].len();

        // println!("\n input dim: {} {}", input.len(), input[0].len());
        // println!("\n gradient dim: {} {}", gradient.len(), gradient[0].len());
        // println!("\n gradient decompre dim: {} {}", gradient_decompr.len(), gradient_decompr[0].len());

        let mut result = Vec::with_capacity(gradient.len());

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
}
