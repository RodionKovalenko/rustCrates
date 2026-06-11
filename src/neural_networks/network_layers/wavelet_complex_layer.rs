use core::fmt::Debug;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_layers::default_layer::LayerInterface;
use crate::{
    neural_networks::network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    neural_networks::utils::dtype::{c_from_f64, c_to_f64, real_from_f64, C},
    wavelet_transform::{
        cwt_complex::{cwt_2d, cwt_2d_full, get_wavelet_derivative, get_wavelet_derivative_full, wavefun_complex, CWTComplex},
        cwt_types::ContinuousWaletetType,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexWaveletLayer {
    pub wavelet: CWTComplex,
    pub is_full_mode: bool,

    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<num::Complex<f64>>>>>,
    #[serde(skip)]
    pub previous_gradient_input_batch: Option<Vec<Vec<Vec<num::Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<num::Complex<f64>>>>>,
}

impl ComplexWaveletLayer {
    pub fn new() -> Self {
        let wavelet = CWTComplex {
            scales: vec![1.0],
            cw_type: ContinuousWaletetType::CGAU5,
            sampling_period: 1.0,
            fc: 1.0,
            fb: 1.0,
            m: 1.0,
            frequencies: vec![],
        };

        Self {
            input_batch: None,
            previous_gradient_input_batch: None,
            gradient: None,
            output_batch: None,
            time_step: 0,
            wavelet,
            is_full_mode: false,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        if !layer_input.has_non_empty_input_batch() {
            if layer_input.get_input_batch_rm_ref().is_some_and(|rm| !rm.is_empty()) {
                panic!("ComplexWaveletLayer received RM-only input; use ComplexWaveletLayerRm instead");
            }
        }

        let input_batch_c: Vec<Vec<Vec<C>>> = layer_input.get_input_batch();
        let input_batch: Vec<Vec<Vec<num::Complex<f64>>>> = input_batch_c
            .iter()
            .map(|matrix| matrix.iter().map(|row| row.iter().copied().map(c_to_f64).collect()).collect())
            .collect();

        let output_batch_f64: Vec<Vec<Vec<num::Complex<f64>>>> = input_batch
            .par_iter()
            .map(|input| {
                let output;
                if self.is_full_mode {
                    let (wavelet_output, _frequencies) = cwt_2d_full(input, &self.wavelet);
                    output = wavelet_output[0].to_vec();
                } else {
                    let (wavelet_output, _frequencies) = cwt_2d(input, &self.wavelet);
                    output = wavelet_output[0].to_vec();
                }

                // let causal_mask = self.create_causal_mask(output.len(), output[0].len());
                // self.apply_attention_mask_inplace(&mut output, &causal_mask);

                output
            })
            .collect();

        let output_batch: Vec<Vec<Vec<C>>> = output_batch_f64.iter().map(|m| m.iter().map(|row| row.iter().copied().map(c_from_f64).collect()).collect()).collect();

        self.input_batch = Some(input_batch);
        self.time_step = layer_input.get_time_step();
        self.output_batch = Some(output_batch_f64);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        if previous_gradient.get_gradient_input_batch().is_empty() {
            if previous_gradient.get_gradient_input_batch_rm_ref().is_some_and(|rm| !rm.is_empty()) {
                panic!("ComplexWaveletLayer received RM-only gradients; use ComplexWaveletLayerRm instead");
            }
        }

        let input_batch = self.input_batch.as_ref().expect("Input batch not found");

        let wavefun_result: Vec<Vec<num::Complex<f64>>> = wavefun_complex(&10, &self.wavelet);

        let prev_grad_c: Vec<Vec<Vec<C>>> = previous_gradient.get_gradient_input_batch();
        let prev_grad: Vec<Vec<Vec<num::Complex<f64>>>> = prev_grad_c.iter().map(|m| m.iter().map(|row| row.iter().copied().map(c_to_f64).collect()).collect()).collect();

        let input_gradient_batch_f64: Vec<Vec<Vec<num::Complex<f64>>>> = prev_grad
            .iter()
            .zip(input_batch)
            .map(|(previous_gradient, input)| {
                if self.is_full_mode {
                    get_wavelet_derivative_full(&input, &wavefun_result, &self.wavelet.scales[0], &previous_gradient)
                } else {
                    previous_gradient
                        .iter()
                        .enumerate()
                        .map(|(row_ind, prev_grad_row)| get_wavelet_derivative(&input[row_ind], &wavefun_result, &self.wavelet.scales[0], &prev_grad_row))
                        .collect()
                }
            })
            .collect();

        let input_gradient_batch: Vec<Vec<Vec<C>>> = input_gradient_batch_f64
            .iter()
            .map(|m| m.iter().map(|row| row.iter().copied().map(c_from_f64).collect()).collect())
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(input_gradient_batch);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn apply_attention_mask_inplace(&self, attention_scores: &mut Vec<Vec<C>>, mask: &Vec<Vec<u8>>) {
        let ln = -1.0e12;
        let large_negative = C::new(real_from_f64(ln), real_from_f64(ln));

        for row in 0..attention_scores.len() {
            for col in 0..attention_scores[row].len() {
                if mask[row % mask.len()][col % mask[0].len()] == 0 {
                    attention_scores[row][col] = large_negative;
                }
            }
        }
    }

    pub fn create_causal_mask(&self, rows: usize, cols: usize) -> Vec<Vec<u8>> {
        let mut mask: Vec<Vec<u8>> = vec![vec![0; cols]; rows];

        for i in 0..rows {
            for j in 0..=i.min(cols - 1) {
                mask[i][j] = 1;
            }
        }

        mask
    }
}

impl ComplexWaveletLayer {
    pub fn update_parameters(&mut self) {}
}

impl LayerInterface for ComplexWaveletLayer {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        ComplexWaveletLayer::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        ComplexWaveletLayer::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        ComplexWaveletLayer::update_parameters(self)
    }
}
