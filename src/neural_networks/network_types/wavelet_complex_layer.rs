use core::fmt::Debug;
use num::Complex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    neural_networks::utils::matrix::RowMajorMatrix,
    wavelet_transform::{
        cwt_complex::{cwt_2d, cwt_2d_full, cwt_2d_full_rm, cwt_2d_rm, get_wavelet_derivative, get_wavelet_derivative_full, get_wavelet_derivative_full_rm, get_wavelet_derivative_slice, wavefun_complex, CWTComplex},
        cwt_types::ContinuousWaletetType,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexWaveletLayer {
    pub wavelet: CWTComplex,
    pub is_full_mode: bool,

    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    input_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    #[serde(skip)]
    pub previous_gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,

    #[serde(skip)]
    pub last_rm_strict: bool,
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
            input_batch_rm: None,
            previous_gradient_input_batch: None,
            gradient: None,
            output_batch: None,
            output_batch_rm: None,
            time_step: 0,
            wavelet,
            is_full_mode: false,
            last_rm_strict: false,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        self.last_rm_strict = layer_input.get_rm_strict();
        let input_batch_rm_ref = layer_input.get_input_batch_rm_ref();
        let input_batch_ref = layer_input.get_input_batch_ref();
        let use_rm = input_batch_ref.is_none() && input_batch_rm_ref.is_some();

        if use_rm {
            let input_batch_rm = input_batch_rm_ref.unwrap();
            if !input_batch_rm.is_empty() {
                return self.forward_rm(layer_input);
            }
        }

        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();

        let output_batch: Vec<Vec<Vec<Complex<f64>>>> = input_batch
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

        self.input_batch = Some(input_batch.clone());
        self.input_batch_rm = None;
        self.time_step = layer_input.get_time_step();
        self.output_batch = Some(output_batch.clone());
        self.output_batch_rm = None;

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn forward_rm(&mut self, layer_input: &LayerInput) -> LayerOutput {
        self.last_rm_strict = layer_input.get_rm_strict();
        let input_batch_rm = layer_input
            .get_input_batch_rm_ref()
            .expect("ComplexWaveletLayer::forward_rm expects RM input")
            .to_vec();

        let output_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = input_batch_rm
            .par_iter()
            .map(|input| {
                if self.is_full_mode {
                    let (wavelet_output, _frequencies) = cwt_2d_full_rm(input, &self.wavelet);
                    wavelet_output[0].clone()
                } else {
                    let (wavelet_output, _frequencies) = cwt_2d_rm(input, &self.wavelet);
                    wavelet_output[0].clone()
                }
            })
            .collect();

        self.input_batch = None;
        self.input_batch_rm = Some(input_batch_rm);
        self.time_step = layer_input.get_time_step();
        self.output_batch = None;
        self.output_batch_rm = Some(output_batch_rm.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        if let Some(prev_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
            if !prev_rm.is_empty() {
                // Only valid to run RM backward if RM cache exists (forward_rm ran).
                if self.input_batch_rm.is_some() {
                    return self.backward_rm(prev_rm);
                }

                // Mixed mode: Vec forward but caller provided RM gradients.
                if self.last_rm_strict {
                    panic!(
                        "RM strict mode violation: ComplexWaveletLayer backward would convert RM gradients to Vec (forward was Vec)"
                    );
                }
                let prev_vec: Vec<Vec<Vec<Complex<f64>>>> = prev_rm.iter().map(|m| m.to_rows()).collect();
                let mut prev_legacy = Gradient::new_default();
                prev_legacy.set_time_step(previous_gradient.get_time_step());
                prev_legacy.set_gradient_input_batch(prev_vec);
                return self.backward(&prev_legacy);
            }
        }

        // Mixed mode: RM forward but caller provided Vec gradients.
        if self.input_batch_rm.is_some() && !previous_gradient.get_gradient_input_batch().is_empty() {
            if self.last_rm_strict {
                panic!(
                    "RM strict mode violation: ComplexWaveletLayer backward would convert Vec gradients to RM (forward was RM)"
                );
            }
            let prev_rm: Vec<RowMajorMatrix<Complex<f64>>> = previous_gradient
                .get_gradient_input_batch()
                .iter()
                .map(|m| RowMajorMatrix::from_rows(m))
                .collect();
            return self.backward_rm(&prev_rm);
        }

        let input_batch = self.input_batch.as_ref().expect("Input batch not found");

        let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, &self.wavelet);

        let input_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient
            .get_gradient_input_batch()
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

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(input_gradient_batch);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<Complex<f64>>]) -> Gradient {
        let input_batch_rm = self
            .input_batch_rm
            .as_ref()
            .expect("ComplexWaveletLayer::backward_rm expects RM cache from forward_rm");
        assert_eq!(input_batch_rm.len(), previous_gradient_batch_rm.len(), "ComplexWaveletLayer::backward_rm batch mismatch");

        let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, &self.wavelet);

        let input_gradient_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = previous_gradient_batch_rm
            .iter()
            .zip(input_batch_rm.iter())
            .map(|(prev_grad_rm, input_rm)| {
                if self.is_full_mode {
                    get_wavelet_derivative_full_rm(input_rm, &wavefun_result, &self.wavelet.scales[0], prev_grad_rm)
                } else {
                    let mut out = RowMajorMatrix::from_data(input_rm.rows, input_rm.cols, vec![Complex::new(0.0, 0.0); input_rm.rows * input_rm.cols]);
                    for r in 0..input_rm.rows {
                        let row = input_rm.row_range(r);
                        let g = get_wavelet_derivative_slice(
                            &input_rm.data[row.clone()],
                            &wavefun_result,
                            &self.wavelet.scales[0],
                            &prev_grad_rm.data[row.clone()],
                        );
                        out.data[row].copy_from_slice(&g);
                    }
                    out
                }
            })
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn apply_attention_mask_inplace(&self, attention_scores: &mut Vec<Vec<Complex<f64>>>, mask: &Vec<Vec<u8>>) {
        let large_negative = Complex::new(-1e12, -1e12);

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
