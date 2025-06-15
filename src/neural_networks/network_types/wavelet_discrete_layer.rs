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
        dwt::{dwt_2d_full, dwt_2d_partial, get_ll_hh, get_ll_hl_lh_hh, grad_dwt_2d, grad_dwt_2d_partial},
        dwt_types::DiscreteWaletetType,
        modes::WaveletMode,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletDiscreteLayer {
    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
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
    pub approx_details: Option<Vec<Vec<Vec<Complex<f64>>>>>,
}

impl WaveletDiscreteLayer {
    pub fn new() -> Self {
        Self {
            input_batch: None,
            previous_gradient_input_batch: None,
            gradient: None,
            output_batch: None,
            time_step: 0,
            wavelet: DiscreteWaletetType::DB2,
            wavelet_mode: WaveletMode::SYMMETRIC,
            is_full_mode: false,
            approx_details: None,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();

        let (output_batch, approx_details): (Vec<_>, Vec<_>) = input_batch
            .par_iter()
            .map(|input| {
                if self.is_full_mode {
                    let dwt_full = dwt_2d_full(input, &self.wavelet, &self.wavelet_mode);
                    let ll_hl_lh_hh = get_ll_hl_lh_hh(&dwt_full);
                    (ll_hl_lh_hh[0].to_vec(), vec![]) // No detail saved in full mode
                } else {
                    let dwt_partial = dwt_2d_partial(&transpose(&input), &self.wavelet, &self.wavelet_mode);
                    let wav_hh_ll = get_ll_hh(&dwt_partial);
                    let detail = wav_hh_ll[1].clone(); // Save HH detail
                    let approx = transpose(&wav_hh_ll[0]); // Approximation (LL)
                    (approx, detail)
                }
            })
            .unzip();

        self.input_batch = Some(input_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.output_batch = Some(output_batch.clone());
        self.approx_details = Some(approx_details);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found");
        let _approx_details = self.approx_details.as_ref().expect("no approximation details found");
        let grad_output_batch = previous_gradient.get_gradient_input_batch();

        assert_eq!(input_batch.len(), grad_output_batch.len(), "Input and gradient batch size mismatch");

        let grad_input_batch: Vec<Vec<Vec<Complex<f64>>>> = grad_output_batch
            .par_iter()
            .zip(input_batch.par_iter())
            .map(|(grad_output, input)| {
                if self.is_full_mode {
                    grad_dwt_2d(grad_output, &self.wavelet, &self.wavelet_mode)
                } else {
                    let gradient_partial = grad_dwt_2d_partial(&transpose(grad_output), &self.wavelet, &self.wavelet_mode);
                    let gradient_transposed = transpose(&gradient_partial);

                    println!("\n gradient non aligned: {:?}", gradient_transposed);
                    if gradient_transposed.len() == input_batch[0].len() {
                        gradient_transposed
                    } else {
                        self.align_gradient_rows_complex(input, &gradient_transposed)
                    }
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
    pub fn align_gradient_rows_complex(&self, input: &Vec<Vec<Complex<f64>>>, gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let input_rows = input.len();
        let grad_rows = gradient.len();
        let cols = gradient[0].len();

        // Handle the case where gradient has fewer rows (pad with zeros or handle as needed)
        assert!(grad_rows >= input_rows, "Gradient must have at least as many rows as needed to align to input");

        let mut result = Vec::with_capacity(input_rows);

        // Copy first input_rows - 1 rows as-is
        for i in 0..(input_rows - 1) {
            result.push(gradient[i].clone());
        }

        // Sum remaining rows into the last row
        let mut last_row = vec![Complex::<f64>::new(0.0, 0.0); cols];
        for i in (input_rows - 1)..grad_rows {
            for j in 0..cols {
                last_row[j] += gradient[i][j];
            }
        }
        result.push(last_row);

        result
    }
}
