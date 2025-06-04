use core::fmt::Debug;
use num::Complex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    wavelet_transform::{
        cwt_complex::{cwt_2d_full, get_wavelet_derivative_full, wavefun_complex, CWTComplex},
        cwt_types::ContinuousWaletetType,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletLayer {
    #[serde(skip)]
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub previous_gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    gradient: Option<Gradient>,
    time_step: usize,
    #[serde(skip)]
    output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub wavelet: CWTComplex,
}

impl WaveletLayer {
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
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();

        let output_batch: Vec<Vec<Vec<Complex<f64>>>> = input_batch
            .par_iter()
            .map(|input| {
                //let (transform_cwt, _frequencies) = cwt_complex(input, &self.wavelet).unwrap();
                // let wavelet_output = convert_to_c_array_f64_3d(transform_cwt);
                let (wavelet_output, _frequencies) = cwt_2d_full(input, &self.wavelet);
                wavelet_output[0].to_vec()
            })
            .collect();

        self.input_batch = Some(input_batch.clone());
        self.time_step = layer_input.get_time_step();
        self.output_batch = Some(output_batch.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch not found");

        let wavefun_result: Vec<Vec<Complex<f64>>> = wavefun_complex(&10, &self.wavelet);

        let input_gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient
            .get_gradient_input_batch()
            .iter()
            .zip(input_batch)
            .map(|(previous_gradient, input)| {
                //previous_gradient.iter().enumerate().map(|(row_ind, prev_grad_row)| get_wavelet_derivative(&input[row_ind], &wavefun_result, &self.wavelet.scales[0], &prev_grad_row)).collect()
                get_wavelet_derivative_full(&input, &wavefun_result, &self.wavelet.scales[0], &previous_gradient)
            })
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(input_gradient_batch);

        self.gradient = Some(gradient.clone());
        gradient
    }
}
