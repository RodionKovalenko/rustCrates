use core::fmt::Debug;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
        network_layers::default_layer::LayerInterface,
        utils::{dtype::{c_from_f64, c_to_f64, C}, matrix::RowMajorMatrix},
    },
    wavelet_transform::{
        cwt_complex::{
            cwt_2d_full_rm, cwt_2d_rm, get_wavelet_derivative_full_rm, get_wavelet_derivative_slice, wavefun_complex, CWTComplex,
        },
        cwt_types::ContinuousWaletetType,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexWaveletLayerRm {
    pub wavelet: CWTComplex,
    pub is_full_mode: bool,

    #[serde(skip)]
    input_batch_rm: Option<Vec<RowMajorMatrix<num::Complex<f64>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
}

impl ComplexWaveletLayerRm {
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
            input_batch_rm: None,
            gradient: None,
            time_step: 0,
            wavelet,
            is_full_mode: false,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        if layer_input.has_non_empty_input_batch() {
            panic!("ComplexWaveletLayerRm received Vec input batch; use ComplexWaveletLayer instead");
        }

        let input_batch_rm_c: Vec<RowMajorMatrix<C>> = layer_input
            .get_input_batch_rm_ref()
            .expect("ComplexWaveletLayerRm::forward expects RM input")
            .to_vec();

        let input_batch_rm: Vec<RowMajorMatrix<num::Complex<f64>>> = input_batch_rm_c
            .iter()
            .map(|m| RowMajorMatrix::from_data(m.rows, m.cols, m.data.iter().copied().map(c_to_f64).collect()))
            .collect();

        let output_batch_rm_f64: Vec<RowMajorMatrix<num::Complex<f64>>> = input_batch_rm
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

        let output_batch_rm: Vec<RowMajorMatrix<C>> = output_batch_rm_f64
            .iter()
            .map(|m| RowMajorMatrix::from_data(m.rows, m.cols, m.data.iter().copied().map(c_from_f64).collect()))
            .collect();

        self.input_batch_rm = Some(input_batch_rm);
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        // Strict RM-only: require RM gradients; do not use get_gradient_input_batch() because it will
        // convert RM->Vec and appear "non-empty" even when only RM is provided.
        let Some(prev_rm_ref) = previous_gradient.get_gradient_input_batch_rm_ref() else {
            if previous_gradient
                .get_gradient_input_batch_ref()
                .is_some_and(|vec_g| !vec_g.is_empty())
            {
                panic!("ComplexWaveletLayerRm received Vec gradients; use ComplexWaveletLayer instead");
            }

            let mut gradient = Gradient::new_default();
            gradient.set_time_step(previous_gradient.get_time_step());
            gradient.set_gradient_input_batch_rm(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        };

        let prev_rm: Vec<RowMajorMatrix<C>> = prev_rm_ref.to_vec();

        if prev_rm.is_empty() {
            let mut gradient = Gradient::new_default();
            gradient.set_time_step(previous_gradient.get_time_step());
            gradient.set_gradient_input_batch_rm(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let input_batch_rm = self
            .input_batch_rm
            .as_ref()
            .expect("ComplexWaveletLayerRm::backward expects RM cache from forward");
        assert_eq!(input_batch_rm.len(), prev_rm.len(), "ComplexWaveletLayerRm::backward batch mismatch");

        let prev_rm_f64: Vec<RowMajorMatrix<num::Complex<f64>>> = prev_rm
            .iter()
            .map(|m| RowMajorMatrix::from_data(m.rows, m.cols, m.data.iter().copied().map(c_to_f64).collect()))
            .collect();

        let wavefun_result: Vec<Vec<num::Complex<f64>>> = wavefun_complex(&10, &self.wavelet);

        let input_gradient_batch_rm_f64: Vec<RowMajorMatrix<num::Complex<f64>>> = prev_rm_f64
            .iter()
            .zip(input_batch_rm.iter())
            .map(|(prev_grad_rm, input_rm)| {
                if self.is_full_mode {
                    get_wavelet_derivative_full_rm(input_rm, &wavefun_result, &self.wavelet.scales[0], prev_grad_rm)
                } else {
                    let mut out = RowMajorMatrix::from_data(
                        input_rm.rows,
                        input_rm.cols,
                        vec![num::Complex::new(0.0, 0.0); input_rm.rows * input_rm.cols],
                    );
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

        let input_gradient_batch_rm: Vec<RowMajorMatrix<C>> = input_gradient_batch_rm_f64
            .iter()
            .map(|m| RowMajorMatrix::from_data(m.rows, m.cols, m.data.iter().copied().map(c_from_f64).collect()))
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);

        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {}
}

impl LayerInterface for ComplexWaveletLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        ComplexWaveletLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        ComplexWaveletLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        ComplexWaveletLayerRm::update_parameters(self)
    }
}
