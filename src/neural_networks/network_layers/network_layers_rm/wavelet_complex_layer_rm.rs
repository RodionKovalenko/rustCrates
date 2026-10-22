use core::fmt::Debug;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
        network_layers::default_layer::LayerInterface,
        utils::{
            dtype::{C, CF64},
            matrix::RowMajorMatrix,
            matrix_rm::{rm_from_f64, rm_to_f64},
        },
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
    input_batch_rm: Option<Vec<RowMajorMatrix<CF64>>>,
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

    /// RM counterpart of `ComplexWaveletLayer::forward`.
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm: Vec<RowMajorMatrix<CF64>> = layer_input.get_input_batch_rm().iter().map(rm_to_f64).collect();

        let output_batch_rm: Vec<RowMajorMatrix<C>> = input_batch_rm
            .par_iter()
            .map(|input| {
                let (wavelet_output, _frequencies) = if self.is_full_mode { cwt_2d_full_rm(input, &self.wavelet) } else { cwt_2d_rm(input, &self.wavelet) };
                rm_from_f64(&wavelet_output[0])
            })
            .collect();

        self.input_batch_rm = Some(input_batch_rm);
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch_rm);
        layer_output
    }

    /// RM counterpart of `ComplexWaveletLayer::backward`.
    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("ComplexWaveletLayerRm::backward expects RM cache from forward");
        let previous_gradient_rm = previous_gradient.get_gradient_input_batch_rm_cow();
        assert_eq!(input_batch_rm.len(), previous_gradient_rm.len(), "ComplexWaveletLayerRm::backward batch mismatch");

        let wavefun_result: Vec<Vec<CF64>> = wavefun_complex(&10, &self.wavelet);
        let scale = &self.wavelet.scales[0];

        let input_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient_rm
            .iter()
            .zip(input_batch_rm.iter())
            .map(|(previous_gradient_rm, input_rm)| {
                let previous_gradient_rm = rm_to_f64(previous_gradient_rm);
                if self.is_full_mode {
                    rm_from_f64(&get_wavelet_derivative_full_rm(input_rm, &wavefun_result, scale, &previous_gradient_rm))
                } else {
                    let mut out = RowMajorMatrix::from_data(input_rm.rows, input_rm.cols, vec![CF64::new(0.0, 0.0); input_rm.rows * input_rm.cols]);
                    for r in 0..input_rm.rows {
                        let row = input_rm.row_range(r);
                        let g = get_wavelet_derivative_slice(&input_rm.data[row.clone()], &wavefun_result, scale, &previous_gradient_rm.data[row.clone()]);
                        out.data[row].copy_from_slice(&g);
                    }
                    rm_from_f64(&out)
                }
            })
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
