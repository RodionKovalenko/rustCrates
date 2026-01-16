use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::{
        network_components::{gradient_struct::Gradient, layer::LayerEnum, layer_input_struct::LayerInput, layer_output_struct::LayerOutput, norm_layer::NormalNormLayer},
        utils::matrix::{add_matrix_2d_c, add_matrix_3d, transpose, transpose_rm, RowMajorMatrix},
    },
    wavelet_transform::{
        dwt::{dwt_1d, dwt_2d_partial, dwt_2d_partial_rm, get_ll_hh, get_ll_hh_1d, get_ll_hh_rm, grad_dwt_2d_partial, inverse_dwt_2d_partial, inverse_dwt_2d_partial_rm},
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
    pub is_linear_layer: bool,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    #[serde(skip)]
    pub compression_levels_used: usize,
    #[serde(skip)]
    pub input_only_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub input_only_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
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
    pub trend_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub trend_input_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    #[serde(skip)]
    pub trend_batch_coefficients: Option<Vec<Vec<Vec<Vec<Complex<f64>>>>>>,
    #[serde(skip)]
    pub details_batch_coefficients: Option<Vec<Vec<Vec<Vec<Complex<f64>>>>>>,
    #[serde(skip)]
    pub details_batch_coefficients_rm: Option<Vec<Vec<RowMajorMatrix<Complex<f64>>>>>,
    #[serde(skip)]
    pub trend_batch_result: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub details_batch_result: Option<Vec<Vec<Vec<Complex<f64>>>>>,
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

    // When true, any RM<->Vec fallback conversion in backward is treated as a bug and must panic.
    #[serde(skip)]
    pub last_rm_strict: bool,
}

impl DiscreteWaveletLayer {
    pub fn new() -> Self {
        let _norm_layer = Some(LayerEnum::Norm(Box::new(NormalNormLayer::new(16, 1e-10, 0.001))));

        Self {
            input_batch: None,
            input_batch_rm: None,
            trend_input_batch: None,
            trend_input_batch_rm: None,
            input_only_batch: None,
            input_only_batch_rm: None,
            previous_gradient_input_batch: None,
            gradient: None,
            output_batch: None,
            output_batch_rm: None,
            time_step: 0,
            wavelet: DiscreteWaveletType::DB4,
            wavelet_size: 16,
            compression_levels: 1,
            compression_levels_used: 0,
            wavelet_mode: WaveletMode::ZERO,
            add_details: false,
            is_full_mode: false,
            is_linear_layer: false,
            trend_batch_result: None,
            details_batch_result: None,
            trend_batch_coefficients: None,
            details_batch_coefficients: None,
            details_batch_coefficients_rm: None,
            compression_dims: None,
            compressed_padding_mask_b: None,
            padding_mask_batch: None,
            target_batch: None,
            target_batch_ids: None,
            norm_layer: None,

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
        let target_batch_ids: Vec<Vec<u32>> = layer_input.get_target_batch_ids();
        let forward_only = layer_input.get_forward_only();
        let mut padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
        let time_step = layer_input.get_time_step();

        let mut trend_batch = vec![];
        let mut details_batch = vec![];
        let mut compression_dims = vec![];
        let mut comp_pad_mask_b: Vec<Vec<u32>> = vec![];
        let input_only = input_batch.clone();

        if padding_mask_batch.is_empty() || padding_mask_batch[0].is_empty() {
            padding_mask_batch = vec![vec![1; input_batch[0].len()]; input_batch.len()];
        }

        if !forward_only || (forward_only && time_step == 0) {
            for (batch_ind, input) in input_batch.iter().enumerate() {
                let (new_trend, new_details, compression_dim) = self.compress_partial(&input, batch_ind);

                trend_batch.push(new_trend);
                details_batch.push(new_details);
                compression_dims.push(compression_dim);
                comp_pad_mask_b.push(self.compress_padding_mask(&padding_mask_batch[batch_ind]));
            }
        }

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

        self.input_batch = Some(input_batch);
        self.input_only_batch = Some(input_only);
        self.trend_input_batch = Some(trend_batch.clone());
        self.details_batch_result = Some(details_batch.clone());
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

    pub fn forward_rm(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = layer_input
            .get_input_batch_rm_ref()
            .expect("DiscreteWaveletLayer::forward_rm expects RM input")
            .to_vec();

        let target_batch_ids: Vec<Vec<u32>> = layer_input.get_target_batch_ids();
        let forward_only = layer_input.get_forward_only();
        let mut padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
        let time_step = layer_input.get_time_step();

        if padding_mask_batch.is_empty() || padding_mask_batch[0].is_empty() {
            padding_mask_batch = vec![vec![1; input_batch_rm[0].rows]; input_batch_rm.len()];
        }

        let mut trend_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = Vec::new();
        let mut details_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = Vec::new();
        let mut compression_dims: Vec<Vec<usize>> = Vec::new();
        let mut comp_pad_mask_b: Vec<Vec<u32>> = Vec::new();
        let mut details_coeffs_batch: Vec<Vec<RowMajorMatrix<Complex<f64>>>> = Vec::new();

        if !forward_only || (forward_only && time_step == 0) {
            for (batch_ind, input_rm) in input_batch_rm.iter().enumerate() {
                let (new_trend, new_details, compression_dim, detail_coeffs_levels) = self.compress_partial_rm(input_rm);
                trend_batch_rm.push(new_trend);
                details_batch_rm.push(new_details);
                compression_dims.push(compression_dim);
                comp_pad_mask_b.push(self.compress_padding_mask(&padding_mask_batch[batch_ind]));

                details_coeffs_batch.push(detail_coeffs_levels);
            }
        }

        // Apply optional norm in RM
        let mut trend_batch_rm_normed = trend_batch_rm.clone();
        if self.norm_layer.is_some() {
            let mut li_norm = layer_input.clone();
            li_norm.clear_input_batch();
            li_norm.set_input_batch_rm(trend_batch_rm.clone());
            li_norm.set_input_batch_before_rm(trend_batch_rm.clone());
            li_norm.set_padding_mask_batch(comp_pad_mask_b.clone());

            if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
                match norm_layer_enum {
                    LayerEnum::RMSNorm(rms_norm_layer) => {
                        let out = rms_norm_layer.forward(&li_norm);
                        trend_batch_rm_normed = out.get_output_batch_rm();
                    }
                    LayerEnum::Norm(norm_layer) => {
                        let mut out = norm_layer.forward(&li_norm);
                        trend_batch_rm_normed = out.take_output_batch_rm().unwrap_or_else(|| vec![]);
                    }
                    _ => {}
                }
            }
        }

        self.input_batch = None;
        self.input_only_batch = None;
        self.trend_input_batch = None;
        self.output_batch = None;

        self.input_batch_rm = Some(input_batch_rm.clone());
        self.input_only_batch_rm = Some(input_batch_rm.clone());
        self.trend_input_batch_rm = Some(trend_batch_rm_normed.clone());
        self.details_batch_result = None;
        self.details_batch_coefficients_rm = Some(details_coeffs_batch);
        self.time_step = layer_input.get_time_step();
        self.output_batch_rm = Some(trend_batch_rm_normed.clone());
        self.target_batch_ids = Some(target_batch_ids);
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.compressed_padding_mask_b = Some(comp_pad_mask_b.clone());
        self.compression_dims = Some(compression_dims);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(trend_batch_rm_normed.clone());
        layer_output.set_padding_mask_batch(padding_mask_batch.clone());

        if trend_batch_rm_normed[0].rows != padding_mask_batch[0].len() {
            layer_output.set_padding_mask_batch(comp_pad_mask_b.clone());
        }

        layer_output
    }
    pub fn forward_inverse(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = layer_input.get_input_batch();
        let mut decompressed_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::with_capacity(input_batch.len());
        let compression_dims = self.compression_dims.as_ref().expect("no compression dwt found").clone();

        for (batch_ind, input) in input_batch.iter().enumerate() {
            let compression_dim = &compression_dims[batch_ind];
            let decompressed = self.decompress_partial(input, compression_dim, batch_ind);
            decompressed_batch.push(decompressed);
        }

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(decompressed_batch.clone());
        layer_output.set_padding_mask_batch(self.padding_mask_batch.clone().expect("no padding mask found"));

        layer_output
    }
    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        if let Some(prev_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
            if !prev_rm.is_empty() {
                // Only valid to run RM backward if RM cache exists (forward_rm ran).
                if self.input_batch_rm.is_some() {
                    return self.backward_rm(previous_gradient);
                }

                // Mixed mode: Vec forward but caller provided RM gradients.
                if self.last_rm_strict {
                    panic!(
                        "RM strict mode violation: DiscreteWaveletLayer backward would convert RM gradients to Vec (forward was Vec)"
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
                    "RM strict mode violation: DiscreteWaveletLayer backward would convert Vec gradients to RM (forward was RM)"
                );
            }
            let prev_rm: Vec<RowMajorMatrix<Complex<f64>>> = previous_gradient
                .get_gradient_input_batch()
                .iter()
                .map(|m| RowMajorMatrix::from_rows(m))
                .collect();
            let mut g = Gradient::new_default();
            g.set_time_step(previous_gradient.get_time_step());
            g.set_gradient_input_batch_rm(prev_rm);
            return self.backward_rm(&g);
        }

        let input_batch = self.input_batch.as_ref().expect("Input batch not found");
        let mut grad_output_batch = previous_gradient.get_gradient_input_batch();
        let compression_dims = self.compression_dims.as_ref().expect("no compression dwt found").clone();

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

        let mut grad_input_batch: Vec<Vec<Vec<Complex<f64>>>> = Vec::with_capacity(grad_output_batch.len());
        for (batch_ind, grad_output) in grad_output_batch.iter_mut().enumerate() {
            let compression_dim = &compression_dims[batch_ind];
            let gradient_decompr = self.decompress_partial_gradient(grad_output, compression_dim, batch_ind);
            grad_input_batch.push(gradient_decompr);
        }

        if !output_gradient_norm.is_empty() {
            grad_input_batch = add_matrix_3d(&grad_input_batch, &grad_input_batch);
        }

        // if self.gradient.is_some() {
        //     let previous_gradient = self.gradient.as_ref().expect("");
        //     grad_input_batch = add_matrix_3d(&grad_input_batch, &previous_gradient.get_gradient_input_batch());
        // }

        let mut gradient = Gradient::new_default();
        gradient.set_time_step(self.time_step);
        gradient.set_gradient_input_batch(grad_input_batch.clone());

        self.previous_gradient_input_batch = Some(grad_input_batch);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn backward_rm(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch_rm = self
            .input_batch_rm
            .as_ref()
            .expect("DiscreteWaveletLayer::backward_rm expects RM cache from forward_rm")
            .to_vec();

        let mut grad_output_batch_rm = previous_gradient
            .get_gradient_input_batch_rm_ref()
            .expect("DiscreteWaveletLayer::backward_rm expects RM gradient")
            .to_vec();
        let compression_dims = self.compression_dims.as_ref().expect("no compression dwt found").clone();

        assert_eq!(input_batch_rm.len(), grad_output_batch_rm.len(), "Input and gradient batch size mismatch");

        // Apply norm backward if present
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(grad_output_batch_rm.clone());

        if let Some(norm_layer) = &mut self.norm_layer {
            match norm_layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let norm_gradient = rms_norm_layer.backward_rm(&grad_output_batch_rm);
                    grad_output_batch_rm = norm_gradient.get_gradient_input_batch_rm();
                    gradient.set_gradient_input_batch_rm(grad_output_batch_rm.clone());
                }
                LayerEnum::Norm(norm_layer) => {
                    let mut g = Gradient::new_default();
                    g.set_gradient_input_batch_rm(grad_output_batch_rm);
                    let norm_gradient = norm_layer.backward(&g);
                    grad_output_batch_rm = norm_gradient.get_gradient_input_batch_rm();
                    gradient.set_gradient_input_batch_rm(grad_output_batch_rm.clone());
                }
                _ => {}
            }
        }

        let mut grad_input_batch_rm: Vec<RowMajorMatrix<Complex<f64>>> = Vec::with_capacity(grad_output_batch_rm.len());
        for (batch_ind, grad_output_rm) in grad_output_batch_rm.iter().enumerate() {
            let compression_dim = &compression_dims[batch_ind];
            let gradient_decompr = self.decompress_partial_gradient_rm(grad_output_rm, compression_dim, batch_ind);
            grad_input_batch_rm.push(gradient_decompr);
        }

        let mut out = Gradient::new_default();
        out.set_time_step(self.time_step);
        out.set_gradient_input_batch_rm(grad_input_batch_rm.clone());

        self.gradient = Some(out.clone());
        out
    }

    fn compress_partial_rm(
        &mut self,
        input_rm: &RowMajorMatrix<Complex<f64>>,
    ) -> (
        RowMajorMatrix<Complex<f64>>,
        RowMajorMatrix<Complex<f64>>,
        Vec<usize>,
        Vec<RowMajorMatrix<Complex<f64>>>,
    ) {
        let mut wav_out = input_rm.clone();
        let mut details_last = RowMajorMatrix::from_data(0, 0, vec![]);
        let mut compression_dims: Vec<usize> = Vec::new();
        let mut detail_coefficients_levels: Vec<RowMajorMatrix<Complex<f64>>> = Vec::new();

        for level in 0..self.compression_levels {
            let wav_t = transpose_rm(&wav_out); // dim x seq
            let dwt_partial = dwt_2d_partial_rm(&wav_t, &self.wavelet, &self.wavelet_mode);
            let (ll, hh) = get_ll_hh_rm(&dwt_partial);

            let trend = transpose_rm(&ll); // seq/2 x dim
            let details = transpose_rm(&hh);

            let mut next = trend.clone();
            if self.add_details {
                assert_eq!((trend.rows, trend.cols), (details.rows, details.cols));
                for i in 0..next.data.len() {
                    next.data[i] += details.data[i];
                }
            }

            wav_out = next;
            details_last = details.clone();
            detail_coefficients_levels.push(details);
            compression_dims.push(wav_out.rows);

            self.compression_levels_used = level + 1;
            if wav_out.rows <= 24 {
                break;
            }
        }

        (wav_out, details_last, compression_dims, detail_coefficients_levels)
    }

    fn decompress_partial_gradient_rm(
        &mut self,
        grad_output_rm: &RowMajorMatrix<Complex<f64>>,
        compression_dim: &Vec<usize>,
        batch_ind: usize,
    ) -> RowMajorMatrix<Complex<f64>> {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("Input batch RM not found");
        let original_seq_len = input_batch_rm[batch_ind].rows;

        let mut grad_transp = transpose_rm(grad_output_rm); // dim x trend_len

        for _ in (0..compression_dim.len()).rev() {
            let ll_len = grad_transp.cols;
            let mut combined = RowMajorMatrix::from_data(grad_transp.rows, ll_len * 2, vec![Complex::new(0.0, 0.0); grad_transp.rows * ll_len * 2]);

            for r in 0..grad_transp.rows {
                let row_ll = grad_transp.row_range(r);
                let dst = combined.row_range(r);

                // LL
                combined.data[dst.start..dst.start + ll_len].copy_from_slice(&grad_transp.data[row_ll.clone()]);

                // HH
                if self.add_details {
                    combined.data[dst.start + ll_len..dst.start + 2 * ll_len].copy_from_slice(&grad_transp.data[row_ll]);
                }
            }

            grad_transp = inverse_dwt_2d_partial_rm(&combined, &self.wavelet, &self.wavelet_mode, 0);
        }

        let mut grad = transpose_rm(&grad_transp); // seq_len x dim

        if grad.rows > original_seq_len {
            let mut truncated = RowMajorMatrix::from_data(original_seq_len, grad.cols, vec![Complex::new(0.0, 0.0); original_seq_len * grad.cols]);
            truncated.data.copy_from_slice(&grad.data[0..original_seq_len * grad.cols]);
            grad = truncated;
        }

        grad
    }
    pub fn backward_inverse(&mut self, prev_gradient: &Gradient) -> Gradient {
        let gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = prev_gradient.get_gradient_input_batch();
        let mut batch_output: Vec<Vec<Vec<Complex<f64>>>> = Vec::with_capacity(gradient_input_batch.len());

        for (_batch_ind, gradient_input) in gradient_input_batch.iter().enumerate() {
            // Transpose to align data dimensions with grad_dwt_2d_partial expectation
            let mut wav_out: Vec<Vec<Complex<f64>>> = transpose(&gradient_input.clone());

            // Apply multi-level DWT backward gradient propagation
            for _ in 0..self.compression_levels_used {
                // Forward DWT (adjoint of inverse DWT) applied to gradient tensor
                let dwt_partial = dwt_2d_partial(&wav_out, &self.wavelet, &self.wavelet_mode);
                let ll_hh = get_ll_hh(&dwt_partial);

                let trend = ll_hh[0].clone();
                let details = ll_hh[1].clone();

                if self.add_details {
                    wav_out = add_matrix_2d_c(&trend, &details);
                } else {
                    wav_out = trend.clone();
                }
            }

            // Transpose back to original axis order expected downstream
            batch_output.push(transpose(&wav_out));
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(batch_output);

        gradient
    }
    pub fn compress_partial(&mut self, input: &[Vec<Complex<f64>>], batch_ind: usize) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>, Vec<usize>) {
        let mut wav_out: Vec<Vec<Complex<f64>>> = input.to_vec();
        let mut details: Vec<Vec<Complex<f64>>> = Vec::new();
        let mut compression_dims: Vec<usize> = Vec::new();
        let mut detail_coefficients_batch: Vec<Vec<Vec<Vec<Complex<f64>>>>> = self.details_batch_coefficients.clone().unwrap_or_else(|| Vec::new());
        let mut detail_coefficients: Vec<Vec<Vec<Complex<f64>>>> = Vec::new();

        //  println!("compression_________________________________________________");
        for _i in 0..self.compression_levels {
            //println!("trend dim: {} {}", trend.len(), trend[0].len());
            let dwt_partial = dwt_2d_partial(&transpose(&wav_out), &self.wavelet, &self.wavelet_mode);
            let ll_hh = get_ll_hh(&dwt_partial);

            let trend = transpose(&ll_hh[0]);
            details = transpose(&ll_hh[1].clone());

            if self.add_details {
                wav_out = add_matrix_2d_c(&trend, &details);
            } else {
                wav_out = trend.clone();
            }
            detail_coefficients.push(details.clone());
            compression_dims.push(trend.len());

            if trend.len() <= 24 {
                self.compression_levels_used = _i + 1;
                break;
            }

            self.compression_levels_used = _i + 1;
        }

        if detail_coefficients.is_empty() {
            self.details_batch_coefficients = Some(vec![detail_coefficients]);
        } else if detail_coefficients_batch.len() > batch_ind {
            detail_coefficients_batch[batch_ind] = detail_coefficients;
            self.details_batch_coefficients = Some(detail_coefficients_batch);
        } else {
            detail_coefficients_batch.push(detail_coefficients);
            self.details_batch_coefficients = Some(detail_coefficients_batch);
        }

        //println!("trend dim: {} {}", trend.len(), trend[0].len());
        (wav_out, details, compression_dims)
    }

    pub fn decompress_partial_gradient(
        &mut self,
        grad_output: &Vec<Vec<Complex<f64>>>, // gradient w.r.t. reconstructed signal for one batch entry
        compression_dim: &Vec<usize>,         // per-level trend sizes (stored during forward)
        batch_ind: usize,
    ) -> Vec<Vec<Complex<f64>>> {
        let gradient_without_target = grad_output.to_vec();
        let mut gradient_transp = transpose(&gradient_without_target);

        // load stored detail coefficients for this batch
        // let detail_coefficients_batch = self.details_batch_coefficients.as_ref().expect("no details_batch_coefficients found").clone();
        // let mut detail_coefficients: Vec<Vec<Vec<Complex<f64>>>> = detail_coefficients_batch[batch_ind].clone();
        let input_batch = self.input_batch.as_ref().expect("Input batch not found");

        // iterate levels in reverse (from coarsest back to original resolution)
        for _i in (0..compression_dim.len()).rev() {
            // the stored detail coefficients were saved in forward; transpose to match gradient_transp layout
            // detail_coefficients[level_idx_rev] = transpose(&detail_coefficients[level_idx_rev]);

            for row in 0..gradient_transp.len() {
                // build HH (detail) vector to pair with LL (trend) gradient
                let mut detail_extension: Vec<Complex<f64>>;

                if self.add_details {
                    // forward: wav_out = trend + details
                    // backward through addition: both operands receive the same gradient
                    detail_extension = gradient_transp[row].to_vec();
                } else {
                    // forward used stored details as constants; supply them here to form the proper HH input
                    detail_extension = vec![Complex::new(0.0, 0.0); gradient_transp[row].len()];
                }

                // ensure both vectors have the same length before concatenation
                self.align_vectors(&mut gradient_transp[row], &mut detail_extension);

                // append HH after LL to form the pair for grad_dwt_2d_partial
                gradient_transp[row].extend_from_slice(&detail_extension);
            }

            // apply the gradient (adjoint) of the partial inverse DWT step
            gradient_transp = inverse_dwt_2d_partial(&gradient_transp, &self.wavelet, &self.wavelet_mode, 0);
        }

        // transpose back to original layout
        let mut gradient_transposed = transpose(&gradient_transp);

        // truncate any padding added during forward
        if gradient_transposed.len() > input_batch[batch_ind].len() {
            self.truncate_matrix(&mut gradient_transposed, input_batch[batch_ind].len());
        }

        gradient_transposed
    }

    pub fn decompress_partial(&mut self, grad_output: &Vec<Vec<Complex<f64>>>, compression_dim: &Vec<usize>, batch_ind: usize) -> Vec<Vec<Complex<f64>>> {
        let gradient_without_target = grad_output.to_vec();
        let mut gradient_transp = transpose(&gradient_without_target);
        let detail_coefficients_batch = self.details_batch_coefficients.as_ref().expect("no details_batch_coefficients found").clone();
        let mut detail_coefficients: Vec<Vec<Vec<Complex<f64>>>> = detail_coefficients_batch[batch_ind].clone();
        let input_batch = self.input_batch.as_ref().expect("Input batch not found");

        for _l in (0..compression_dim.len()).rev() {
            detail_coefficients[_l] = transpose(&detail_coefficients[_l]);
            for i in 0..gradient_transp.len() {
                let mut detail_extension: Vec<Complex<f64>>;

                if self.add_details {
                    detail_extension = gradient_transp[i].to_vec();
                } else {
                    let _detail_coeff_len = detail_coefficients[_l].len();
                    detail_extension = detail_coefficients[_l][0].clone();

                    if !self.is_linear_layer {
                        detail_extension = detail_coefficients[_l][i % _detail_coeff_len].clone();
                    }
                }

                self.align_vectors(&mut gradient_transp[i], &mut detail_extension);

                gradient_transp[i].extend_from_slice(&detail_extension);
            }

            gradient_transp = grad_dwt_2d_partial(&gradient_transp, &self.wavelet, &self.wavelet_mode);
        }

        let mut gradient_transposed = transpose(&gradient_transp);

        if gradient_transposed.len() > input_batch[batch_ind].len() {
            self.truncate_matrix(&mut gradient_transposed, input_batch[batch_ind].len());
        }

        gradient_transposed
    }
    pub fn align_vectors(&self, a: &mut Vec<Complex<f64>>, b: &mut Vec<Complex<f64>>) {
        let len_a = a.len();
        let len_b = b.len();

        if len_a < len_b {
            self.truncate_vectors(b, len_a);
            // a.extend(std::iter::repeat(Complex::new(0.0, 0.0)).take(len_b - len_a));
        } else {
            self.truncate_vectors(a, len_b);
            // b.extend(std::iter::repeat(Complex::new(0.0, 0.0)).take(len_a - len_b));
        }
    }
    pub fn truncate_vectors(&self, a: &mut Vec<Complex<f64>>, b_len: usize) {
        let min_len = a.len().min(b_len);
        a.truncate(min_len);
    }
    pub fn truncate_matrix(&self, matrix: &mut Vec<Vec<Complex<f64>>>, target_rows: usize) {
        // Truncate rows
        let min_rows = matrix.len().min(target_rows);
        matrix.truncate(min_rows);
    }
    pub fn compress_padding_mask(&self, padding_mask: &Vec<u32>) -> Vec<u32> {
        let input_f64: Vec<f64> = padding_mask.iter().map(|v| *v as f64).collect();
        let mut dwt_partial: Vec<f64> = input_f64.clone();

        for _ in 0..self.compression_levels_used {
            let dwt = dwt_1d(&dwt_partial, &self.wavelet, &self.wavelet_mode);
            let wav_hh_ll: Vec<Vec<f64>> = get_ll_hh_1d(&dwt);
            dwt_partial = wav_hh_ll[0].clone(); // Approximation (LL)
        }

        // Step 4: Threshold the approximation to get new mask
        //  println!("original dwt: {:?}", dwt_partial);
        let compressed_mask: Vec<u32> = vec![1; dwt_partial.len()];

        // println!("original padding_mask: {:?}", padding_mask);
        // println!("compressed mask: {:?}", compressed_mask);
        compressed_mask
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
