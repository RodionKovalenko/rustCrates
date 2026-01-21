use core::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::{
    neural_networks::{
        network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
        network_layers::layer::LayerEnum,
        utils::{
            dtype::{c_from_f64, c_to_f64, C, CF64},
            matrix::{transpose_rm, RowMajorMatrix},
        },
    },
    wavelet_transform::{
        dwt::{dwt_1d, dwt_2d_partial_rm, get_ll_hh_1d, get_ll_hh_rm, inverse_dwt_2d_partial_rm},
        dwt_types::DiscreteWaveletType,
        modes::WaveletMode,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteWaveletLayerRm {
    pub wavelet: DiscreteWaveletType,
    pub wavelet_mode: WaveletMode,
    pub is_full_mode: bool,
    pub wavelet_size: usize,
    pub compression_levels: usize,
    pub norm_layer: Option<LayerEnum>,
    pub add_details: bool,
    pub is_linear_layer: bool,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub compression_levels_used: usize,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub compression_dims: Option<Vec<Vec<usize>>>,
    #[serde(skip)]
    pub compressed_padding_mask_b: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub target_batch_ids: Option<Vec<Vec<u32>>>,
}

impl DiscreteWaveletLayerRm {
    pub fn new() -> Self {
        Self {
            input_batch_rm: None,
            gradient: None,
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
            compression_dims: None,
            compressed_padding_mask_b: None,
            padding_mask_batch: None,
            target_batch_ids: None,
            norm_layer: None,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        if layer_input.has_non_empty_input_batch() {
            panic!("DiscreteWaveletLayerRm received Vec input batch; use DiscreteWaveletLayer instead");
        }

        let input_batch_rm: Vec<RowMajorMatrix<C>> = layer_input
            .get_input_batch_rm_ref()
            .expect("DiscreteWaveletLayerRm::forward expects RM input")
            .to_vec();

        let target_batch_ids: Vec<Vec<u32>> = layer_input.get_target_batch_ids();
        let forward_only = layer_input.get_forward_only();
        let mut padding_mask_batch: Vec<Vec<u32>> = layer_input.get_padding_mask_batch();
        let time_step = layer_input.get_time_step();

        if padding_mask_batch.is_empty() || padding_mask_batch[0].is_empty() {
            padding_mask_batch = vec![vec![1; input_batch_rm[0].rows]; input_batch_rm.len()];
        }

        let mut trend_batch_rm: Vec<RowMajorMatrix<C>> = Vec::new();
        let mut compression_dims: Vec<Vec<usize>> = Vec::new();
        let mut comp_pad_mask_b: Vec<Vec<u32>> = Vec::new();

        if !forward_only || (forward_only && time_step == 0) {
            for (batch_ind, input_rm) in input_batch_rm.iter().enumerate() {
                let (new_trend, _details_last, compression_dim) = self.compress_partial_rm(input_rm);
                trend_batch_rm.push(new_trend);
                compression_dims.push(compression_dim);
                comp_pad_mask_b.push(self.compress_padding_mask(&padding_mask_batch[batch_ind]));
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
                    LayerEnum::NormRm(norm_layer_rm) => {
                        let out = norm_layer_rm.forward(&li_norm);
                        trend_batch_rm_normed = out.get_output_batch_rm();
                    }
                    LayerEnum::Norm(_) => {
                        panic!("DiscreteWaveletLayerRm has Vec Norm layer; use NormRm instead");
                    }
                    _ => {}
                }
            }
        }

        self.input_batch_rm = Some(input_batch_rm);
        self.time_step = layer_input.get_time_step();
        self.output_batch_rm = Some(trend_batch_rm_normed.clone());
        self.target_batch_ids = Some(target_batch_ids);
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.compressed_padding_mask_b = Some(comp_pad_mask_b.clone());
        self.compression_dims = Some(compression_dims);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(trend_batch_rm_normed.clone());
        layer_output.set_padding_mask_batch(padding_mask_batch.clone());

        if !trend_batch_rm_normed.is_empty() && trend_batch_rm_normed[0].rows != padding_mask_batch[0].len() {
            layer_output.set_padding_mask_batch(comp_pad_mask_b.clone());
        }

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        // Strict RM-only: require RM gradients; do not use get_gradient_input_batch() because it will
        // convert RM->Vec and appear "non-empty" even when only RM is provided.
        let Some(grad_output_batch_rm_ref) = previous_gradient.get_gradient_input_batch_rm_ref() else {
            if previous_gradient
                .get_gradient_input_batch_ref()
                .is_some_and(|vec_g| !vec_g.is_empty())
            {
                panic!("DiscreteWaveletLayerRm received Vec gradients; use DiscreteWaveletLayer instead");
            }

            let mut out = Gradient::new_default();
            out.set_time_step(previous_gradient.get_time_step());
            out.set_gradient_input_batch_rm(vec![]);
            self.gradient = Some(out.clone());
            return out;
        };

        let mut grad_output_batch_rm = grad_output_batch_rm_ref.to_vec();

        if grad_output_batch_rm.is_empty() {
            let mut out = Gradient::new_default();
            out.set_time_step(previous_gradient.get_time_step());
            out.set_gradient_input_batch_rm(vec![]);
            self.gradient = Some(out.clone());
            return out;
        }

        let input_batch_rm = self
            .input_batch_rm
            .as_ref()
            .expect("DiscreteWaveletLayerRm::backward expects RM cache from forward")
            .to_vec();
        let compression_dims = self.compression_dims.as_ref().expect("no compression dwt found").clone();

        assert_eq!(input_batch_rm.len(), grad_output_batch_rm.len(), "Input and gradient batch size mismatch");

        // Apply norm backward if present
        if let Some(norm_layer) = &mut self.norm_layer {
            match norm_layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let norm_gradient = rms_norm_layer.backward_rm(&grad_output_batch_rm);
                    grad_output_batch_rm = norm_gradient.get_gradient_input_batch_rm();
                }
                LayerEnum::NormRm(norm_layer_rm) => {
                    let mut g = Gradient::new_default();
                    g.set_gradient_input_batch_rm(grad_output_batch_rm);
                    let norm_gradient = norm_layer_rm.backward(&g);
                    grad_output_batch_rm = norm_gradient.get_gradient_input_batch_rm();
                }
                LayerEnum::Norm(_) => {
                    panic!("DiscreteWaveletLayerRm has Vec Norm layer; use NormRm instead");
                }
                _ => {}
            }
        }

        let mut grad_input_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(grad_output_batch_rm.len());
        for (batch_ind, grad_output_rm) in grad_output_batch_rm.iter().enumerate() {
            let compression_dim = &compression_dims[batch_ind];
            let gradient_decompr = self.decompress_partial_gradient_rm(grad_output_rm, compression_dim, batch_ind);
            grad_input_batch_rm.push(gradient_decompr);
        }

        let mut out = Gradient::new_default();
        out.set_time_step(self.time_step);
        out.set_gradient_input_batch_rm(grad_input_batch_rm);

        self.gradient = Some(out.clone());
        out
    }

    fn compress_partial_rm(&mut self, input_rm: &RowMajorMatrix<C>) -> (RowMajorMatrix<C>, RowMajorMatrix<C>, Vec<usize>) {
        let mut wav_out_f64 = RowMajorMatrix::from_data(input_rm.rows, input_rm.cols, input_rm.data.iter().copied().map(c_to_f64).collect());
        let mut details_last_f64: RowMajorMatrix<CF64> = RowMajorMatrix::from_data(0, 0, vec![]);
        let mut compression_dims: Vec<usize> = Vec::new();

        for level in 0..self.compression_levels {
            let wav_t = transpose_rm(&wav_out_f64); // dim x seq
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

            wav_out_f64 = next;
            details_last_f64 = details.clone();
            compression_dims.push(wav_out_f64.rows);

            self.compression_levels_used = level + 1;
            if wav_out_f64.rows <= 24 {
                break;
            }
        }

        let wav_out_c = RowMajorMatrix::from_data(wav_out_f64.rows, wav_out_f64.cols, wav_out_f64.data.into_iter().map(c_from_f64).collect());
        let details_last_c = RowMajorMatrix::from_data(details_last_f64.rows, details_last_f64.cols, details_last_f64.data.into_iter().map(c_from_f64).collect());

        (wav_out_c, details_last_c, compression_dims)
    }

    fn decompress_partial_gradient_rm(&mut self, grad_output_rm: &RowMajorMatrix<C>, compression_dim: &Vec<usize>, batch_ind: usize) -> RowMajorMatrix<C> {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("Input batch RM not found");
        let original_seq_len = input_batch_rm[batch_ind].rows;

        let grad_output_f64 = RowMajorMatrix::from_data(grad_output_rm.rows, grad_output_rm.cols, grad_output_rm.data.iter().copied().map(c_to_f64).collect());
        let mut grad_transp = transpose_rm(&grad_output_f64); // dim x trend_len

        for _ in (0..compression_dim.len()).rev() {
            let ll_len = grad_transp.cols;
            let mut combined = RowMajorMatrix::from_data(
                grad_transp.rows,
                ll_len * 2,
                vec![CF64::new(0.0, 0.0); grad_transp.rows * ll_len * 2],
            );

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
            let mut truncated = RowMajorMatrix::from_data(
                original_seq_len,
                grad.cols,
                vec![CF64::new(0.0, 0.0); original_seq_len * grad.cols],
            );
            truncated.data.copy_from_slice(&grad.data[0..original_seq_len * grad.cols]);
            grad = truncated;
        }

        RowMajorMatrix::from_data(grad.rows, grad.cols, grad.data.into_iter().map(c_from_f64).collect())
    }

    fn compress_padding_mask(&self, padding_mask: &Vec<u32>) -> Vec<u32> {
        let input_f64: Vec<f64> = padding_mask.iter().map(|v| *v as f64).collect();
        let mut dwt_partial: Vec<f64> = input_f64.clone();

        for _ in 0..self.compression_levels_used {
            let dwt = dwt_1d(&dwt_partial, &self.wavelet, &self.wavelet_mode);
            let wav_hh_ll: Vec<Vec<f64>> = get_ll_hh_1d(&dwt);
            dwt_partial = wav_hh_ll[0].clone();
        }

        vec![1; dwt_partial.len()]
    }
}
