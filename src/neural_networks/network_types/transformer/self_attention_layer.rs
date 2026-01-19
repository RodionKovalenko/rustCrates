use super::masked_attention_head::MaskedAttentionHead;
use crate::neural_networks::{
    network_components::{
        add_rms_norm_layer::RMSNormLayer,
        gradient_struct::Gradient,
        layer::{LayerEnum, LayerType},
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
        norm_layer::NormalNormLayer,
    },
    network_types::{transformer::{transformer_updater::calculate_alpha}, wavelet_discrete_layer::DiscreteWaveletLayer},
    utils::matrix::{add_matrix_3d, scale_matrix_3d_by_scalar, RowMajorMatrix},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::dtype::{r, C, Real, ZERO};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer {
    pub attention_heads: Vec<MaskedAttentionHead>,
    pub activated_output: Vec<Vec<C>>,
    pub norm_layer: Option<LayerEnum>,
    pub discrete_wavelet_layer: Option<DiscreteWaveletLayer>,
    pub alpha: Real,
    pub beta: Real,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
}

impl SelfAttentionLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(num_heads: usize, rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<MaskedAttentionHead> = vec![];
        let head_cols = cols / num_heads; // Columns per attention head

        for _i in 0..num_heads {
            let attention_head = MaskedAttentionHead::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer, learning_rate);
            attention_heads.push(attention_head);
        }

        let epsilon: f64 = 0.000000000001;
        let _norm_layer_rms = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(cols, epsilon, learning_rate))));
        let _norm_layer = Some(LayerEnum::Norm(Box::new(NormalNormLayer::new(cols, epsilon, learning_rate))));
        let _dwt_layer = Some(DiscreteWaveletLayer::new());

        let alpha = calculate_alpha();
        let beta = r(1.0) / alpha;

        Self {
            attention_heads,
            activated_output: vec![],
            norm_layer: _norm_layer,
            discrete_wavelet_layer: None,
            input_batch: None,
            input_batch_rm: None,
            output_batch: None,
            output_batch_rm: None,
            gradient: None,
            time_step: 0,
            alpha,
            beta,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl SelfAttentionLayer {
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm_ref = layer_input.get_input_batch_rm_ref();
        let input_batch_ref = layer_input.get_input_batch_ref();
        let use_rm = input_batch_ref.is_none() && input_batch_rm_ref.is_some();

        if use_rm {
            let input_batch_rm = input_batch_rm_ref.unwrap();
            let mut batch_output_rm: Vec<RowMajorMatrix<C>> = input_batch_rm.to_vec();
            let input_batch_rm_original = input_batch_rm.to_vec();
            let padding_mask_batch = layer_input.get_padding_mask_batch();

            // DWT path is currently Vec-only; skip if configured.
            if self.discrete_wavelet_layer.is_some() {
                panic!("DiscreteWaveletLayer is not supported for RM SelfAttentionLayer yet");
            }

            let mut local_input = layer_input.clone();
            local_input.clear_input_batch();
            local_input.set_input_batch_rm(batch_output_rm.clone());
            local_input.set_padding_mask_batch(padding_mask_batch.clone());

            if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
                match norm_layer_enum {
                    LayerEnum::RMSNorm(_rms_norm_layer) => {
                        // RMSNorm is still Vec-only in this codepath.
                        panic!("RMSNorm is not supported for RM SelfAttentionLayer yet");
                    }
                    LayerEnum::Norm(norm_layer) => {
                        let output = norm_layer.forward(&local_input);
                        if let Some(out_rm) = output.get_output_batch_rm_ref() {
                            batch_output_rm = out_rm.to_vec();
                            local_input.set_input_batch_rm(batch_output_rm.clone());
                        } else {
                            panic!("NormalNormLayer did not return RM output");
                        }
                    }
                    _ => {}
                }
            }

            let attention_head_outputs_rm: Vec<Vec<RowMajorMatrix<C>>> = self
                .attention_heads
                .par_iter_mut()
                .map(|attention_head| {
                    let attention_output = attention_head.forward(&local_input);
                    attention_output.get_output_batch_rm()
                })
                .collect();

            let combined_rm = concat_heads_rm(&attention_head_outputs_rm);

            // Residual: beta * out + input
            let mut out = combined_rm;
            for (b, m) in out.iter_mut().enumerate() {
                assert_eq!(m.rows, input_batch_rm_original[b].rows);
                assert_eq!(m.cols, input_batch_rm_original[b].cols);
                for i in 0..m.data.len() {
                    m.data[i] = m.data[i] * self.beta + input_batch_rm_original[b].data[i];
                }
            }

            self.input_batch = None;
            self.output_batch = None;
            self.input_batch_rm = Some(input_batch_rm_original);
            self.output_batch_rm = Some(out.clone());
            self.time_step = layer_input.get_time_step();

            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch_rm(out);
            layer_output.set_padding_mask_batch(padding_mask_batch.clone());
            return layer_output;
        }

        let mut batch_output = layer_input.get_input_batch();
        let input_batch = layer_input.get_input_batch();
        let mut padding_mask_batch = layer_input.get_padding_mask_batch();

        // Apply DWT if the layer is present
        if self.discrete_wavelet_layer.is_some() {
            if let Some(dwt_layer) = self.discrete_wavelet_layer.as_mut() {
                let dwt_output = dwt_layer.forward(&layer_input);
                batch_output = dwt_output.get_output_batch();
                padding_mask_batch = dwt_output.get_padding_mask_batch();
            }
        }

        let mut layer_input = layer_input.clone();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        // Process the dense layers
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let output = rms_norm_layer.forward(&layer_input);
                    batch_output = output.get_output_batch();
                }
                LayerEnum::Norm(norm_layer) => {
                    let output = norm_layer.forward(&layer_input);
                    batch_output = output.get_output_batch();
                    //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
                }
                _ => {}
            }
        }

        layer_input.set_input_batch(batch_output.clone());

        let batch_size = input_batch.len();
        let sequence_size = input_batch[0].len();
        let mut batch_output: Vec<Vec<Vec<C>>> = vec![vec![vec![]; sequence_size]; batch_size];
        let batch_size = input_batch.len();

        //println!("padding mask batch: {:?}", &padding_mask_batch);
        let attention_head_outputs: Vec<_> = self
            .attention_heads
            .par_iter_mut() // Use rayon's parallel iterator
            .map(|attention_head| {
                let attention_output = attention_head.forward(&layer_input);
                attention_output.get_output_batch()
            })
            .collect(); // Collect the results into a vector

        // println!("attention head outputs: {:?}", &attention_head_outputs);

        // Combine the outputs of the attention heads (e.g., concatenating horizontally)
        for b in 0..batch_size {
            for i in 0..sequence_size {
                let mut combined_output: Vec<C> = Vec::new();
                for head_output in &attention_head_outputs {
                    combined_output.extend_from_slice(&head_output[b][i]);
                }
                batch_output[b][i] = combined_output;
            }
        }

        layer_input.set_input_batch(batch_output.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        // Decompress Wavelet if the layer is present
        if self.discrete_wavelet_layer.is_some() {
            if let Some(dwt_layer) = self.discrete_wavelet_layer.as_mut() {
                layer_input.set_input_batch(batch_output.clone());
                layer_input.set_padding_mask_batch(padding_mask_batch.clone());

                let wavelet_inverse_output = dwt_layer.forward_inverse(&layer_input);

                batch_output = wavelet_inverse_output.get_output_batch();
                padding_mask_batch = wavelet_inverse_output.get_padding_mask_batch();
            }
        }

        // Residual connection
        let batch_output_scaled = scale_matrix_3d_by_scalar(&batch_output, self.beta);
        batch_output = add_matrix_3d(&batch_output_scaled, &input_batch);

        self.input_batch = Some(input_batch.clone());
        self.input_batch_rm = None;
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output.clone());
        layer_output.set_padding_mask_batch(padding_mask_batch.clone());
        self.output_batch = Some(batch_output.clone());
        self.output_batch_rm = None;

        layer_output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut scaled_upstream: Vec<RowMajorMatrix<C>> = previous_gradient_batch_rm.to_vec();
        for m in scaled_upstream.iter_mut() {
            for v in m.data.iter_mut() {
                *v *= self.beta;
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(previous_gradient_batch_rm.to_vec());

        if self.discrete_wavelet_layer.is_some() {
            panic!("DiscreteWaveletLayer is not supported for RM SelfAttentionLayer backward yet");
        }

        let num_heads = self.attention_heads.len();
        assert!(num_heads > 0, "No attention heads found in self-attention layer!");
        let previous_gradient_head_splitted = split_gradient_into_heads_rm(&scaled_upstream, num_heads);

        let gradient_input_batches_rm: Vec<Vec<RowMajorMatrix<C>>> = self
            .attention_heads
            .par_iter_mut()
            .enumerate()
            .map(|(head_ind, attention_head)| {
                let previous_head_gradient_batch = &previous_gradient_head_splitted[head_ind];
                let g = attention_head.backward_rm(previous_head_gradient_batch);
                g.get_gradient_input_batch_rm()
            })
            .collect();

        let mut combined: Vec<RowMajorMatrix<C>> = gradient_input_batches_rm[0].clone();
        for h in 1..gradient_input_batches_rm.len() {
            for b in 0..combined.len() {
                for i in 0..combined[b].data.len() {
                    combined[b].data[i] += gradient_input_batches_rm[h][b].data[i];
                }
            }
        }

        // Norm backward (RM-capable)
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(_rms_norm_layer) => {
                    panic!("RMSNorm is not supported for RM SelfAttentionLayer backward yet");
                }
                LayerEnum::Norm(norm_layer) => {
                    let mut g = Gradient::new_default();
                    g.set_gradient_input_batch_rm(combined);
                    combined = norm_layer.backward(&g).get_gradient_input_batch_rm();
                }
                _ => {}
            }
        }

        // Residual gradient: upstream + attention path
        let mut out = combined;
        for b in 0..out.len() {
            for i in 0..out[b].data.len() {
                out[b].data[i] += previous_gradient_batch_rm[b].data[i];
            }
        }

        gradient.set_gradient_input_batch_rm(out);
        gradient
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        let mut gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient_batch.clone();
        gradient_input_batch = scale_matrix_3d_by_scalar(&gradient_input_batch, self.beta);

        let mut gradient: Gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(previous_gradient_batch.clone());

        // Apply DWT if the layer is present
        if self.discrete_wavelet_layer.is_some() {
            if let Some(dwt_layer) = self.discrete_wavelet_layer.as_mut() {
                let gradient_inverse: Gradient = dwt_layer.backward_inverse(&gradient);
                gradient_input_batch = gradient_inverse.get_gradient_input_batch();
                gradient.set_gradient_input_batch(gradient_input_batch.clone());
            }
        }

        let num_heads = self.attention_heads.len();
        assert!(num_heads > 0, "No attention heads found in self-attention layer!");

        let previous_gradient_head_splitted = self.split_gradient_into_heads(&gradient_input_batch);

        let gradient_input_batches: Vec<Vec<Vec<Vec<C>>>> = self
            .attention_heads
            .par_iter_mut()
            .enumerate()
            .map(|(head_ind, attention_head)| {
                // It's better to borrow if possible, not clone — clone only if needed
                let previous_head_gradient_batch = &previous_gradient_head_splitted[head_ind];
                let gradient = attention_head.backward(previous_head_gradient_batch);
                gradient.get_gradient_input_batch()
            })
            .collect();

        let mut combined_gradient_input_batch: Vec<Vec<Vec<C>>> = vec![
            vec![
                vec![C::new(ZERO, ZERO); gradient_input_batches[0][0][0].len()];
                gradient_input_batches[0][0].len()
            ];
            gradient_input_batches[0].len()
        ];

        for h in 0..gradient_input_batches.len() {
            for b in 0..gradient_input_batches[h].len() {
                for s in 0..gradient_input_batches[h][b].len() {
                    for d in 0..gradient_input_batches[h][b][s].len() {
                        combined_gradient_input_batch[b][s][d] += gradient_input_batches[h][b][s][d];
                    }
                }
            }
        }

        // let max = combined_gradient_input_batch.iter().flat_map(|v| v.iter().flat_map(|w| w.iter())).max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Less));
        // let min = combined_gradient_input_batch.iter().flat_map(|v| v.iter().flat_map(|w| w.iter())).min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Greater));

        // println!("max in backward self-attention layer gradient batch: {:?}", max);
        // println!("min in backward self-attention layer gradient batch: {:?}", min);

        if self.discrete_wavelet_layer.is_some() {
            if let Some(dwt_layer) = self.discrete_wavelet_layer.as_mut() {
                gradient.set_gradient_input_batch(combined_gradient_input_batch.clone());

                let dwt_gradient = dwt_layer.backward(&gradient);
                combined_gradient_input_batch = dwt_gradient.get_gradient_input_batch();
            }
        }

        gradient.set_gradient_input_batch(combined_gradient_input_batch.clone());
        gradient_input_batch = combined_gradient_input_batch.clone();

        // Process the dense layers
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let norm_gradient = rms_norm_layer.backward(&gradient_input_batch);
                    gradient_input_batch = norm_gradient.get_gradient_input_batch();
                    gradient.set_gradient_input_batch(gradient_input_batch.clone());
                }
                LayerEnum::Norm(norm_layer) => {
                    let norm_gradient = norm_layer.backward(&gradient);
                    gradient_input_batch = norm_gradient.get_gradient_input_batch();
                    gradient.set_gradient_input_batch(gradient_input_batch.clone());
                }
                _ => {}
            }
        }

        combined_gradient_input_batch = add_matrix_3d(previous_gradient_batch, &gradient_input_batch);

        // Return the final gradient
        gradient.set_gradient_input_batch(combined_gradient_input_batch);

        gradient
    }

    pub fn split_gradient_into_heads(&self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Vec<Vec<Vec<Vec<C>>>> {
        let batch_size = previous_gradient_batch.len();
        let seq_len = previous_gradient_batch[0].len();
        let dim = previous_gradient_batch[0][0].len();
        let num_heads = self.attention_heads.len();
        let head_dim = dim / num_heads;

        assert!(dim % num_heads == 0, "dim={} must be divisible by num_heads={}", dim, num_heads);

        // Initialize a vector to store gradients for each attention head
        let mut grad_heads = vec![vec![vec![vec![C::new(ZERO, ZERO); head_dim]; seq_len]; batch_size]; num_heads];

        for batch_ind in 0..batch_size {
            for seq_in in 0..seq_len {
                for head in 0..num_heads {
                    let start_idx = head * head_dim;
                    let end_idx = start_idx + head_dim;

                    grad_heads[head][batch_ind][seq_in] = previous_gradient_batch[batch_ind][seq_in][start_idx..end_idx].to_vec();
                }
            }
        }

        //println!("grad_heads: {}, {}, {}, {}", &grad_heads.len(), grad_heads[0].len(), grad_heads[0][0].len(), grad_heads[0][0][0].len());

        grad_heads
    }

    pub fn update_parameters(&mut self) {
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    rms_norm_layer.update_parameters();
                }
                LayerEnum::Norm(norm_layer) => {
                    norm_layer.update_parameters();
                }
                _ => {}
            }
        }

        self.attention_heads.par_iter_mut().for_each(|attention_head| attention_head.update_parameters());
    }
}

fn concat_heads_rm(head_outputs: &[Vec<RowMajorMatrix<C>>]) -> Vec<RowMajorMatrix<C>> {
    assert!(!head_outputs.is_empty(), "no head outputs");
    let batch_size = head_outputs[0].len();
    let num_heads = head_outputs.len();
    let mut out = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let rows = head_outputs[0][b].rows;
        let total_cols: usize = (0..num_heads).map(|h| head_outputs[h][b].cols).sum();
        let mut data = vec![C::new(ZERO, ZERO); rows * total_cols];

        for r in 0..rows {
            let mut col_off = 0;
            for h in 0..num_heads {
                let m = &head_outputs[h][b];
                let cols = m.cols;
                let src = &m.data[r * cols..(r + 1) * cols];
                let dst_start = r * total_cols + col_off;
                data[dst_start..dst_start + cols].copy_from_slice(src);
                col_off += cols;
            }
        }

        out.push(RowMajorMatrix::from_data(rows, total_cols, data));
    }

    out
}

fn split_gradient_into_heads_rm(previous_gradient_batch_rm: &[RowMajorMatrix<C>], num_heads: usize) -> Vec<Vec<RowMajorMatrix<C>>> {
    let batch_size = previous_gradient_batch_rm.len();
    let seq_len = previous_gradient_batch_rm[0].rows;
    let dim = previous_gradient_batch_rm[0].cols;
    let head_dim = dim / num_heads;
    assert!(dim % num_heads == 0, "dim={} must be divisible by num_heads={}", dim, num_heads);

    let mut grad_heads: Vec<Vec<RowMajorMatrix<C>>> = vec![Vec::with_capacity(batch_size); num_heads];
    for _h in 0..num_heads {
        // placeholder; will fill below
    }

    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let mut per_batch = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let m = &previous_gradient_batch_rm[b];
            let mut data = vec![C::new(ZERO, ZERO); seq_len * head_dim];
            for r in 0..seq_len {
                let src = &m.data[r * dim + start..r * dim + end];
                let dst = &mut data[r * head_dim..(r + 1) * head_dim];
                dst.copy_from_slice(src);
            }
            per_batch.push(RowMajorMatrix::from_data(seq_len, head_dim, data));
        }
        grad_heads[h] = per_batch;
    }
    grad_heads
}
