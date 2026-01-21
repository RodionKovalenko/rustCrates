use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{network_layers_rm::norm_layer_rm::NormalNormLayerRm},
    utils::{
        dtype::{r, C, Real, ZERO},
        matrix::RowMajorMatrix,
    },
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use super::{
    sparse_masked_attention_head_rm::SparseMaskedAttentionHeadRm,
    transformer_updater::calculate_alpha,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSelfAttentionLayerRm {
    pub attention_heads: Vec<SparseMaskedAttentionHeadRm>,
    pub norm_layer: NormalNormLayerRm,
    pub alpha: Real,
    pub beta: Real,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
}

impl SparseSelfAttentionLayerRm {
    pub fn new(num_heads: usize, rows: usize, cols: usize, window_size: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<SparseMaskedAttentionHeadRm> = vec![];
        let head_cols = cols / num_heads;
        for _ in 0..num_heads {
            attention_heads.push(SparseMaskedAttentionHeadRm::new(rows, head_cols, window_size, learning_rate));
        }

        let epsilon: f64 = 1e-12;
        let norm_layer = NormalNormLayerRm::new(cols, epsilon, learning_rate);

        let alpha = calculate_alpha();
        let beta = r(1.0) / alpha;

        Self {
            attention_heads,
            norm_layer,
            alpha,
            beta,
            input_batch_rm: None,
            output_batch_rm: None,
            gradient: None,
            time_step: 0,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm = layer_input
            .get_input_batch_rm_ref()
            .expect("SparseSelfAttentionLayerRm requires RM input");

        let input_batch_rm_original = input_batch_rm.to_vec();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

        // Norm
        let mut local_input = layer_input.clone();
        local_input.clear_input_batch();
        local_input.set_input_batch_rm(input_batch_rm_original.clone());
        local_input.set_padding_mask_batch(padding_mask_batch.clone());

        let mut norm_out = self.norm_layer.forward(&local_input);
        let batch_output_rm = norm_out.take_output_batch_rm().expect("NormRm did not return RM output");

        local_input.set_input_batch_rm(batch_output_rm.clone());

        let attention_head_outputs_rm: Vec<Vec<RowMajorMatrix<C>>> = self
            .attention_heads
            .par_iter_mut()
            .map(|attention_head| {
                let attention_output = attention_head.forward(&local_input);
                attention_output
                    .get_output_batch_rm_ref()
                    .expect("SparseMaskedAttentionHeadRm did not return RM output")
                    .to_vec()
            })
            .collect();

        let out = concat_heads_rm(&attention_head_outputs_rm);

        // Residual: beta*out + input
        let mut out_scaled = out.clone();
        for b in 0..out_scaled.len() {
            assert_eq!(out_scaled[b].cols, input_batch_rm_original[b].cols);
            let in_aligned = if layer_input.get_forward_only() {
                let seq_len_aligned = out_scaled[b].rows;
                if input_batch_rm_original[b].rows > seq_len_aligned {
                    let start = (input_batch_rm_original[b].rows - seq_len_aligned) * input_batch_rm_original[b].cols;
                    RowMajorMatrix::from_data(
                        seq_len_aligned,
                        input_batch_rm_original[b].cols,
                        input_batch_rm_original[b].data[start..].to_vec(),
                    )
                } else {
                    input_batch_rm_original[b].clone()
                }
            } else {
                assert_eq!(out_scaled[b].rows, input_batch_rm_original[b].rows);
                input_batch_rm_original[b].clone()
            };

            for i in 0..out_scaled[b].data.len() {
                out_scaled[b].data[i] = out_scaled[b].data[i] * self.beta + in_aligned.data[i];
            }
        }

        self.input_batch_rm = Some(input_batch_rm_original);
        self.output_batch_rm = Some(out_scaled.clone());
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(out_scaled);
        layer_output.set_padding_mask_batch(padding_mask_batch);
        layer_output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut scaled_upstream: Vec<RowMajorMatrix<C>> = previous_gradient_batch_rm.to_vec();
        for m in scaled_upstream.iter_mut() {
            for v in m.data.iter_mut() {
                *v *= self.beta;
            }
        }

        let num_heads = self.attention_heads.len();
        assert!(num_heads > 0);
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

        // Norm backward
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(combined.clone());
        let norm_gradient = self.norm_layer.backward(&gradient);
        combined = norm_gradient.get_gradient_input_batch_rm();

        // Residual: add unscaled upstream
        let mut combined_with_residual = combined.clone();
        for b in 0..combined_with_residual.len() {
            for i in 0..combined_with_residual[b].data.len() {
                combined_with_residual[b].data[i] += previous_gradient_batch_rm[b].data[i];
            }
        }

        let mut out = Gradient::new_default();
        out.set_gradient_input_batch_rm(combined_with_residual);
        out.set_total_valid_tokens(1);
        self.gradient = Some(out.clone());
        out
    }

    pub fn update_parameters(&mut self) {
        self.norm_layer.update_parameters();
        self.attention_heads.par_iter_mut().for_each(|head| head.update_parameters());
    }

    pub fn clear_cache(&mut self) {
        self.attention_heads.par_iter_mut().for_each(|head| head.clear_cache());
    }
}

fn concat_heads_rm(attention_head_outputs: &[Vec<RowMajorMatrix<C>>]) -> Vec<RowMajorMatrix<C>> {
    assert!(!attention_head_outputs.is_empty());
    let num_heads = attention_head_outputs.len();
    let batch_size = attention_head_outputs[0].len();
    assert!(batch_size > 0);

    let seq_len = attention_head_outputs[0][0].rows;
    let head_dim = attention_head_outputs[0][0].cols;
    for h in 0..num_heads {
        assert_eq!(attention_head_outputs[h].len(), batch_size);
        for b in 0..batch_size {
            assert_eq!(attention_head_outputs[h][b].rows, seq_len);
            assert_eq!(attention_head_outputs[h][b].cols, head_dim);
        }
    }

    let total_dim = head_dim * num_heads;
    let mut out: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let mut data = vec![C::new(ZERO, ZERO); seq_len * total_dim];
        for i in 0..seq_len {
            for h in 0..num_heads {
                let src = &attention_head_outputs[h][b];
                let src_row = src.row_range(i);
                let dst_offset = i * total_dim + h * head_dim;
                data[dst_offset..dst_offset + head_dim].copy_from_slice(&src.data[src_row]);
            }
        }
        out.push(RowMajorMatrix::from_data(seq_len, total_dim, data));
    }
    out
}

fn split_gradient_into_heads_rm(previous_gradient_batch: &[RowMajorMatrix<C>], num_heads: usize) -> Vec<Vec<RowMajorMatrix<C>>> {
    assert!(num_heads > 0);
    let batch_size = previous_gradient_batch.len();
    assert!(batch_size > 0);

    let seq_len = previous_gradient_batch[0].rows;
    let dim = previous_gradient_batch[0].cols;
    assert!(dim % num_heads == 0);
    let head_dim = dim / num_heads;

    let mut out: Vec<Vec<RowMajorMatrix<C>>> = Vec::with_capacity(num_heads);
    for _ in 0..num_heads {
        out.push(Vec::with_capacity(batch_size));
    }

    for b in 0..batch_size {
        let g = &previous_gradient_batch[b];
        for h in 0..num_heads {
            let mut data = vec![C::new(ZERO, ZERO); seq_len * head_dim];
            for i in 0..seq_len {
                let src_offset = i * dim + h * head_dim;
                let dst_offset = i * head_dim;
                data[dst_offset..dst_offset + head_dim].copy_from_slice(&g.data[src_offset..src_offset + head_dim]);
            }
            out[h].push(RowMajorMatrix::from_data(seq_len, head_dim, data));
        }
    }

    out
}
