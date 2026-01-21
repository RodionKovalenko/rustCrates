use crate::neural_networks::{
    network_components::{
        gradient_struct::Gradient,
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
    },
    network_layers::{
        add_rms_norm_layer::RMSNormLayer,
        layer::{LayerEnum, LayerType},
        network_layers_rm::norm_layer_rm::NormalNormLayerRm,
    },
    network_types::transformer::masked_attention_head_approximation::MaskedAttentionHeadApproximation,
    utils::matrix::RowMajorMatrix,
};

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::dtype::{C, ZERO};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayerApproximationRm {
    pub attention_heads: Vec<MaskedAttentionHeadApproximation>,
    pub activated_output: Vec<Vec<C>>,
    pub norm_layer: Option<LayerEnum>,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub time_step: usize,
}

impl SelfAttentionLayerApproximationRm {
    pub fn new(num_heads: usize, rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<MaskedAttentionHeadApproximation> = vec![];
        let head_cols = cols / num_heads;

        for _ in 0..num_heads {
            let attention_head =
                MaskedAttentionHeadApproximation::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer, learning_rate);
            attention_heads.push(attention_head);
        }

        let epsilon: f64 = 0.000000000001;
        let norm_layer = Some(LayerEnum::NormRm(Box::new(NormalNormLayerRm::new(cols, epsilon, learning_rate))));
        let _norm_layer_rms = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(cols, epsilon, learning_rate))));

        Self {
            attention_heads,
            activated_output: vec![],
            norm_layer,
            input_batch_rm: None,
            output_batch_rm: None,
            time_step: 0,
        }
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm = layer_input
            .get_input_batch_rm_ref()
            .expect("SelfAttentionLayerApproximationRm::forward expects RM input_batch_rm");

        assert!(!input_batch_rm.is_empty(), "SelfAttentionLayerApproximationRm::forward got empty RM batch");

        let padding_mask_batch = layer_input.get_padding_mask_batch();
        self.input_batch_rm = Some(input_batch_rm.to_vec());
        self.time_step = layer_input.get_time_step();

        let batch_size = input_batch_rm.len();
        let sequence_size = input_batch_rm[0].rows;
        let d_model = input_batch_rm[0].cols;

        let mut li = layer_input.clone();
        li.clear_input_batch();
        li.set_input_batch_rm(input_batch_rm.to_vec());
        li.set_padding_mask_batch(padding_mask_batch.clone());

        let attention_head_outputs_rm: Vec<Vec<RowMajorMatrix<C>>> = self
            .attention_heads
            .par_iter_mut()
            .map(|attention_head| {
                let attention_output = attention_head.forward(&li);
                attention_output.get_output_batch_rm()
            })
            .collect();

        let num_heads = attention_head_outputs_rm.len();
        assert!(num_heads > 0);
        let head_dim = attention_head_outputs_rm[0][0].cols;
        assert_eq!(head_dim * num_heads, d_model);

        let mut combined_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut out = RowMajorMatrix::from_data(sequence_size, d_model, vec![C::new(ZERO, ZERO); sequence_size * d_model]);
            for t in 0..sequence_size {
                let out_row = out.row_range(t);
                let mut col_off = 0;
                for h in 0..num_heads {
                    let src_row = attention_head_outputs_rm[h][b].row_range(t);
                    let src = &attention_head_outputs_rm[h][b].data[src_row];
                    out.data[out_row.start + col_off..out_row.start + col_off + head_dim].copy_from_slice(src);
                    col_off += head_dim;
                }
            }
            combined_rm.push(out);
        }

        let mut li2 = layer_input.clone();
        li2.clear_input_batch();
        li2.set_input_batch_rm(combined_rm.clone());
        li2.set_input_batch_before_rm(input_batch_rm.to_vec());
        li2.set_padding_mask_batch(padding_mask_batch.clone());

        let mut out_rm = combined_rm;
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let output = rms_norm_layer.forward(&li2);
                    out_rm = output.get_output_batch_rm();
                }
                LayerEnum::NormRm(norm_layer) => {
                    let mut output = norm_layer.forward(&li2);
                    out_rm = output.take_output_batch_rm().expect("NormRm did not return RM output");
                }
                LayerEnum::Norm(_) => {
                    panic!("SelfAttentionLayerApproximationRm requires NormRm (Vec Norm not allowed)");
                }
                _ => {}
            }
        }

        self.output_batch_rm = Some(out_rm.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(out_rm);
        layer_output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(previous_gradient_batch_rm.to_vec());

        let mut grad_input_rm: Vec<RowMajorMatrix<C>> = previous_gradient_batch_rm.to_vec();
        let mut output_gradient_norm_rm: Vec<RowMajorMatrix<C>> = Vec::new();

        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let norm_gradient = rms_norm_layer.backward_rm(&grad_input_rm);
                    grad_input_rm = norm_gradient.get_gradient_input_batch_rm();
                    output_gradient_norm_rm = grad_input_rm.clone();
                }
                LayerEnum::NormRm(norm_layer) => {
                    let mut g = Gradient::new_default();
                    g.set_gradient_input_batch_rm(grad_input_rm);
                    let norm_gradient = norm_layer.backward(&g);
                    grad_input_rm = norm_gradient.get_gradient_input_batch_rm();
                    output_gradient_norm_rm = grad_input_rm.clone();
                }
                LayerEnum::Norm(_) => {
                    panic!("SelfAttentionLayerApproximationRm backward_rm requires NormRm");
                }
                _ => {}
            }
        }

        let num_heads = self.attention_heads.len();
        assert!(num_heads > 0, "No attention heads found in self-attention approximation layer!");
        let batch_size = grad_input_rm.len();
        let seq_len = grad_input_rm[0].rows;
        let dim = grad_input_rm[0].cols;
        let head_dim = dim / num_heads;
        assert!(dim % num_heads == 0);

        // Split into per-head RM gradients
        let mut grad_heads_rm: Vec<Vec<RowMajorMatrix<C>>> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let mut per_batch = Vec::with_capacity(batch_size);
            for b in 0..batch_size {
                let mut m = RowMajorMatrix::from_data(seq_len, head_dim, vec![C::new(ZERO, ZERO); seq_len * head_dim]);
                for t in 0..seq_len {
                    let src_row = grad_input_rm[b].row_range(t);
                    let dst_row = m.row_range(t);
                    let start = src_row.start + h * head_dim;
                    let end = start + head_dim;
                    m.data[dst_row].copy_from_slice(&grad_input_rm[b].data[start..end]);
                }
                per_batch.push(m);
            }
            grad_heads_rm.push(per_batch);
        }

        // Backpropagate per-head
        let mut grad_inputs_per_head: Vec<Vec<RowMajorMatrix<C>>> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let head_grad = self.attention_heads[h].backward_rm(&grad_heads_rm[h]);
            grad_inputs_per_head.push(head_grad.get_gradient_input_batch_rm());
        }

        // Combine head grads back into d_model
        let mut combined: Vec<RowMajorMatrix<C>> = vec![
            RowMajorMatrix::from_data(seq_len, dim, vec![C::new(ZERO, ZERO); seq_len * dim]);
            batch_size
        ];

        for h in 0..num_heads {
            for b in 0..batch_size {
                for t in 0..seq_len {
                    let dst_row = combined[b].row_range(t);
                    let src_row = grad_inputs_per_head[h][b].row_range(t);
                    let start = dst_row.start + h * head_dim;
                    let end = start + head_dim;
                    combined[b].data[start..end].copy_from_slice(&grad_inputs_per_head[h][b].data[src_row]);
                }
            }
        }

        // Residual behavior (matches old hybrid impl)
        if !output_gradient_norm_rm.is_empty() {
            for b in 0..batch_size {
                for idx in 0..combined[b].data.len() {
                    combined[b].data[idx] += output_gradient_norm_rm[b].data[idx];
                }
            }
        }

        gradient.set_gradient_input_batch_rm(combined);
        gradient
    }

    pub fn update_parameters(&mut self) {
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    rms_norm_layer.update_parameters();
                }
                LayerEnum::NormRm(norm_layer) => {
                    norm_layer.update_parameters();
                }
                _ => {}
            }
        }

        self.attention_heads.par_iter_mut().for_each(|attention_head| attention_head.update_parameters());
    }
}
