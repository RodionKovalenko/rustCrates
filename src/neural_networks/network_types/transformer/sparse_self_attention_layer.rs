use crate::neural_networks::{
    network_components::{
        add_rms_norm_layer::RMSNormLayer,
        gradient_struct::Gradient,
        layer::{LayerEnum, LayerType},
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
        norm_layer::NormalNormLayer,
    },
    network_types::transformer::{sparse_masked_attention_head::SparseMaskedAttentionHead, transformer_updater::calculate_alpha},
    utils::matrix::{add_matrix_3d, scale_matrix_3d_by_scalar, RowMajorMatrix},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::dtype::{r, Real, C, ZERO};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSelfAttentionLayer {
    pub attention_heads: Vec<SparseMaskedAttentionHead>,
    pub activated_output: Vec<Vec<C>>,
    pub norm_layer: Option<LayerEnum>,
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
    #[serde(skip)]
    pub last_rm_strict: bool,
}

impl SparseSelfAttentionLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(num_heads: usize, rows: usize, cols: usize, window_size: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<SparseMaskedAttentionHead> = vec![];
        let head_cols = cols / num_heads; // Columns per attention head

        for _i in 0..num_heads {
            //let window_overlaping_shift = window_size + (_i) * window_size;
            let attention_head = SparseMaskedAttentionHead::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer, window_size, learning_rate);
            attention_heads.push(attention_head);
        }

        let epsilon: f64 = 0.000000000001;
        let _norm_layer_rms = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(cols, epsilon, learning_rate))));
        let _norm_layer = Some(LayerEnum::Norm(Box::new(NormalNormLayer::new(cols, epsilon, learning_rate))));

        let alpha = calculate_alpha();
        let beta = r(1.0) / alpha;

        Self {
            attention_heads,
            activated_output: vec![],
            norm_layer: _norm_layer,
            input_batch: None,
            input_batch_rm: None,
            output_batch: None,
            output_batch_rm: None,
            gradient: None,
            time_step: 0,
            last_rm_strict: false,
            alpha,
            beta,
        }
    }
}
impl SparseSelfAttentionLayer {
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        self.last_rm_strict = layer_input.get_rm_strict();
        let input_batch_rm_ref = layer_input.get_input_batch_rm_ref();
        let input_batch_ref = layer_input.get_input_batch_ref();
        let use_rm = input_batch_ref.is_none() && input_batch_rm_ref.is_some();

        if use_rm {
            let input_batch_rm = input_batch_rm_ref.unwrap();
            let mut batch_output_rm: Vec<RowMajorMatrix<C>> = input_batch_rm.to_vec();
            let input_batch_rm_original = input_batch_rm.to_vec();
            let padding_mask_batch = layer_input.get_padding_mask_batch();

            let mut local_input = layer_input.clone();
            local_input.clear_input_batch();
            local_input.set_input_batch_rm(batch_output_rm.clone());
            local_input.set_padding_mask_batch(padding_mask_batch.clone());

            if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
                match norm_layer_enum {
                    LayerEnum::RMSNorm(_rms_norm_layer) => {
                        panic!("RMSNorm is not supported for RM SparseSelfAttentionLayer yet");
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
                    attention_output.get_output_batch_rm_ref().expect("SparseMaskedAttentionHead did not return RM output").to_vec()
                })
                .collect();

            let out = concat_heads_rm(&attention_head_outputs_rm);

            // Residual connection: beta * out + input
            let mut out_scaled = out.clone();
            for b in 0..out_scaled.len() {
                assert_eq!(out_scaled[b].cols, input_batch_rm_original[b].cols);
                if layer_input.get_forward_only() {
                    // Align input to the last seq_len rows if output is trimmed.
                    let seq_len_aligned = out_scaled[b].rows;
                    let in_aligned = if input_batch_rm_original[b].rows > seq_len_aligned {
                        let start = (input_batch_rm_original[b].rows - seq_len_aligned) * input_batch_rm_original[b].cols;
                        RowMajorMatrix::from_data(seq_len_aligned, input_batch_rm_original[b].cols, input_batch_rm_original[b].data[start..].to_vec())
                    } else {
                        input_batch_rm_original[b].clone()
                    };

                    for i in 0..out_scaled[b].data.len() {
                        out_scaled[b].data[i] = out_scaled[b].data[i] * self.beta + in_aligned.data[i];
                    }
                } else {
                    assert_eq!(out_scaled[b].rows, input_batch_rm_original[b].rows);
                    for i in 0..out_scaled[b].data.len() {
                        out_scaled[b].data[i] = out_scaled[b].data[i] * self.beta + input_batch_rm_original[b].data[i];
                    }
                }
            }

            self.input_batch = None;
            self.output_batch = None;
            self.input_batch_rm = Some(input_batch_rm_original);
            self.output_batch_rm = Some(out_scaled.clone());
            self.time_step = layer_input.get_time_step();

            let mut layer_output = LayerOutput::new_default();
            layer_output.set_output_batch_rm(out_scaled);
            layer_output.set_padding_mask_batch(padding_mask_batch);
            return layer_output;
        }

        let mut batch_output = layer_input.get_input_batch();
        let input_batch = layer_input.get_input_batch();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

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

        let seq_len_aligned = attention_head_outputs[0][0].len();
        let mut batch_output: Vec<Vec<Vec<C>>> = vec![vec![vec![]; seq_len_aligned]; batch_size];

        // [head][batch][sequence][dim]
        // Combine the outputs of the attention heads (e.g., concatenating horizontally)
        for b in 0..batch_size {
            for i in 0..seq_len_aligned {
                let mut combined_output: Vec<C> = Vec::new();
                for head_output in &attention_head_outputs {
                    combined_output.extend_from_slice(&head_output[b][i]);
                }
                batch_output[b][i] = combined_output;
            }
        }

        layer_input.set_input_batch(batch_output.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        // Residual connection
        let batch_output_scaled = scale_matrix_3d_by_scalar(&batch_output, self.beta);

        if layer_input.get_forward_only() {
            let input_batch_aligned = input_batch
                .iter()
                .map(|q_seq| {
                    let seq_len = q_seq.len();
                    if seq_len > seq_len_aligned {
                        q_seq[seq_len - seq_len_aligned..].to_vec()
                    } else {
                        q_seq.clone()
                    }
                })
                .collect();

            batch_output = add_matrix_3d(&batch_output_scaled, &input_batch_aligned);
        } else {
            batch_output = add_matrix_3d(&batch_output_scaled, &input_batch);
        }

        self.input_batch = Some(input_batch.clone());
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output.clone());
        layer_output.set_padding_mask_batch(padding_mask_batch.clone());
        self.output_batch = Some(batch_output.clone());

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

        let num_heads = self.attention_heads.len();
        assert!(num_heads > 0, "No attention heads found in sparse self-attention layer!");
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
                assert_eq!(combined[b].rows, gradient_input_batches_rm[h][b].rows);
                assert_eq!(combined[b].cols, gradient_input_batches_rm[h][b].cols);
                for i in 0..combined[b].data.len() {
                    combined[b].data[i] += gradient_input_batches_rm[h][b].data[i];
                }
            }
        }

        gradient.set_gradient_input_batch_rm(combined.clone());

        // Norm backward
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(_rms_norm_layer) => {
                    panic!("RMSNorm is not supported for RM SparseSelfAttentionLayer backward yet");
                }
                LayerEnum::Norm(norm_layer) => {
                    let norm_gradient = norm_layer.backward(&gradient);
                    if let Some(g_rm) = norm_gradient.get_gradient_input_batch_rm_ref() {
                        combined = g_rm.to_vec();
                    } else {
                        if self.last_rm_strict {
                            panic!("RM strict mode violation: SparseSelfAttentionLayer backward_rm would convert Vec gradients from Norm into RM");
                        }
                        combined = norm_gradient.get_gradient_input_batch().iter().map(|m| RowMajorMatrix::from_rows(m)).collect();
                    }
                    gradient.set_gradient_input_batch_rm(combined.clone());
                }
                _ => {}
            }
        }

        // Residual: add unscaled upstream
        let mut combined_with_residual = combined.clone();
        for b in 0..combined_with_residual.len() {
            assert_eq!(combined_with_residual[b].rows, previous_gradient_batch_rm[b].rows);
            assert_eq!(combined_with_residual[b].cols, previous_gradient_batch_rm[b].cols);
            for i in 0..combined_with_residual[b].data.len() {
                combined_with_residual[b].data[i] += previous_gradient_batch_rm[b].data[i];
            }
        }

        gradient.set_gradient_input_batch_rm(combined_with_residual);
        gradient
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        let mut gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient_batch.clone();
        gradient_input_batch = scale_matrix_3d_by_scalar(&gradient_input_batch, self.beta);

        let mut gradient: Gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(previous_gradient_batch.clone());

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

        let mut combined_gradient_input_batch: Vec<Vec<Vec<C>>> =
            vec![vec![vec![C::new(ZERO, ZERO); gradient_input_batches[0][0][0].len()]; gradient_input_batches[0][0].len()]; gradient_input_batches[0].len()];

        for h in 0..gradient_input_batches.len() {
            for b in 0..gradient_input_batches[h].len() {
                for s in 0..gradient_input_batches[h][b].len() {
                    for d in 0..gradient_input_batches[h][b][s].len() {
                        combined_gradient_input_batch[b][s][d] += gradient_input_batches[h][b][s][d];
                    }
                }
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

        combined_gradient_input_batch = add_matrix_3d(&previous_gradient_batch, &gradient_input_batch);

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
    assert!(dim % num_heads == 0, "dim={} must be divisible by num_heads={}", dim, num_heads);
    let head_dim = dim / num_heads;

    let mut out: Vec<Vec<RowMajorMatrix<C>>> = Vec::with_capacity(num_heads);
    for _ in 0..num_heads {
        out.push(Vec::with_capacity(batch_size));
    }

    for b in 0..batch_size {
        let g = &previous_gradient_batch[b];
        assert_eq!(g.rows, seq_len);
        assert_eq!(g.cols, dim);

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
