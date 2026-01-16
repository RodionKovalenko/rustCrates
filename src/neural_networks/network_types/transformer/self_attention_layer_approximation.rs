use crate::neural_networks::{
    network_components::{
        add_rms_norm_layer::RMSNormLayer,
        gradient_struct::Gradient,
        layer::{LayerEnum, LayerType},
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
        norm_layer::NormalNormLayer,
    },
    network_types::transformer::masked_attention_head_approximation::MaskedAttentionHeadApproximation,
    utils::matrix::{add_matrix_3d, RowMajorMatrix},
};
use num::Complex;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayerApproximation {
    pub attention_heads: Vec<MaskedAttentionHeadApproximation>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub norm_layer: Option<LayerEnum>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    #[serde(skip)]
    pub time_step: usize,
}

impl SelfAttentionLayerApproximation {
    // Constructor to initialize multiple attention heads
    pub fn new(num_heads: usize, rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<MaskedAttentionHeadApproximation> = vec![];
        let head_cols = cols / num_heads; // Columns per attention head

        for _i in 0..num_heads {
            let attention_head = MaskedAttentionHeadApproximation::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer, learning_rate);
            attention_heads.push(attention_head);
        }

        let epsilon: f64 = 0.000000000001;
        let _norm_layer_rms = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(cols, epsilon, learning_rate))));
        let _norm_layer = Some(LayerEnum::Norm(Box::new(NormalNormLayer::new(cols, epsilon, learning_rate))));

        Self {
            attention_heads,
            activated_output: vec![],
            norm_layer: _norm_layer,
            input_batch: None,
            input_batch_rm: None,
            output_batch: None,
            output_batch_rm: None,
            time_step: 0,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl SelfAttentionLayerApproximation {
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm_ref = layer_input.get_input_batch_rm_ref();
        let input_batch_ref = layer_input.get_input_batch_ref();
        let use_rm = input_batch_ref.is_none() && input_batch_rm_ref.is_some();

        if use_rm {
            let input_batch_rm = input_batch_rm_ref.unwrap();
            if !input_batch_rm.is_empty() {
                let padding_mask_batch = layer_input.get_padding_mask_batch();
                self.input_batch = None;
                self.input_batch_rm = Some(input_batch_rm.to_vec());
                self.time_step = layer_input.get_time_step();

                let batch_size = input_batch_rm.len();
                let sequence_size = input_batch_rm[0].rows;
                let d_model = input_batch_rm[0].cols;

                let mut li = layer_input.clone();
                li.clear_input_batch();
                li.set_input_batch_rm(input_batch_rm.to_vec());
                li.set_padding_mask_batch(padding_mask_batch.clone());

                let attention_head_outputs_rm: Vec<Vec<RowMajorMatrix<Complex<f64>>>> = self
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

                let mut combined_rm: Vec<RowMajorMatrix<Complex<f64>>> = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let mut out = RowMajorMatrix::from_data(sequence_size, d_model, vec![Complex::new(0.0, 0.0); sequence_size * d_model]);
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

                // Norm
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
                        LayerEnum::Norm(norm_layer) => {
                            let output = norm_layer.forward(&li2);
                            out_rm = output.get_output_batch_rm();
                        }
                        _ => {}
                    }
                }

                self.output_batch = None;
                self.output_batch_rm = Some(out_rm.clone());

                let mut layer_output = LayerOutput::new_default();
                layer_output.set_output_batch_rm(out_rm);
                return layer_output;
            }
        }

        let input_batch_before = layer_input.get_input_batch();
        let input_batch = input_batch_before.clone();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

        self.input_batch = Some(input_batch.clone());
        self.input_batch_rm = None;
        self.time_step = layer_input.get_time_step();

        let batch_size = input_batch.len();
        let sequence_size = input_batch[0].len();
        let mut batch_output: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![]; sequence_size]; batch_size];
        let batch_size = input_batch.len();

        // Apply the attention mechanism for each head
        let mut layer_input = layer_input.clone();
        layer_input.set_input_batch(input_batch.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

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
                let mut combined_output: Vec<Complex<f64>> = Vec::new();
                for head_output in &attention_head_outputs {
                    combined_output.extend_from_slice(&head_output[b][i]);
                }
                batch_output[b][i] = combined_output;
            }
        }

        layer_input.set_input_batch(batch_output.clone());
        layer_input.set_input_batch_before(input_batch_before.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        self.output_batch = Some(batch_output.clone());

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

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output.clone());
        self.output_batch = Some(batch_output.clone());
        self.output_batch_rm = None;

        layer_output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<Complex<f64>>]) -> Gradient {
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(previous_gradient_batch_rm.to_vec());

        let mut grad_input_rm: Vec<RowMajorMatrix<Complex<f64>>> = previous_gradient_batch_rm.to_vec();
        let mut output_gradient_norm_rm: Vec<RowMajorMatrix<Complex<f64>>> = Vec::new();

        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let norm_gradient = rms_norm_layer.backward_rm(&grad_input_rm);
                    grad_input_rm = norm_gradient.get_gradient_input_batch_rm();
                    output_gradient_norm_rm = grad_input_rm.clone();
                }
                LayerEnum::Norm(norm_layer) => {
                    let mut g = Gradient::new_default();
                    g.set_gradient_input_batch_rm(grad_input_rm);
                    let norm_gradient = norm_layer.backward(&g);
                    grad_input_rm = norm_gradient.get_gradient_input_batch_rm();
                    output_gradient_norm_rm = grad_input_rm.clone();
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
        let mut grad_heads_rm: Vec<Vec<RowMajorMatrix<Complex<f64>>>> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let mut per_batch = Vec::with_capacity(batch_size);
            for b in 0..batch_size {
                let mut m = RowMajorMatrix::from_data(seq_len, head_dim, vec![Complex::new(0.0, 0.0); seq_len * head_dim]);
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

        // Backprop through each head
        let mut grad_inputs_per_head: Vec<Vec<RowMajorMatrix<Complex<f64>>>> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let head_grad = self.attention_heads[h].backward_rm(&grad_heads_rm[h]);
            grad_inputs_per_head.push(head_grad.get_gradient_input_batch_rm());
        }

        // Sum gradients from heads
        let d_model = grad_inputs_per_head[0][0].cols;
        let mut combined: Vec<RowMajorMatrix<Complex<f64>>> = vec![
            RowMajorMatrix::from_data(seq_len, d_model, vec![Complex::new(0.0, 0.0); seq_len * d_model]);
            batch_size
        ];
        for h in 0..num_heads {
            for b in 0..batch_size {
                for idx in 0..combined[b].data.len() {
                    combined[b].data[idx] += grad_inputs_per_head[h][b].data[idx];
                }
            }
        }

        // Preserve the same residual behavior as the Vec path.
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

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        // let input_batch = self.input_batch.as_ref().expect("Input batch not found in self-attention layer backward");
        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient_batch.clone();

        let mut gradient: Gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(previous_gradient_batch.clone());

        let mut output_gradient_norm = vec![];

        // Process the dense layers
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let norm_gradient = rms_norm_layer.backward(&gradient_input_batch);
                    gradient_input_batch = norm_gradient.get_gradient_input_batch();
                    gradient.set_gradient_input_batch(gradient_input_batch.clone());
                    output_gradient_norm = gradient_input_batch.clone();
                }
                LayerEnum::Norm(norm_layer) => {
                    let norm_gradient = norm_layer.backward(&gradient);
                    gradient_input_batch = norm_gradient.get_gradient_input_batch();
                    gradient.set_gradient_input_batch(gradient_input_batch.clone());
                    output_gradient_norm = gradient_input_batch.clone();
                }
                _ => {}
            }
        }

        let num_heads = self.attention_heads.len();
        assert!(num_heads > 0, "No attention heads found in self-attention layer!");

        let previous_gradient_head_splitted = self.split_gradient_into_heads(&gradient_input_batch);
        let mut gradient_input_batches: Vec<Vec<Vec<Vec<Complex<f64>>>>> = Vec::new();

        // Backpropagate gradients through each attention head
        for (head_ind, attention_head) in self.attention_heads.iter_mut().enumerate() {
            let previous_head_gradient_batch = previous_gradient_head_splitted[head_ind].clone();

            gradient = attention_head.backward(&previous_head_gradient_batch);

            gradient_input_batches.push(gradient.get_gradient_input_batch());

            // println!("gradient input head {:?}", &gradient.get_gradient_input_batch());
        }

        let mut combined_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); gradient_input_batches[0][0][0].len()]; gradient_input_batches[0][0].len()]; gradient_input_batches[0].len()];

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

        if !output_gradient_norm.is_empty() {
            combined_gradient_input_batch = add_matrix_3d(&combined_gradient_input_batch, &output_gradient_norm);
        }

        // Return the final gradient
        gradient.set_gradient_input_batch(combined_gradient_input_batch);

        gradient
    }

    pub fn split_gradient_into_heads(&self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Vec<Complex<f64>>>>> {
        let batch_size = previous_gradient_batch.len();
        let seq_len = previous_gradient_batch[0].len();
        let dim = previous_gradient_batch[0][0].len();
        let num_heads = self.attention_heads.len();
        let head_dim = dim / num_heads;

        assert!(dim % num_heads == 0, "dim={} must be divisible by num_heads={}", dim, num_heads);

        // Initialize a vector to store gradients for each attention head
        let mut grad_heads = vec![vec![vec![vec![Complex::new(0.0, 0.0); head_dim]; seq_len]; batch_size]; num_heads];

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
