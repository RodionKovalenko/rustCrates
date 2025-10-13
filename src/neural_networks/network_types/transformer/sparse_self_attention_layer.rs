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
    network_types::wavelet_discrete_layer::DiscreteWaveletLayer,
    utils::matrix::add_matrix_3d,
};
use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSelfAttentionLayer {
    pub attention_heads: Vec<MaskedAttentionHead>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub norm_layer: Option<LayerEnum>,
    pub discrete_wavelet_layer: Option<DiscreteWaveletLayer>,
    pub input_partition_order: usize,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
}

impl SparseSelfAttentionLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(num_heads: usize, rows: usize, cols: usize, partition_shift: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<MaskedAttentionHead> = vec![];
        let head_cols = cols;

        for _i in 0..num_heads {
            let attention_head = MaskedAttentionHead::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer, learning_rate);
            attention_heads.push(attention_head);
        }

        let epsilon: f64 = 0.000000000001;
        let _norm_layer_rms = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(cols, epsilon, learning_rate))));
        let _norm_layer = Some(LayerEnum::Norm(Box::new(NormalNormLayer::new(cols, epsilon, learning_rate))));
        let _dwt_layer = Some(DiscreteWaveletLayer::new());

        Self {
            attention_heads,
            activated_output: vec![],
            norm_layer: _norm_layer,
            discrete_wavelet_layer: None,
            input_partition_order: partition_shift,
            input_batch: None,
            output_batch: None,
            gradient: None,
            time_step: 0,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl SparseSelfAttentionLayer {
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
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

        // Apply the attention mechanism for each head
        let mut layer_input = layer_input.clone();
        layer_input.set_input_batch(batch_output.clone());
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

        let input_splitted_heads = self.split_input_into_partitions(&batch_output);
        let padding_mask_splitted_heads = self.split_padding_mask_into_partitions(&padding_mask_batch);

        //println!("padding mask batch: {:?}", &padding_mask_batch);
        let attention_head_outputs: Vec<_> = self
            .attention_heads
            .par_iter_mut() // Use rayon's parallel iterator
            .enumerate()
            .map(|(head_index, attention_head)| {
                let mut layer_input = layer_input.clone();
                layer_input.set_input_batch(input_splitted_heads[head_index].clone());
                layer_input.set_padding_mask_batch(padding_mask_splitted_heads[head_index].clone());
                let attention_output = attention_head.forward(&layer_input);
                attention_output.get_output_batch()
            })
            .collect(); // Collect the results into a vector

        // println!("attention head outputs: {:?}", &attention_head_outputs);
        let batch_size = input_batch.len();
        let sequence_size = input_batch[0].len();
        let feature_dim = input_batch[0][0].len();

        batch_output = vec![vec![vec![Complex::new(0.0, 0.0); feature_dim]; sequence_size]; batch_size];

        if attention_head_outputs.len() < self.attention_heads.len() {
            println!("Attention head output size is smaller than number of heads, something went wrong!");
            // Combine the outputs of the attention heads (e.g., concatenating horizontally)
            for head_output in attention_head_outputs {
                for b in 0..batch_size {
                    for row in 0..head_output[b].len() {
                        for val in 0..head_output[b][row].len() {
                            batch_output[b][row][val] += head_output[b][row][val];
                        }
                    }
                }
            }
        } else {
            // Combine the outputs of the attention heads (e.g., concatenating horizontally)

            batch_output = Vec::new();

            // Take all rows from each head and glue them
            for head_output in attention_head_outputs {
                for (head_batch_index, head_batch) in head_output.iter().enumerate() {
                    for row in head_batch {
                        if batch_output.len() <= head_batch_index {
                            batch_output.push(vec![]);
                        }
                        batch_output[head_batch_index].push(row.clone());
                    }
                }
            }

            // for b in 0..batch_size {
            //     println!("dimension after concat: {}, {}, {}", batch_output.len(), &batch_output[b].len(), &batch_output[b][0].len());
            //     println!("input dimension: {}, {}, {}", input_batch.len(), &input_batch[b].len(), &input_batch[b][0].len());
            // }
        }

        layer_input.set_input_batch(batch_output.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        // println!("batch output after self-attention: {}, {}, {}", &batch_output.len(), &batch_output[0].len(), &batch_output[0][0].len());

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

        batch_output = add_matrix_3d(&batch_output, &input_batch);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output.clone());
        layer_output.set_padding_mask_batch(padding_mask_batch.clone());

        self.output_batch = Some(batch_output.clone());
        self.input_batch = Some(input_batch.clone());
        self.time_step = layer_input.get_time_step();

        layer_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient_batch.clone();

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

        let previous_gradient_head_splitted: Vec<Vec<Vec<Vec<Complex<f64>>>>> = self.split_input_into_partitions(&gradient_input_batch);
        let mut gradient_input_batches: Vec<Vec<Vec<Vec<Complex<f64>>>>> = Vec::new();

        // Backpropagate gradients through each attention head
        for (head_ind, attention_head) in self.attention_heads.iter_mut().enumerate() {
            let previous_head_gradient_batch = previous_gradient_head_splitted[head_ind].clone();
            gradient = attention_head.backward(&previous_head_gradient_batch);
            gradient_input_batches.push(gradient.get_gradient_input_batch());
            // println!("gradient input head {:?}", &gradient.get_gradient_input_batch());
        }

        let batch_size = gradient_input_batch.len();
        let seq_len = gradient_input_batch[0].len();
        let feature_dim = gradient_input_batch[0][0].len();

        let mut combined_gradient_input_batch = vec![vec![vec![Complex::new(0.0, 0.0); feature_dim]; seq_len]; batch_size];

        let num_input_per_partition = seq_len / num_heads;
        let rest_partitions = seq_len % num_heads;
        let input_shift = self.input_partition_order * num_input_per_partition;

        let mut start_idx = input_shift;
        let num_heads = self.attention_heads.len();

        for head_idx in 0..num_heads {
            for b in 0..batch_size {
                let grad_batch = &gradient_input_batches[head_idx][b];
                let seq_len_h = grad_batch.len();
                for i in 0..seq_len_h {
                    let global_idx = (start_idx + i) % seq_len;
                    combined_gradient_input_batch[b][global_idx] = grad_batch[i].clone();
                }
            }
            // Move start_idx by current partition length; partition gets extra 1 if head_idx < rest_partitions
            start_idx += num_input_per_partition + if head_idx < rest_partitions { 1 } else { 0 };
        }

        gradient.set_gradient_input_batch(combined_gradient_input_batch.clone());

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
                }
                LayerEnum::Norm(norm_layer) => {
                    let norm_gradient = norm_layer.backward(&gradient);
                    gradient_input_batch = norm_gradient.get_gradient_input_batch();
                }
                _ => {}
            }
        }

        combined_gradient_input_batch = add_matrix_3d(&previous_gradient_batch, &gradient_input_batch);

        // Return the final gradient
        gradient.set_gradient_input_batch(combined_gradient_input_batch);

        gradient
    }

    pub fn split_input_into_partitions(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Vec<Complex<f64>>>>> {
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        let num_partitions = self.attention_heads.len();
        // seq length can have rest by division with num_partitions
        let mut num_input_per_partition = seq_len / num_partitions;
        let mut rest_partitions = seq_len % num_partitions;

        // the last partition will have more input then the previous one is there is rest
        let mut partitions: Vec<Vec<Vec<Vec<Complex<f64>>>>> = vec![vec![vec![]; batch_size]; num_partitions];
        let mut input_shift: usize = self.input_partition_order * num_input_per_partition;

        if input_shift > 5 {
            input_shift -= 5;
        }

        let mut start_idx: usize = input_shift;
        let mut end_idx: usize = start_idx;

        if seq_len <= num_partitions {
            num_input_per_partition = seq_len;
            rest_partitions = 0;
        }

        for part in 0..num_partitions {
            for batch_ind in 0..batch_size {
                end_idx = start_idx + num_input_per_partition + if part < rest_partitions { 1 } else { 0 };

                for seq_in in start_idx..end_idx {
                    partitions[part][batch_ind].push(input_batch[batch_ind][seq_in % seq_len].clone());
                }
            }
            start_idx = end_idx;
        }

        // println!("partitions input: {}, {}, {}, {}", &partitions.len(), &partitions[0].len(), &partitions[0][0].len(), &partitions[0][0][0].len());

        partitions
    }

    pub fn split_padding_mask_into_partitions(&mut self, padding_mask_batch: &Vec<Vec<u32>>) -> Vec<Vec<Vec<u32>>> {
        let batch_size = padding_mask_batch.len();
        let seq_len = padding_mask_batch[0].len();

        let num_partitions = self.attention_heads.len();
        // seq length can have rest by division with num_partitions
        let mut num_input_per_partition = seq_len / num_partitions;
        let mut rest_partitions = seq_len % num_partitions;

        // the last partition will have more input then the previous one is there is rest
        let mut partitions: Vec<Vec<Vec<u32>>> = vec![vec![vec![]; batch_size]; num_partitions];
        let mut input_shift: usize = (self.input_partition_order * num_input_per_partition) % seq_len;

        if input_shift > 5 {
            input_shift -= 5;
        }

        let mut start_idx: usize = input_shift;
        let mut end_idx: usize = start_idx;

        if seq_len <= num_partitions {
            num_input_per_partition = seq_len;
            rest_partitions = 0;
        }

        for part in 0..num_partitions {
            for batch_ind in 0..batch_size {
                end_idx = start_idx + num_input_per_partition + if part < rest_partitions { 1 } else { 0 };

                for seq_in in start_idx..end_idx {
                    partitions[part][batch_ind].push(padding_mask_batch[batch_ind][seq_in % seq_len].clone());
                }
            }
            start_idx = end_idx;
        }

        // println!("partitions padding mask: {}, {}, {}", &partitions.len(), &partitions[0].len(), &partitions[0][0].len());

        partitions
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
