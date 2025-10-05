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
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer {
    pub attention_heads: Vec<MaskedAttentionHead>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub norm_layer: Option<LayerEnum>,
    pub discrete_wavelet_layer: Option<DiscreteWaveletLayer>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
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

        Self {
            attention_heads,
            activated_output: vec![],
            norm_layer: _norm_layer,
            discrete_wavelet_layer: None,
            input_batch: None,
            output_batch: None,
            gradient: None,
            time_step: 0,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl SelfAttentionLayer {
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
        let mut batch_output: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![]; sequence_size]; batch_size];
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
                let mut combined_output: Vec<Complex<f64>> = Vec::new();
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
        batch_output = add_matrix_3d(&batch_output, &input_batch);

        self.input_batch = Some(input_batch.clone());
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output.clone());
        layer_output.set_padding_mask_batch(padding_mask_batch.clone());
        self.output_batch = Some(batch_output.clone());

        layer_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        // let input_batch = self.input_batch.as_ref().expect("Input batch not found in self-attention layer backward");
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

        combined_gradient_input_batch = add_matrix_3d(&previous_gradient_batch, &gradient_input_batch);

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
