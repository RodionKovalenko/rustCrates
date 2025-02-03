use super::masked_attention_head::MaskedAttentionHead;
use crate::neural_networks::network_components::{
    add_rms_norm_layer::RMSNormLayer,
    layer::{LayerEnum, LayerType},
};
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer {
    pub attention_heads: Vec<MaskedAttentionHead>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub dense_layers: Vec<LayerEnum>,
}

impl SelfAttentionLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(num_heads: usize, rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<MaskedAttentionHead> = vec![];
        let mut layers: Vec<LayerEnum> = vec![];
        let head_cols = cols / num_heads; // Columns per attention head

        for _i in 0..num_heads {
            let attention_head = MaskedAttentionHead::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer);
            attention_heads.push(attention_head);
        }

        let epsilon: f64 = 0.000001;
        let rms_norm_layer = RMSNormLayer::new(cols, epsilon, learning_rate);

        layers.push(LayerEnum::RMSNorm(Box::new(rms_norm_layer)));

        Self {
            attention_heads,
            activated_output: vec![],
            dense_layers: layers,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl SelfAttentionLayer {
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, padding_mask_batch: &Vec<Vec<u32>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = input_batch
            .par_iter()
            .zip(padding_mask_batch)
            .map(|(input, padding_mask)| {
                let mut output: Vec<Vec<Complex<f64>>> = vec![];

                // Apply the attention mechanism for each head
                let mut attention_head_outputs: Vec<Vec<Vec<Complex<f64>>>> = vec![];

                //println!("padding mask batch: {:?}", &padding_mask_batch);
                for attention_head in &self.attention_heads {
                    let attention_output = attention_head.forward(input, &padding_mask); // Get output for each attention head
                    attention_head_outputs.push(attention_output);
                }

                // Combine the outputs of the attention heads (e.g., concatenating horizontally)
                for i in 0..input.len() {
                    let mut combined_output: Vec<Complex<f64>> = Vec::new();
                    for head_output in &attention_head_outputs {
                        combined_output.extend_from_slice(&head_output[i]);
                    }
                    output.push(combined_output);
                }

                output // Return the output for the current input
            })
            .collect();

        // println!("output batch size: {} {} {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());
        // println!("input batch size: {} {} {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());

        // Process the dense layers
        for layer in self.dense_layers.iter_mut() {
            match layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let rms_norm_layer = Some(rms_norm_layer).unwrap();

                    // Forward pass for the dense layer
                    output_batch = rms_norm_layer.forward(&output_batch, input_batch);
                }
                _ => {}
            }
        }

        output_batch
    }

    pub fn backward(&self, gradients: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let mut backpropagated_gradients = gradients.clone();

        // Backpropagate gradients through each attention head
        for attention_head in &self.attention_heads {
            backpropagated_gradients = attention_head.backward(&backpropagated_gradients);
            // Call backward on each head
        }

        backpropagated_gradients
    }
}
