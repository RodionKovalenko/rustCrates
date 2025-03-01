use super::masked_attention_head::MaskedAttentionHead;
use crate::neural_networks::{
    network_components::{
        add_rms_norm_layer::RMSNormLayer,
        gradient_struct::Gradient,
        layer::{LayerEnum, LayerType},
    },
    utils::matrix::check_nan_or_inf_3d,
};
use num::Complex;
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer {
    pub attention_heads: Vec<MaskedAttentionHead>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub dense_layers: Vec<LayerEnum>,
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
}

impl SelfAttentionLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(num_heads: usize, rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<MaskedAttentionHead> = vec![];
        let mut layers: Vec<LayerEnum> = vec![];
        let head_cols = cols / num_heads; // Columns per attention head

        for _i in 0..num_heads {
            let attention_head = MaskedAttentionHead::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer, learning_rate);
            attention_heads.push(attention_head);
        }

        let epsilon: f64 = 0.000001;
        let rms_norm_layer = RMSNormLayer::new(cols, epsilon, learning_rate);

        layers.push(LayerEnum::RMSNorm(Box::new(rms_norm_layer)));

        Self {
            attention_heads,
            activated_output: vec![],
            dense_layers: layers,
            input_batch: None,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl SelfAttentionLayer {
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, padding_mask_batch: &Vec<Vec<u32>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch = Some(input_batch.clone());
        let batch_size = input_batch.len();
        let sequence_size = input_batch[0].len();

        let mut batch_output: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![]; sequence_size]; batch_size];
        // Apply the attention mechanism for each head
        let mut attention_head_outputs: Vec<Vec<Vec<Vec<Complex<f64>>>>> = vec![];

        //println!("padding mask batch: {:?}", &padding_mask_batch);
        for attention_head in self.attention_heads.iter_mut() {
            let attention_output = attention_head.forward(input_batch, &padding_mask_batch);
            attention_head_outputs.push(attention_output);
        }

        // println!("input batch outputs: {:?}", &input_batch);
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

        // println!("output batch size: {} {} {}", output_batch.len(), output_batch[0].len(), output_batch[0][0].len());

        // Process the dense layers
        // for layer in self.dense_layers.iter_mut() {
        //     match layer {
        //         LayerEnum::RMSNorm(rms_norm_layer) => {
        //             batch_output = rms_norm_layer.forward(&batch_output, input_batch);

        //             check_nan_or_inf_3d(&mut batch_output, "check rms norm in self attention layer");
        //         }
        //         _ => {}
        //     }
        // }

        batch_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        // let input_batch = self.input_batch.as_ref().expect("Input batch not found in self-attention layer backward");
        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient_batch.clone();

        let mut gradient: Gradient = Gradient::new_default();

        // Process the dense layers
        // for layer in self.dense_layers.iter_mut() {
        //     match layer {
        //         LayerEnum::RMSNorm(rms_norm_layer) => {
        //             check_nan_or_inf_3d(&mut gradient_input_batch, "gradient input batch has NaN values in selft attention layer backward rnorm");

        //             let gradient = rms_norm_layer.backward(&gradient_input_batch);
        //             gradient_input_batch = gradient.get_gradient_input_batch();
        //         }
        //         _ => {}
        //     }
        // }

        let previous_gradient_head_splitted = self.split_gradient_into_heads(&gradient_input_batch);
        let mut gradient_input_batches: Vec<Vec<Vec<Vec<Complex<f64>>>>> = Vec::new();

        // Backpropagate gradients through each attention head
        for (head_ind, attention_head) in self.attention_heads.iter_mut().enumerate() {
            let mut previous_head_gradient_batch = previous_gradient_head_splitted[head_ind].clone();

            check_nan_or_inf_3d(&mut previous_head_gradient_batch, "previous_head_gradient_batch NaN values in selft attention layer backward attention head");

            // println!("backward previous head gradient batch: {} {} {}", &previous_head_gradient_batch.len(), previous_head_gradient_batch[0].len(), previous_head_gradient_batch[0][0].len());
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

        grad_heads
    }

    pub fn update_parameters(&mut self) {
        // for layer in self.dense_layers.iter_mut() {
        //     match layer {
        //         LayerEnum::RMSNorm(rms_norm_layer) => {
        //             rms_norm_layer.update_parameters();
        //         }
        //         _ => {}
        //     }
        // }

        for attention_head in self.attention_heads.iter_mut() {
            attention_head.update_parameters();
        }
    }
}
