use super::attention_head::AttentionHead;
use crate::neural_networks::network_components::{
    add_rms_norm_layer::RMSNormLayer,
    layer::{BaseLayer, LayerEnum, LayerType},
};
use num::Complex;
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer {
    pub attention_heads: Vec<AttentionHead>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub dense_layers: Vec<LayerEnum>,
}

impl SelfAttentionLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(num_heads: usize, rows: usize, cols: usize, learning_rate: f64) -> Self {
        let mut attention_heads: Vec<AttentionHead> = vec![];
        let mut layers: Vec<LayerEnum> = vec![];
        let head_cols = cols / num_heads; // Columns per attention head

        for _i in 0..num_heads {
            let attention_head = AttentionHead::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer);
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
impl BaseLayer for SelfAttentionLayer {
    fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let mut output: Vec<Vec<Complex<f64>>> = vec![];

        // Apply the attention mechanism for each head
        let mut attention_head_outputs: Vec<Vec<Vec<Complex<f64>>>> = vec![];

        for attention_head in &self.attention_heads {
            let attention_output = attention_head.forward(input); // Get output for each attention head
            attention_head_outputs.push(attention_output);
        }

        // Combine the outputs of the attention heads (you could concatenate or sum the outputs)
        // For simplicity, let's concatenate the outputs horizontally (along columns)
        for i in 0..input.len() {
            let mut combined_output: Vec<Complex<f64>> = Vec::new();
            for head_output in &attention_head_outputs {
                combined_output.extend_from_slice(&head_output[i]);
            }
            output.push(combined_output);
        }

        for layer in self.dense_layers.iter() {
            match layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let rms_norm_layer = Some(rms_norm_layer).unwrap();
                    let previous_output = &output;
                    println!("Previous output: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                    // Forward pass for the dense layer (make sure dense accepts Vec<Vec<Complex<f64>>>)
                    let output_rms = rms_norm_layer.forward(&previous_output, &input);

                    println!("RMS output: {:?}, {:?}", &output_rms.len(), &output_rms[0].len());

                    //println!("RMS output: {:?}", &output_rms);
                    output = output_rms;
                }
                _ => {}
            }
        }

        println!("output of self attention layer: {:?} x {:?}", output.len(), output[0].len());
        output
    }

    fn backward(&self, gradients: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let mut backpropagated_gradients = gradients.clone();

        // Backpropagate gradients through each attention head
        for attention_head in &self.attention_heads {
            backpropagated_gradients = attention_head.backward(&backpropagated_gradients);
            // Call backward on each head
        }

        backpropagated_gradients
    }
}
