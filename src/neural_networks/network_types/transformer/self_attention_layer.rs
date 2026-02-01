use super::masked_attention_head::MaskedAttentionHead;
use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        add_rms_norm_layer::RMSNormLayer,
        layer::{LayerEnum, LayerType},
        norm_layer::NormalNormLayer,
    },
    network_types::transformer::transformer_updater::calculate_alpha,
    utils::matrix::{RowMajorMatrix, add_matrix_3d_in_place, scale_matrix_3d_by_scalar_in_place},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::dtype::{r, Real, C, ZERO};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer {
    pub attention_heads: Vec<MaskedAttentionHead>,
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
            alpha,
            beta,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl SelfAttentionLayer {
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let mut batch_output = layer_input.get_input_batch();
        let input_batch = layer_input.get_input_batch();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

        let mut layer_input = layer_input.clone();

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

        // Residual connection
        scale_matrix_3d_by_scalar_in_place(&mut batch_output, self.beta);
        add_matrix_3d_in_place(&mut batch_output, &input_batch);

        self.input_batch = Some(input_batch.clone());
        self.time_step = layer_input.get_time_step();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output.clone());
        layer_output.set_padding_mask_batch(padding_mask_batch.clone());
        self.output_batch = Some(batch_output.clone());

        layer_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        let mut gradient_input_batch: Vec<Vec<Vec<C>>> = previous_gradient_batch.clone();
        scale_matrix_3d_by_scalar_in_place(&mut gradient_input_batch, self.beta);

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
        
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let norm_gradient = rms_norm_layer.backward(&combined_gradient_input_batch);
                    combined_gradient_input_batch = norm_gradient.get_gradient_input_batch();
                }
                LayerEnum::Norm(norm_layer) => {
                    gradient.set_gradient_input_batch(combined_gradient_input_batch);
                    let norm_gradient = norm_layer.backward(&gradient);
                    combined_gradient_input_batch = norm_gradient.get_gradient_input_batch();
                }
                _ => {}
            }
        }

        add_matrix_3d_in_place(&mut combined_gradient_input_batch, &previous_gradient_batch);

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
