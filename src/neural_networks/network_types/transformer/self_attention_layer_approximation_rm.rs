use crate::neural_networks::{
    network_components::{
        gradient_struct::Gradient,
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
    },
    network_layers::{
        layer::{LayerEnum, LayerType},
        network_layers_rm::{norm_layer_rm::NormalNormLayerRm, rms_norm_layer_rm::RMSNormLayerRm},
    },
    network_types::transformer::masked_attention_head_approximation_rm::MaskedAttentionHeadApproximationRm,
    utils::{
        matrix_rm::RowMajorMatrix,
        matrix_rm::{add_assign_rm, concat_heads_rm, split_heads_rm, sum_heads_rm},
    },
};

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::dtype::C;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayerApproximationRm {
    pub attention_heads: Vec<MaskedAttentionHeadApproximationRm>,
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
        let mut attention_heads: Vec<MaskedAttentionHeadApproximationRm> = vec![];
        let head_cols = cols / num_heads;

        for _ in 0..num_heads {
            let attention_head =
                MaskedAttentionHeadApproximationRm::create_default_attention_layer(rows, head_cols, LayerType::AttentionLayer, learning_rate);
            attention_heads.push(attention_head);
        }

        let epsilon: f64 = 0.000000000001;
        let norm_layer = Some(LayerEnum::NormRm(Box::new(NormalNormLayerRm::new(cols, epsilon, learning_rate))));
        let _norm_layer_rms = Some(LayerEnum::RMSNormRm(Box::new(RMSNormLayerRm::new(cols, epsilon, learning_rate))));

        Self {
            attention_heads,
            activated_output: vec![],
            norm_layer,
            input_batch_rm: None,
            output_batch_rm: None,
            time_step: 0,
        }
    }

    /// RM counterpart of `SelfAttentionLayerApproximation::forward`: heads -> concat -> norm.
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch_rm = layer_input.get_input_batch_rm();
        assert!(!input_batch_rm.is_empty(), "SelfAttentionLayerApproximationRm::forward got empty RM batch");

        let padding_mask_batch = layer_input.get_padding_mask_batch();
        self.time_step = layer_input.get_time_step();

        let mut li = layer_input.clone();
        li.clear_input_batch();
        li.set_input_batch_rm(input_batch_rm.clone());
        li.set_padding_mask_batch(padding_mask_batch.clone());

        let head_outputs: Vec<Vec<RowMajorMatrix<C>>> = self.attention_heads.par_iter_mut().map(|head| head.forward(&li).get_output_batch_rm()).collect();
        let mut output = concat_heads_rm(&head_outputs);

        li.set_input_batch_rm(output.clone());
        li.set_input_batch_before_rm(input_batch_rm.clone());

        match self.norm_layer.as_mut() {
            Some(LayerEnum::RMSNormRm(rms_norm_layer)) => output = rms_norm_layer.forward(&li).get_output_batch_rm(),
            Some(LayerEnum::NormRm(norm_layer)) => output = norm_layer.forward(&li).take_output_batch_rm().expect("NormRm did not return RM output"),
            Some(LayerEnum::Norm(_)) => panic!("SelfAttentionLayerApproximationRm requires NormRm (Vec Norm not allowed)"),
            _ => {}
        }

        self.input_batch_rm = Some(input_batch_rm);
        self.output_batch_rm = Some(output.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output);
        layer_output
    }

    /// RM counterpart of `SelfAttentionLayerApproximation::backward`.
    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let mut gradient_input_rm: Vec<RowMajorMatrix<C>> = previous_gradient_batch_rm.to_vec();
        let mut output_gradient_norm_rm: Option<Vec<RowMajorMatrix<C>>> = None;

        match self.norm_layer.as_mut() {
            Some(LayerEnum::RMSNormRm(rms_norm_layer)) => {
                gradient_input_rm = rms_norm_layer.backward_rm(&gradient_input_rm).get_gradient_input_batch_rm();
                output_gradient_norm_rm = Some(gradient_input_rm.clone());
            }
            Some(LayerEnum::NormRm(norm_layer)) => {
                let mut g = Gradient::new_default();
                g.set_gradient_input_batch_rm(gradient_input_rm);
                gradient_input_rm = norm_layer.backward(&g).get_gradient_input_batch_rm();
                output_gradient_norm_rm = Some(gradient_input_rm.clone());
            }
            Some(LayerEnum::Norm(_)) => panic!("SelfAttentionLayerApproximationRm backward_rm requires NormRm"),
            _ => {}
        }

        // Every head sees the full input, so the input gradients of the heads are summed.
        let head_gradients = split_heads_rm(&gradient_input_rm, self.attention_heads.len());
        let head_input_gradients: Vec<Vec<RowMajorMatrix<C>>> = self
            .attention_heads
            .iter_mut()
            .zip(head_gradients.iter())
            .map(|(head, gradient)| head.backward_rm(gradient).get_gradient_input_batch_rm())
            .collect();
        let mut combined = sum_heads_rm(head_input_gradients);

        // Residual behavior (matches the Vec layer)
        if let Some(output_gradient_norm_rm) = output_gradient_norm_rm {
            for (c, n) in combined.iter_mut().zip(output_gradient_norm_rm.iter()) {
                add_assign_rm(c, n);
            }
        }

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(combined);
        gradient
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        self.backward_rm(&previous_gradient.get_gradient_input_batch_rm_cow())
    }

    pub fn update_parameters(&mut self) {
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNormRm(rms_norm_layer) => {
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
