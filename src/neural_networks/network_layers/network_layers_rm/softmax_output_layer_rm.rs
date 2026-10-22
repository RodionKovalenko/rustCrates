use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::default_layer::LayerInterface,
    network_layers::softmax_output_layer::count_valid_target_tokens,
    network_types::neural_network_generic::OperationMode,
    utils::{
        activation_rm::{softmax_backward_real_with_gradient_rm, softmax_last_row_rm},
        dtype::C,
        matrix::RowMajorMatrix,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxLayerRm {
    pub learning_rate: f64,
    pub operation_mode: OperationMode,

    #[serde(skip)]
    pub softmax_output_batch: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub cross_entropy_loss_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl SoftmaxLayerRm {
    pub fn new(learning_rate: f64, operation_mode: OperationMode, _feature_dim: usize) -> Self {
        Self {
            learning_rate,
            operation_mode,
            softmax_output_batch: None,
            cross_entropy_loss_batch: None,
            gradient: None,
            padding_mask_batch: None,
            time_step: 0,
            batch_size: 1,
        }
    }

    /// RM counterpart of `SoftmaxLayer::forward_inner`. TRAINING computes the CE loss and the logit
    /// gradients, PRODUCTION the softmax of the last row.
    pub fn forward_inner(&mut self, layer_input: &LayerInput, padding_mask_option: Option<Vec<Vec<u32>>>, target_token_ids: Option<Vec<Vec<u32>>>) -> Vec<RowMajorMatrix<C>> {
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();

        let input_batch_rm = layer_input.get_input_batch_rm();
        let batch_size = input_batch_rm.len();
        let seq_len = input_batch_rm.first().map(|m| m.rows).unwrap_or(0);

        let padding_mask_batch = padding_mask_option.unwrap_or_else(|| vec![vec![1; seq_len]; batch_size]);
        let target_token_batch_ids = target_token_ids.unwrap_or_default();
        let output_indices_batch = layer_input.get_output_indices();

        let mut total_valid_tokens: usize = 0;

        let (layer_output_batch, losses, input_gradient_batch_rm): (Vec<RowMajorMatrix<C>>, Vec<Vec<Vec<C>>>, Vec<RowMajorMatrix<C>>) = match self.operation_mode {
            OperationMode::PRODUCTION => (input_batch_rm.par_iter().map(softmax_last_row_rm).collect(), Vec::new(), Vec::new()),
            OperationMode::TRAINING => {
                total_valid_tokens = count_valid_target_tokens(&padding_mask_batch, &target_token_batch_ids);

                let (losses, gradients): (Vec<Vec<Vec<C>>>, Vec<RowMajorMatrix<C>>) = (0..batch_size)
                    .into_par_iter()
                    .map(|batch_ind| {
                        let output_indices: &[Vec<usize>] = if output_indices_batch.is_empty() { &[] } else { &output_indices_batch[batch_ind] };
                        softmax_backward_real_with_gradient_rm(
                            &input_batch_rm[batch_ind],
                            &target_token_batch_ids[batch_ind],
                            &padding_mask_batch[batch_ind],
                            total_valid_tokens,
                            output_indices,
                        )
                    })
                    .unzip();

                (Vec::new(), losses, gradients)
            }
        };

        self.padding_mask_batch = Some(padding_mask_batch);
        self.softmax_output_batch = Some(layer_output_batch.clone());
        self.cross_entropy_loss_batch = Some(losses);

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);
        gradient.set_total_valid_tokens(total_valid_tokens);
        self.gradient = Some(gradient);

        layer_output_batch
    }

    pub fn update_parameters(&mut self) {}

    pub fn backward_inner(&mut self, _target_token_ids: &Vec<Vec<u32>>) -> Gradient {
        self.gradient.as_ref().unwrap().clone()
    }

    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let padding_mask = layer_input.get_padding_mask_batch();
        let target_ids = layer_input.get_target_batch_ids();
        let softmax_output = SoftmaxLayerRm::forward_inner(
            self,
            layer_input,
            if padding_mask.is_empty() { None } else { Some(padding_mask) },
            if target_ids.is_empty() { None } else { Some(target_ids) },
        );
        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(softmax_output);
        output
    }

    pub fn backward(&mut self, _previous_gradient: &Gradient) -> Gradient {
        self.gradient.as_ref().expect("SoftmaxLayerRm: gradient not set; call forward first").clone()
    }
}

impl LayerInterface for SoftmaxLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        SoftmaxLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        SoftmaxLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        SoftmaxLayerRm::update_parameters(self)
    }
}
