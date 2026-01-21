use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
    network_types::neural_network_generic::OperationMode,
    utils::{
        activation::softmax_backward_real_with_gradient_rm,
        dtype::{Real, C},
        matrix::RowMajorMatrix,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxLayerRm {
    pub learning_rate: f64,
    pub operation_mode: OperationMode,

    #[serde(skip)]
    pub softmax_output_batch: Option<Vec<Vec<Vec<Real>>>>,
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

    pub fn forward(&mut self, layer_input: &LayerInput, padding_mask_option: Option<Vec<Vec<u32>>>, target_token_ids: Option<Vec<Vec<u32>>>) -> Vec<Vec<Vec<Real>>> {
        self.time_step = layer_input.get_time_step();
        self.batch_size = layer_input.get_batch_size();

        let input_batch_rm_ref = layer_input
            .get_input_batch_rm_ref()
            .expect("SoftmaxLayerRm::forward expects input_batch_rm");

        if input_batch_rm_ref.is_empty() {
            self.padding_mask_batch = Some(vec![]);
            self.softmax_output_batch = Some(vec![]);
            self.cross_entropy_loss_batch = Some(vec![]);

            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_total_valid_tokens(0);
            self.gradient = Some(gradient);
            return vec![];
        }

        let batch_size = input_batch_rm_ref.len();
        let seq_len = input_batch_rm_ref[0].rows;

        let padding_mask_batch = padding_mask_option.unwrap_or_else(|| vec![vec![1; seq_len]; batch_size]);
        let target_token_batch_ids = target_token_ids.unwrap_or(Vec::new());

        let output_indices_batch = layer_input.get_output_indices();

        let mut total_valid_tokens: usize = 0;

        let (layer_output_batch, losses, input_gradient_batch_rm): (Vec<Vec<Vec<Real>>>, Vec<Vec<Vec<C>>>, Vec<RowMajorMatrix<C>>) = match self.operation_mode {
            OperationMode::PRODUCTION => {
                // Keep behavior consistent with Vec path (softmax only last row). We return an empty batch here
                // because transformer inference path already bypasses this function.
                (Vec::new(), Vec::new(), Vec::new())
            }
            OperationMode::TRAINING => {
                total_valid_tokens = padding_mask_batch
                    .iter()
                    .zip(target_token_batch_ids.iter())
                    .map(|(mask, targets)| {
                        let target_len = targets.len();
                        let valid_seq_len = mask.iter().filter(|&&m| m != 0).count();
                        let offset = valid_seq_len.saturating_sub(target_len);

                        targets
                            .iter()
                            .enumerate()
                            .filter(|(i, &target_id)| target_id != 1 && mask[offset + i] != 0)
                            .count()
                    })
                    .sum();

                let per_batch: Vec<(Vec<Vec<C>>, RowMajorMatrix<C>)> = (0..batch_size)
                    .into_par_iter()
                    .map(|batch_ind| {
                        let output_indices = if !output_indices_batch.is_empty() {
                            &output_indices_batch[batch_ind]
                        } else {
                            &vec![]
                        };

                        softmax_backward_real_with_gradient_rm(
                            &input_batch_rm_ref[batch_ind],
                            &target_token_batch_ids[batch_ind],
                            &padding_mask_batch[batch_ind],
                            total_valid_tokens,
                            output_indices,
                        )
                    })
                    .collect();

                let mut losses_batch: Vec<Vec<Vec<C>>> = Vec::with_capacity(batch_size);
                let mut grads_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_size);
                for (l, g) in per_batch {
                    losses_batch.push(l);
                    grads_batch_rm.push(g);
                }

                (Vec::new(), losses_batch, grads_batch_rm)
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

    pub fn backward(&mut self, _target_token_ids: &Vec<Vec<u32>>) -> Gradient {
        self.gradient.as_ref().unwrap().clone()
    }
}
