use core::fmt::Debug;
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_layers::adaptive_pooling::adaptive_avg_pool1d_layer::CompressionMetadata;
use crate::neural_networks::utils::dtype::{C, Real};
use crate::neural_networks::utils::matrix::RowMajorMatrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOutput {
    output_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    output_batch_real: Option<Vec<Vec<Vec<Real>>>>,
    output_record: Option<Vec<Vec<C>>>,
    l2_regularization: Option<Vec<Vec<C>>>,
    input_gradient_batch: Option<Vec<Vec<Vec<C>>>>,
    padding_mask_batch: Option<Vec<Vec<u32>>>,
    pooling_metadata: Option<CompressionMetadata>,
    cross_entropy_loss_batch: Option<Vec<Vec<Vec<C>>>>,
    set_output_indices: Option<Vec<Vec<Vec<usize>>>>,
}

impl LayerOutput {
    pub fn new_default() -> Self {
        LayerOutput {
            output_batch: None,
            output_batch_rm: None,
            output_batch_real: None,
            output_record: None,
            l2_regularization: None,
            input_gradient_batch: None,
            padding_mask_batch: None,
            pooling_metadata: None,
            cross_entropy_loss_batch: None,
            set_output_indices: None,
        }
    }

    pub fn set_output_batch(&mut self, output_batch: Vec<Vec<Vec<C>>>) {
        self.output_batch = Some(output_batch);
    }

    pub fn set_output_batch_rm(&mut self, output_batch_rm: Vec<RowMajorMatrix<C>>) {
        self.output_batch_rm = Some(output_batch_rm);
    }
    pub fn set_output_batch_real(&mut self, output_batch: Vec<Vec<Vec<Real>>>) {
        self.output_batch_real = Some(output_batch);
    }
    pub fn set_output_record(&mut self, output_record: Vec<Vec<C>>) {
        self.output_record = Some(output_record);
    }
    pub fn set_l2_regularization(&mut self, l2_regularization: Vec<Vec<C>>) {
        self.l2_regularization = Some(l2_regularization);
    }
    pub fn set_input_gradient_batch(&mut self, input_gradient_batch: Vec<Vec<Vec<C>>>) {
        self.input_gradient_batch = Some(input_gradient_batch);
    }
    pub fn set_padding_mask_batch(&mut self, padding_mask_batch: Vec<Vec<u32>>) {
        self.padding_mask_batch = Some(padding_mask_batch);
    }

    pub fn set_cross_entropy_loss_batch(&mut self, cross_entropy_loss_batch: Vec<Vec<Vec<C>>>) {
        self.cross_entropy_loss_batch = Some(cross_entropy_loss_batch);
    }

    pub fn set_output_indices(&mut self, output_indices: Vec<Vec<Vec<usize>>>) {
        self.set_output_indices = Some(output_indices);
    }
    pub fn get_output_indices(&self) -> Vec<Vec<Vec<usize>>> {
        self.set_output_indices.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_output_batch(&self) -> Vec<Vec<Vec<C>>> {
        if let Some(output_batch) = &self.output_batch {
            return output_batch.clone();
        }

        if let Some(output_batch_rm) = &self.output_batch_rm {
            return output_batch_rm.iter().map(|m| m.to_rows()).collect();
        }

        vec![]
    }

    pub fn take_output_batch(&mut self) -> Option<Vec<Vec<Vec<C>>>> {
        self.output_batch.take()
    }

    pub fn get_output_batch_rm(&self) -> Vec<RowMajorMatrix<C>> {
        if let Some(output_batch_rm) = &self.output_batch_rm {
            return output_batch_rm.clone();
        }

        if let Some(output_batch) = &self.output_batch {
            if output_batch.is_empty() {
                return vec![];
            }

            let mut out = Vec::with_capacity(output_batch.len());
            for m in output_batch {
                match RowMajorMatrix::try_from_rows(m) {
                    Some(rm) => out.push(rm),
                    None => return vec![],
                }
            }

            return out;
        }

        vec![]
    }

    pub fn take_output_batch_rm(&mut self) -> Option<Vec<RowMajorMatrix<C>>> {
        self.output_batch_rm.take()
    }

    pub fn get_output_batch_rm_ref(&self) -> Option<&[RowMajorMatrix<C>]> {
        self.output_batch_rm.as_deref()
    }
    pub fn get_output_batch_real(&self) -> Vec<Vec<Vec<Real>>> {
        self.output_batch_real.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_output_record(&self) -> Vec<Vec<C>> {
        self.output_record.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_l2_regularization(&self) -> Vec<Vec<C>> {
        self.l2_regularization.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_input_gradient_batch(&self) -> Vec<Vec<Vec<C>>> {
        self.input_gradient_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_padding_mask_batch(&self) -> Vec<Vec<u32>> {
        self.padding_mask_batch.clone().unwrap_or_else(|| vec![])
    }

    pub fn take_padding_mask_batch(&mut self) -> Option<Vec<Vec<u32>>> {
        self.padding_mask_batch.take()
    }
    pub fn set_pooling_metadata(&mut self, pooling_metadata: CompressionMetadata) {
        self.pooling_metadata = Some(pooling_metadata);
    }
    pub fn get_pooling_metadata(&self) -> Option<CompressionMetadata> {
        self.pooling_metadata.clone()
    }
    pub fn get_cross_entropy_loss_batch(&self) -> Vec<Vec<Vec<C>>> {
        self.cross_entropy_loss_batch.clone().unwrap_or_else(|| vec![])
    }

    pub fn take_cross_entropy_loss_batch(&mut self) -> Option<Vec<Vec<Vec<C>>>> {
        self.cross_entropy_loss_batch.take()
    }
}
