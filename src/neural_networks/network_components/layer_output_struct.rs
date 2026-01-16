use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_components::adaptive_pooling::adaptive_avg_pool1d_layer::CompressionMetadata;
use crate::neural_networks::utils::matrix::RowMajorMatrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOutput {
    output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    output_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    output_batch_f64: Option<Vec<Vec<Vec<f64>>>>,
    output_record: Option<Vec<Vec<Complex<f64>>>>,
    l2_regularization: Option<Vec<Vec<Complex<f64>>>>,
    input_gradient_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    padding_mask_batch: Option<Vec<Vec<u32>>>,
    pooling_metadata: Option<CompressionMetadata>,
    cross_entropy_loss_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    set_output_indices: Option<Vec<Vec<Vec<usize>>>>,
}

impl LayerOutput {
    pub fn new_default() -> Self {
        LayerOutput {
            output_batch: None,
            output_batch_rm: None,
            output_batch_f64: None,
            output_record: None,
            l2_regularization: None,
            input_gradient_batch: None,
            padding_mask_batch: None,
            pooling_metadata: None,
            cross_entropy_loss_batch: None,
            set_output_indices: None,
        }
    }

    pub fn set_output_batch(&mut self, output_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.output_batch = Some(output_batch);
    }

    pub fn set_output_batch_rm(&mut self, output_batch_rm: Vec<RowMajorMatrix<Complex<f64>>>) {
        self.output_batch_rm = Some(output_batch_rm);
    }
    pub fn set_output_batch_f64(&mut self, output_batch: Vec<Vec<Vec<f64>>>) {
        self.output_batch_f64 = Some(output_batch);
    }
    pub fn set_output_record(&mut self, output_record: Vec<Vec<Complex<f64>>>) {
        self.output_record = Some(output_record);
    }
    pub fn set_l2_regularization(&mut self, l2_regularization: Vec<Vec<Complex<f64>>>) {
        self.l2_regularization = Some(l2_regularization);
    }
    pub fn set_input_gradient_batch(&mut self, input_gradient_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.input_gradient_batch = Some(input_gradient_batch);
    }
    pub fn set_padding_mask_batch(&mut self, padding_mask_batch: Vec<Vec<u32>>) {
        self.padding_mask_batch = Some(padding_mask_batch);
    }

    pub fn set_cross_entropy_loss_batch(&mut self, cross_entropy_loss_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.cross_entropy_loss_batch = Some(cross_entropy_loss_batch);
    }

    pub fn set_output_indices(&mut self, output_indices: Vec<Vec<Vec<usize>>>) {
        self.set_output_indices = Some(output_indices);
    }
    pub fn get_output_indices(&self) -> Vec<Vec<Vec<usize>>> {
        self.set_output_indices.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_output_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        if let Some(output_batch) = &self.output_batch {
            return output_batch.clone();
        }

        if let Some(output_batch_rm) = &self.output_batch_rm {
            return output_batch_rm.iter().map(|m| m.to_rows()).collect();
        }

        vec![]
    }

    pub fn take_output_batch(&mut self) -> Option<Vec<Vec<Vec<Complex<f64>>>>> {
        self.output_batch.take()
    }

    pub fn get_output_batch_rm(&self) -> Vec<RowMajorMatrix<Complex<f64>>> {
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

    pub fn take_output_batch_rm(&mut self) -> Option<Vec<RowMajorMatrix<Complex<f64>>>> {
        self.output_batch_rm.take()
    }

    pub fn get_output_batch_rm_ref(&self) -> Option<&[RowMajorMatrix<Complex<f64>>]> {
        self.output_batch_rm.as_deref()
    }
    pub fn get_output_batch_f64(&self) -> Vec<Vec<Vec<f64>>> {
        self.output_batch_f64.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_output_record(&self) -> Vec<Vec<Complex<f64>>> {
        self.output_record.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_l2_regularization(&self) -> Vec<Vec<Complex<f64>>> {
        self.l2_regularization.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_input_gradient_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
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
    pub fn get_cross_entropy_loss_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.cross_entropy_loss_batch.clone().unwrap_or_else(|| vec![])
    }

    pub fn take_cross_entropy_loss_batch(&mut self) -> Option<Vec<Vec<Vec<Complex<f64>>>>> {
        self.cross_entropy_loss_batch.take()
    }
}
