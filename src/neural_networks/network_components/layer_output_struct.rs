use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOutput {
    output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    output_batch_f64: Option<Vec<Vec<Vec<f64>>>>,
    output_record: Option<Vec<Vec<Complex<f64>>>>,
    l2_regularization: Option<Vec<Vec<Complex<f64>>>>,
    input_gradient_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    padding_mask_batch: Option<Vec<Vec<u32>>>,
}

impl LayerOutput {
    pub fn new_default() -> Self {
        LayerOutput {
            output_batch: None,
            output_batch_f64: None,
            output_record: None,
            l2_regularization: None,
            input_gradient_batch: None,
            padding_mask_batch: None,
        }
    }

    pub fn set_output_batch(&mut self, output_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.output_batch = Some(output_batch);
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

    pub fn get_output_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.output_batch.clone().unwrap_or_else(|| vec![])
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
}
