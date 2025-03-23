use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInput {
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    padding_mask_batch: Option<Vec<Vec<u32>>>,
    input_record: Option<Vec<Vec<Complex<f64>>>>,
}

impl LayerInput {
    pub fn new_default() -> Self {
        LayerInput {
            input_batch: None,
            padding_mask_batch: None,
            input_record: None,
        }
    }
    pub fn set_input_batch(&mut self, input_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.input_batch = Some(input_batch);
    }
    pub fn set_padding_mask_batch(&mut self, padding_mask_batch: Vec<Vec<u32>>) {
        self.padding_mask_batch = Some(padding_mask_batch);
    }
    pub fn set_input_record(&mut self, input_record: Vec<Vec<Complex<f64>>>) {
        self.input_record = Some(input_record);
    }

    pub fn get_padding_mask_batch(&self) -> Vec<Vec<u32>> {
        self.padding_mask_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_input_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_input_record(&self) -> Vec<Vec<Complex<f64>>> {
        self.input_record.clone().unwrap_or_else(|| vec![])
    }
}
