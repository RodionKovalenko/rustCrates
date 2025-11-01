use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_components::adaptive_pooling::adaptive_avg_pool1d_layer::CompressionMetadata;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInput {
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    batch_ids: Option<Vec<Vec<u32>>>,
    target_batch_ids: Option<Vec<Vec<u32>>>,
    input_batch_before: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    previous_gradient_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    padding_mask_batch: Option<Vec<Vec<u32>>>,
    input_record: Option<Vec<Vec<Complex<f64>>>>,
    target_tokens_len: usize,
    batch_size: usize,
    time_step: usize,
    forward_only: bool,
    calculate_gradient: bool,
    record_ind: usize,
    pooling_metadata: Option<CompressionMetadata>,
    calculate_k_v_cache: bool,
}

impl LayerInput {
    pub fn new_default() -> Self {
        LayerInput {
            input_batch: None,
            batch_ids: None,
            target_batch_ids: None,
            input_batch_before: None,
            padding_mask_batch: None,
            input_record: None,
            previous_gradient_input_batch: None,
            time_step: 0,
            target_tokens_len: 0,
            forward_only: false,
            calculate_gradient: true,
            record_ind: 0,
            batch_size: 25,
            pooling_metadata: None,
            calculate_k_v_cache: false,
        }
    }
    pub fn set_input_batch(&mut self, input_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.input_batch = Some(input_batch);
    }
    pub fn set_forward_only(&mut self, forward_only: bool) {
        self.forward_only = forward_only;
    }
    pub fn set_input_batch_before(&mut self, input_batch_before: Vec<Vec<Vec<Complex<f64>>>>) {
        self.input_batch_before = Some(input_batch_before);
    }
    pub fn set_previous_gradient_input_batch(&mut self, previous_gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        self.previous_gradient_input_batch = Some(previous_gradient_input_batch);
    }
    pub fn set_padding_mask_batch(&mut self, padding_mask_batch: Vec<Vec<u32>>) {
        self.padding_mask_batch = Some(padding_mask_batch);
    }
    pub fn set_input_record(&mut self, input_record: Vec<Vec<Complex<f64>>>) {
        self.input_record = Some(input_record);
    }
    pub fn set_batch_ids(&mut self, batch_ids: Vec<Vec<u32>>) {
        self.batch_ids = Some(batch_ids);
    }
    pub fn set_time_step(&mut self, time_step: usize) {
        self.time_step = time_step;
    }
    pub fn set_target_token_len(&mut self, target_token_len: usize) {
        self.target_tokens_len = target_token_len;
    }
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }
    pub fn set_record_index(&mut self, record_ind: usize) {
        self.record_ind = record_ind;
    }
    pub fn set_calculate_gradient(&mut self, calculate_gradient: bool) {
        self.calculate_gradient = calculate_gradient;
    }
    pub fn set_target_batch_ids(&mut self, target_ids: Vec<Vec<u32>>) {
        self.target_batch_ids = Some(target_ids);
    }

    pub fn get_padding_mask_batch(&self) -> Vec<Vec<u32>> {
        self.padding_mask_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_input_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_input_batch_before(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch_before.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_target_batch_ids(&self) -> Vec<Vec<u32>> {
        self.target_batch_ids.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_previous_gradient_input_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.previous_gradient_input_batch.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_input_record(&self) -> Vec<Vec<Complex<f64>>> {
        self.input_record.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_batch_ids(&self) -> Vec<Vec<u32>> {
        self.batch_ids.clone().unwrap_or_else(|| vec![])
    }
    pub fn get_forward_only(&self) -> bool {
        self.forward_only
    }
    pub fn get_calculate_gradient(&self) -> bool {
        self.calculate_gradient
    }
    pub fn get_time_step(&self) -> usize {
        self.time_step
    }
    pub fn get_target_token_len(&self) -> usize {
        self.target_tokens_len
    }
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
    pub fn get_record_index(&self) -> usize {
        self.record_ind
    }
    pub fn set_pooling_metadata(&mut self, metadata: Option<CompressionMetadata>) {
        self.pooling_metadata = metadata;
    }
    pub fn get_pooling_metadata(&self) -> Option<CompressionMetadata> {
        self.pooling_metadata.clone()
    }

    pub fn set_calculate_k_v_cache(&mut self, calculate_k_v_cache: bool) {
        self.calculate_k_v_cache = calculate_k_v_cache;
    }
    pub fn get_calculate_k_v_cache(&self) -> bool {
        self.calculate_k_v_cache
    }
}
