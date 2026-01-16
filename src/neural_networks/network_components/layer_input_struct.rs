use core::fmt::Debug;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::matrix::RowMajorMatrix;
use crate::neural_networks::{network_components::adaptive_pooling::adaptive_avg_pool1d_layer::CompressionMetadata, network_types::transformer::transformer_network::TOP_K_SIZE};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInput {
    input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    input_batch_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
    batch_ids: Option<Vec<Vec<u32>>>,
    target_batch_ids: Option<Vec<Vec<u32>>>,
    input_batch_before: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    input_batch_before_rm: Option<Vec<RowMajorMatrix<Complex<f64>>>>,
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
    total_valid_tokens: usize,
    top_k_size: Option<usize>,
    output_indices: Option<Vec<Vec<Vec<usize>>>>,

    // When true, any RM->Vec fallback conversion should be treated as a bug and must panic.
    rm_strict: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputRepresentation {
    RowMajor,
    Vec,
    Empty,
}

impl LayerInput {
    pub fn new_default() -> Self {
        LayerInput {
            input_batch: None,
            input_batch_rm: None,
            batch_ids: None,
            target_batch_ids: None,
            input_batch_before: None,
            input_batch_before_rm: None,
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
            total_valid_tokens: 1,
            top_k_size: None,
            output_indices: None,

            rm_strict: false,
        }
    }

    pub fn set_rm_strict(&mut self, rm_strict: bool) {
        self.rm_strict = rm_strict;
    }

    pub fn get_rm_strict(&self) -> bool {
        self.rm_strict
    }
    pub fn set_input_batch(&mut self, input_batch: Vec<Vec<Vec<Complex<f64>>>>) {
        // Only build RM cache for rectangular inputs. Training can legitimately produce
        // variable-length (ragged) rows (e.g. top-k / sparse representations), in which case
        // RM conversion is not possible.
        if input_batch.is_empty() {
            self.input_batch_rm = None;
        } else {
            let mut input_rm = Vec::with_capacity(input_batch.len());
            let mut ok = true;
            for m in &input_batch {
                match RowMajorMatrix::try_from_rows(m) {
                    Some(rm) => input_rm.push(rm),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }

            if ok {
                self.input_batch_rm = Some(input_rm);
            } else {
                if self.rm_strict {
                    panic!(
                        "LayerInput::set_input_batch: cannot build RowMajorMatrix from ragged rows while rm_strict=true"
                    );
                }
                self.input_batch_rm = None;
            }
        }

        self.input_batch = Some(input_batch);
    }

    pub fn clear_input_batch(&mut self) {
        self.input_batch = None;
    }

    pub fn set_input_batch_rm(&mut self, input_batch_rm: Vec<RowMajorMatrix<Complex<f64>>>) {
        self.input_batch_rm = Some(input_batch_rm);
    }

    pub fn clear_input_batch_rm(&mut self) {
        self.input_batch_rm = None;
    }
    pub fn set_forward_only(&mut self, forward_only: bool) {
        self.forward_only = forward_only;
    }
    pub fn set_input_batch_before(&mut self, input_batch_before: Vec<Vec<Vec<Complex<f64>>>>) {
        self.input_batch_before = Some(input_batch_before);
    }

    pub fn set_input_batch_before_rm(&mut self, input_batch_before_rm: Vec<RowMajorMatrix<Complex<f64>>>) {
        self.input_batch_before_rm = Some(input_batch_before_rm);
    }

    pub fn clear_input_batch_before_rm(&mut self) {
        self.input_batch_before_rm = None;
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

    pub fn set_total_valid_tokens(&mut self, total_valid_tokens: usize) {
        self.total_valid_tokens = total_valid_tokens;
    }

    pub fn set_top_k_size(&mut self, top_k_size: usize) {
        self.top_k_size = Some(top_k_size);
    }
    pub fn get_top_k_size(&self) -> usize {
        self.top_k_size.unwrap_or(TOP_K_SIZE)
    }

    pub fn set_output_indices(&mut self, output_indices: Vec<Vec<Vec<usize>>>) {
        self.output_indices = Some(output_indices);
    }

    pub fn get_output_indices(&self) -> Vec<Vec<Vec<usize>>> {
        self.output_indices.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_padding_mask_batch(&self) -> Vec<Vec<u32>> {
        self.padding_mask_batch.clone().unwrap_or_else(|| vec![])
    }

    pub fn get_padding_mask_batch_ref(&self) -> Option<&[Vec<u32>]> {
        self.padding_mask_batch.as_deref()
    }
    pub fn get_input_batch(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        if let Some(input_batch) = &self.input_batch {
            return input_batch.clone();
        }

        if let Some(input_batch_rm) = &self.input_batch_rm {
            if self.rm_strict {
                panic!("LayerInput: attempted RM->Vec conversion for input_batch while rm_strict=true");
            }
            return input_batch_rm.iter().map(|m| m.to_rows()).collect();
        }

        vec![]
    }

    pub fn get_input_batch_ref(&self) -> Option<&[Vec<Vec<Complex<f64>>>]> {
        self.input_batch.as_deref()
    }

    pub fn get_input_batch_rm(&self) -> Vec<RowMajorMatrix<Complex<f64>>> {
        if let Some(input_batch_rm) = &self.input_batch_rm {
            return input_batch_rm.clone();
        }

        if let Some(input_batch) = &self.input_batch {
            if input_batch.is_empty() {
                return vec![];
            }

            let mut out = Vec::with_capacity(input_batch.len());
            for m in input_batch {
                match RowMajorMatrix::try_from_rows(m) {
                    Some(rm) => out.push(rm),
                    None => {
                        if self.rm_strict {
                            panic!(
                                "LayerInput::get_input_batch_rm: cannot build RowMajorMatrix from ragged rows while rm_strict=true"
                            );
                        }
                        return vec![];
                    }
                }
            }

            return out;
        }

        vec![]
    }

    pub fn take_input_batch_rm(&mut self) -> Option<Vec<RowMajorMatrix<Complex<f64>>>> {
        self.input_batch_rm.take()
    }

    pub fn take_input_batch(&mut self) -> Option<Vec<Vec<Vec<Complex<f64>>>>> {
        self.input_batch.take()
    }

    pub fn get_input_batch_rm_ref(&self) -> Option<&[RowMajorMatrix<Complex<f64>>]> {
        self.input_batch_rm.as_deref()
    }

    pub fn has_non_empty_input_batch(&self) -> bool {
        self.input_batch.as_ref().is_some_and(|b| !b.is_empty())
    }

    pub fn has_non_empty_input_batch_rm(&self) -> bool {
        self.input_batch_rm.as_ref().is_some_and(|b| !b.is_empty())
    }

    /// Returns which input representation should be used.
    /// Preference rule: use RM when present & non-empty; otherwise use Vec when present & non-empty.
    pub fn input_representation(&self) -> InputRepresentation {
        if self.has_non_empty_input_batch_rm() {
            return InputRepresentation::RowMajor;
        }
        if self.has_non_empty_input_batch() {
            return InputRepresentation::Vec;
        }
        InputRepresentation::Empty
    }

    pub fn assert_has_input(&self, layer_name: &str) {
        if self.input_representation() == InputRepresentation::Empty {
            panic!("{}: LayerInput has no input (Vec and RM are empty)", layer_name);
        }
    }
    pub fn get_input_batch_before(&self) -> Vec<Vec<Vec<Complex<f64>>>> {
        if let Some(input_batch_before) = &self.input_batch_before {
            return input_batch_before.clone();
        }

        if let Some(input_batch_before_rm) = &self.input_batch_before_rm {
            if self.rm_strict {
                panic!("LayerInput: attempted RM->Vec conversion for input_batch_before while rm_strict=true");
            }
            return input_batch_before_rm.iter().map(|m| m.to_rows()).collect();
        }

        vec![]
    }

    pub fn get_input_batch_before_rm(&self) -> Vec<RowMajorMatrix<Complex<f64>>> {
        self.input_batch_before_rm.clone().unwrap_or_else(|| vec![])
    }

    pub fn take_input_batch_before_rm(&mut self) -> Option<Vec<RowMajorMatrix<Complex<f64>>>> {
        self.input_batch_before_rm.take()
    }

    pub fn get_input_batch_before_rm_ref(&self) -> Option<&[RowMajorMatrix<Complex<f64>>]> {
        self.input_batch_before_rm.as_deref()
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

    pub fn get_total_valid_tokens(&self) -> usize {
        self.total_valid_tokens
    }
}
