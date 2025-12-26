use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{complex_to_linear_layer::ComplexToLinearLayer, layer_input_struct::LayerInput},
    network_types::neural_network_generic::OperationMode,
    utils::activation::{softmax_backward_real_with_gradient, softmax_last_row},
};

use super::gradient_struct::Gradient;

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxLayer {
    pub learning_rate: f64,
    pub operation_mode: OperationMode,
    pub complex_to_linear_layer: Option<ComplexToLinearLayer>,

    #[serde(skip)]
    pub softmax_output_batch: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(skip)]
    pub cross_entropy_loss_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl SoftmaxLayer {
    pub fn new(learning_rate: f64, operation_mode: OperationMode, _feature_dim: usize) -> Self {
        Self {
            learning_rate,
            operation_mode,
            complex_to_linear_layer: None,
            softmax_output_batch: None,
            input_batch: None,
            gradient: None,
            padding_mask_batch: None,
            cross_entropy_loss_batch: None,
            time_step: 0,
            batch_size: 1,
        }
    }
    pub fn forward(&mut self, layer_input: &LayerInput, padding_mask_option: Option<Vec<Vec<u32>>>, target_token_ids: Option<Vec<Vec<u32>>>) -> Vec<Vec<Vec<f64>>> {
        let input_batch = layer_input.get_input_batch();
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        let padding_mask_batch = padding_mask_option.unwrap_or_else(|| vec![vec![1; seq_len]; batch_size]);

        let target_token_batch_ids = target_token_ids.unwrap_or(Vec::new());

        let input_batch_linear = if let Some(complex_to_linear_layer) = &mut self.complex_to_linear_layer {
            complex_to_linear_layer.forward(&layer_input).get_output_batch()
        } else {
            input_batch.clone()
        };

        let (layer_output_batch, losses, mut input_gradient_batch) = match self.operation_mode {
            OperationMode::PRODUCTION => {
                let output: Vec<Vec<Vec<f64>>> = input_batch
                    .par_iter()
                    .map(|input| softmax_last_row(input)) // Apply `softmax_last_row` to each input
                    .collect(); // Collect results into a Vec

                (output, Vec::new(), Vec::new())
            }
            OperationMode::TRAINING => {
                // Compute total valid tokens across entire batch for proper normalization
                let total_valid_tokens: usize = padding_mask_batch
                    .iter()
                    .zip(target_token_batch_ids.iter())
                    .map(|(mask, targets)| {
                        let seq_len = mask.len();
                        let target_len = targets.len();
                        let offset = seq_len - target_len;
                        mask.iter().skip(offset).filter(|&&m| m == 1).count()
                    })
                    .sum();

                let output_gradients = (0..batch_size)
                    .into_par_iter()
                    .map(|batch_ind| {
                        let outputs = softmax_backward_real_with_gradient(&input_batch_linear[batch_ind], &target_token_batch_ids[batch_ind], &padding_mask_batch[batch_ind], total_valid_tokens);
                        outputs
                    })
                    .unzip();

                (Vec::new(), output_gradients.0, output_gradients.1)
            }
        };

        if let Some(complex_to_linear_layer) = &mut self.complex_to_linear_layer {
            let mut complex_to_linear_gradient = Gradient::new_default();
            complex_to_linear_gradient.set_gradient_input_batch(input_gradient_batch);

            complex_to_linear_gradient = complex_to_linear_layer.backward(&complex_to_linear_gradient);
            input_gradient_batch = complex_to_linear_gradient.get_gradient_input_batch()
        }

        self.padding_mask_batch = Some(padding_mask_batch);
        self.softmax_output_batch = Some(layer_output_batch.clone());
        self.cross_entropy_loss_batch = Some(losses);
        self.input_batch = Some(input_batch.clone());

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(input_gradient_batch);
        self.gradient = Some(gradient);

        layer_output_batch
    }

    pub fn update_parameters(&mut self) {
        if let Some(complex_to_linear_layer) = &mut self.complex_to_linear_layer {
            complex_to_linear_layer.update_parameters();
        }
    }

    pub fn backward(&mut self, _target_token_ids: &Vec<Vec<u32>>) -> Gradient {
        // let softmax_output_batch: &Vec<Vec<Vec<f64>>> = self.softmax_output_batch.as_ref().expect("Softmax output batch is missing in softmax layer");
        // let _input_batch: &Vec<Vec<Vec<Complex<f64>>>> = self.input_batch.as_ref().expect("Input batch is missing in softmax layer");
        // let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Input batch is missing in softmax layer");

        // let mut batch_size = self.batch_size;
        // let seq_len = softmax_output_batch[0].len();
        // let vocab_dim = softmax_output_batch[0][0].len();
        // let mut softmax_gradient;

        // if softmax_output_batch.len() > batch_size {
        //     batch_size = softmax_output_batch.len();
        // }

        // let mut gradient_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); vocab_dim]; seq_len]; batch_size];

        // for (batch_index, (softmax_output, target_tokens)) in softmax_output_batch.iter().zip(target_token_ids.iter()).enumerate() {
        //     let padding_mask = &padding_mask_batch[batch_index];

        //     let mut _sequence_len_unpadded: usize = 0;
        //     for padding in padding_mask.iter() {
        //         if *padding != 0 {
        //             _sequence_len_unpadded += 1;
        //         }
        //     }

        //     let mut target_len_unpadded = 0.0;
        //     for (_t, &target_class) in target_tokens.iter().enumerate() {
        //         if target_class != 1 {
        //             target_len_unpadded += 1.0;
        //         }
        //     }

        //     // let target_len = target_tokens.len();
        //     let seq_ind_start = _sequence_len_unpadded - target_len_unpadded as usize;
        //     let seq_end = _sequence_len_unpadded;

        //     // let seq_ind_start = softmax_output.len() - target_len;
        //     // let seq_end = softmax_output.len();

        //     let normalizer: f64 = batch_size as f64 * target_len_unpadded;

        //     for (t, &target_class) in target_tokens.iter().enumerate() {
        //         if target_class as usize >= vocab_dim {
        //             panic!("Target token ID {} exceeds vocabulary dimension {}", target_class, vocab_dim);
        //         }

        //         if target_class == 1 {
        //             continue; // Skip padding token
        //         }

        //         let seq_ind = seq_ind_start + t;

        //         if seq_ind >= seq_end {
        //             break;
        //         }

        //         for (c, softmax_prob) in softmax_output[seq_ind].iter().enumerate() {
        //             // for log softmax
        //             // let prob = softmax_prob.exp();
        //             // for softmax
        //             //let prob = softmax_prob;

        //             // let real_part_gradient = input_batch[batch_index][seq_ind][c].re / input_batch[batch_index][seq_ind][c].norm();
        //             // let im_part_gradient = input_batch[batch_index][seq_ind][c].im / input_batch[batch_index][seq_ind][c].norm();

        //             if target_class == c as u32 {
        //                 softmax_gradient = (softmax_prob - 1.0) / normalizer;
        //             } else {
        //                 softmax_gradient = softmax_prob / normalizer;
        //             };

        //             gradient_batch[batch_index][seq_ind][c] += Complex::new(softmax_gradient, 0.0);
        //         }
        //     }
        // }

        // let mut gradient = Gradient::new_default();
        // gradient.set_gradient_input_batch(gradient_batch);
        // self.gradient = Some(gradient.clone());

        // gradient
        self.gradient.as_ref().unwrap().clone()
    }
}
