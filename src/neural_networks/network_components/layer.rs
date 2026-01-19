use crate::neural_networks::{
    network_components::{
        adaptive_pooling::adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer, complex_to_linear_layer::ComplexToLinearLayer, multi_linear_layer::MultiLinearLayer,
        sparse_linear_layer::SparseLinearLayer,
    },
    network_types::{
        feedforward_layer::FeedForwardLayer,
        transformer::{self_attention_layer::SelfAttentionLayer, self_attention_layer_approximation::SelfAttentionLayerApproximation, sparse_self_attention_layer::SparseSelfAttentionLayer},
        wavelet_complex_layer::ComplexWaveletLayer,
        wavelet_discrete_layer::DiscreteWaveletLayer,
    },
    utils::{
        activation::{activate_output_complex, swish},
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        derivative::{get_gradient_complex, get_gradient_swish},
        matrix::{
            add_matrix, add_vector, add_vector_rm, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate, conjugate_transpose, conjugate_transpose_rm,
            hadamard_product_2d_c, multiply_complex, multiply_complex_rm, split_data_by_columns, RowMajorMatrix,
        },
        weights_initializer::initialize_weights_complex,
    },
};
use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::dtype::{r, C, Real, ONE, ZERO};

use super::{
    add_rms_norm_layer::RMSNormLayer, embedding_layer::EmbeddingLayer, gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput, linear_layer::LinearLayer,
    norm_layer::NormalNormLayer, positional_encoding_layer::PositionalEncodingLayer, softmax_output_layer::SoftmaxLayer,
};

use crate::neural_networks::utils::activation::{sigmoid_complex, tanh_complex};

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::TANH // or any other variant you prefer as default
    }
}

// https://encord.com/blog/activation-functions-neural-networks/
// Activation Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    SIGMOID,
    TANH,
    LINEAR,
    SOFTMAX,
    RELU,
    // Leaky Relu
    LEAKYRELU,
    // Exponental Linear Unit Function
    ELU,
    // Scaled Exponental Linear Unit Function
    SELU,
    // Gaussian Error Linear Units used in Chat-GTP-3, Albert und Roberta
    GELU,
    SWiGLU,
    SOFTSIGN,
    SOFTPLUS,
    PROBIT,
    RANDOM,
}

// Base Layer trait
pub trait BaseLayer: Debug + Clone {
    fn forward(&self, input: &Vec<Vec<Vec<C>>>) -> Vec<Vec<Vec<C>>>;
    fn backward(&self, gradient: &Vec<Vec<Vec<C>>>) -> Vec<Vec<Vec<C>>>;
}

// Layer Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
    AttentionLayer,
    DenseLayer,
    LinearLayer,
}

// Implement Default for LayerType
impl Default for LayerType {
    fn default() -> Self {
        LayerType::InputLayer // or another sensible default
    }
}

// Layer Enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerEnum {
    AdaptiveAvgPool1d(Box<AdaptiveAvgPool1dLayer>),
    Embedding(Box<EmbeddingLayer>),
    PositionalEncoding(Box<PositionalEncodingLayer>),
    Dense(Box<Layer>),
    FeedForward(Box<FeedForwardLayer>),
    RMSNorm(Box<RMSNormLayer>),
    Norm(Box<NormalNormLayer>),
    SelfAttention(Box<SelfAttentionLayer>),
    SparseSelfAttention(Box<SparseSelfAttentionLayer>),
    SelfAttentionApproximation(Box<SelfAttentionLayerApproximation>),
    Linear(Box<LinearLayer>),
    SparseLinear(Box<SparseLinearLayer>),
    MultiLinear(Box<MultiLinearLayer>),
    DiscreteWavelet(Box<DiscreteWaveletLayer>),
    ComplexToLinear(Box<ComplexToLinearLayer>),
    Wavelet(Box<ComplexWaveletLayer>),
    Softmax(Box<SoftmaxLayer>),
}

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<C>>,
    pub bias: Vec<C>,
    pub activation_type: ActivationType,
    pub layer_type: LayerType,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<C>>>>,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,

    pub weights_rm: Option<RowMajorMatrix<C>>,
    #[serde(skip)]
    pub inactivated_input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub inactivated_input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

// Helper function to determine the type of layer
pub fn get_layer_type(layer_idx: usize, total_layers: usize) -> LayerType {
    match layer_idx {
        0 => LayerType::InputLayer,
        x if x == total_layers - 1 => LayerType::OutputLayer,
        _ => LayerType::HiddenLayer,
    }
}

// Implement BaseLayer for Layer struct
impl Layer {
    pub fn new(rows: usize, cols: usize, learning_rate: &f64, activation: &ActivationType, layer_type: LayerType) -> Self {
        Layer {
            activation_type: activation.clone(),
            layer_type,
            ..Layer::default(rows, cols, learning_rate) // Fill the rest with default values
        }
    }

    pub fn prepare_for_save(&mut self) {
        let needs_rebuild = self.weights_rm.is_none() || self.weights_rm.as_ref().is_some_and(|w| w.rows != self.weights.len() || w.cols != self.weights[0].len());

        if needs_rebuild {
            self.weights_rm = Some(RowMajorMatrix::from_rows(&self.weights));
        }
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch_rm_ref = input.get_input_batch_rm_ref();
        let input_batch_ref = input.get_input_batch_ref();
        let rm_available = input_batch_rm_ref.is_some() && input_batch_ref.map_or(true, |v| v.is_empty());
        let activation_supported_rm = matches!(
            self.activation_type,
            ActivationType::LINEAR | ActivationType::TANH | ActivationType::RELU |  ActivationType::LEAKYRELU | ActivationType::SIGMOID | ActivationType::GELU | ActivationType::SWiGLU
        );
        let use_rm = rm_available && activation_supported_rm;

        if use_rm {
            let input_rm = input_batch_rm_ref.unwrap();
            let needs_rebuild = self.weights_rm.is_none() || self.weights_rm.as_ref().is_some_and(|w| w.rows != self.weights.len() || w.cols != self.weights[0].len());
            if needs_rebuild {
                self.weights_rm = Some(RowMajorMatrix::from_rows(&self.weights));
            }
            let weights_rm = self.weights_rm.as_ref().unwrap();

            let calculate_gradient = input.get_calculate_gradient();

            let needs_raw_pre_activation_rm = matches!(self.activation_type, ActivationType::SWiGLU | ActivationType::GELU);

            let mut output_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(input_rm.len());
            let mut raw_output_batch_rm: Vec<RowMajorMatrix<C>> = if calculate_gradient && needs_raw_pre_activation_rm {
                Vec::with_capacity(input_rm.len())
            } else {
                Vec::new()
            };

            for m in input_rm.iter() {
                let mut out = multiply_complex_rm(m, weights_rm);
                add_vector_rm(&mut out, &self.bias);
                if calculate_gradient && needs_raw_pre_activation_rm {
                    raw_output_batch_rm.push(out.clone());
                }
                let out_act = activate_output_complex_rm(out, &self.activation_type);
                output_batch_rm.push(out_act);
            }

            self.batch_size = input.get_batch_size();
            self.time_step = input.get_time_step();
            self.input_batch = None;
            self.inactivated_input_batch = None;
            self.output_batch = None;

            self.input_batch_rm = if calculate_gradient { Some(input_rm.to_vec()) } else { None };
            self.inactivated_input_batch_rm = if calculate_gradient && needs_raw_pre_activation_rm { Some(raw_output_batch_rm) } else { None };
            self.output_batch_rm = if calculate_gradient { Some(output_batch_rm.clone()) } else { None };
            self.padding_mask_batch = Some(input.get_padding_mask_batch());

            let mut output = LayerOutput::new_default();
            output.set_output_batch_rm(output_batch_rm);
            return output;
        }

        // RM input is present but activation isn't supported in RM path.
        // Fallback to legacy path (keeps correctness, but may reintroduce conversions for this layer only).
        if rm_available && !activation_supported_rm {
            if input.get_rm_strict() {
                panic!(
                    "RM strict mode violation: Dense layer activation {:?} lacks RM support (would fall back to Vec conversion)",
                    self.activation_type
                );
            }
            let input_rm = input_batch_rm_ref.unwrap();
            let input_batch_vec: Vec<Vec<Vec<C>>> = input_rm.iter().map(|m| m.to_rows()).collect();
            let mut input_legacy = input.clone();
            input_legacy.clear_input_batch_rm();
            input_legacy.set_input_batch(input_batch_vec);
            return self.forward(&input_legacy);
        }

        let input_batch = input.get_input_batch();
        let padding_mask_batch = input.get_padding_mask_batch();
        self.batch_size = input.get_batch_size();

        self.input_batch = Some(input_batch.clone());
        self.time_step = input.get_time_step();

        let inactivated_batch_output: Vec<Vec<Vec<C>>> = input_batch
            .par_iter()
            .map(|input| {
            let mut output: Vec<Vec<C>> = multiply_complex(input, &self.weights);

                // Add bias to the result
                add_vector(&mut output, &self.bias);
                output
            })
            .collect();

        let batch_output: Vec<Vec<Vec<C>>> = inactivated_batch_output
            .par_iter()
            .map(|input| {
                // Apply activation if the layer type is DenseLayer
                let activated_output = activate_output_complex(&input, self.activation_type.clone());
                activated_output
            })
            .collect();

        self.output_batch = Some(batch_output.clone());
        self.inactivated_input_batch = Some(inactivated_batch_output);
        self.input_batch_rm = None;
        self.inactivated_input_batch_rm = None;
        self.output_batch_rm = None;
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        let mut output = LayerOutput::new_default();
        output.set_output_batch(batch_output);

        output
    }

    pub fn backward_rm(&mut self, previous_gradient_batch_rm: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch_rm = self.input_batch_rm.as_ref().expect("Input RM batch is missing in dense layer");
        let batch_len = input_batch_rm.len();
        if batch_len == 0 {
            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        // If the upstream gradient batch is missing/empty or has a different length, pad/truncate with zeros.
        let (out_rows, out_cols) = if let Some(out_rm) = self.output_batch_rm.as_ref().and_then(|v| v.first()) {
            (out_rm.rows, out_rm.cols)
        } else {
            (input_batch_rm[0].rows, self.weights[0].len())
        };

        let mut prev_grads: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_len);
        for b in 0..batch_len {
            if let Some(g) = previous_gradient_batch_rm.get(b) {
                prev_grads.push(g.clone());
            } else {
                prev_grads.push(RowMajorMatrix::from_data(out_rows, out_cols, vec![C::new(ZERO, ZERO); out_rows * out_cols]));
            }
        }

        let needs_rebuild = self.weights_rm.is_none() || self.weights_rm.as_ref().is_some_and(|w| w.rows != self.weights.len() || w.cols != self.weights[0].len());
        if needs_rebuild {
            self.weights_rm = Some(RowMajorMatrix::from_rows(&self.weights));
        }
        let weights_rm = self.weights_rm.as_ref().unwrap();
        let weights_h_rm = conjugate_transpose_rm(weights_rm);

        let mut gradient = Gradient::new_default();

        let batch_len = input_batch_rm.len();
        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()]; batch_len];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; batch_len];

        let mut input_gradient_batch_rm: Vec<RowMajorMatrix<C>> = Vec::with_capacity(batch_len);

        match &self.activation_type {
            ActivationType::SWiGLU => {
                let raw_output_batch_rm = self.inactivated_input_batch_rm.as_ref().expect("Raw output RM batch is missing in dense SWiGLU layer");

                let cols = weights_rm.cols;
                assert!(cols % 2 == 0, "SWiGLU expects even column count");
                let half = cols / 2;

                // Split weights and bias into two halves.
                let weights_1_rm = split_columns_rm(weights_rm, 0, half);
                let weights_2_rm = split_columns_rm(weights_rm, half, cols);
                let weights_1_h_rm = conjugate_transpose_rm(&weights_1_rm);
                let weights_2_h_rm = conjugate_transpose_rm(&weights_2_rm);

                for batch_ind in 0..batch_len {
                    let input_rm = &input_batch_rm[batch_ind];
                    let grad_y_rm = &prev_grads[batch_ind];

                    let raw_rm = &raw_output_batch_rm[batch_ind];
                    assert_eq!(raw_rm.cols, cols);
                    assert_eq!(grad_y_rm.cols, half, "SWiGLU gradient must match activated output cols");

                    let a_rm = split_columns_rm(raw_rm, 0, half);
                    let b_rm = split_columns_rm(raw_rm, half, cols);

                    let swish_b = swish_rm(&b_rm);
                    let grad_swish_b = swish_grad_rm(&b_rm);

                    let dl_da = hadamard_rm(grad_y_rm, &conjugate_rm(&swish_b));
                    let dl_db_1 = hadamard_rm(grad_y_rm, &conjugate_rm(&a_rm));
                    let dl_db = hadamard_rm(&dl_db_1, &conjugate_rm(&grad_swish_b));

                    let input_h_rm = conjugate_transpose_rm(input_rm);
                    let wgrad1_rm = multiply_complex_rm(&input_h_rm, &dl_da);
                    let wgrad2_rm = multiply_complex_rm(&input_h_rm, &dl_db);

                    // Bias gradients: sum rows
                    accumulate_bias_from_rm(&mut bias_gradients[batch_ind][0..half], &dl_da);
                    accumulate_bias_from_rm(&mut bias_gradients[batch_ind][half..cols], &dl_db);

                    // Input gradients
                    let gx_a = multiply_complex_rm(&dl_da, &weights_1_h_rm);
                    let gx_b = multiply_complex_rm(&dl_db, &weights_2_h_rm);
                    let gx = add_rm(&gx_a, &gx_b);
                    input_gradient_batch_rm.push(gx);

                    // Combine weight gradients
                    let mut rows = wgrad1_rm.to_rows();
                    let rows2 = wgrad2_rm.to_rows();
                    for (r, row2) in rows2.into_iter().enumerate() {
                        rows[r].extend_from_slice(&row2);
                    }
                    weight_gradients[batch_ind] = rows;
                }
            }
            _ => {
                let output_batch_rm = self.output_batch_rm.as_ref().expect("Output RM batch is missing in dense layer");

                let raw_output_batch_rm = if self.activation_type == ActivationType::GELU {
                    Some(self.inactivated_input_batch_rm.as_ref().expect("Raw output RM batch is missing in dense GELU layer"))
                } else {
                    None
                };

                for batch_ind in 0..batch_len {
                    let input_rm = &input_batch_rm[batch_ind];
                    let grad_y_rm = &prev_grads[batch_ind];
                    let out_act_rm = &output_batch_rm[batch_ind];
                    assert_eq!(grad_y_rm.rows, out_act_rm.rows);
                    assert_eq!(grad_y_rm.cols, out_act_rm.cols);

                    let d_act = if self.activation_type == ActivationType::GELU {
                        let raw_rm = &raw_output_batch_rm.unwrap()[batch_ind];
                        activation_derivative_from_raw_rm(raw_rm, &self.activation_type)
                    } else {
                        activation_derivative_rm(out_act_rm, &self.activation_type)
                    };
                    let dz = hadamard_rm(grad_y_rm, &conjugate_rm(&d_act));

                    let input_h_rm = conjugate_transpose_rm(input_rm);
                    let wgrad_rm = multiply_complex_rm(&input_h_rm, &dz);
                    weight_gradients[batch_ind] = wgrad_rm.to_rows();

                    accumulate_bias_from_rm(&mut bias_gradients[batch_ind], &dz);

                    let gx_rm = multiply_complex_rm(&dz, &weights_h_rm);
                    input_gradient_batch_rm.push(gx_rm);
                }
            }
        }

        gradient.set_gradient_input_batch_rm(input_gradient_batch_rm);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        if self.input_batch.is_none() {
            if let Some(_input_batch_rm) = self.input_batch_rm.as_ref() {
                // RM forward was used; fall back to RM backward by converting the incoming Vec gradient.
                let previous_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient_batch.iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();

                let mut gradient = self.backward_rm(&previous_gradient_batch_rm);

                // Keep legacy callers working by materializing Vec gradients.
                let legacy_gx: Vec<Vec<Vec<C>>> = gradient.get_gradient_input_batch_rm().iter().map(|m| m.to_rows()).collect();
                gradient.set_gradient_input_batch(legacy_gx);
                return gradient;
            }

            let mut gradient = Gradient::new_default();
            gradient.set_gradient_input_batch(vec![]);
            gradient.set_gradient_input_batch_rm(vec![]);
            gradient.set_gradient_weight_batch(vec![]);
            gradient.set_gradient_bias_batch(vec![]);
            self.gradient = Some(gradient.clone());
            return gradient;
        }

        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in dense layer");
        let raw_output_batch = self.inactivated_input_batch.as_ref().expect("Raw output batch is missing in dense layer");
        let output_batch = self.output_batch.as_ref().expect("Output batch is missing in dense layer");

        let mut gradient = Gradient::new_default();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; input_batch.len()];
        let mut input_gradient_batch = vec![vec![vec![C::new(ZERO, ZERO); previous_gradient_batch[0][0].len()]; previous_gradient_batch[0].len()]; input_batch.len()];

        let previous_gradient_batch_padded: Vec<Vec<Vec<C>>> = previous_gradient_batch.clone();

        // println!("\n\n\nprevious gradient batch padded: {:?}", previous_gradient_batch_padded);

        match &self.activation_type {
            ActivationType::SWiGLU => {
                let (weights_1, weights_2) = split_data_by_columns(&self.weights);
                let mut gradient_weights_1 = vec![vec![vec![C::new(ZERO, ZERO); weights_1[0].len()]; weights_1.len()]; input_batch.len()];
                let mut gradient_weights_2 = vec![vec![vec![C::new(ZERO, ZERO); weights_2[0].len()]; weights_2.len()]; input_batch.len()];

                let mut gradient_bias_1 = vec![vec![C::new(ZERO, ZERO); bias_gradients[0].len() / 2]; input_batch.len()];
                let mut gradient_bias_2 = vec![vec![C::new(ZERO, ZERO); bias_gradients[0].len() / 2]; input_batch.len()];

                for batch_ind in 0..input_batch.len() {
                    let (a, b) = split_data_by_columns(&raw_output_batch[batch_ind]);

                    let swish_b = swish(&b);
                    let gradient_b = get_gradient_swish(&b);

                    //5x10 hadamard 5x10 = 5x10
                    let dl_da = hadamard_product_2d_c(&previous_gradient_batch[batch_ind], &conjugate(&swish_b));
                    // 16x5 * 5x10 = 16x10
                    gradient_weights_1[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &dl_da);

                    // 5x10 hadamard 5x10 = 5x10
                    let dl_db_1 = hadamard_product_2d_c(&previous_gradient_batch[batch_ind], &conjugate(&a));
                    // 5x10 hadamard 5x10 = 5x10
                    let dl_db_2 = hadamard_product_2d_c(&dl_db_1, &conjugate(&gradient_b));

                    // 16x5 * 5x10 = 16x10
                    gradient_weights_2[batch_ind] = multiply_complex(&conjugate_transpose(&input_batch[batch_ind]), &dl_db_2);

                    for grad_row in dl_da.iter() {
                        for (k, grad_val) in grad_row.iter().enumerate() {
                            gradient_bias_1[batch_ind][k] += grad_val;
                        }
                    }
                    for grad_row in dl_db_2.iter() {
                        for (k, grad_val) in grad_row.iter().enumerate() {
                            gradient_bias_2[batch_ind][k] += grad_val;
                        }
                    }

                    // 5x10 * 10x16 = 5x16
                    let gradient_input_a = multiply_complex(&dl_da, &conjugate_transpose(&weights_1));
                    // 5x10 * 10x16 = 5x16
                    let gradient_input_b = multiply_complex(&dl_db_2, &conjugate_transpose(&weights_2));
                    input_gradient_batch[batch_ind] = add_matrix(&gradient_input_a, &gradient_input_b);
                }

                for batch_ind in 0..input_batch.len() {
                    for (row_ind, row) in gradient_weights_1[batch_ind].iter_mut().enumerate() {
                        row.extend_from_slice(&gradient_weights_2[batch_ind][row_ind]);
                    }
                    gradient_bias_1[batch_ind].extend_from_slice(&gradient_bias_2[batch_ind]);
                }

                weight_gradients = gradient_weights_1;
                bias_gradients = gradient_bias_1;
            }
            _ => {
                for (batch_ind, (input, previous_gradient)) in input_batch.iter().zip(&previous_gradient_batch_padded).enumerate() {
                    let gradient_output = get_gradient_complex(&output_batch[batch_ind], &raw_output_batch[batch_ind], self.activation_type.clone());
                    let gradient_output_conj = conjugate(&gradient_output);

                    input_gradient_batch[batch_ind] = hadamard_product_2d_c(previous_gradient, &gradient_output_conj);
                    weight_gradients[batch_ind] = multiply_complex(&conjugate_transpose(&input), &input_gradient_batch[batch_ind]);

                    //Accumulate gradients for biases
                    for grad_row in input_gradient_batch[batch_ind].iter() {
                        for (k, grad_val) in grad_row.iter().enumerate() {
                            bias_gradients[batch_ind][k] += grad_val;
                        }
                    }

                    input_gradient_batch[batch_ind] = multiply_complex(&input_gradient_batch[batch_ind], &conjugate_transpose(&self.weights));
                }
            }
        }

        // if self.gradient.is_some() {
        //     let previous_gradient = self.gradient.as_ref().expect("");
        //     weight_gradients = add_matrix_3d(&weight_gradients, &previous_gradient.get_gradient_weight_batch());
        //     bias_gradients = add_matrix(&bias_gradients, &previous_gradient.get_gradient_bias_batch());
        // }

        gradient.set_gradient_input_batch(input_gradient_batch);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());

        let total_valid_tokens: Real = r(gradient.get_total_valid_tokens().max(1) as f64);

        clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        weight_gradients = average_matrix_by_scalar(&weight_gradients, total_valid_tokens);
        bias_gradients = average_vector_by_scalar(&bias_gradients, total_valid_tokens);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights, mut prev_v_weights_hat, mut prev_v_bias_hat) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_bias(),
                previous_gradient.get_prev_v_bias(),
                previous_gradient.get_prev_m_weights(),
                previous_gradient.get_prev_v_weights(),
                previous_gradient.get_prev_v_weights_hat(),
                previous_gradient.get_prev_v_bias_hat(),
            )
        } else {
            // Initialize to zeros on first step
            (
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
            )
        };

        calculate_adam_w_bias(
            &mut self.bias,
            &gradient.get_gradient_bias(),
            &mut prev_m_bias,
            &mut prev_v_bias,
            &mut prev_v_bias_hat,
            learning_rate,
            time_step,
        );
        calculate_adam_w(
            &mut self.weights,
            &gradient.get_gradient_weights(),
            &mut prev_m_weights,
            &mut prev_v_weights,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
        );

        // We updated weights in-place; cached row-major view is now stale.
        // Rebuild lazily on the next RM forward/backward.
        self.weights_rm = None;

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);
        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}

fn activate_output_complex_rm(mut data: RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    match activation {
        ActivationType::LINEAR => data,
        ActivationType::TANH => {
            for v in data.data.iter_mut() {
                *v = tanh_complex(*v);
            }
            data
        }
        ActivationType::RELU => {
            for v in data.data.iter_mut() {
                *v = if v.re > ZERO { *v } else { C::new(ZERO, ZERO) };
            }
            data
        }
        ActivationType::LEAKYRELU => {
            for v in data.data.iter_mut() {
                *v = if v.re > ZERO { *v } else { *v * r(0.01) };
            }
            data
        }
        ActivationType::SIGMOID => {
            for v in data.data.iter_mut() {
                *v = sigmoid_complex(v);
            }
            data
        }
        ActivationType::GELU => {
            for v in data.data.iter_mut() {
                *v = C::new(gelu_real(v.re), v.im);
            }
            data
        }
        ActivationType::SWiGLU => {
            let cols = data.cols;
            assert!(cols % 2 == 0, "SWiGLU expects an even column count");
            let half = cols / 2;
            let mut out = RowMajorMatrix::from_data(data.rows, half, vec![C::new(ZERO, ZERO); data.rows * half]);
            for r in 0..data.rows {
                let row = data.row_range(r);
                let out_row = out.row_range(r);
                for c in 0..half {
                    let a = data.data[row.start + c];
                    let b = data.data[row.start + half + c];
                    out.data[out_row.start + c] = a * b * sigmoid_complex(&b);
                }
            }
            out
        }
        _ => {
            // Keep legacy behavior for unsupported activations in RM path.
            // (Callers should only use RM forward with activations they know are supported.)
            data
        }
    }
}

fn activation_derivative_rm(activated: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(activated.rows, activated.cols, vec![C::new(ZERO, ZERO); activated.rows * activated.cols]);
    match activation {
        ActivationType::LINEAR => {
            for v in out.data.iter_mut() {
                *v = C::new(ONE, ZERO);
            }
        }
        ActivationType::TANH => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = C::new(ONE, ZERO) - (z * z);
            }
        }
        ActivationType::SIGMOID => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = z * (C::new(ONE, ZERO) - z);
            }
        }
        ActivationType::RELU => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = if z.re > ZERO { C::new(ONE, ZERO) } else { C::new(ZERO, ZERO) };
            }
        }
        ActivationType::LEAKYRELU => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = if z.re > ZERO { C::new(ONE, ZERO) } else { C::new(r(0.01), ZERO) };
            }
        }
        _ => {
            // Unsupported derivative in RM path
        }
    }
    out
}

fn activation_derivative_from_raw_rm(raw: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(raw.rows, raw.cols, vec![C::new(ZERO, ZERO); raw.rows * raw.cols]);
    match activation {
        ActivationType::GELU => {
            for (dst, &z) in out.data.iter_mut().zip(raw.data.iter()) {
                *dst = C::new(gelu_derivative_real(z.re), ZERO);
            }
        }
        _ => {
            // Not implemented; callers should fall back to activation_derivative_rm.
        }
    }
    out
}

fn gelu_real(x: Real) -> Real {
    // tanh-based approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let xf = x as f64;
    let x3 = xf * xf * xf;
    let inner = SQRT_2_OVER_PI * (xf + 0.044_715 * x3);
    r(0.5 * xf * (1.0 + inner.tanh()))
}

fn gelu_derivative_real(x: Real) -> Real {
    // Derivative of the tanh-based approximation.
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    let xf = x as f64;
    let x2 = xf * xf;
    let _x3 = x2 * xf;
    let inner = SQRT_2_OVER_PI * (xf + 0.044_715 * (xf * xf * xf));
    let t = inner.tanh();
    let sech2 = 1.0 - t * t;
    let inner_prime = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044_715 * x2);
    r(0.5 * (1.0 + t) + 0.5 * xf * sech2 * inner_prime)
}

fn conjugate_rm(m: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(m.rows, m.cols, vec![C::new(ZERO, ZERO); m.rows * m.cols]);
    for (dst, &v) in out.data.iter_mut().zip(m.data.iter()) {
        *dst = v.conj();
    }
    out
}

fn hadamard_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = RowMajorMatrix::from_data(a.rows, a.cols, vec![C::new(ZERO, ZERO); a.rows * a.cols]);
    for i in 0..out.data.len() {
        out.data[i] = a.data[i] * b.data[i];
    }
    out
}

fn add_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = RowMajorMatrix::from_data(a.rows, a.cols, vec![C::new(ZERO, ZERO); a.rows * a.cols]);
    for i in 0..out.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }
    out
}

fn split_columns_rm(matrix: &RowMajorMatrix<C>, start_col: usize, end_col: usize) -> RowMajorMatrix<C> {
    assert!(start_col <= end_col);
    assert!(end_col <= matrix.cols);
    let cols = end_col - start_col;
    let mut data = vec![C::new(ZERO, ZERO); matrix.rows * cols];
    for r in 0..matrix.rows {
        let src_row = matrix.row_range(r);
        let dst_row_start = r * cols;
        let src_start = src_row.start + start_col;
        let src_end = src_row.start + end_col;
        data[dst_row_start..dst_row_start + cols].copy_from_slice(&matrix.data[src_start..src_end]);
    }
    RowMajorMatrix::from_data(matrix.rows, cols, data)
}

fn swish_rm(b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(b.rows, b.cols, vec![C::new(ZERO, ZERO); b.rows * b.cols]);
    for i in 0..out.data.len() {
        let z = b.data[i];
        out.data[i] = z * sigmoid_complex(&z);
    }
    out
}

fn swish_grad_rm(b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(b.rows, b.cols, vec![C::new(ZERO, ZERO); b.rows * b.cols]);
    let one = C::new(ONE, ZERO);
    for i in 0..out.data.len() {
        let z = b.data[i];
        let sigma = sigmoid_complex(&z);
        out.data[i] = sigma + z * sigma * (one - sigma);
    }
    out
}

fn accumulate_bias_from_rm(dst: &mut [C], grad: &RowMajorMatrix<C>) {
    assert_eq!(dst.len(), grad.cols);
    for r in 0..grad.rows {
        let row = grad.row_range(r);
        for c in 0..grad.cols {
            dst[c] += grad.data[row.start + c];
        }
    }
}

impl Layer {
    pub fn default(rows: usize, cols: usize, learning_rate: &f64) -> Self {
        let mut weights: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let bias: Vec<C> = vec![C::new(ONE, ZERO); cols];

        initialize_weights_complex(rows, cols, &mut weights); // 2D matrix

        Layer {
            weights,
            bias,
            activation_type: ActivationType::SIGMOID,
            layer_type: LayerType::InputLayer,
            input_batch: None,
            input_batch_rm: None,
            weights_rm: None,
            output_batch: None,
            output_batch_rm: None,
            gradient: None,
            previous_gradient: None,
            inactivated_input_batch: None,
            inactivated_input_batch_rm: None,
            learning_rate: *learning_rate,
            padding_mask_batch: None,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }
}
