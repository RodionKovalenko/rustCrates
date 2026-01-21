use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        adaptive_pooling::adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer,
        complex_to_linear_layer::ComplexToLinearLayer,
        complex_to_linear_layer_rm::ComplexToLinearLayerRm,
        feedforward_layer::FeedForwardLayer,
        feedforward_layer_rm::FeedForwardLayerRm,
        embedding_layer_rm::EmbeddingLayerRm,
        layer_rm::LayerRm,
        linear_layer_rm::LinearLayerRm,
        multi_linear_layer::MultiLinearLayer,
        sparse_linear_layer::SparseLinearLayer,
        sparse_linear_layer_rm::SparseLinearLayerRm,
        wavelet_complex_layer::ComplexWaveletLayer,
        wavelet_complex_layer_rm::ComplexWaveletLayerRm,
        wavelet_discrete_layer::DiscreteWaveletLayer,
        wavelet_discrete_layer_rm::DiscreteWaveletLayerRm,
    },
    network_types::transformer::{
        self_attention_layer::SelfAttentionLayer,
        self_attention_layer_approximation::SelfAttentionLayerApproximation,
        self_attention_layer_approximation_rm::SelfAttentionLayerApproximationRm,
        sparse_self_attention_layer::SparseSelfAttentionLayer,
        sparse_self_attention_layer_rm::SparseSelfAttentionLayerRm,
    },
    utils::{
        activation::{activate_output_complex, swish},
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        derivative::{get_gradient_complex, get_gradient_swish},
        matrix::{
            add_matrix, add_vector, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate, conjugate_transpose,
            hadamard_product_2d_c, multiply_complex, split_data_by_columns,
        },
        weights_initializer::initialize_weights_complex,
    },
};
use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::dtype::{r, Real, C, ONE, ZERO};

use super::{
    add_rms_norm_layer::RMSNormLayer, embedding_layer::EmbeddingLayer, linear_layer::LinearLayer, norm_layer::NormalNormLayer, norm_layer_rm::NormalNormLayerRm,
    positional_encoding_layer::PositionalEncodingLayer, positional_encoding_layer_rm::PositionalEncodingLayerRm, softmax_output_layer::SoftmaxLayer,
    softmax_output_layer_rm::SoftmaxLayerRm,
};

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
    EmbeddingRm(Box<EmbeddingLayerRm>),
    PositionalEncoding(Box<PositionalEncodingLayer>),
    PositionalEncodingRm(Box<PositionalEncodingLayerRm>),
    Dense(Box<Layer>),
    DenseRm(Box<LayerRm>),
    FeedForward(Box<FeedForwardLayer>),
    FeedForwardRm(Box<FeedForwardLayerRm>),
    RMSNorm(Box<RMSNormLayer>),
    Norm(Box<NormalNormLayer>),
    NormRm(Box<NormalNormLayerRm>),
    SelfAttention(Box<SelfAttentionLayer>),
    SparseSelfAttention(Box<SparseSelfAttentionLayer>),
    SparseSelfAttentionRm(Box<SparseSelfAttentionLayerRm>),
    SelfAttentionApproximation(Box<SelfAttentionLayerApproximation>),
    SelfAttentionApproximationRm(Box<SelfAttentionLayerApproximationRm>),
    Linear(Box<LinearLayer>),
    LinearRm(Box<LinearLayerRm>),
    SparseLinear(Box<SparseLinearLayer>),
    SparseLinearRm(Box<SparseLinearLayerRm>),
    MultiLinear(Box<MultiLinearLayer>),
    DiscreteWavelet(Box<DiscreteWaveletLayer>),
    DiscreteWaveletRm(Box<DiscreteWaveletLayerRm>),
    ComplexToLinear(Box<ComplexToLinearLayer>),
    ComplexToLinearRm(Box<ComplexToLinearLayerRm>),
    Wavelet(Box<ComplexWaveletLayer>),
    WaveletRm(Box<ComplexWaveletLayerRm>),
    Softmax(Box<SoftmaxLayer>),
    SoftmaxRm(Box<SoftmaxLayerRm>),
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
    pub inactivated_input_batch: Option<Vec<Vec<Vec<C>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<C>>>>,
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

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
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
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        let mut output = LayerOutput::new_default();
        output.set_output_batch(batch_output);

        output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<C>>>) -> Gradient {
        if self.input_batch.is_none() {
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
            output_batch: None,
            gradient: None,
            previous_gradient: None,
            inactivated_input_batch: None,
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
