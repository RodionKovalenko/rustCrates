use crate::neural_networks::{
    network_components::{adaptive_pooling::adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer, multi_linear_layer::MultiLinearLayer},
    network_types::{
        feedforward_layer::FeedForwardLayer,
        transformer::{self_attention_layer::SelfAttentionLayer, self_attention_layer_approximation::SelfAttentionLayerApproximation, transformer_network::EMA_SCALER},
        wavelet_complex_layer::ComplexWaveletLayer,
        wavelet_discrete_layer::DiscreteWaveletLayer,
    },
    utils::{
        activation::activate_output_complex_padding,
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        derivative::get_gradient_complex,
        matrix::{add_matrix, add_matrix_3d, add_vector, apply_padding_mask_batch, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, compute_global_norm, conjugate_transpose, hadamard_product_2d_c, multiply_complex, transpose},
        weights_initializer::initialize_weights_complex,
    },
};
use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    add_rms_norm_layer::RMSNormLayer, embedding_layer::EmbeddingLayer, gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput, linear_layer::LinearLayer, norm_layer::NormalNormLayer, positional_encoding_layer::PositionalEncodingLayer,
    softmax_output_layer::SoftmaxLayer,
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
    SOFTSIGN,
    SOFTPLUS,
    PROBIT,
    RANDOM,
}

// Base Layer trait
pub trait BaseLayer: Debug + Clone {
    fn forward(&self, input: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>>;
    fn backward(&self, gradient: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>>;
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
    SelfAttentionApproximation(Box<SelfAttentionLayerApproximation>),
    Linear(Box<LinearLayer>),
    MultiLinear(Box<MultiLinearLayer>),
    DiscreteWavelet(Box<DiscreteWaveletLayer>),
    Wavelet(Box<ComplexWaveletLayer>),
    Softmax(Box<SoftmaxLayer>),
}

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub bias: Vec<Complex<f64>>,
    pub activation_type: ActivationType,
    pub layer_type: LayerType,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub inactivated_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub previous_gradient: Option<Gradient>,
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

        let inactivated_batch_output: Vec<Vec<Vec<Complex<f64>>>> = input_batch
            .par_iter()
            .map(|input| {
                let output: Vec<Vec<Complex<f64>>> = multiply_complex(input, &self.weights);

                // Add bias to the result
                let raw_output: Vec<Vec<Complex<f64>>> = add_vector(&output, &self.bias);
                raw_output
            })
            .collect();

        let batch_output: Vec<Vec<Vec<Complex<f64>>>> = inactivated_batch_output
            .par_iter()
            .zip(padding_mask_batch.par_iter())
            .map(|(input, padding_mask)| {
                // Apply activation if the layer type is DenseLayer
                activate_output_complex_padding(&input, self.activation_type.clone(), padding_mask)
            })
            .collect();

        self.output_batch = Some(batch_output.clone());
        self.inactivated_input_batch = Some(inactivated_batch_output);
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        let mut output = LayerOutput::new_default();
        output.set_output_batch(batch_output);

        output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in dense layer");
        let raw_output_batch = self.inactivated_input_batch.as_ref().expect("Raw output batch is missing in dense layer");
        let output_batch = self.output_batch.as_ref().expect("Output batch is missing in dense layer");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("No padding mask batch found");

        let mut gradient = Gradient::new_default();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.bias.len()]; input_batch.len()];
        let mut input_gradient_batch = vec![vec![vec![Complex::new(0.0, 0.0); previous_gradient_batch[0][0].len()]; previous_gradient_batch[0].len()]; input_batch.len()];

        let mut previous_gradient_batch_padded: Vec<Vec<Vec<Complex<f64>>>> = previous_gradient_batch.clone();

        apply_padding_mask_batch(&mut previous_gradient_batch_padded, padding_mask_batch);

        // println!("\n\n\nprevious gradient batch padded: {:?}", previous_gradient_batch_padded);

        for (batch_ind, (input, previous_gradient)) in input_batch.iter().zip(&previous_gradient_batch_padded).enumerate() {
            let gradient_output = get_gradient_complex(&output_batch[batch_ind], &raw_output_batch[batch_ind], self.activation_type.clone());
            let gradient_output_conj = transpose(&conjugate_transpose(&gradient_output));

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

        if self.gradient.is_some() {
            let previous_gradient = self.gradient.as_ref().expect("");
            // input_gradient_batch = add_matrix_3d(&input_gradient_batch, &previous_gradient.get_gradient_input_batch());
            weight_gradients = add_matrix_3d(&weight_gradients, &previous_gradient.get_gradient_weight_batch());
            bias_gradients = add_matrix(&bias_gradients, &previous_gradient.get_gradient_bias_batch());
        }

        gradient.set_gradient_input_batch(input_gradient_batch);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());

        let input_batch = gradient.get_gradient_input_batch();
        let mut batch_size = input_batch.len() as f64;

        if self.batch_size > 0 {
            batch_size = self.batch_size as f64;
        }

        let mut all_gradients = vec![weight_gradients];
        let global_norm = compute_global_norm(&all_gradients, &bias_gradients);
        self.ema = self.smoothing * self.ema + (1.0 - self.smoothing) * global_norm;
        let max_norm = self.ema * EMA_SCALER;
        clip_all_gradients_by_global_norm_2d(&mut all_gradients, &mut bias_gradients, global_norm, max_norm);

        let mut weight_gradients: Vec<Vec<Complex<f64>>> = all_gradients[0].clone();

        weight_gradients = average_matrix_by_scalar(&weight_gradients, batch_size);
        bias_gradients = average_vector_by_scalar(&bias_gradients, batch_size);

        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (previous_gradient.get_prev_m_bias(), previous_gradient.get_prev_v_bias(), previous_gradient.get_prev_m_weights(), previous_gradient.get_prev_v_weights())
        } else {
            // Initialize to zeros on first step
            (
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![Complex::new(0.0, 0.0); self.bias.len()],
                vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()],
                vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()],
            )
        };

        calculate_adam_w_bias(&mut self.bias, &gradient.get_gradient_bias(), &mut prev_m_bias, &mut prev_v_bias, learning_rate, time_step);
        calculate_adam_w(&mut self.weights, &gradient.get_gradient_weights(), &mut prev_m_weights, &mut prev_v_weights, learning_rate, time_step);

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        self.previous_gradient = Some(gradient.clone());
    }
}

impl Layer {
    pub fn default(rows: usize, cols: usize, learning_rate: &f64) -> Self {
        let mut weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let bias: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

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
        }
    }
}
