use crate::neural_networks::{
    network_types::{feedforward_layer::FeedForwardLayer, transformer::self_attention_layer::SelfAttentionLayer},
    utils::{
        activation::activate_output_complex_padding,
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        derivative::get_gradient_complex,
        matrix::{add_vector, apply_padding_mask_batch, check_nan_or_inf, check_nan_or_inf_3d, clip_gradient_1d, clip_gradients, hadamard_product_2d_c, is_nan_or_inf, multiply_complex, transpose},
        weights_initializer::initialize_weights_complex,
    },
};
use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

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
    Embedding(Box<EmbeddingLayer>),
    PositionalEncoding(Box<PositionalEncodingLayer>),
    Dense(Box<Layer>),
    FeedForward(Box<FeedForwardLayer>),
    RMSNorm(Box<RMSNormLayer>),
    Norm(Box<NormalNormLayer>),
    SelfAttention(Box<SelfAttentionLayer>),
    Linear(Box<LinearLayer>),
    Softmax(Box<SoftmaxLayer>),
}

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<Complex<f64>>>,
    pub bias: Vec<Complex<f64>>,
    pub activation_type: ActivationType,
    pub layer_type: LayerType,

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
    pub learning_rate: f64,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    pub time_step: usize,
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

        self.input_batch = Some(input_batch.clone());

        let mut inactivated_batch_output: Vec<Vec<Vec<Complex<f64>>>> = input_batch
            .par_iter()
            .map(|input| {
                let mut output: Vec<Vec<Complex<f64>>> = multiply_complex(input, &self.weights);

                check_nan_or_inf(&mut output, "check dense inactivated_batch_output without bias");

                // Add bias to the result
                let mut raw_output: Vec<Vec<Complex<f64>>> = add_vector(&output, &self.bias);

                if check_nan_or_inf(&mut raw_output, "check dense inactivated_batch_output with bias") {
                    let max = &self.weights.iter().flatten().max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));
                    let min = &self.weights.iter().flatten().min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));

                    println!("max value in weights dense: {:?}", max);
                    println!("max value in weights dense: {:?}", min);

                    let max = input.iter().flatten().max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));
                    let min = input.iter().flatten().min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));

                    println!("max value in input dense: {:?}", max);
                    println!("max value in input dense: {:?}", min);

                    let max = raw_output.iter().flatten().max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));
                    let min = raw_output.iter().flatten().min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));
                    println!("max value in raw output with bias dense: {:?}", max);
                    println!("max value in raw output with bias dense: {:?}", min);

                    let max = &self.bias.iter().max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));
                    let min = &self.bias.iter().min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal));
                    println!("max value in bias dense: {:?}", max);
                    println!("max value in bias dense: {:?}", min);
                }
                raw_output
            })
            .collect();

        check_nan_or_inf_3d(&mut inactivated_batch_output, "check dense inactivated_batch_output");

        let mut batch_output: Vec<Vec<Vec<Complex<f64>>>> = inactivated_batch_output
            .par_iter() // Process the input batch in parallel
            .zip(padding_mask_batch.par_iter())
            .map(|(input, padding_mask)| {
                // Apply activation if the layer type is DenseLayer
                activate_output_complex_padding(&input, self.activation_type.clone(), padding_mask)
            })
            .collect(); // Collect the results back into a Vec

        check_nan_or_inf_3d(&mut batch_output, "check dense batch_output");

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

        // For each input sample in the batch
        for (batch_ind, (input, previous_gradient)) in input_batch.iter().zip(&previous_gradient_batch_padded).enumerate() {
            // Multiply the transposed input sample with previous gradients (for weight gradients)

            // 7, 7
            let gradient_output = get_gradient_complex(&output_batch[batch_ind], &raw_output_batch[batch_ind], self.activation_type.clone());

            check_nan_or_inf(&mut previous_gradient.clone(), "previous gradient in dense layer backward is not valid");
            check_nan_or_inf(&mut gradient_output.clone(), "gradient output in dense layer backward is not valid");

            // 7,7 hadamard 7, 7 = 7,7
            input_gradient_batch[batch_ind] = hadamard_product_2d_c(&previous_gradient, &gradient_output);
            // 7,8 * 7, 7 = 8, 7 * 7, 7 = 8, 7
            check_nan_or_inf(&mut input_gradient_batch[batch_ind], "previous gradient in dense layer backward is not valid");

            weight_gradients[batch_ind] = multiply_complex(&transpose(&input), &input_gradient_batch[batch_ind]);

            //Accumulate gradients for biases
            for grad_row in input_gradient_batch[batch_ind].iter() {
                for (k, grad_val) in grad_row.iter().enumerate() {
                    bias_gradients[batch_ind][k] += grad_val.clone(); // Sum the gradients for biases
                }
            }

            input_gradient_batch[batch_ind] = multiply_complex(&input_gradient_batch[batch_ind], &transpose(&self.weights));

            // let scaling_factor = Complex::new(1.0 / 50254.0, 1.0 / 50254.0);
            // input_gradient_batch[batch_ind] = multiply_scalar_with_matrix::<Complex<f64>>(scaling_factor, &input_gradient_batch[batch_ind]);
        }

        gradient.set_gradient_input_batch(input_gradient_batch);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in linear layer");
        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());

        let input_batch = gradient.get_gradient_input_batch();
        let batch_size = input_batch.len() as f64;

        let threshold = 0.5;
        clip_gradients(&mut weight_gradients, threshold);
        clip_gradient_1d(&mut bias_gradients, threshold);

        let mut prev_m_bias: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.bias.len()];
        let mut prev_v_bias: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.bias.len()];

        let mut prev_m_weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()];
        let mut prev_v_weights: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); self.weights[0].len()]; self.weights.len()];

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        if let Some(previous_gradient) = &mut self.previous_gradient {
            prev_m_bias = previous_gradient.get_prev_m_bias();
            prev_v_bias = previous_gradient.get_prev_v_bias();

            prev_m_weights = previous_gradient.get_prev_m_weights();
            prev_v_weights = previous_gradient.get_prev_v_weights();

            self.bias = calculate_adam_w_bias(&self.bias, &gradient.get_gradient_bias(), &mut prev_m_bias, &mut prev_v_bias, learning_rate, time_step);
            self.weights = calculate_adam_w(&self.weights, &gradient.get_gradient_weights(), &mut prev_m_weights, &mut prev_v_weights, learning_rate, time_step);
        } else {
            // Update weights and biases using gradient descent
            for (i, row) in self.weights.iter_mut().enumerate() {
                for (j, weight_value) in row.iter_mut().enumerate() {
                    if !is_nan_or_inf(&weight_gradients[i][j]) {
                        *weight_value -= self.learning_rate * (weight_gradients[i][j] / batch_size);
                    }
                }
            }

            for (i, value) in self.bias.iter_mut().enumerate() {
                if !is_nan_or_inf(&bias_gradients[i]) {
                    *value -= self.learning_rate * (bias_gradients[i] / batch_size);
                }
            }
        }

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
        }
    }
}
