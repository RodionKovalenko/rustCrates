use crate::neural_networks::{
    network_types::{feedforward_layer::FeedForwardLayer, transformer::self_attention_layer::SelfAttentionLayer},
    utils::{
        activation::activate_output_complex_padding,
        derivative::get_gradient_complex,
        matrix::{add_vector, apply_padding_mask_batch, check_nan_or_inf_3d, clip_gradient_1d, clip_gradients, hadamard_product_2d_c, is_nan_or_inf, multiply_complex, transpose},
        weights_initializer::initialize_weights_complex,
    },
};
use core::fmt::Debug;
use num::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{norm_layer::NormalNormLayer, add_rms_norm_layer::RMSNormLayer, embedding_layer::EmbeddingLayer, gradient_struct::Gradient, layer_input_struct::LayerInput, linear_layer::LinearLayer, layer_output_struct::LayerOutput, positional_encoding_layer::PositionalEncodingLayer, softmax_output_layer::SoftmaxLayer};

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
    pub inactivated_output: Vec<Vec<Complex<f64>>>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub gradient_w: Vec<Vec<Complex<f64>>>,
    pub errors: Vec<Vec<Complex<f64>>>,
    pub previous_gradient: Vec<Vec<Complex<f64>>>,
    pub m1: Vec<Vec<Complex<f64>>>,
    pub v1: Vec<Vec<Complex<f64>>>,
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub inactivated_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub gradient: Option<Gradient>,
    pub learning_rate: f64,
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
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
                // Multiply input with weights
                // check_nan_or_inf(&mut input, "check dense input");
                // check_nan_or_inf(&self.weights, "check dense weights");

                let output: Vec<Vec<Complex<f64>>> = multiply_complex(input, &self.weights);

                // println!("check output_dense raw");
                // check_nan_or_inf(&output);

                // Add bias to the result
                let raw_output: Vec<Vec<Complex<f64>>> = add_vector(&output, &self.bias);

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
            // 7,7 hadamard 7, 7 = 7,7
            input_gradient_batch[batch_ind] = hadamard_product_2d_c(&previous_gradient, &gradient_output);
            // 7,8 * 7, 7 = 8, 7 * 7, 7 = 8, 7
            weight_gradients[batch_ind] = multiply_complex(&transpose(&input), &input_gradient_batch[batch_ind]);

            // println!("previous gradient in ffn backward: {} {}", previous_gradient.len(), previous_gradient[0].len());
            // println!("gradient_output in ffn backward: {} {}", gradient_output.len(), gradient_output[0].len());
            // println!("weights in ffn in ffn backward: {} {}", &self.weights.len(), &self.weights[0].len());
            // println!("input sample in ffn backward: {} {}", input_sample.len(), input_sample[0].len());

            //Accumulate gradients for biases
            for grad_row in input_gradient_batch[batch_ind].iter() {
                for (k, grad_val) in grad_row.iter().enumerate() {
                    bias_gradients[batch_ind][k] += grad_val.clone(); // Sum the gradients for biases
                }
            }

            input_gradient_batch[batch_ind] = multiply_complex(&input_gradient_batch[batch_ind], &transpose(&self.weights));
        }

        gradient.set_gradient_input_batch(input_gradient_batch);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.as_ref().expect("No Gradient found in linear layer");
        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());

        let input_batch = gradient.get_gradient_input_batch();
        let batch_size = input_batch.len() as f64;

        let threshold = 1.0;
        clip_gradients(&mut weight_gradients, threshold);
        clip_gradient_1d(&mut bias_gradients, threshold);

        // println!("-----------------------------------------------------");
        // println!("weights gradients dense: {:?}", &weight_gradients);
        // println!("bias gradients dense: {:?}", &bias_gradients);
        // println!("-----------------------------------------------------");

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

        // self.learning_rate *= 0.99;
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
            inactivated_output: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            activated_output: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            gradient_w: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            errors: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            previous_gradient: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            input_batch: None,
            output_batch: None,
            gradient: None,
            inactivated_input_batch: None,
            learning_rate: *learning_rate,
            padding_mask_batch: None,
        }
    }
}
