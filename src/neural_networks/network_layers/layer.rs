use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        default_layer::LayerInterface,
        adaptive_linear_layer::AdaptiveLinearLayer, adaptive_pooling::adaptive_avg_pool1d_layer::AdaptiveAvgPool1dLayer, clustering_linear_layer::ClusteringLinearLayer, complex_to_linear_layer::ComplexToLinearLayer, eml_linear_layer::EmlLinearLayer, feedforward_layer::FeedForwardLayer, multi_linear_layer::MultiLinearLayer, network_layers_rm::{
            complex_to_linear_layer_rm::ComplexToLinearLayerRm, embedding_layer_rm::EmbeddingLayerRm, feedforward_layer_rm::FeedForwardLayerRm, layer_rm::LayerRm, linear_layer_rm::LinearLayerRm,
            sparse_linear_layer_rm::SparseLinearLayerRm, wavelet_complex_layer_rm::ComplexWaveletLayerRm, wavelet_discrete_layer_rm::DiscreteWaveletLayerRm,
        }, wavelet_complex_layer::ComplexWaveletLayer, wavelet_discrete_layer::DiscreteWaveletLayer
    },
    network_types::transformer::{
        self_attention_layer::SelfAttentionLayer, self_attention_layer_approximation::SelfAttentionLayerApproximation, self_attention_layer_approximation_rm::SelfAttentionLayerApproximationRm,
        sparse_self_attention_layer::SparseSelfAttentionLayer, sparse_self_attention_layer_rm::SparseSelfAttentionLayerRm,
    },
    network_types::neural_network_generic::OperationMode,
    utils::{
        activation::{activate_output_complex, swish},
        adam_w::{calculate_adam_w, calculate_adam_w_bias},
        derivative::{get_gradient_complex, get_gradient_swish},
        matrix::{
            add_matrix, add_vector, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate, conjugate_transpose, hadamard_product_2d_c, multiply_complex,
            normalize_bias, normalize_gradients, split_data_by_columns, RowMajorMatrix,
        },
        weights_initializer::initialize_weights_complex,
    },
};
use core::fmt::Debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::neural_networks::utils::dtype::{r, Real, C, ZERO};

use super::{
    add_rms_norm_layer::RMSNormLayer,
    embedding_layer::EmbeddingLayer,
    linear_layer::LinearLayer,
    network_layers_rm::{norm_layer_rm::NormalNormLayerRm, positional_encoding_layer_rm::PositionalEncodingLayerRm, softmax_output_layer_rm::SoftmaxLayerRm},
    norm_layer::NormalNormLayer,
    positional_encoding_layer::PositionalEncodingLayer,
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
    ClusteringLinear(Box<ClusteringLinearLayer>),
    AdaptiveLinear(Box<AdaptiveLinearLayer>),
    Eml(Box<EmlLinearLayer>),
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
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in dense layer");
        let raw_output_batch = self.inactivated_input_batch.as_ref().expect("Raw output batch is missing in dense layer");
        let output_batch = self.output_batch.as_ref().expect("Output batch is missing in dense layer");

        let mut gradient = Gradient::new_default();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<Vec<Vec<C>>> = vec![vec![vec![C::new(ZERO, ZERO); self.weights[0].len()]; self.weights.len()]; input_batch.len()];
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; input_batch.len()];

        // Gradient w.r.t input has the input feature dimension (weights rows).
        let in_features = self.weights.len();
        let mut input_gradient_batch: Vec<Vec<Vec<C>>> = (0..input_batch.len())
            .map(|b| {
                let seq_len = input_batch[b].len();
                vec![vec![C::new(ZERO, ZERO); in_features]; seq_len]
            })
            .collect();

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

        normalize_gradients(&mut weight_gradients);
        normalize_bias(&mut bias_gradients);

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
            &bias_gradients,
            &mut prev_m_bias,
            &mut prev_v_bias,
            &mut prev_v_bias_hat,
            learning_rate,
            time_step,
        );
        calculate_adam_w(
            &mut self.weights,
            &weight_gradients,
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

#[inline]
fn complex_batch_to_real_batch(data: &[Vec<Vec<C>>]) -> Vec<Vec<Vec<Real>>> {
    data.iter().map(|seq| seq.iter().map(|row| row.iter().map(|z| z.re).collect()).collect()).collect()
}

impl LayerEnum {
    /// Short human-readable layer name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            LayerEnum::AdaptiveAvgPool1d(_) => "AdaptiveAvgPool1d",
            LayerEnum::Embedding(_) => "Embedding",
            LayerEnum::EmbeddingRm(_) => "EmbeddingRm",
            LayerEnum::PositionalEncoding(_) => "PositionalEncoding",
            LayerEnum::PositionalEncodingRm(_) => "PositionalEncodingRm",
            LayerEnum::Dense(_) => "Dense",
            LayerEnum::DenseRm(_) => "DenseRm",
            LayerEnum::FeedForward(_) => "FeedForward",
            LayerEnum::FeedForwardRm(_) => "FeedForwardRm",
            LayerEnum::RMSNorm(_) => "RMSNorm",
            LayerEnum::Norm(_) => "Norm",
            LayerEnum::NormRm(_) => "NormRm",
            LayerEnum::SelfAttention(_) => "SelfAttention",
            LayerEnum::SparseSelfAttention(_) => "SparseSelfAttention",
            LayerEnum::SparseSelfAttentionRm(_) => "SparseSelfAttentionRm",
            LayerEnum::SelfAttentionApproximation(_) => "SelfAttentionApproximation",
            LayerEnum::SelfAttentionApproximationRm(_) => "SelfAttentionApproximationRm",
            LayerEnum::Linear(_) => "Linear",
            LayerEnum::LinearRm(_) => "LinearRm",
            LayerEnum::ClusteringLinear(_) => "ClusteringLinear",
            LayerEnum::AdaptiveLinear(_) => "AdaptiveLinear",
            LayerEnum::Eml(_) => "Eml",
            LayerEnum::SparseLinearRm(_) => "SparseLinearRm",
            LayerEnum::MultiLinear(_) => "MultiLinear",
            LayerEnum::DiscreteWavelet(_) => "DiscreteWavelet",
            LayerEnum::DiscreteWaveletRm(_) => "DiscreteWaveletRm",
            LayerEnum::ComplexToLinear(_) => "ComplexToLinear",
            LayerEnum::ComplexToLinearRm(_) => "ComplexToLinearRm",
            LayerEnum::Wavelet(_) => "Wavelet",
            LayerEnum::WaveletRm(_) => "WaveletRm",
            LayerEnum::Softmax(_) => "Softmax",
            LayerEnum::SoftmaxRm(_) => "SoftmaxRm",
        }
    }
}

/// Uniform layer dispatch used by the network forward/backward loops.
///
/// Every variant funnels into the concrete layer's own forward/backward; the
/// per-variant glue (Vec<->RowMajor gradient conversion, terminal-head
/// specifics for Softmax/EML/AdaptiveLinear) lives here so the network code in
/// `transformer_network.rs` stays a plain loop.
impl LayerInterface for LayerEnum {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let forward_only = layer_input.get_forward_only();

        match self {
            LayerEnum::AdaptiveAvgPool1d(layer) => layer.forward(layer_input),
            LayerEnum::Embedding(layer) => layer.forward(layer_input),
            LayerEnum::EmbeddingRm(layer) => layer.forward(layer_input),
            LayerEnum::PositionalEncoding(layer) => layer.forward(layer_input),
            LayerEnum::PositionalEncodingRm(layer) => layer.forward(layer_input),
            LayerEnum::Dense(layer) => layer.forward(layer_input),
            LayerEnum::DenseRm(layer) => LayerInterface::forward(&mut **layer, layer_input),
            LayerEnum::FeedForward(layer) => layer.forward(layer_input),
            LayerEnum::FeedForwardRm(layer) => layer.forward(layer_input),
            LayerEnum::RMSNorm(layer) => LayerInterface::forward(&mut **layer, layer_input),
            LayerEnum::Norm(layer) => layer.forward(layer_input),
            LayerEnum::NormRm(layer) => layer.forward(layer_input),
            LayerEnum::SelfAttention(layer) => layer.forward(layer_input),
            LayerEnum::SparseSelfAttention(layer) => layer.forward(layer_input),
            LayerEnum::SparseSelfAttentionRm(layer) => layer.forward(layer_input),
            LayerEnum::SelfAttentionApproximation(layer) => layer.forward(layer_input),
            LayerEnum::SelfAttentionApproximationRm(layer) => layer.forward(layer_input),
            LayerEnum::Linear(layer) => layer.forward(layer_input),
            LayerEnum::LinearRm(layer) => LayerInterface::forward(&mut **layer, layer_input),
            LayerEnum::ClusteringLinear(layer) => layer.forward(layer_input),
            LayerEnum::AdaptiveLinear(layer) => {
                let mut output = layer.forward(layer_input);
                if forward_only {
                    let logits = output.get_output_batch();
                    output.set_output_batch_real(complex_batch_to_real_batch(&logits));
                }
                output
            }
            LayerEnum::Eml(layer) => {
                let mut output = layer.forward(layer_input);
                if forward_only {
                    // Inference: the head emits sparse top-k token scores + indices, exactly
                    // like AdaptiveLinear, so the greedy decoder can argmax and map to ids.
                    let logits = output.get_output_batch();
                    output.set_output_batch_real(complex_batch_to_real_batch(&logits));
                } else {
                    // Terminal teacher-forced head on the training path: no dense logits, so
                    // pass the hidden states through unchanged for any downstream consumer.
                    output.set_output_batch(layer_input.get_input_batch());
                }
                output
            }
            LayerEnum::SparseLinearRm(layer) => layer.forward(layer_input),
            LayerEnum::MultiLinear(layer) => layer.forward(layer_input),
            LayerEnum::DiscreteWavelet(layer) => layer.forward(layer_input),
            LayerEnum::DiscreteWaveletRm(layer) => layer.forward(layer_input),
            LayerEnum::ComplexToLinear(layer) => layer.forward(layer_input),
            LayerEnum::ComplexToLinearRm(layer) => layer.forward(layer_input),
            LayerEnum::Wavelet(layer) => layer.forward(layer_input),
            LayerEnum::WaveletRm(layer) => layer.forward(layer_input),
            LayerEnum::Softmax(layer) => {
                let mut output = LayerOutput::new_default();
                if !forward_only {
                    // Training always requires CE loss + gradients; ensure Softmax is in TRAINING
                    // even if a prior inference call switched it to PRODUCTION.
                    layer.operation_mode = OperationMode::TRAINING;
                    let padding_mask = layer_input.get_padding_mask_batch();
                    let padding_mask_option = if padding_mask.is_empty() { None } else { Some(padding_mask) };
                    let _softmax_result = layer.forward_inner(layer_input, padding_mask_option, Some(layer_input.get_target_batch_ids()));
                    output.set_cross_entropy_loss_batch(layer.cross_entropy_loss_batch.clone().unwrap());
                } else {
                    output.set_output_batch_real(complex_batch_to_real_batch(&layer_input.get_input_batch()));
                }
                output
            }
            LayerEnum::SoftmaxRm(layer) => {
                let mut output = LayerOutput::new_default();
                if !forward_only {
                    layer.operation_mode = OperationMode::TRAINING;
                    let padding_mask = layer_input.get_padding_mask_batch();
                    let padding_mask_option = if padding_mask.is_empty() { None } else { Some(padding_mask) };
                    let _softmax_result = layer.forward_inner(layer_input, padding_mask_option, Some(layer_input.get_target_batch_ids()));
                    output.set_cross_entropy_loss_batch(layer.cross_entropy_loss_batch.clone().unwrap());
                }
                // Inference: the RM logits are consumed here without emitting anything,
                // matching the previous network-level behavior.
                output
            }
        }
    }

    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        match self {
            LayerEnum::AdaptiveAvgPool1d(layer) => layer.backward(previous_gradient),
            LayerEnum::Embedding(layer) => {
                let grad_vec: Vec<Vec<Vec<C>>> = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref().filter(|g| !g.is_empty()) {
                    gr_rm.iter().map(|m| m.to_rows()).collect()
                } else {
                    previous_gradient.get_gradient_input_batch()
                };
                layer.backward_inner(&grad_vec)
            }
            LayerEnum::EmbeddingRm(layer) => {
                let gr_rm = previous_gradient.get_gradient_input_batch_rm_ref().filter(|g| !g.is_empty()).expect("EmbeddingRm expects RM gradients");
                layer.backward_inner(gr_rm)
            }
            LayerEnum::PositionalEncoding(layer) => layer.backward(previous_gradient),
            LayerEnum::PositionalEncodingRm(layer) => {
                let gr_rm = if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                    gr_rm.to_vec()
                } else {
                    previous_gradient.get_gradient_input_batch_rm()
                };
                layer.backward_inner(&gr_rm)
            }
            LayerEnum::Dense(layer) => layer.backward(&previous_gradient.get_gradient_input_batch()),
            LayerEnum::DenseRm(layer) => LayerInterface::backward(&mut **layer, previous_gradient),
            LayerEnum::FeedForward(layer) => layer.backward(previous_gradient),
            LayerEnum::FeedForwardRm(layer) => layer.backward(previous_gradient),
            LayerEnum::RMSNorm(layer) => LayerInterface::backward(&mut **layer, previous_gradient),
            LayerEnum::Norm(layer) => {
                if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                    let mut pg = previous_gradient.clone();
                    pg.set_gradient_input_batch(gr_rm.iter().map(|m| m.to_rows()).collect());
                    pg.set_gradient_input_batch_rm(vec![]);
                    layer.backward(&pg)
                } else {
                    layer.backward(previous_gradient)
                }
            }
            LayerEnum::NormRm(layer) => {
                if previous_gradient.get_gradient_input_batch_rm_ref().is_some() {
                    layer.backward(previous_gradient)
                } else {
                    let mut pg = previous_gradient.clone();
                    let gr_rm = pg.get_gradient_input_batch_rm();
                    pg.set_gradient_input_batch_rm(gr_rm);
                    layer.backward(&pg)
                }
            }
            LayerEnum::SelfAttention(layer) => layer.backward(previous_gradient),
            LayerEnum::SparseSelfAttention(layer) => layer.backward(previous_gradient),
            LayerEnum::SparseSelfAttentionRm(layer) => {
                if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                    layer.backward_rm(gr_rm)
                } else {
                    let previous_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient.get_gradient_input_batch().iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
                    layer.backward_rm(&previous_gradient_batch_rm)
                }
            }
            LayerEnum::SelfAttentionApproximation(layer) => {
                if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                    let previous_gradient_batch: Vec<Vec<Vec<C>>> = gr_rm.iter().map(|m| m.to_rows()).collect();
                    layer.backward(&previous_gradient_batch)
                } else {
                    layer.backward(&previous_gradient.get_gradient_input_batch())
                }
            }
            LayerEnum::SelfAttentionApproximationRm(layer) => {
                if let Some(gr_rm) = previous_gradient.get_gradient_input_batch_rm_ref() {
                    layer.backward_rm(gr_rm)
                } else {
                    let previous_gradient_batch_rm: Vec<RowMajorMatrix<C>> = previous_gradient.get_gradient_input_batch().iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
                    layer.backward_rm(&previous_gradient_batch_rm)
                }
            }
            LayerEnum::Linear(layer) => layer.backward(previous_gradient),
            LayerEnum::LinearRm(layer) => LayerInterface::backward(&mut **layer, previous_gradient),
            LayerEnum::ClusteringLinear(layer) => layer.backward(previous_gradient),
            LayerEnum::AdaptiveLinear(layer) => layer.backward(previous_gradient),
            LayerEnum::Eml(layer) => layer.backward(previous_gradient),
            LayerEnum::SparseLinearRm(layer) => layer.backward(previous_gradient),
            LayerEnum::MultiLinear(layer) => layer.backward(previous_gradient),
            LayerEnum::DiscreteWavelet(layer) => layer.backward(previous_gradient),
            LayerEnum::DiscreteWaveletRm(layer) => {
                if previous_gradient.get_gradient_input_batch_rm_ref().is_some() {
                    layer.backward(previous_gradient)
                } else {
                    let previous_gradient_batch: Vec<RowMajorMatrix<C>> = previous_gradient.get_gradient_input_batch().iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
                    let mut g = Gradient::new_default();
                    g.set_gradient_input_batch_rm(previous_gradient_batch);
                    g.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());
                    layer.backward(&g)
                }
            }
            LayerEnum::ComplexToLinear(layer) => layer.backward(previous_gradient),
            LayerEnum::ComplexToLinearRm(layer) => LayerInterface::backward(&mut **layer, previous_gradient),
            LayerEnum::Wavelet(layer) => layer.backward(previous_gradient),
            LayerEnum::WaveletRm(layer) => {
                if previous_gradient.get_gradient_input_batch_rm_ref().is_some() {
                    layer.backward(previous_gradient)
                } else {
                    let previous_gradient_batch: Vec<RowMajorMatrix<C>> = previous_gradient.get_gradient_input_batch().iter().map(|rows| RowMajorMatrix::from_rows(rows)).collect();
                    let mut g = Gradient::new_default();
                    g.set_gradient_input_batch_rm(previous_gradient_batch);
                    g.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());
                    layer.backward(&g)
                }
            }
            LayerEnum::Softmax(layer) => layer.backward(previous_gradient),
            LayerEnum::SoftmaxRm(layer) => LayerInterface::backward(&mut **layer, previous_gradient),
        }
    }

    fn update_parameters(&mut self) {
        match self {
            LayerEnum::AdaptiveAvgPool1d(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::Embedding(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::EmbeddingRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::PositionalEncoding(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::PositionalEncodingRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::Dense(layer) => layer.update_parameters(),
            LayerEnum::DenseRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::FeedForward(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::FeedForwardRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::RMSNorm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::Norm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::NormRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::SelfAttention(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::SparseSelfAttention(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::SparseSelfAttentionRm(layer) => layer.update_parameters(),
            LayerEnum::SelfAttentionApproximation(layer) => layer.update_parameters(),
            LayerEnum::SelfAttentionApproximationRm(layer) => layer.update_parameters(),
            LayerEnum::Linear(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::LinearRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::ClusteringLinear(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::AdaptiveLinear(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::Eml(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::SparseLinearRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::MultiLinear(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::DiscreteWavelet(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::DiscreteWaveletRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::ComplexToLinear(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::ComplexToLinearRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::Wavelet(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::WaveletRm(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::Softmax(layer) => LayerInterface::update_parameters(&mut **layer),
            LayerEnum::SoftmaxRm(layer) => LayerInterface::update_parameters(&mut **layer),
        }
    }
}

impl Layer {
    pub fn default(rows: usize, cols: usize, learning_rate: &f64) -> Self {
        let mut weights: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        let bias: Vec<C> = vec![C::new(0.001, ZERO); cols];

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
