use crate::neural_networks::utils::weights_initializer::initialize_weights_complex;
use core::fmt::{self, Debug};
use std::fmt::Formatter;
use num::Complex;
use serde::ser::SerializeSeq;
use serde::{de::{SeqAccess, Visitor}, Deserialize, Deserializer, Serialize, Serializer};


// Implement Serialize manually
impl<const N: usize, const M: usize> Serialize for Layer<N, M> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(N * M * 9 + M + 2))?; // 2 for activation_type and layer_type

        // Serialize 2D arrays
        for array in &[
            &self.weights, &self.inactivated_output, &self.activated_output,
            &self.gradient, &self.gradient_w, &self.errors,
            &self.previous_gradient, &self.m1, &self.v1
        ] {
            for row in array.iter() {
                for element in row.iter() {
                    seq.serialize_element(element)?;
                }
            }
        }

        // Serialize 1D array
        for element in &self.layer_bias {
            seq.serialize_element(element)?;
        }

        // Serialize remaining fields
        seq.serialize_element(&self.activation_type)?;
        seq.serialize_element(&self.layer_type)?;

        seq.end() // Finish the sequence
    }
}

// Implement Deserialize manually
impl<'de, const M: usize, const N: usize> Deserialize<'de> for Layer<M, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LayerVisitor<const M: usize, const N: usize>;

        impl<'de, const M: usize, const N: usize> Visitor<'de> for LayerVisitor<M, N> {
            type Value = Layer<M, N>;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("a Layer struct with 2D arrays of Complex<f64>")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Layer<M, N>, A::Error>
            where
                A: SeqAccess<'de>,
            {

                // Initialize arrays
                let mut weights = [[Complex::new(0.0, 0.0); M]; N];
                let mut inactivated_output = [[Complex::new(0.0, 0.0); M]; N];
                let mut activated_output = [[Complex::new(0.0, 0.0); M]; N];
                let mut gradient = [[Complex::new(0.0, 0.0); M]; N];
                let mut gradient_w = [[Complex::new(0.0, 0.0); M]; N];
                let mut errors = [[Complex::new(0.0, 0.0); M]; N];
                let mut previous_gradient = [[Complex::new(0.0, 0.0); M]; N];
                let mut m1 = [[Complex::new(0.0, 0.0); M]; N];
                let mut v1 = [[Complex::new(0.0, 0.0); M]; N];
                let mut layer_bias = [Complex::new(0.0, 0.0); M];

                // Deserialize 2D arrays
                for array in [
                    &mut weights, &mut inactivated_output, &mut activated_output,
                    &mut gradient, &mut gradient_w, &mut errors,
                    &mut previous_gradient, &mut m1, &mut v1,
                ].iter_mut() {
                    for row in array.iter_mut() {
                        for element in row.iter_mut() {
                            *element = seq.next_element()?.ok_or_else(|| {
                                serde::de::Error::invalid_length(N * M, &self)
                            })?;
                        }
                    }
                }

                // Deserialize layer_bias
                for element in layer_bias.iter_mut() {
                    *element = seq.next_element()?.ok_or_else(|| {
                        serde::de::Error::invalid_length(M, &self)
                    })?;
                }

                // Deserialize activation_type and layer_type
                let activation_type: ActivationType = seq.next_element()?.ok_or_else(|| {
                    serde::de::Error::missing_field("activation_type")
                })?;
                let layer_type: LayerType = seq.next_element()?.ok_or_else(|| {
                    serde::de::Error::missing_field("layer_type")
                })?;

                Ok(Layer {
                    weights,
                    layer_bias,
                    activation_type,
                    layer_type,
                    inactivated_output,
                    activated_output,
                    gradient,
                    gradient_w,
                    errors,
                    previous_gradient,
                    m1,
                    v1,
                })
            }
        }

        deserializer.deserialize_seq(LayerVisitor)
    }
}

// Base Layer trait
pub trait BaseLayer<const M: usize, const N: usize>: Debug {
    fn forward(&mut self, input: &[[Complex<f64>; M]; N]);
    fn backward(&mut self, gradient: &[[Complex<f64>; M]; N]);
}

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::TANH // or any other variant you prefer as default
    }
}

// Activation Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    SIGMOID,
    TANH,
    LINEAR,
    RANDOM,
}

// Layer Type Enum
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
}

// Implement Default for LayerType
impl Default for LayerType {
    fn default() -> Self {
        LayerType::InputLayer // or another sensible default
    }
}

// Layer Enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerEnum<const M: usize, const N: usize> {
    Dense(Box<Layer<M, N>>),
    RMSNorm(Box<Layer<M, N>>),
    SelfAttention(Box<Layer<M, N>>),
}

// Layer struct
#[derive(Debug, Clone)]
pub struct Layer<const M: usize, const N: usize>  {
    pub weights: [[Complex<f64>; M]; N],
    pub layer_bias: [Complex<f64>; M],
    pub activation_type: ActivationType,
    pub layer_type: LayerType,
    pub inactivated_output: [[Complex<f64>; M]; N],
    pub activated_output: [[Complex<f64>; M]; N],
    pub gradient: [[Complex<f64>; M]; N],
    pub gradient_w: [[Complex<f64>; M]; N],
    pub errors: [[Complex<f64>; M]; N],
    pub previous_gradient: [[Complex<f64>; M]; N],
    pub m1: [[Complex<f64>; M]; N],
    pub v1: [[Complex<f64>; M]; N],
}

// Implement BaseLayer for Layer struct
impl<const M: usize, const N: usize> BaseLayer<M, N>  for Layer<M, N> {
    fn forward(&mut self, input: &[[Complex<f64>; M]; N]) {
        // Implement forward pass logic for the layer
    }

    fn backward(&mut self, gradient: &[[Complex<f64>; M]; N]) {
        // Implement backward pass logic for the layer
    }
}

impl<const M: usize, const N: usize> Default for Layer<M, N> {
    fn default() -> Self {
        Layer {
            weights: [[Complex::new(0.0, 0.0); M]; N],
            layer_bias: [Complex::new(0.0, 0.0); M],
            activation_type: ActivationType::SIGMOID, // Default activation type
            layer_type: LayerType::InputLayer,         // Default layer type
            inactivated_output: [[Complex::new(0.0, 0.0); M]; N],
            activated_output: [[Complex::new(0.0, 0.0); M]; N],
            gradient: [[Complex::new(0.0, 0.0); M]; N],
            gradient_w: [[Complex::new(0.0, 0.0); M]; N],
            errors: [[Complex::new(0.0, 0.0); M]; N],
            previous_gradient: [[Complex::new(0.0, 0.0); M]; N],
            m1: [[Complex::new(0.0, 0.0); M]; N],
            v1: [[Complex::new(0.0, 0.0); M]; N],
        }
    }
}

// Self-Attention Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionLayer<const M: usize, const N: usize> {
    base_layer: Layer<M, N>,
}

// Implement BaseLayer for SelfAttentionLayer
impl<const M: usize, const N: usize> BaseLayer<M, N> for SelfAttentionLayer<M, N> {
    fn forward(&mut self, input: &[[Complex<f64>; M]; N]) {
        // Implement forward logic for self-attention
    }

    fn backward(&mut self, gradient: &[[Complex<f64>; M]; N]) {
        // Implement backward logic for self-attention
    }
}

// RMSNorm Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormLayer<const M: usize, const N: usize> {
    base_layer: Layer<M, N>,
}

// Implement BaseLayer for RMSNormLayer
impl<const M: usize, const N: usize> BaseLayer<M, N> for RMSNormLayer<M, N> {
    fn forward(&mut self, input: &[[Complex<f64>; M]; N]) {
        // Implement forward logic for RMSNorm
    }

    fn backward(&mut self, gradient: &[[Complex<f64>; M]; N]) {
        // Implement backward logic for RMSNorm
    }
}

// Layer initialization function with Box<dyn BaseLayer>
pub fn initialize_default_layers<const M: usize, const N: usize>(
    num_outputs: &usize,
    num_h_layers: &usize,
    activation: &ActivationType,
) -> Vec<LayerEnum<M, N>> {
    let mut layers: Vec<LayerEnum<M, N>> = Vec::new();
    let total_layers: usize = *num_h_layers + 2;

    if *num_h_layers == 0 {
        panic!("Number of hidden layers cannot be zero.");
    }

    for l in 0..total_layers {
        let layer_type = get_layer_type(l, total_layers);
        let layer = create_default_layer::<M, N>(activation, layer_type);

        layers.push(LayerEnum::Dense(Box::new(layer)));
    }

    layers
}

// Layer creation function
pub fn create_default_layer<const M: usize, const N: usize>(
    activation: &ActivationType,
    layer_type: LayerType,
) -> Layer<M, N> {
    let mut weights: [[Complex<f64>; M]; N] = [[Complex::new(0.0, 0.0); M]; N];

    initialize_weights_complex::<M, N>(&mut weights); // 2D matrix

    Layer {
        weights,
        layer_bias: [Complex::new(0.0, 0.0); M], // 1D vector
        activation_type: activation.clone(),
        layer_type,
        ..Default::default() // Fill the rest with default values
    }
}

// Helper function to determine the type of layer
pub fn get_layer_type(layer_idx: usize, total_layers: usize) -> LayerType {
    match layer_idx {
        0 => LayerType::InputLayer,
        x if x == total_layers - 1 => LayerType::OutputLayer,
        _ => LayerType::HiddenLayer,
    }
}