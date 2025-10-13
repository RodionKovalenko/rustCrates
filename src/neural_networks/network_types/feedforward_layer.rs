use crate::neural_networks::{
    network_components::{
        add_rms_norm_layer::RMSNormLayer,
        gradient_struct::Gradient,
        layer::{ActivationType, Layer, LayerEnum, LayerType},
        layer_input_struct::LayerInput,
        layer_output_struct::LayerOutput,
        linear_layer::LinearLayer,
        norm_layer::NormalNormLayer,
    },
    utils::matrix::add_matrix_3d,
};
use num::Complex;
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardLayer {
    pub layers: Vec<LayerEnum>,
    pub norm_layer: Option<LayerEnum>,
    pub gradient: Option<Gradient>,
    pub learning_rate: f64,

    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl FeedForwardLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let epsilon: f64 = 0.00000001;
        let mut cols_swinglu = cols;
        let activation = ActivationType::SWiGLU;

        if activation == ActivationType::SWiGLU {
            cols_swinglu *= 2;
        }

        let mut layers: Vec<LayerEnum> = vec![];
        let dense_layer: Layer = Layer::new(rows, cols_swinglu, &learning_rate, &activation, LayerType::DenseLayer);

        let _dense_layer_2: Layer = Layer::new(cols, rows, &learning_rate, &ActivationType::GELU, LayerType::DenseLayer);
        let _linear_layer = LinearLayer::new(learning_rate, cols, rows);
        let _norm_layer = Some(LayerEnum::Norm(Box::new(NormalNormLayer::new(rows, epsilon, learning_rate))));
        let _rms_norm_layer = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(rows, epsilon, learning_rate))));

        layers.push(LayerEnum::Dense(Box::new(dense_layer)));
        // layers.push(LayerEnum::Dense(Box::new(_dense_layer_2)));
        layers.push(LayerEnum::Linear(Box::new(_linear_layer)));

        Self {
            layers,
            gradient: None,
            norm_layer: _norm_layer,
            learning_rate,
            input_batch: None,
            output_batch: None,
            padding_mask_batch: None,
            time_step: 0,
            batch_size: 0,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl FeedForwardLayer {
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = input.get_input_batch();
        self.batch_size = input.get_batch_size();

        self.input_batch = Some(input_batch.clone());
        self.time_step = input.get_time_step();

        let mut output: Vec<Vec<Vec<Complex<f64>>>> = input_batch.clone();
        let mut padding_mask_batch = input.get_padding_mask_batch();

        if padding_mask_batch.is_empty() {
            padding_mask_batch = vec![vec![1; input_batch[0].len()]; input_batch.len()];
        }
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        // Apply the RMS normalization layer
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    let rms_output = rms_norm_layer.forward(input);
                    output = rms_output.get_output_batch();
                    //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
                }
                LayerEnum::Norm(norm_layer) => {
                    let layer_output = norm_layer.forward(input);
                    output = layer_output.get_output_batch();

                    //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
                }
                _ => {}
            }
        }

        let mut layer_input = input.clone();
        layer_input.set_input_batch(output.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        // println!("padding mask in feedforward layer: {:?}", padding_mask_batch);

        // Apply all layers sequentially
        for layer in self.layers.iter_mut() {
            match layer {
                LayerEnum::Dense(dense_layer) => {
                    layer_input.set_input_batch(output.clone());
                    let output_dense = dense_layer.forward(&layer_input);
                    output = output_dense.get_output_batch();
                    //println!("Output FFN Dense layer: {:?}, {:?},  {:?}", &output.len(), &output[0].len(), &output[0][0].len());
                }
                LayerEnum::Linear(linear_layer) => {
                    layer_input.set_input_batch(output.clone());

                    // println!("padding mask in linear layer: {:?}", padding_mask_batch);
                    let output_linear = linear_layer.forward(&layer_input);
                    output = output_linear.get_output_batch();
                    // println!("gradient input batch in linear layer: {} {} {}", input_gradient_batch.len(), input_gradient_batch[0].len(), input_gradient_batch[0][0].len());
                }
                _ => {}
            }
        }

        // Residual connection
        output = add_matrix_3d(&output, &input_batch);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output.clone());
        self.output_batch = Some(output);

        layer_output
    }

    pub fn backward(&mut self, prev_gradients: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let mut output_gradients = prev_gradients.clone();

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch(prev_gradients.clone());

        // forward -> Dense, Linear
        // backward -> Linear, Dense
        for layer in self.layers.iter_mut().rev() {
            match layer {
                LayerEnum::Dense(dense_layer) => {
                    gradient = dense_layer.backward(&output_gradients);
                    output_gradients = gradient.get_gradient_input_batch();
                    //println!("Gradient input batch FFN Dense Layer: {:?}, {:?},  {:?}", &output_gradients.len(), &output_gradients[0].len(), &output_gradients[0][0].len());
                }
                LayerEnum::Linear(linear_layer) => {
                    gradient = linear_layer.backward(&gradient);
                    output_gradients = gradient.get_gradient_input_batch();
                    //println!("Gradient input batch FFN Linear Layer: {:?}, {:?},  {:?}", &output_gradients.len(), &output_gradients[0].len(), &output_gradients[0][0].len());
                }
                _ => {}
            }
        }

        gradient.set_gradient_input_batch(output_gradients.clone());

        //Apply RMSNorm backpropagation if it's present
        if let Some(norm_layer) = &mut self.norm_layer {
            match norm_layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    gradient = rms_norm_layer.backward(&output_gradients);
                    output_gradients = gradient.get_gradient_input_batch();
                    // println!("FFN, gradient from RMS Norm backward: {}, {}, {}", output_gradients.len(), output_gradients[0].len(), output_gradients[0][0].len());
                }
                LayerEnum::Norm(norm_layer) => {
                    gradient = norm_layer.backward(&gradient);
                    output_gradients = gradient.get_gradient_input_batch();
                    //println!("FFN, gradient from Norm backward: {}, {}, {}", output_gradients.len(), output_gradients[0].len(), output_gradients[0][0].len());
                }
                _ => {}
            }
        }

        output_gradients = add_matrix_3d(&prev_gradients, &output_gradients);

        gradient.set_gradient_input_batch(output_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn update_parameters(&mut self) {
        // Apply RMSNorm backpropagation if it's present
        if let Some(layer_enum) = &mut self.norm_layer {
            match layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    rms_norm_layer.update_parameters();
                }
                LayerEnum::Norm(norm_layer) => {
                    norm_layer.update_parameters();
                }
                _ => {}
            }
        }

        //1. Linear
        //2. Dense
        for layer in self.layers.iter_mut().rev() {
            match layer {
                LayerEnum::Dense(dense) => {
                    dense.update_parameters();
                }
                LayerEnum::Linear(linear) => {
                    linear.update_parameters();
                }
                _ => {}
            }
        }
    }
}
