use crate::neural_networks::{
    network_components::{
        add_and_norm_layer::NormalNormLayer, add_rms_norm_layer::RMSNormLayer, gradient_struct::Gradient, input_struct::LayerInput, layer::{ActivationType, Layer, LayerEnum, LayerType}, linear_layer::LinearLayer, output_struct::LayerOutput
    },
    utils::matrix::check_nan_or_inf_3d,
};
use num::Complex;
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardLayer {
    pub layers: Vec<LayerEnum>,
    pub norm_layer: Option<LayerEnum>,
    pub learning_rate: f64,
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
}

impl FeedForwardLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let epsilon: f64 = 0.0000000001;

        let mut layers: Vec<LayerEnum> = vec![];
        let dense_layer: Layer = Layer::new(rows, cols, &learning_rate, &ActivationType::GELU, LayerType::DenseLayer);
        let linear_layer = LinearLayer::new(learning_rate, cols, rows);
        let _norm_layer = Some(LayerEnum::Norm(Box::new(NormalNormLayer::new(rows, epsilon, learning_rate))));
        let _rms_norm_layer = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(rows, epsilon, learning_rate))));

        layers.push(LayerEnum::Dense(Box::new(dense_layer)));
        layers.push(LayerEnum::Linear(Box::new(linear_layer)));

        Self {
            layers,
            norm_layer: None,
            learning_rate,
            input_batch: None,
            padding_mask_batch: None,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl FeedForwardLayer {
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = input.get_input_batch();
        self.input_batch = Some(input_batch.clone());
        let mut output: Vec<Vec<Vec<Complex<f64>>>> = input_batch.clone();
        let padding_mask_batch = self.padding_mask_batch.clone().unwrap_or_else(|| vec![vec![1; input_batch[0].len()]; input_batch.len()]);
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        // println!("input batch: {} {} {}", input_batch.len(), input_batch[0].len(), input_batch[0][0].len());

        // Apply all layers sequentially
        for layer in self.layers.iter_mut() {
            match layer {
                LayerEnum::Dense(dense_layer) => {
                    let mut dense_layer_input = LayerInput::new_default();
                    dense_layer_input.set_input_batch(output.clone());
                    dense_layer_input.set_padding_mask_batch(padding_mask_batch.clone());

                    let output_dense = dense_layer.forward(&dense_layer_input);
                    output = output_dense.get_output_batch();
                    //println!("Output FFN Dense layer: {:?}, {:?},  {:?}", &output.len(), &output[0].len(), &output[0][0].len());
                }
                LayerEnum::Linear(linear_layer) => {
                    let mut linear_layer_input = LayerInput::new_default();
                    linear_layer_input.set_input_batch(output.clone());
                    
                    let output_linear = linear_layer.forward(&linear_layer_input);
                    output = output_linear.get_output_batch();
                }
                _ => {}
            }
        }

        // Apply the RMS normalization layer
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    output = rms_norm_layer.forward(&output, &input_batch);
                    //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
                },
                LayerEnum::Norm(norm_layer) => {
                    output = norm_layer.forward(&output, &input_batch);
                    //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
                }
                _ => {}
            }
        }

        
        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output);

        layer_output
    }

    pub fn backward(&mut self, prev_gradients: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let mut output_gradients = prev_gradients.clone();

        let mut gradient = Gradient::new_default();

        //Apply RMSNorm backpropagation if it's present
        if let Some(norm_layer) = &mut self.norm_layer {
            match norm_layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    gradient = rms_norm_layer.backward(&output_gradients);
                    let gradient_input_batch = gradient.get_gradient_input_batch();

                    output_gradients = gradient_input_batch;

                    // println!("FFN, gradient from RMS Norm backward: {}, {}, {}", output_gradients.len(), output_gradients[0].len(), output_gradients[0][0].len());
                },
                LayerEnum::Norm(norm_layer) => {
                    gradient = norm_layer.backward(&output_gradients);
                    let gradient_input_batch = gradient.get_gradient_input_batch();

                    output_gradients = gradient_input_batch;

                    // println!("FFN, gradient from Norm backward: {}, {}, {}", output_gradients.len(), output_gradients[0].len(), output_gradients[0][0].len());
                }
                _ => {}
            }
        }

        // forward -> Dense, Linear
        // backward -> Linear, Dense
        for layer in self.layers.iter_mut().rev() {
            match layer {
                LayerEnum::Dense(dense) => {
                    let dense_layer = Some(dense).unwrap();

                    check_nan_or_inf_3d(&mut output_gradients, "previous gradients in ffn dense layer has None values");
                    gradient = dense_layer.backward(&output_gradients);
                    let mut gradient_input_batch = gradient.get_gradient_input_batch();

                    // println!("Gradient input batch FFN Dense Layer: {:?}, {:?},  {:?}", &gradient_input_batch.len(), &gradient_input_batch[0].len(), &gradient_input_batch[0][0].len());
                    check_nan_or_inf_3d(&mut gradient_input_batch, "output gradients in ffn dense layer has None values");
                    output_gradients = gradient_input_batch;
                }
                LayerEnum::Linear(linear) => {
                    let linear_layer = Some(linear).unwrap();

                    check_nan_or_inf_3d(&mut output_gradients, "output gradients in linear layer has None values:");
                    gradient = linear_layer.backward(&output_gradients);
                    let gradient_input_batch = gradient.get_gradient_input_batch();

                    // println!("Gradient input batch FFN Linear Layer: {:?}, {:?},  {:?}", &gradient_input_batch.len(), &gradient_input_batch[0].len(), &gradient_input_batch[0][0].len());
                    output_gradients = gradient_input_batch;
                }
                _ => {}
            }
        }

        gradient
    }

    pub fn update_parameters(&mut self) {
        // Apply RMSNorm backpropagation if it's present
        if let Some(layer_enum) = &mut self.norm_layer {
            match layer_enum {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    rms_norm_layer.update_parameters();
                },
                LayerEnum::Norm(norm_layer) => {
                    norm_layer.update_parameters();
                },
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
