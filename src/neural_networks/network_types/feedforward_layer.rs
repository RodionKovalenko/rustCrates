use crate::neural_networks::network_components::{
    add_rms_norm_layer::RMSNormLayer,
    gradient_struct::Gradient,
    layer::{ActivationType, Layer, LayerEnum, LayerType},
    linear_layer::LinearLayer,
};
use num::Complex;
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardLayer {
    pub layers: Vec<LayerEnum>,
    pub rms_norm_layer: Option<LayerEnum>,
    pub learning_rate: f64,
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
}

impl FeedForwardLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let epsilon: f64 = 0.000001;

        let mut layers: Vec<LayerEnum> = vec![];
        let dense_layer: Layer = Layer::new(rows, cols, &learning_rate, &ActivationType::GELU, LayerType::DenseLayer);
        let linear_layer = LinearLayer::new(learning_rate, cols, rows);
        let rms_norm_layer = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(rows, epsilon, learning_rate))));

        layers.push(LayerEnum::Dense(Box::new(dense_layer)));
        layers.push(LayerEnum::Linear(Box::new(linear_layer)));

        Self {
            layers,
            rms_norm_layer,
            learning_rate,
            input_batch: None,
            padding_mask_batch: None,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl FeedForwardLayer {
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        self.input_batch = Some(input_batch.clone());
        let mut output: Vec<Vec<Vec<Complex<f64>>>> = input_batch.clone();
        let padding_mask_batch = self.padding_mask_batch.clone().unwrap_or_else(|| vec![vec![1; input_batch[0].len()]; input_batch.len()]);
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        // Apply all layers sequentially
        for layer in self.layers.iter_mut() {
            match layer {
                LayerEnum::Dense(dense_layer) => {
                    let output_dense = dense_layer.forward(&output, &padding_mask_batch);

                    //println!("Output FFN Dense layer: {:?}, {:?},  {:?}", &output_dense.len(), &output_dense[0].len(), &output_dense[0][0].len());
                    output = output_dense;
                }
                LayerEnum::Linear(linear_layer) => {
                    let output_linear = linear_layer.forward(&output);

                    //println!("Output FFN Linear layer: {:?}, {:?},  {:?}", &output_linear.len(), &output_linear[0].len(), &output_linear[0][0].len());
                    output = output_linear;
                }
                _ => {}
            }
        }

        // Apply the RMS normalization layer
        let rms_norm_layer_enum = self.rms_norm_layer.as_mut().unwrap();
        match rms_norm_layer_enum {
            LayerEnum::RMSNorm(rms_norm_layer) => {
                output = rms_norm_layer.forward(&output, &input_batch);
                //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
            }
            _ => {}
        }

        output
    }

    pub fn backward(&mut self, prev_gradients: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let mut output_gradients = prev_gradients.clone();

        let mut gradient = Gradient::new_default();

        //Apply RMSNorm backpropagation if it's present
        if let Some(rms_norm_layer) = &mut self.rms_norm_layer {
            match rms_norm_layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    gradient = rms_norm_layer.backward(&output_gradients);
                    output_gradients = gradient.get_gradient_input_batch();

                    println!("FFN, gradient from RMS Norm backward: {}, {}, {}", output_gradients.len(), output_gradients[0].len(), output_gradients[0][0].len());
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
                    gradient = dense_layer.backward(&output_gradients);
                    let gradient_input_batch = gradient.get_gradient_input_batch();

                    println!("Gradient input batch FFN Dense Layer: {:?}, {:?},  {:?}", &gradient_input_batch.len(), &gradient_input_batch[0].len(), &gradient_input_batch[0][0].len());
                    output_gradients = gradient_input_batch;
                }
                LayerEnum::Linear(linear) => {
                    let linear_layer = Some(linear).unwrap();
                    gradient = linear_layer.backward(&output_gradients);
                    let gradient_input_batch = gradient.get_gradient_input_batch();

                    println!("Gradient input batch FFN Linear Layer: {:?}, {:?},  {:?}", &gradient_input_batch.len(), &gradient_input_batch[0].len(), &gradient_input_batch[0][0].len());
                    output_gradients = gradient_input_batch;
                }
                _ => {}
            }
        }

        gradient
    }

    pub fn update_parameters(&mut self) {
        // Apply RMSNorm backpropagation if it's present
        if let Some(rms_norm_layer) = &mut self.rms_norm_layer {
            match rms_norm_layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                    rms_norm_layer.update_parameters();
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
