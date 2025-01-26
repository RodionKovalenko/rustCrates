use crate::neural_networks::network_components::{
    add_rms_norm_layer::RMSNormLayer,
    layer::{create_default_layer, ActivationType, Layer, LayerEnum, LayerType},
};
use num::Complex;
use serde::{Deserialize, Serialize};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardLayer {
    pub layers: Vec<Layer>,
    pub rms_norm_layer: Option<LayerEnum>,
    pub learning_rate: f64,
}

impl FeedForwardLayer {
    // Constructor to initialize multiple attention heads
    pub fn new(rows: usize, cols: usize, output_cols: usize, learning_rate: f64) -> Self {
        let epsilon: f64 = 0.000001;

        let mut layers: Vec<Layer> = vec![];
        let dense_layer: Layer = create_default_layer(rows, cols, &ActivationType::GELU, LayerType::DenseLayer);
        let linear_layer: Layer = create_default_layer(cols, output_cols, &ActivationType::LINEAR, LayerType::LinearLayer);
        let rms_norm_layer = Some(LayerEnum::RMSNorm(Box::new(RMSNormLayer::new(cols, epsilon, learning_rate))));

        layers.push(dense_layer);
        layers.push(linear_layer);

        Self {
            layers,
            rms_norm_layer,
            learning_rate,
        }
    }
}

// Implement BaseLayer for SelfAttentionLayer
impl FeedForwardLayer {
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        let mut output: Vec<Vec<Vec<Complex<f64>>>> = input_batch.clone();

        // Apply all layers sequentially
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
            // Debugging information (optional):
            println!("layer weights in ffn: {:?}, {}, {}", &layer.layer_type, layer.weights.len(), layer.weights[0].len());
            println!("output in ffn layer: {:?}, {}, {}", &layer.layer_type, output.len(), output[0].len());
        }

       
        // Apply the RMS normalization layer
        let rms_norm_layer_enum = self.rms_norm_layer.as_mut().unwrap();
        match rms_norm_layer_enum {
            LayerEnum::RMSNorm(rms_norm_layer) => {
                let rms_norm_layer = Some(rms_norm_layer).unwrap();
               
                output = rms_norm_layer.forward(&output, input_batch);
                //println!("RMS NORM input in ffn: {:?}, {:?}", &output.len(), &output[0].len());
            }
            _ => {}
        }

        output
    }

    pub fn backward(&mut self, gradients: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let mut output_gradients = gradients.clone();

        // Apply RMSNorm backpropagation if it's present
        if let Some(rms_norm_layer) = &mut self.rms_norm_layer {
            match rms_norm_layer {
                LayerEnum::RMSNorm(rms_norm_layer) => {
                     output_gradients = rms_norm_layer.backward(&output_gradients);
                }
                _ => {}
            }
        }

        // Backpropagate through the linear layer (second layer)
        let mut linear_layer_gradients = output_gradients.clone();
        if let Some(linear_layer) = self.layers.get(1) {
            linear_layer_gradients = linear_layer.backward(&output_gradients);
        }

        // Backpropagate through the dense layer (first layer with GELU activation)
        let mut dense_layer_gradients = linear_layer_gradients.clone();
        if let Some(dense_layer) = self.layers.get(0) {
            dense_layer_gradients = dense_layer.backward(&dense_layer_gradients);
            // The GELU activation's gradient will also need to be computed here.
            dense_layer_gradients = self.apply_activation_gradient(&dense_layer_gradients);
        }

        dense_layer_gradients
    }

    fn apply_activation_gradient(&self, gradients: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        // This applies the gradient of the GELU activation function.
        // You will need to implement the GELU gradient or use an existing method in your framework.
        gradients.clone() // Placeholder for actual GELU gradient computation
    }
}
