use crate::neural_networks::network_components::{
    add_rms_norm_layer::RMSNormLayer,
    layer::{create_default_layer, ActivationType, BaseLayer, Layer, LayerEnum, LayerType},
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
impl BaseLayer for FeedForwardLayer {
    fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        let mut output: Vec<Vec<Complex<f64>>> = input.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
            // println!("layer in ffn: {:?}, {}, {}", &layer.layer_type, layer.weights.len(), layer.weights[0].len());
            // println!("output in ffn layer: {:?}, {}, {}", &layer.layer_type, output.len(), output[0].len());
        }

        let rms_norm_layer = &self.rms_norm_layer.as_ref().unwrap();
        match rms_norm_layer {
            LayerEnum::RMSNorm(rms_norm_layer) => {
                let rms_norm_layer = Some(rms_norm_layer).unwrap();
                let previous_output = &output;
                println!("Previous output: {:?}, {:?}", &previous_output.len(), &previous_output[0].len());

                let output_rms = rms_norm_layer.forward(&previous_output, &input);

                println!("RMS output in ffn: {:?}, {:?}", &output_rms.len(), &output_rms[0].len());

                //println!("RMS output: {:?}", &output_rms);
                output = output_rms;
            }
            _ => {}
        }

        output
    }

    fn backward(&self, gradients: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        gradients.clone()
    }
}
