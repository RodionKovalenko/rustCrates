use num::Complex;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::layer::{BaseLayer, LayerType},
    utils::{activation::softmax_complex, matrix::multiply_complex, weights_initializer::initialize_weights_complex},
};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHead {
    pub weights_q: Vec<Vec<Complex<f64>>>,
    pub weights_k: Vec<Vec<Complex<f64>>>,
    pub weights_v: Vec<Vec<Complex<f64>>>,

    pub bias_q: Vec<Complex<f64>>,
    pub bias_k: Vec<Complex<f64>>,
    pub bias_v: Vec<Complex<f64>>,

    pub layer_type: LayerType,

    pub inactivated_output: Vec<Vec<Complex<f64>>>,
    pub activated_output: Vec<Vec<Complex<f64>>>,
    pub gradient: Vec<Vec<Complex<f64>>>,
    pub gradient_w: Vec<Vec<Complex<f64>>>,
    pub errors: Vec<Vec<Complex<f64>>>,
    pub previous_gradient: Vec<Vec<Complex<f64>>>,

    pub m1: Vec<Vec<Complex<f64>>>,
    pub v1: Vec<Vec<Complex<f64>>>,
}

impl AttentionHead {
    fn default(rows: usize, cols: usize) -> Self {
        let mut weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];
        let mut weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

        initialize_weights_complex(rows, cols, &mut weights_q);
        initialize_weights_complex(rows, cols, &mut weights_k);
        initialize_weights_complex(rows, cols, &mut weights_v);

        let bias_q: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let bias_k: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];
        let bias_v: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); cols];

        AttentionHead {
            weights_q,
            weights_k,
            weights_v,

            bias_q,
            bias_k,
            bias_v,
            layer_type: LayerType::InputLayer,

            inactivated_output: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            activated_output: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            gradient: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            gradient_w: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            errors: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            previous_gradient: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
        }
    }

    fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }

    pub fn create_default_attention_layer(
        rows: usize,
        cols: usize,
        layer_type: LayerType,
    ) -> AttentionHead {
        let mut attention_layer: AttentionHead = AttentionHead::default(rows, cols);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl BaseLayer for AttentionHead {
    fn forward(&self, input: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
       // println!("output in weights_q : {:?}, {:?}", &self.weights_q.len(), &self.weights_q[0].len());

        let q = multiply_complex(input, &self.weights_q);
        let k = multiply_complex(input, &self.weights_k);
        let v = multiply_complex(input, &self.weights_v);

        let attention_scores = multiply_complex(&q, &k);
        let attention_scores_scales = scale_attention_scores(&attention_scores, k[0].len() as f64);
        let attention_weights = softmax_complex(&attention_scores_scales);
      
        // Multiply attention weights with value (V)
        let output = multiply_complex(&attention_weights, &v);

        // println!("output in attenthion score: {:?}, {:?}", &attention_scores.len(), &attention_scores[0].len());
        // println!("output in q : {:?}, {:?}", &q.len(), &q[0].len());
        // println!("output in attenthion head attention weigths: {:?}, {:?}", &attention_weights.len(), &attention_weights[0].len());
        // println!("output in attenthion head: {:?}", &output);
        output
    }

    fn backward(&self, _gradient: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
        // Implement backward pass logic for the layer

        self.gradient.clone()
    }
}

fn scale_attention_scores(attention_scores: &Vec<Vec<Complex<f64>>>, d_k: f64) -> Vec<Vec<Complex<f64>>> {
    let scaling_factor = Complex::new(1.0 / d_k.sqrt(), 0.0); // Scaling by 1 / sqrt(d_k)
    let mut scaled_scores = attention_scores.clone();

    // Scale each attention score
    for row in 0..scaled_scores.len() {
        for col in 0..scaled_scores[row].len() {
            scaled_scores[row][col] = scaled_scores[row][col] * scaling_factor;
        }
    }

    scaled_scores
}