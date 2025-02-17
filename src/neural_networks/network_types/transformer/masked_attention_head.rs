use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer::LayerType},
    utils::{activation::softmax_complex_padding, matrix::multiply_complex, weights_initializer::initialize_weights_complex},
};

// Layer struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedAttentionHead {
    pub weights_q: Vec<Vec<Complex<f64>>>,
    pub weights_k: Vec<Vec<Complex<f64>>>,
    pub weights_v: Vec<Vec<Complex<f64>>>,

    pub bias_q: Vec<Complex<f64>>,
    pub bias_k: Vec<Complex<f64>>,
    pub bias_v: Vec<Complex<f64>>,

    pub layer_type: LayerType,

    pub gradient: Option<Gradient>,
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub inactivated_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,

    pub m1: Vec<Vec<Complex<f64>>>,
    pub v1: Vec<Vec<Complex<f64>>>,
}

impl MaskedAttentionHead {
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

        MaskedAttentionHead {
            weights_q,
            weights_k,
            weights_v,

            bias_q,
            bias_k,
            bias_v,
            layer_type: LayerType::InputLayer,

            gradient: None,
            input_batch: None,
            inactivated_input_batch: None,
            output_batch: None,
            padding_mask_batch: None,
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
        }
    }

    fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType) -> MaskedAttentionHead {
        let mut attention_layer: MaskedAttentionHead = MaskedAttentionHead::default(rows, cols);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl MaskedAttentionHead {
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, padding_mask_batch: &Vec<Vec<u32>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        // println!("input len in masked_attention : {:?}, {:?}", &input.len(), &input[0].len());
        // println!("weights_q : {:?}, {:?}", &self.weights_q.len(), &self.weights_q[0].len());

        self.input_batch = Some(input_batch.clone());

        let batch_output: Vec<Vec<Vec<Complex<f64>>>> = input_batch
            .par_iter()
            .zip(padding_mask_batch) 
            .map(|(input, padding_mask)| {
                let q = multiply_complex(input, &self.weights_q);
                let k = multiply_complex(input, &self.weights_k);
                let v = multiply_complex(input, &self.weights_v);

                let mask = create_causal_mask(input.len(), input.len());
                let attention_scores = multiply_complex(&q, &k);
                let mut attention_scores_scales = scale_attention_scores(&attention_scores, k[0].len() as f64);

                // Apply the mask to the scaled attention scores
                apply_attention_mask_inplace(&mut attention_scores_scales, &mask);

                let attention_weights = softmax_complex_padding(&attention_scores_scales, padding_mask);
                //  println!("padding mask: {:?}", &padding_mask);
                //  println!("output in attenthion head attention weigths: {:?}", &attention_weights);

                // Multiply attention weights with value (V)
                let output = multiply_complex(&attention_weights, &v);

                // println!("output in attenthion score: {:?}, {:?}", &attention_scores.len(), &attention_scores[0].len());
                // println!("output in q : {:?}, {:?}", &q.len(), &q[0].len());
                // println!("output in attenthion head: {:?}", &output);
                output
            })
            .collect();

        self.output_batch = Some(batch_output.clone());

        batch_output
    }

    pub fn backward(&mut self, previous_gradient: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        let output_batch = self.output_batch.as_ref().expect("Output batch is missing in attention head layer");
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in lattention head inear layer");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Padding mask batch is missing in attention head ");
    
        // Initialize gradients for each parameter (weights and biases)
        let mut gradient_q = vec![vec![Complex::new(0.0, 0.0); self.weights_q[0].len()]; self.weights_q.len()];
        let mut gradient_k = vec![vec![Complex::new(0.0, 0.0); self.weights_k[0].len()]; self.weights_k.len()];
        let mut gradient_v = vec![vec![Complex::new(0.0, 0.0); self.weights_v[0].len()]; self.weights_v.len()];
    
        let mut gradient_attention_weights = vec![vec![Complex::new(0.0, 0.0); self.weights_q.len()]; self.weights_q.len()];
    
        // Initialize the gradient of the attention output
        let mut gradient_attention_output = vec![vec![Complex::new(0.0, 0.0); output_batch[0].len()]; output_batch.len()];
    
        // Gradient of the softmax output
        let mut gradient_attention_weights_softmax = vec![vec![Complex::new(0.0, 0.0); output_batch[0].len()]; output_batch.len()];
    
       



        // Compute the gradients for the parameters and store them
        let mut gradient = Gradient::new_default();
    
        // Optionally, store the gradients for weights (q, k, v) here if needed
    
        self.gradient = Some(gradient.clone());
    
        gradient
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

fn create_causal_mask(rows: usize, cols: usize) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0; cols]; rows]; // Initialize with zeros

    for i in 0..rows {
        for j in 0..=i.min(cols - 1) {
            mask[i][j] = 1; // Allow attention to current and previous tokens
        }
    }

    mask
}

fn apply_attention_mask_inplace(attention_scores: &mut Vec<Vec<Complex<f64>>>, mask: &Vec<Vec<u8>>) {
    let large_negative = Complex::new(-1e9, 0.0);

    for row in 0..attention_scores.len() {
        for col in 0..attention_scores[row].len() {
            if mask[row][col] == 0 {
                attention_scores[row][col] = large_negative; // Apply the mask
            }
        }
    }
}
