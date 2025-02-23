use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{
        gradient_struct::Gradient,
        layer::{ActivationType, LayerType},
    },
    utils::{
        activation::{activate_output_complex, softmax_complex_padding},
        derivative::{backpropagate_softmax_masked, softmax_derivative_complex_jacobian},
        matrix::{multiply_complex, transpose},
        weights_initializer::initialize_weights_complex,
    },
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
    pub attention_weights_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub inactivated_input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,

    pub m1: Vec<Vec<Complex<f64>>>,
    pub v1: Vec<Vec<Complex<f64>>>,
}

impl MaskedAttentionHead {
    pub fn new(rows: usize, cols: usize) -> Self {
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
            attention_weights_batch: None,
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
        }
    }

    fn set_layer_type(&mut self, layer_type: LayerType) {
        self.layer_type = layer_type;
    }

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType) -> MaskedAttentionHead {
        let mut attention_layer: MaskedAttentionHead = MaskedAttentionHead::new(rows, cols);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl MaskedAttentionHead {
    // Input shape e.g. [2][5] and out shape of weights [5][4] => we get final output [2][4]
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, padding_mask_batch: &Vec<Vec<u32>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        // println!("input len in masked_attention : {:?}, {:?}", &input.len(), &input[0].len());
        // println!("weights_q : {:?}, {:?}", &self.weights_q.len(), &self.weights_q[0].len());

        self.input_batch = Some(input_batch.clone());
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        let attention_weights_batch_tuple: (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<Vec<Complex<f64>>>>) = input_batch
            .par_iter()
            .zip(padding_mask_batch)
            .map(|(input, padding_mask)| {
                let q = multiply_complex(input, &self.weights_q);
                let k = multiply_complex(input, &self.weights_k);
                let v = multiply_complex(input, &self.weights_v);

                //[[1, 0], [1, 1]]
                let mask = create_causal_mask(input.len(), input.len());
                let attention_scores = multiply_complex(&q, &transpose(&k));
                let mut attention_scores_scales = scale_attention_scores(&attention_scores, k[0].len() as f64);

                // Apply the mask to the scaled attention scores
                apply_attention_mask_inplace(&mut attention_scores_scales, &mask);

                let attention_weights = softmax_complex_padding(&attention_scores_scales, padding_mask);

                (attention_weights, v)
            })
            .collect();

        let batch_output: Vec<Vec<Vec<Complex<f64>>>> = attention_weights_batch_tuple
            .par_iter()
            .map(|(attention_weights, v)| {
                let output = multiply_complex(&attention_weights, &v);
                output
            })
            .collect();

        let (attention_weights_batch, _) = attention_weights_batch_tuple;
        self.attention_weights_batch = Some(attention_weights_batch);
        self.output_batch = Some(batch_output.clone());

        batch_output
    }

    pub fn backward(&mut self, previous_gradient_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Gradient {
        // Input shape e.g. [2][5] and out shape of weights [5][4] => we get final output [2][4]
        let output_batch = self.output_batch.as_ref().expect("Output batch is missing in attention head layer");
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in lattention head inear layer");
        let padding_mask_batch = self.padding_mask_batch.as_ref().expect("Padding mask batch is missing in attention head ");

        // dimensions [seq_len][seq_len] -> A
        let attention_weights_batch = self.attention_weights_batch.as_ref().expect("Attention weights batch is missing in attention head");

        let batch_size = output_batch.len();

        // Initialize gradients for each parameter (weights and biases)
        let mut input_gradient_batch = vec![vec![vec![Complex::new(0.0, 0.0); previous_gradient_batch[0][0].len()]; previous_gradient_batch[0].len()]; input_batch.len()];
        let mut gradient_q_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_v[0].len()]; self.weights_v.len()]; batch_size];

        for (batch_ind, previous_gradient) in previous_gradient_batch.iter().enumerate() {
            // Compute gradient of Wv
            let grad_wv = multiply_complex(&attention_weights_batch[batch_ind], &input_batch[batch_ind]);
            gradient_v_batch[batch_ind] = multiply_complex(&grad_wv, &previous_gradient);

            let q: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_q);
            let k: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_k);
            let v: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_v);

            let d_k = k[0].len() as f64;
            // Compute gradient of k_scaled w.r.t. k
            let k_scaled: Vec<Vec<Complex<f64>>> = scale_attention_scores(&transpose(&k), d_k);
            let q_scaled: Vec<Vec<Complex<f64>>> = scale_attention_scores(&q, d_k);

            // Compute activation derivative softmax
            let softmax_derivative: Vec<Vec<Vec<Complex<f64>>>> = softmax_derivative_complex_jacobian(&attention_weights_batch[batch_ind]);

            /*
                A = Q*KT/sqtr(dk)
                S = sigma(A) * V
                O = S * V

                dl/ds = Gt * VT 
                dl/dq = dl/ds * ds/dq

                dl/dwq = XT * dl/dq
             */

            // Gradient Wq
            // 2, 4 * 2, 4 = 2,2
            let dl_ds: Vec<Vec<Complex<f64>>> = multiply_complex(previous_gradient, &v);
            // 2,2 * 2,2  = 2,2
            let dl_da: Vec<Vec<Complex<f64>>> = backpropagate_softmax_masked(&softmax_derivative, &dl_ds, &padding_mask_batch[batch_ind]);
            // 2,2 * 4, 2 = 2,4
            let dl_dq: Vec<Vec<Complex<f64>>> = multiply_complex(&dl_da, &k_scaled);
            // 2,5 * 2, 4 = 4,5
            let dl_dwq: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &dl_dq);
            gradient_q_batch[batch_ind] = dl_dwq;

            // Gradient Wk
            // 2, 4 * 2,2 = 4,2
            let dl_dk: Vec<Vec<Complex<f64>>> = multiply_complex(&q_scaled, &dl_da) ;
            // 2,5 * 4,2 = 5, 4
            let dl_dwk: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &transpose(&dl_dk));
            gradient_k_batch[batch_ind] = dl_dwk;
        }

        // Compute the gradients for the parameters and store them
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_weights_v_batch(gradient_v_batch);
        gradient.set_gradient_weights_q_batch(gradient_q_batch);
        gradient.set_gradient_weights_k_batch(gradient_k_batch);

        self.gradient = Some(gradient.clone());

        gradient
    }
    pub fn update_parameters(&mut self) {}
}

fn scale_attention_scores(attention_scores: &Vec<Vec<Complex<f64>>>, d_k: f64) -> Vec<Vec<Complex<f64>>> {
    let scaling_factor = 1.0 / d_k.sqrt();
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
