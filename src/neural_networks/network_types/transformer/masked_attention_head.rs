use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer::LayerType},
    utils::{
        activation::softmax_complex_padding,
        derivative::{backpropagate_softmax_masked, softmax_derivative_complex_jacobian},
        matrix::{add_matrix, check_nan_or_inf, clip_gradients, multiply_complex, transpose},
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
    pub learning_rate: f64,

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
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
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
            learning_rate: learning_rate,

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

    pub fn create_default_attention_layer(rows: usize, cols: usize, layer_type: LayerType, learning_rate: f64) -> MaskedAttentionHead {
        let mut attention_layer: MaskedAttentionHead = MaskedAttentionHead::new(rows, cols, learning_rate);
        attention_layer.set_layer_type(layer_type);

        attention_layer
    }
}

// Implement BaseLayer for Layer struct
impl MaskedAttentionHead {
    // Input shape e.g. [2][5] and out shape of weights [5][4] => we get final output [2][4]
    pub fn forward(&mut self, input_batch: &Vec<Vec<Vec<Complex<f64>>>>, padding_mask_batch: &Vec<Vec<u32>>) -> Vec<Vec<Vec<Complex<f64>>>> {
        // println!("input len in masked_attention : {:?}, {:?}, {}", &input_batch.len(), &input_batch[0].len(), &input_batch[0][0].len());
        // println!("weights_q : {:?}, {:?}", &self.weights_q.len(), &self.weights_q[0].len());

        self.input_batch = Some(input_batch.clone());
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        let (attention_weights_batch, v_batch): (Vec<Vec<Vec<Complex<f64>>>>, Vec<Vec<Vec<Complex<f64>>>>) = input_batch
            .par_iter()
            .zip(padding_mask_batch)
            .map(|(input, padding_mask)| {
                // check_nan_or_inf(&mut self.weights_q, "check weights q in attention head");
                // check_nan_or_inf(&mut self.weights_k, "check weights k in attention head");
                // check_nan_or_inf(&mut self.weights_v, "check weights v in attention head");

                // 1, 16 * 16, 4 = 1, 4
                let mut q = multiply_complex(input, &self.weights_q);
                let mut k = multiply_complex(input, &self.weights_k);
                let mut v = multiply_complex(input, &self.weights_v);

                check_nan_or_inf(&mut q, "check q in attention head");
                check_nan_or_inf(&mut k, "check k in attention head");
                check_nan_or_inf(&mut v, "check v in attention head");

                //[[1, 0], [1, 1]]
                let mask: Vec<Vec<u8>> = create_causal_mask(input.len(), input.len());

                // 1, 4 * 4, 1 = 1, 1
                let mut attention_scores = multiply_complex(&q, &transpose(&k));
                // println!("q in attention head: {:?}", &q);
                // println!("k in attention head: {:?}", &transpose(&k));
                check_nan_or_inf(&mut attention_scores, "check attention scores");
                // 1,1
                let mut attention_scores_scales = scale_attention_scores(&attention_scores, k[0].len() as f64);

                check_nan_or_inf(&mut attention_scores_scales, "check attention_scores_scales");

                // Apply the mask to the scaled attention scores
                // 1,1
                apply_attention_mask_inplace(&mut attention_scores_scales, &mask);

                check_nan_or_inf(&mut attention_scores_scales, "check attention_scores_scales mask inplace");

                // 1,1
                let mut attention_weights = softmax_complex_padding(&attention_scores_scales, padding_mask);

                check_nan_or_inf(&mut attention_weights, "check attention_weights in attention head forward");

                (attention_weights, v)
            })
            .collect();

        let batch_output: Vec<Vec<Vec<Complex<f64>>>> = attention_weights_batch
            .par_iter()
            .zip(v_batch)
            .map(|(attention_weights, v)| {
                // 1, 1 * 1, 4 = 1, 4
                let mut output = multiply_complex(&attention_weights, &v);

                check_nan_or_inf(&mut output, "check output in masked attention head");
                output
            })
            .collect();

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
        let mut gradient_input_batch = vec![vec![vec![Complex::new(0.0, 0.0); previous_gradient_batch[0][0].len()]; previous_gradient_batch[0].len()]; input_batch.len()];
        let mut gradient_q_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_q[0].len()]; self.weights_q.len()]; batch_size];
        let mut gradient_k_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_k[0].len()]; self.weights_k.len()]; batch_size];
        let mut gradient_v_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); self.weights_v[0].len()]; self.weights_v.len()]; batch_size];

        for (batch_ind, previous_gradient) in previous_gradient_batch.iter().enumerate() {
            let q: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_q);
            let k: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_k);
            let v: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &self.weights_v);

            /*
               A = Q*KT/sqtr(dk)
               S = sigma(A) * V
               O = S * V

               dl/ds = Gt * VT
               dl/da = Gt * VT * ds/da = dl/ds * ds/da
               dl/dq = dl/ds * ds/da * da/dq = dl/da  * da/dq
               dl/dwq = XT * dl/ds * ds/da * da/dq = XT * dl/dq

               => dl/dwq = XT * (Gt * VT * grad(A) * Kt/sqtr(dk))
            */

            // 2, 4 * 2, 4 = 2,2
            let dl_ds: Vec<Vec<Complex<f64>>> = multiply_complex(previous_gradient, &v);

            // Compute gradient of Wv
            // 2,4 * 2, 2 = 4, 2
            let grad_wv = multiply_complex(&previous_gradient, &attention_weights_batch[batch_ind]);
            // 5,2 * 4, 2 = 5, 2 * 2, 4 = 5, 4
            gradient_v_batch[batch_ind] = multiply_complex(&transpose(&input_batch[batch_ind]), &grad_wv);

            let d_k = k[0].len() as f64;
            // Compute gradient of k_scaled w.r.t. k
            let k_scaled: Vec<Vec<Complex<f64>>> = scale_attention_scores(&transpose(&k), d_k);
            let q_scaled: Vec<Vec<Complex<f64>>> = scale_attention_scores(&q, d_k);

            // Compute activation derivative softmax
            let softmax_derivative: Vec<Vec<Vec<Complex<f64>>>> = softmax_derivative_complex_jacobian(&attention_weights_batch[batch_ind]);

            // println!("softmax_derivative dim: {}, {}, {}", &softmax_derivative.len(), &softmax_derivative[0].len(),  &softmax_derivative[0][0].len());

            // Gradient Wq
            // 2,2 * 2,2  = 2,2
            let dl_da: Vec<Vec<Complex<f64>>> = backpropagate_softmax_masked(&softmax_derivative, &dl_ds, &padding_mask_batch[batch_ind]);
            // println!("dl_da dim: {}, {}", &dl_da.len(), &dl_da[0].len(),);
            // 2,2 * 4, 2 = 2,4
            let dl_dq: Vec<Vec<Complex<f64>>> = multiply_complex(&dl_da, &k_scaled);
            // println!("dl_dq dim: {}, {}", &dl_dq.len(), &dl_dq[0].len(),);
            // 2,2 * 4, 2 = 2,4
            // 2,5 * 2, 4 = 4,5
            let dl_dwq: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &dl_dq);
            // println!("dl_dwq dim: {}, {}", &dl_dwq.len(), &dl_dwq[0].len());
            gradient_q_batch[batch_ind] = dl_dwq;

            // Gradient Wk
            // 2, 4 * 2,2 = 4,2
            let dl_dk: Vec<Vec<Complex<f64>>> = multiply_complex(&q_scaled, &dl_da);
            // println!("dl_dk dim: {}, {}", &dl_dk.len(), &dl_dk[0].len());
            // 2,5 * 4,2 = 5, 4
            let dl_dwk: Vec<Vec<Complex<f64>>> = multiply_complex(&input_batch[batch_ind], &transpose(&dl_dk));
            // println!("dl_dwk dim: {}, {}", &dl_dwk.len(), &dl_dwk[0].len());
            gradient_k_batch[batch_ind] = dl_dwk;

            // 2,4 * 5, 4 = 2, 5
            let dl_dqx = multiply_complex(&dl_dq, &self.weights_q);
            // 4,2 * 5,4 = 2,4 * 4, 5 = 2, 5
            let dl_dkx = multiply_complex(&transpose(&dl_dk), &self.weights_k);
            // 4,2 * 5, 4 = 2, 5
            let dl_dvx = multiply_complex(&transpose(&grad_wv), &self.weights_v);

            gradient_input_batch[batch_ind] = add_matrix(&dl_dqx, &dl_dkx);
            // println!("dl_dqx dim: {}, {}", &dl_dqx.len(), &dl_dqx[0].len());
            // println!("dl_dkx dim: {}, {}", &dl_dkx.len(), &dl_dkx[0].len());
            gradient_input_batch[batch_ind] = add_matrix(&gradient_input_batch[batch_ind], &dl_dvx);
        }

        // Compute the gradients for the parameters and store them
        let mut gradient = Gradient::new_default();
        gradient.set_gradient_weights_v_batch(gradient_v_batch);
        gradient.set_gradient_weights_q_batch(gradient_q_batch);
        gradient.set_gradient_weights_k_batch(gradient_k_batch);
        gradient.set_gradient_input_batch(gradient_input_batch);

        self.gradient = Some(gradient.clone());

        gradient
    }
    pub fn update_parameters(&mut self) {
        let gradient = self.gradient.as_ref().expect("Gradient is missing in attention head layer");
        let (mut grad_w_q, mut grad_w_v, mut grad_w_k) = (gradient.get_gradient_weights_q(), gradient.get_gradient_weights_v(), gradient.get_gradient_weights_k());

        let input_batch = gradient.get_gradient_input_batch();
        let batch_size = input_batch.len() as f64;

        let threshold = 1.0;
        clip_gradients(&mut grad_w_q, threshold);
        clip_gradients(&mut grad_w_v, threshold);
        clip_gradients(&mut grad_w_k, threshold);

        check_nan_or_inf(&mut grad_w_q, "check weight gradients q in attention head");
        check_nan_or_inf(&mut grad_w_v, "check weight gradients v in attention head");
        check_nan_or_inf(&mut grad_w_k, "check weight gradients k in attention head");

        // Update weights q
        for (i, row) in self.weights_q.iter_mut().enumerate() {
            for (j, weight_value) in row.iter_mut().enumerate() {
                if !grad_w_q[i][j].re.is_nan() && !grad_w_q[i][j].im.is_nan() {
                    *weight_value -= self.learning_rate * (grad_w_q[i][j] / batch_size);
                }
            }
        }

        // Update weights v
        for (i, row) in self.weights_v.iter_mut().enumerate() {
            for (j, weight_value) in row.iter_mut().enumerate() {
                if !grad_w_v[i][j].re.is_nan() && !grad_w_v[i][j].im.is_nan() {
                    *weight_value -= self.learning_rate * (grad_w_v[i][j] / batch_size);
                }
            }
        }

        // Update weights k
        for (i, row) in self.weights_k.iter_mut().enumerate() {
            for (j, weight_value) in row.iter_mut().enumerate() {
                if !grad_w_k[i][j].re.is_nan() && !grad_w_k[i][j].im.is_nan() {
                    *weight_value -= self.learning_rate * (grad_w_k[i][j] / batch_size);
                }
            }
        }
    }
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
    let large_negative = Complex::new(-1e5, 0.0);

    for row in 0..attention_scores.len() {
        for col in 0..attention_scores[row].len() {
            if mask[row][col] == 0 {
                attention_scores[row][col] = large_negative; // Apply the mask
            }
        }
    }
}
