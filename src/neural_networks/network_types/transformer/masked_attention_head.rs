use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer::LayerType, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    utils::{
        activation::softmax_complex_padding,
        adam_w::calculate_adam_w,
        derivative::{backpropagate_softmax_masked, softmax_derivative_complex_jacobian},
        matrix::{add_matrix, check_nan_or_inf, clip_gradients, is_nan_or_inf, multiply_complex, transpose},
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

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub previous_gradient: Option<Gradient>,
    pub time_step: usize,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub attention_weights_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub output_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
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
            previous_gradient: None,
            input_batch: None,
            output_batch: None,
            padding_mask_batch: None,
            attention_weights_batch: None,
            m1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            v1: vec![vec![Complex::new(0.0, 0.0); cols]; rows],
            time_step: 0,
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
    pub fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        let input_batch = layer_input.get_input_batch();
        let padding_mask_batch = layer_input.get_padding_mask_batch();

        // println!("input len in masked_attention : {:?}, {:?}, {}", &input_batch.len(), &input_batch[0].len(), &input_batch[0][0].len());
        // println!("weights_q : {:?}, {:?}", &self.weights_q.len(), &self.weights_q[0].len());

        self.input_batch = Some(input_batch.clone());
        self.padding_mask_batch = Some(padding_mask_batch.clone());
        self.time_step = layer_input.get_time_step();

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
                let mask: Vec<Vec<u8>> = create_causal_mask(input.len());

                // println!("\ncausal mask: {:?}", &mask);

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
                let mut attention_weights = softmax_complex_padding(&attention_scores_scales, &padding_mask);

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

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(batch_output);

        layer_output
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
               S = sigma(A)
               O = S * V

               dl/ds = Gt * VT
               dl/da = Gt * VT * ds/da = dl/ds * ds/da
               dl/dq = dl/ds * ds/da * da/dq = dl/da  * da/dq
               dl/dwq = XT * dl/ds * ds/da * da/dq = XT * dl/dq

               //Wq
               => dl/dwq = XT * (Gt * VT * grad(A) * Kt/sqtr(dk))
               dl/dwq = dl/ds * ds/da * da/dq * dq/dWq

               //Wv
               => dl/dWv = dl/do * do/dv * dv/dwv

               //Wk
               dl/dWk = dl/ds * ds/da * da/dk * dk/dWk

            */

            // 2, 4 * 2, 4 = 2,2
            let dl_ds: Vec<Vec<Complex<f64>>> = multiply_complex(previous_gradient, &transpose(&v));

            // Compute gradient of Wv
            // 2, 4 * 2, 2 = 4, 2
            let grad_wv = multiply_complex(&transpose(&previous_gradient), &attention_weights_batch[batch_ind]);
            // 5,2 * 4, 2 = 5, 2 * 2, 4 = 5, 4
            gradient_v_batch[batch_ind] = multiply_complex(&transpose(&input_batch[batch_ind]), &transpose(&grad_wv));

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
            // 4,2 * 2, 2 = 4,2
            let dl_dq: Vec<Vec<Complex<f64>>> = multiply_complex(&k_scaled, &transpose(&dl_da));
            // println!("dl_dq dim: {}, {}", &dl_dq.len(), &dl_dq[0].len(),);
            // 2,5 * 4,2 = 5,2 * 2, 4 = 5, 4
            let dl_dwq: Vec<Vec<Complex<f64>>> = multiply_complex(&transpose(&input_batch[batch_ind]), &transpose(&dl_dq));
            // println!("dl_dwq dim: {}, {}", &dl_dwq.len(), &dl_dwq[0].len());
            gradient_q_batch[batch_ind] = dl_dwq;

            // Gradient Wk
            // 2,4 * 2,2 = 4,2 * 2,2 = 4,2
            let dl_dk: Vec<Vec<Complex<f64>>> = multiply_complex(&transpose(&q_scaled),&dl_da);
            // println!("dl_dk dim: {}, {}", &dl_dk.len(), &dl_dk[0].len());
            // 2,5 * 4,2 = 5, 4
            let dl_dwk: Vec<Vec<Complex<f64>>> = multiply_complex(&transpose(&input_batch[batch_ind]), &transpose(&dl_dk));
            // println!("dl_dwk dim: {}, {}", &dl_dwk.len(), &dl_dwk[0].len());
            gradient_k_batch[batch_ind] = dl_dwk;

            // println!("\n weights_q dim: {}, {}", &self.weights_q.len(), &self.weights_q[0].len());
            // println!("\n dl_dq dim: {}, {}", &dl_dq.len(), &dl_dq[0].len());
            // println!("\n dl_dk dim: {}, {}", &dl_dk.len(), &dl_dk[0].len());
            // println!("\n grad_wv dim: {}, {}", &grad_wv.len(), &grad_wv[0].len());

            // 4,2 * 5, 4 = 2, 4 * 4, 5 = 2,5
            let dl_dqx = multiply_complex(&transpose(&dl_dq), &transpose(&self.weights_q));
            // 4,2 * 5, 4 = 2, 4 * 4, 5 = 2,5
            let dl_dkx = multiply_complex(&transpose(&dl_dk), &transpose(&self.weights_k));
             // 4,2 * 5, 4 = 2, 4 * 4, 5 = 2,5
            let dl_dvx = multiply_complex(&transpose(&grad_wv), &transpose(&self.weights_v));

            gradient_input_batch[batch_ind] = add_matrix(&dl_dqx, &dl_dkx);
            // println!("dl_dqx dim: {}, {}", &dl_dqx.len(), &dl_dqx[0].len());
            // println!("dl_dkx dim: {}, {}", &dl_dkx.len(), &dl_dkx[0].len());
            gradient_input_batch[batch_ind] = add_matrix(&gradient_input_batch[batch_ind], &dl_dvx);
        }

        // let max = gradient_input_batch.iter().flat_map(|v| v.iter().flat_map(|w| w.iter())).max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Less));
        // let min = gradient_input_batch.iter().flat_map(|v| v.iter().flat_map(|w| w.iter())).min_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Greater));

        // println!("max in backward attention head gradient batch: {:?}", max);
        // println!("min in backward attention head gradient batch: {:?}", min);

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
        let gradient: &mut Gradient = self.gradient.as_mut().expect("Gradient is missing in attention head layer");
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

        let mut prev_m_weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_q[0].len()]; grad_w_q.len()];
        let mut prev_v_weights_q: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_q[0].len()]; grad_w_q.len()];

        let mut prev_m_weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_k[0].len()]; grad_w_k.len()];
        let mut prev_v_weights_k: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_k[0].len()]; grad_w_k.len()];

        let mut prev_m_weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()];
        let mut prev_v_weights_v: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); grad_w_v[0].len()]; grad_w_v.len()];

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        if let Some(previous_gradient) = &mut self.previous_gradient {
            prev_m_weights_q = previous_gradient.get_prev_m_weigths_q();
            prev_v_weights_q = previous_gradient.get_prev_v_weigths_q();

            prev_m_weights_k = previous_gradient.get_prev_m_weigths_k();
            prev_v_weights_k = previous_gradient.get_prev_v_weigths_k();

            prev_m_weights_v = previous_gradient.get_prev_m_weigths_v();
            prev_v_weights_v = previous_gradient.get_prev_v_weigths_v();

            self.weights_q = calculate_adam_w(&self.weights_q, &grad_w_q, &mut prev_m_weights_q, &mut prev_v_weights_q, learning_rate, time_step);
            self.weights_k = calculate_adam_w(&self.weights_k, &grad_w_k, &mut prev_m_weights_k, &mut prev_v_weights_k, learning_rate, time_step);
            self.weights_v = calculate_adam_w(&self.weights_v, &grad_w_v, &mut prev_m_weights_v, &mut prev_v_weights_v, learning_rate, time_step);
        } else {
            // Update weights q
            for (i, row) in self.weights_q.iter_mut().enumerate() {
                for (j, weight_value) in row.iter_mut().enumerate() {
                    if !is_nan_or_inf(&grad_w_q[i][j]) {
                        *weight_value -= self.learning_rate * (grad_w_q[i][j] / batch_size);
                    }
                }
            }

            // Update weights v
            for (i, row) in self.weights_v.iter_mut().enumerate() {
                for (j, weight_value) in row.iter_mut().enumerate() {
                    if !is_nan_or_inf(&grad_w_v[i][j]) {
                        *weight_value -= self.learning_rate * (grad_w_v[i][j] / batch_size);
                    }
                }
            }

            // Update weights k
            for (i, row) in self.weights_k.iter_mut().enumerate() {
                for (j, weight_value) in row.iter_mut().enumerate() {
                    if !is_nan_or_inf(&grad_w_k[i][j]) {
                        *weight_value -= self.learning_rate * (grad_w_k[i][j] / batch_size);
                    }
                }
            }
        }

        gradient.set_prev_m_weights_q(prev_m_weights_q);
        gradient.set_prev_v_weights_q(prev_v_weights_q);
        gradient.set_prev_m_weights_k(prev_m_weights_k);
        gradient.set_prev_v_weights_k(prev_v_weights_k);
        gradient.set_prev_m_weights_v(prev_m_weights_v);
        gradient.set_prev_v_weights_v(prev_v_weights_v);

        self.previous_gradient = Some(gradient.clone());
    }
}

fn scale_attention_scores(attention_scores: &Vec<Vec<Complex<f64>>>, d_k: f64) -> Vec<Vec<Complex<f64>>> {
    let scaling_factor = 1.0 / (1e-8 + d_k.sqrt());
    let mut scaled_scores = attention_scores.clone();

    // Scale each attention score
    for row in 0..scaled_scores.len() {
        for col in 0..scaled_scores[row].len() {
            scaled_scores[row][col] = scaled_scores[row][col] * scaling_factor;
        }
    }

    scaled_scores
}

fn create_causal_mask(rows: usize) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0; rows]; rows]; // Initialize with zeros

    for i in 0..rows {
        for j in 0..=i {
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
