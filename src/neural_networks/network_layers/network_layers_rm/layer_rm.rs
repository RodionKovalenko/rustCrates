use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        default_layer::LayerInterface,
        layer::{ActivationType, LayerType},
    },
    utils::{
        activation_rm::{activate_output_complex_rm, swish_rm},
        adam_w::{calculate_adam_w_bias, calculate_adam_w_rm},
        derivative_rm::{get_gradient_complex_rm, swish_grad_rm},
        dtype::{r, Real, C, ZERO},
        matrix::{add_vector_rm, average_vector_by_scalar, conjugate_transpose_rm, multiply_complex_rm, normalize_bias, split_data_by_columns_rm, RowMajorMatrix},
        matrix_rm::{
            accumulate_bias_from_rm, add_rm, average_matrix_by_scalar_rm, clip_all_gradients_by_global_norm_rm, concat_columns_rm, conjugate_rm, hadamard_rm, normalize_gradients_rm,
        },
        weights_initializer::initialize_weights_complex,
    },
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRm {
    pub weights: RowMajorMatrix<C>,
    pub bias: Vec<C>,
    pub activation_type: ActivationType,
    pub layer_type: LayerType,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub inactivated_input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl LayerRm {
    pub fn new(rows: usize, cols: usize, learning_rate: &f64, activation: &ActivationType, layer_type: LayerType) -> Self {
        let mut weights_vec: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        initialize_weights_complex(rows, cols, &mut weights_vec);
        let weights = RowMajorMatrix::from_rows(&weights_vec);
        let bias: Vec<C> = vec![C::new(0.001, ZERO); cols];

        Self {
            weights,
            bias,
            activation_type: activation.clone(),
            layer_type,
            learning_rate: *learning_rate,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
            previous_gradient: None,
            input_batch_rm: None,
            inactivated_input_batch_rm: None,
            output_batch_rm: None,
            gradient: None,
            padding_mask_batch: None,
            time_step: 0,
            batch_size: 0,
        }
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<RowMajorMatrix<C>> = input.get_input_batch_rm();
        let padding_mask_batch = input.get_padding_mask_batch();
        self.batch_size = input.get_batch_size();

        self.input_batch_rm = Some(input_batch.clone());
        self.time_step = input.get_time_step();

        let inactivated_batch_output: Vec<RowMajorMatrix<C>> = input_batch
            .par_iter()
            .map(|input| {
                let mut output: RowMajorMatrix<C> = multiply_complex_rm(input, &self.weights);

                // Add bias to the result
                add_vector_rm(&mut output, &self.bias);
                output
            })
            .collect();

        self.inactivated_input_batch_rm = Some(inactivated_batch_output.clone());

        let batch_output: Vec<RowMajorMatrix<C>> = inactivated_batch_output
            .par_iter()
            .map(|input| {
                // Apply activation if the layer type is DenseLayer
                let activated_output = activate_output_complex_rm(input, &self.activation_type);
                activated_output
            })
            .collect();

        self.output_batch_rm = Some(batch_output.clone());
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(batch_output);

        output
    }

    /// RM counterpart of `Layer::backward`.
    pub fn backward_rm(&mut self, previous_gradient_batch: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch = self.input_batch_rm.as_ref().expect("Input batch is missing in dense RM layer");
        let raw_output_batch = self.inactivated_input_batch_rm.as_ref().expect("Raw output batch is missing in dense RM layer");
        let output_batch = self.output_batch_rm.as_ref().expect("Output batch is missing in dense RM layer");

        assert_eq!(previous_gradient_batch.len(), input_batch.len(), "Previous gradient batch size does not match input batch size in dense RM layer");

        let mut gradient = Gradient::new_default();

        let (input_gradient_batch, weight_gradients, bias_gradients): (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>, Vec<Vec<C>>) = match &self.activation_type {
            ActivationType::SWiGLU => {
                let (weights_1, weights_2) = split_data_by_columns_rm(&self.weights);
                let weights_1_t = conjugate_transpose_rm(&weights_1);
                let weights_2_t = conjugate_transpose_rm(&weights_2);
                let half = self.bias.len() / 2;

                let results: Vec<(RowMajorMatrix<C>, RowMajorMatrix<C>, Vec<C>)> = (0..input_batch.len())
                    .into_par_iter()
                    .map(|batch_ind| {
                        let (a, b) = split_data_by_columns_rm(&raw_output_batch[batch_ind]);

                        let swish_b = swish_rm(&b);
                        let gradient_b = swish_grad_rm(&b);
                        let input_t = conjugate_transpose_rm(&input_batch[batch_ind]);

                        //5x10 hadamard 5x10 = 5x10
                        let dl_da = hadamard_rm(&previous_gradient_batch[batch_ind], &conjugate_rm(&swish_b));
                        // 16x5 * 5x10 = 16x10
                        let gradient_weights_1 = multiply_complex_rm(&input_t, &dl_da);

                        // 5x10 hadamard 5x10 = 5x10
                        let dl_db_1 = hadamard_rm(&previous_gradient_batch[batch_ind], &conjugate_rm(&a));
                        // 5x10 hadamard 5x10 = 5x10
                        let dl_db_2 = hadamard_rm(&dl_db_1, &conjugate_rm(&gradient_b));

                        // 16x5 * 5x10 = 16x10
                        let gradient_weights_2 = multiply_complex_rm(&input_t, &dl_db_2);

                        let mut gradient_bias_1 = vec![C::new(ZERO, ZERO); half];
                        let mut gradient_bias_2 = vec![C::new(ZERO, ZERO); half];
                        accumulate_bias_from_rm(&mut gradient_bias_1, &dl_da);
                        accumulate_bias_from_rm(&mut gradient_bias_2, &dl_db_2);
                        gradient_bias_1.extend_from_slice(&gradient_bias_2);

                        // 5x10 * 10x16 = 5x16
                        let gradient_input_a = multiply_complex_rm(&dl_da, &weights_1_t);
                        // 5x10 * 10x16 = 5x16
                        let gradient_input_b = multiply_complex_rm(&dl_db_2, &weights_2_t);

                        (add_rm(&gradient_input_a, &gradient_input_b), concat_columns_rm(&gradient_weights_1, &gradient_weights_2), gradient_bias_1)
                    })
                    .collect();

                unzip_batch(results)
            }
            _ => {
                let weights_t = conjugate_transpose_rm(&self.weights);

                let results: Vec<(RowMajorMatrix<C>, RowMajorMatrix<C>, Vec<C>)> = (0..input_batch.len())
                    .into_par_iter()
                    .map(|batch_ind| {
                        let gradient_output = get_gradient_complex_rm(&output_batch[batch_ind], &raw_output_batch[batch_ind], &self.activation_type);
                        let gradient_output_conj = conjugate_rm(&gradient_output);

                        let delta = hadamard_rm(&previous_gradient_batch[batch_ind], &gradient_output_conj);
                        let weight_gradient = multiply_complex_rm(&conjugate_transpose_rm(&input_batch[batch_ind]), &delta);

                        //Accumulate gradients for biases
                        let mut bias_gradient = vec![C::new(ZERO, ZERO); self.bias.len()];
                        accumulate_bias_from_rm(&mut bias_gradient, &delta);

                        let input_gradient = multiply_complex_rm(&delta, &weights_t);

                        (input_gradient, weight_gradient, bias_gradient)
                    })
                    .collect();

                unzip_batch(results)
            }
        };

        gradient.set_gradient_input_batch_rm(input_gradient_batch);
        gradient.set_gradient_weight_batch_rm(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        // Falls back to converting Vec<Vec<C>> gradients if the previous layer is not an RM layer.
        match previous_gradient.get_gradient_input_batch_rm_ref() {
            Some(previous_gradient_batch) => self.backward_rm(previous_gradient_batch),
            None => {
                let previous_gradient_batch = previous_gradient.get_gradient_input_batch_rm();
                self.backward_rm(&previous_gradient_batch)
            }
        }
    }

    /// RM counterpart of `Layer::update_parameters`.
    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in dense RM layer");

        let mut weight_gradients = gradient.get_gradient_weights_rm().expect("No weight gradients found in dense RM layer");
        let mut bias_gradients = gradient.get_gradient_bias();

        let total_valid_tokens: Real = r(gradient.get_total_valid_tokens().max(1) as f64);

        clip_all_gradients_by_global_norm_rm(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        weight_gradients = average_matrix_by_scalar_rm(&weight_gradients, total_valid_tokens);
        bias_gradients = average_vector_by_scalar(&bias_gradients, total_valid_tokens);

        normalize_gradients_rm(&mut weight_gradients);
        normalize_bias(&mut bias_gradients);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;
        let (rows, cols) = (self.weights.rows, self.weights.cols);

        let (mut prev_m_bias, mut prev_v_bias, mut prev_v_bias_hat, mut prev_m_weights, mut prev_v_weights, mut prev_v_weights_hat) = match &self.previous_gradient {
            Some(previous_gradient) => (
                previous_gradient.get_prev_m_bias(),
                previous_gradient.get_prev_v_bias(),
                previous_gradient.get_prev_v_bias_hat(),
                previous_gradient.get_prev_m_weights_rm().unwrap_or_else(|| RowMajorMatrix::create_empty(rows, cols)),
                previous_gradient.get_prev_v_weights_rm().unwrap_or_else(|| RowMajorMatrix::create_empty(rows, cols)),
                previous_gradient.get_prev_v_weights_hat_rm().unwrap_or_else(|| RowMajorMatrix::create_empty(rows, cols)),
            ),
            None => (
                // Initialize to zeros on first step
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
                RowMajorMatrix::create_empty(rows, cols),
                RowMajorMatrix::create_empty(rows, cols),
                RowMajorMatrix::create_empty(rows, cols),
            ),
        };

        calculate_adam_w_bias(&mut self.bias, &bias_gradients, &mut prev_m_bias, &mut prev_v_bias, &mut prev_v_bias_hat, learning_rate, time_step);
        calculate_adam_w_rm(&mut self.weights, &weight_gradients, &mut prev_m_weights, &mut prev_v_weights, &mut prev_v_weights_hat, learning_rate, time_step);

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);
        gradient.set_prev_m_weights_rm(prev_m_weights);
        gradient.set_prev_v_weights_rm(prev_v_weights);
        gradient.set_prev_v_weights_hat_rm(prev_v_weights_hat);
        gradient.set_gradient_weights_rm(weight_gradients);
        gradient.set_gradient_bias(bias_gradients);
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
}

fn unzip_batch(results: Vec<(RowMajorMatrix<C>, RowMajorMatrix<C>, Vec<C>)>) -> (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>, Vec<Vec<C>>) {
    let mut input_gradients = Vec::with_capacity(results.len());
    let mut weight_gradients = Vec::with_capacity(results.len());
    let mut bias_gradients = Vec::with_capacity(results.len());
    for (input_gradient, weight_gradient, bias_gradient) in results {
        input_gradients.push(input_gradient);
        weight_gradients.push(weight_gradient);
        bias_gradients.push(bias_gradient);
    }
    (input_gradients, weight_gradients, bias_gradients)
}

impl LayerInterface for LayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        LayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        LayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        LayerRm::update_parameters(self)
    }
}
