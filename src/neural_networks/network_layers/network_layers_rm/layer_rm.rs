use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput}, network_layers::{
        default_layer::LayerInterface,
        layer::{ActivationType, LayerType},
    }, utils::{
         activation_rm::{activate_output_complex_rm, swish_rm}, adam_w::{calculate_adam_w, calculate_adam_w_bias}, derivative_rm::swish_grad_rm, dtype::{C, Real, ZERO, r}, matrix::{
            RowMajorMatrix, add_vector_rm, average_matrix_by_scalar, average_vector_by_scalar, clip_all_gradients_by_global_norm_2d, conjugate_transpose_rm, multiply_complex_rm, normalize_bias, normalize_gradients, split_data_by_columns_rm,
        }, matrix_rm::{add_rm, conjugate_rm, hadamard_rm}, weights_initializer::initialize_weights_complex,
    },
};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
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

        let mut inactivated_batch_output: Vec<RowMajorMatrix<C>> = input_batch
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
            .par_iter_mut()
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

    pub fn backward_rm(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch_rm.as_ref().expect("Input batch is missing in dense layer");
        let raw_output_batch = self.inactivated_input_batch_rm.as_ref().expect("Raw output batch is missing in dense layer");
        let output_batch = self.output_batch_rm.as_ref().expect("Output batch is missing in dense layer");
        let previous_gradient_batch = previous_gradient.get_gradient_input_batch_rm_ref().expect("No previous gradient input batch found");

        let mut gradient = Gradient::new_default();

        // Initialize gradients for weights and biases
        let mut weight_gradients: Vec<RowMajorMatrix<C>> = Vec::with_capacity(input_batch.len());
        let mut bias_gradients: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); self.bias.len()]; input_batch.len()];

        // Gradient w.r.t input has the input feature dimension (weights rows).
        let in_features = self.weights.cols;

        let mut input_gradient_batch: Vec<RowMajorMatrix<C>> = RowMajorMatrix::create_from_batch(input_batch);

        match &self.activation_type {
            ActivationType::SWiGLU => {
                let (weights_1, weights_2) = split_data_by_columns_rm(&self.weights);
                let mut gradient_weights_1 = vec![RowMajorMatrix::create_empty(weights_1.rows, weights_1.cols); input_batch.len()];
                let mut gradient_weights_2 =  vec![RowMajorMatrix::create_empty(weights_2.rows, weights_2.cols); input_batch.len()];

                let mut gradient_bias_1 = vec![vec![C::new(ZERO, ZERO); bias_gradients[0].len() / 2]; input_batch.len()];
                let mut gradient_bias_2 = vec![vec![C::new(ZERO, ZERO); bias_gradients[0].len() / 2]; input_batch.len()];

                for batch_ind in 0..input_batch.len() {
                    let (a, b) = split_data_by_columns_rm(&raw_output_batch[batch_ind]);

                    let swish_b = swish_rm(&b);
                    let gradient_b = swish_grad_rm(&b);

                    //5x10 hadamard 5x10 = 5x10
                    let dl_da = hadamard_rm(&previous_gradient_batch[batch_ind], &conjugate_rm(&swish_b));
                    // 16x5 * 5x10 = 16x10
                    gradient_weights_1[batch_ind] = multiply_complex_rm(&conjugate_transpose_rm(&input_batch[batch_ind]), &dl_da);

                    // 5x10 hadamard 5x10 = 5x10
                    let dl_db_1 = hadamard_rm(&previous_gradient_batch[batch_ind], &conjugate_rm(&a));
                    // 5x10 hadamard 5x10 = 5x10
                    let dl_db_2 = hadamard_rm(&dl_db_1, &conjugate_rm(&gradient_b));

                    // 16x5 * 5x10 = 16x10
                    gradient_weights_2[batch_ind] = multiply_complex_rm(&conjugate_transpose_rm(&input_batch[batch_ind]), &dl_db_2);

                    for grad_row in dl_da.iter() {
                        for (k, grad_val) in grad_row.iter().enumerate() {
                            gradient_bias_1[batch_ind][k] += grad_val;
                        }
                    }
                    for grad_row in dl_db_2.iter() {
                        for (k, grad_val) in grad_row.iter().enumerate() {
                            gradient_bias_2[batch_ind][k] += grad_val;
                        }
                    }

                    // 5x10 * 10x16 = 5x16
                    let gradient_input_a = multiply_complex_rm(&dl_da, &conjugate_transpose_rm(&weights_1));
                    // 5x10 * 10x16 = 5x16
                    let gradient_input_b = multiply_complex_rm(&dl_db_2, &conjugate_transpose_rm(&weights_2));
                    input_gradient_batch[batch_ind] = add_rm(&gradient_input_a, &gradient_input_b);
                }

                for batch_ind in 0..input_batch.len() {
                    for (row_ind, row) in gradient_weights_1[batch_ind].iter_mut().enumerate() {
                        row.extend_from_slice(&gradient_weights_2[batch_ind][row_ind]);
                    }
                    gradient_bias_1[batch_ind].extend_from_slice(&gradient_bias_2[batch_ind]);
                }

                weight_gradients = gradient_weights_1;
                bias_gradients = gradient_bias_1;
            }
            _ => {
                for (batch_ind, (input, previous_gradient)) in input_batch.iter().zip(&previous_gradient_batch).enumerate() {
                    let gradient_output = get_gradient_complex(&output_batch[batch_ind], &raw_output_batch[batch_ind], self.activation_type.clone());
                    let gradient_output_conj = conjugate(&gradient_output);

                    input_gradient_batch[batch_ind] = hadamard_product_2d_c(previous_gradient, &gradient_output_conj);
                    weight_gradients[batch_ind] = multiply_complex(&conjugate_transpose(&input), &input_gradient_batch[batch_ind]);

                    //Accumulate gradients for biases
                    for grad_row in input_gradient_batch[batch_ind].iter() {
                        for (k, grad_val) in grad_row.iter().enumerate() {
                            bias_gradients[batch_ind][k] += grad_val;
                        }
                    }

                    input_gradient_batch[batch_ind] = multiply_complex(&input_gradient_batch[batch_ind], &conjugate_transpose(&self.weights));
                }
            }
        }

        // if self.gradient.is_some() {
        //     let previous_gradient = self.gradient.as_ref().expect("");
        //     weight_gradients = add_matrix_3d(&weight_gradients, &previous_gradient.get_gradient_weight_batch());
        //     bias_gradients = add_matrix(&bias_gradients, &previous_gradient.get_gradient_bias_batch());
        // }

        gradient.set_gradient_input_batch(input_gradient_batch);
        gradient.set_gradient_weight_batch(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let prev_rm_ref = previous_gradient.get_gradient_input_batch_rm_ref().expect("LayerRm::backward expects RM gradients");
        self.backward_rm(prev_rm_ref)
    }

    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in dense RM layer");

        let (mut weight_gradients, mut bias_gradients) = (gradient.get_gradient_weights(), gradient.get_gradient_bias());

        let total_valid_tokens: Real = r(gradient.get_total_valid_tokens().max(1) as f64);

        clip_all_gradients_by_global_norm_2d(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        weight_gradients = average_matrix_by_scalar(&weight_gradients, total_valid_tokens);
        bias_gradients = average_vector_by_scalar(&bias_gradients, total_valid_tokens);

        normalize_gradients(&mut weight_gradients);
        normalize_bias(&mut bias_gradients);

        let learning_rate = self.learning_rate;
        let time_step = self.time_step;

        let (mut prev_m_bias, mut prev_v_bias, mut prev_m_weights, mut prev_v_weights, mut prev_v_weights_hat, mut prev_v_bias_hat) = if let Some(previous_gradient) = &mut self.previous_gradient {
            (
                previous_gradient.get_prev_m_bias(),
                previous_gradient.get_prev_v_bias(),
                previous_gradient.get_prev_m_weights(),
                previous_gradient.get_prev_v_weights(),
                previous_gradient.get_prev_v_weights_hat(),
                previous_gradient.get_prev_v_bias_hat(),
            )
        } else {
            (
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![C::new(ZERO, ZERO); self.bias.len()],
                vec![vec![C::new(ZERO, ZERO); self.weights.cols]; self.weights.rows],
                vec![vec![C::new(ZERO, ZERO); self.weights.cols]; self.weights.rows],
                vec![vec![C::new(ZERO, ZERO); self.weights.cols]; self.weights.rows],
                vec![C::new(ZERO, ZERO); self.bias.len()],
            )
        };

        calculate_adam_w_bias(&mut self.bias, &bias_gradients, &mut prev_m_bias, &mut prev_v_bias, &mut prev_v_bias_hat, learning_rate, time_step);

        // Update weights by round-tripping through Vec<Vec<C>> to reuse the existing AdamW.
        let mut weights_vec = self.weights.to_rows();
        calculate_adam_w(
            &mut weights_vec,
            &weight_gradients,
            &mut prev_m_weights,
            &mut prev_v_weights,
            &mut prev_v_weights_hat,
            learning_rate,
            time_step,
        );
        self.weights = RowMajorMatrix::from_rows(&weights_vec);

        gradient.set_prev_m_bias(prev_m_bias);
        gradient.set_prev_v_bias(prev_v_bias);
        gradient.set_prev_m_weights(prev_m_weights);
        gradient.set_prev_v_weights(prev_v_weights);
        gradient.set_prev_v_weights_hat(prev_v_weights_hat);
        gradient.set_prev_v_bias_hat(prev_v_bias_hat);
        gradient.set_gradient_weights(weight_gradients.clone());
        gradient.set_gradient_bias(bias_gradients.clone());
        self.previous_gradient = Some(gradient.clone());

        self.gradient = None;
    }
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
