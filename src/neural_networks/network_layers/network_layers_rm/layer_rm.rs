use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        default_layer::LayerInterface,
        layer::{ActivationType, LayerType},
    },
    utils::{
        activation_rm::{activate_output_complex_rm, swish_rm},
        adam_w_rm::{update_bias_adam_w, update_weights_adam_w_rm},
        derivative_rm::{get_gradient_complex_rm, swish_grad_rm},
        dtype::{r, Real, C, ZERO},
        matrix::{average_vector_by_scalar, normalize_bias}, matrix_rm::RowMajorMatrix,
        matrix_rm::{
            average_matrix_by_scalar_rm, clip_all_gradients_by_global_norm_rm, concat_columns_rm, conjugate_rm, hadamard_rm, linear_backward_batch_rm, linear_forward_batch_rm, normalize_gradients_rm,
            split_data_by_columns_rm,
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
        self.batch_size = input.get_batch_size();
        self.time_step = input.get_time_step();

        let inactivated_batch_output = linear_forward_batch_rm(&input_batch, &self.weights, Some(&self.bias));
        let batch_output: Vec<RowMajorMatrix<C>> = inactivated_batch_output.par_iter().map(|raw| activate_output_complex_rm(raw, &self.activation_type)).collect();

        self.input_batch_rm = Some(input_batch);
        self.inactivated_input_batch_rm = Some(inactivated_batch_output);
        self.output_batch_rm = Some(batch_output.clone());
        self.padding_mask_batch = Some(input.get_padding_mask_batch());

        let mut output = LayerOutput::new_default();
        output.set_output_batch_rm(batch_output);
        output
    }

    /// dL/d(XW + b) for one sample, given the upstream gradient dL/dY.
    fn output_delta(&self, previous_gradient: &RowMajorMatrix<C>, raw_output: &RowMajorMatrix<C>, output: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
        match &self.activation_type {
            ActivationType::SWiGLU => {
                // Y = a * swish(b) with [a | b] = XW + b
                let (a, b) = split_data_by_columns_rm(raw_output);
                let dl_da = hadamard_rm(previous_gradient, &conjugate_rm(&swish_rm(&b)));
                let dl_db = hadamard_rm(&hadamard_rm(previous_gradient, &conjugate_rm(&a)), &conjugate_rm(&swish_grad_rm(&b)));
                concat_columns_rm(&dl_da, &dl_db)
            }
            activation => hadamard_rm(previous_gradient, &conjugate_rm(&get_gradient_complex_rm(output, raw_output, activation))),
        }
    }

    /// RM counterpart of `Layer::backward`.
    pub fn backward_rm(&mut self, previous_gradient_batch: &[RowMajorMatrix<C>]) -> Gradient {
        let input_batch = self.input_batch_rm.as_ref().expect("Input batch is missing in dense RM layer");
        let raw_output_batch = self.inactivated_input_batch_rm.as_ref().expect("Raw output batch is missing in dense RM layer");
        let output_batch = self.output_batch_rm.as_ref().expect("Output batch is missing in dense RM layer");

        assert_eq!(previous_gradient_batch.len(), input_batch.len(), "Previous gradient batch size does not match input batch size in dense RM layer");

        let delta_batch: Vec<RowMajorMatrix<C>> = (0..input_batch.len())
            .into_par_iter()
            .map(|b| self.output_delta(&previous_gradient_batch[b], &raw_output_batch[b], &output_batch[b]))
            .collect();

        let (input_gradient_batch, weight_gradients, bias_gradients) = linear_backward_batch_rm(input_batch, &delta_batch, &self.weights);

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(input_gradient_batch);
        gradient.set_gradient_weight_batch_rm(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        self.gradient = Some(gradient.clone());

        gradient
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        self.backward_rm(&previous_gradient.get_gradient_input_batch_rm_cow())
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

        let previous_gradient = self.previous_gradient.as_ref();
        update_bias_adam_w(&mut self.bias, &bias_gradients, previous_gradient, gradient, self.learning_rate, self.time_step);
        update_weights_adam_w_rm(Gradient::WEIGHTS, &mut self.weights, &weight_gradients, previous_gradient, gradient, self.learning_rate, self.time_step);

        self.previous_gradient = self.gradient.take();
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
