use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::default_layer::LayerInterface,
    utils::{
        adam_w_rm::{update_bias_adam_w, update_weights_adam_w_rm},
        dtype::{r, C, ZERO},
        matrix::{average_vector_by_scalar, normalize_bias}, matrix_rm::RowMajorMatrix,
        matrix_rm::{average_matrix_by_scalar_rm, clip_all_gradients_by_global_norm_rm, linear_backward_batch_rm, linear_forward_batch_rm, normalize_gradients_rm},
        weights_initializer::initialize_weights_complex,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayerRm {
    pub weights: RowMajorMatrix<C>,
    pub bias: Vec<C>,
    pub learning_rate: f64,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl LinearLayerRm {
    pub fn new(learning_rate: f64, rows: usize, cols: usize) -> Self {
        let mut weights_vec: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        initialize_weights_complex(rows, cols, &mut weights_vec);
        let weights = RowMajorMatrix::from_rows(&weights_vec);
        let bias: Vec<C> = vec![C::new(0.001, ZERO); cols];

        Self {
            weights,
            bias,
            learning_rate,
            input_batch_rm: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
            batch_size: 0,
            smoothing: 0.99,
            ema: 0.0,
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    /// RM counterpart of `LinearLayer::forward`.
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch = input.get_input_batch_rm();

        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let output_batch = linear_forward_batch_rm(&input_batch, &self.weights, Some(&self.bias));
        self.input_batch_rm = (!input_batch.is_empty()).then_some(input_batch);

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_batch);
        layer_output
    }

    /// RM counterpart of `LinearLayer::backward`.
    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch_rm.as_ref().expect("No input batch found for LinearLayerRm backward pass");

        let (input_gradient_batch, weight_gradients, bias_gradients) = linear_backward_batch_rm(input_batch, &previous_gradient.get_gradient_input_batch_rm_cow(), &self.weights);

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(input_gradient_batch);
        gradient.set_gradient_weight_batch_rm(weight_gradients);
        gradient.set_gradient_bias_batch(bias_gradients);
        gradient.set_total_valid_tokens(previous_gradient.get_total_valid_tokens());
        self.gradient = Some(gradient.clone());

        gradient
    }

    /// RM counterpart of `LinearLayer::update_parameters`.
    pub fn update_parameters(&mut self) {
        let gradient: &mut Gradient = self.gradient.as_mut().expect("No Gradient found in LinearLayerRm");

        let mut weight_gradients = gradient.get_gradient_weights_rm().expect("No weight gradients found in LinearLayerRm");
        let mut bias_gradients = gradient.get_gradient_bias();
        let total_valid_tokens = r(gradient.get_total_valid_tokens() as f64);

        weight_gradients = average_matrix_by_scalar_rm(&weight_gradients, total_valid_tokens);
        bias_gradients = average_vector_by_scalar(&bias_gradients, total_valid_tokens);

        clip_all_gradients_by_global_norm_rm(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        normalize_gradients_rm(&mut weight_gradients);
        normalize_bias(&mut bias_gradients);

        let previous_gradient = self.previous_gradient.as_ref();
        update_bias_adam_w(&mut self.bias, &bias_gradients, previous_gradient, gradient, self.learning_rate, self.time_step);
        update_weights_adam_w_rm(Gradient::WEIGHTS, &mut self.weights, &weight_gradients, previous_gradient, gradient, self.learning_rate, self.time_step);

        self.previous_gradient = self.gradient.take();
    }
}

impl LayerInterface for LinearLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        LinearLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        LinearLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        LinearLayerRm::update_parameters(self)
    }
}
