use core::fmt::Debug;
use serde::{Deserialize, Serialize};

use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{default_layer::LayerInterface, network_layers_rm::linear_layer_rm::LinearLayerRm},
    utils::{
        adam_w_rm::{update_bias_adam_w, update_weights_adam_w_rm},
        array_splitting::split_sizes,
        dtype::{r, C, ZERO},
        matrix::{average_vector_by_scalar, normalize_bias}, matrix_rm::RowMajorMatrix,
        matrix_rm::{
            average_matrix_by_scalar_rm, clip_all_gradients_by_global_norm_rm, concat_heads_rm, normalize_gradients_rm, split_columns_rm, sum_heads_rm,
        },
        weights_initializer::initialize_weights_complex,
    },
};

/// RM counterpart of `MultiLinearLayer`: a linear layer whose output columns are split over several
/// `LinearLayerRm`s; the parameters are updated as one combined matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLinearLayerRm {
    pub layers: Vec<LinearLayerRm>,
    pub learning_rate: f64,
    pub col_ranges: Vec<(usize, usize)>,
    pub smoothing: f64,
    pub ema: f64,
    pub global_norm: f64,
    pub max_norm: f64,
    pub previous_gradient: Option<Gradient>,

    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl MultiLinearLayerRm {
    pub fn new(learning_rate: f64, rows: usize, cols: usize, num_layers: usize) -> Self {
        let mut weights_vec = vec![vec![C::new(ZERO, ZERO); cols]; rows];
        initialize_weights_complex(rows, cols, &mut weights_vec);
        let weights = RowMajorMatrix::from_rows(&weights_vec);

        let mut layers = Vec::with_capacity(num_layers);
        let mut col_ranges = Vec::with_capacity(num_layers);
        let mut start_col = 0;
        for chunk_size in split_sizes(cols, num_layers) {
            let end_col = start_col + chunk_size;
            let mut linear_layer = LinearLayerRm::new(learning_rate, rows, chunk_size);
            linear_layer.weights = split_columns_rm(&weights, start_col, end_col);
            linear_layer.bias = vec![C::new(ZERO, ZERO); chunk_size];
            layers.push(linear_layer);
            col_ranges.push((start_col, end_col));
            start_col = end_col;
        }

        Self {
            layers,
            learning_rate,
            col_ranges,
            gradient: None,
            previous_gradient: None,
            smoothing: 0.99,
            ema: 0.0,
            time_step: 0,
            batch_size: 0,
            global_norm: 0.0,
            max_norm: 0.0,
        }
    }

    /// Weights and bias of all sublayers combined column-wise.
    pub fn get_combined_weights(&self) -> (RowMajorMatrix<C>, Vec<C>) {
        let weights = concat_heads_rm(&self.layers.iter().map(|l| vec![l.weights.clone()]).collect::<Vec<_>>()).remove(0);
        let bias = self.layers.iter().flat_map(|l| l.bias.iter().copied()).collect();
        (weights, bias)
    }

    /// Gradients of all sublayers combined column-wise and normalized (like `MultiLinearLayer`).
    pub fn get_combined_gradients(&self) -> (RowMajorMatrix<C>, Vec<C>) {
        let weight_gradients: Vec<Vec<RowMajorMatrix<C>>> = self
            .layers
            .iter()
            .map(|l| vec![l.gradient.as_ref().and_then(|g| g.get_gradient_weights_rm()).unwrap_or_else(|| RowMajorMatrix::create_empty(l.weights.rows, l.weights.cols))])
            .collect();
        let mut weights = concat_heads_rm(&weight_gradients).remove(0);
        let mut bias: Vec<C> = self
            .layers
            .iter()
            .flat_map(|l| l.gradient.as_ref().map(|g| g.get_gradient_bias()).unwrap_or_else(|| vec![C::new(ZERO, ZERO); l.bias.len()]))
            .collect();

        normalize_gradients_rm(&mut weights);
        normalize_bias(&mut bias);
        (weights, bias)
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        self.time_step = input.get_time_step();
        self.batch_size = input.get_batch_size();

        let output_chunks: Vec<Vec<RowMajorMatrix<C>>> = self.layers.iter_mut().map(|l| l.forward(input).take_output_batch_rm().expect("LinearLayerRm must return RM output")).collect();

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(concat_heads_rm(&output_chunks));
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let total_valid_tokens = previous_gradient.get_total_valid_tokens();
        let previous_gradient_rm = previous_gradient.get_gradient_input_batch_rm_cow();

        // Each sublayer gets its column block of the upstream gradient; the input gradients are summed.
        let input_gradients: Vec<Vec<RowMajorMatrix<C>>> = self
            .layers
            .iter_mut()
            .zip(self.col_ranges.iter())
            .map(|(layer, &(start_col, end_col))| {
                let mut gradient_sliced = Gradient::new_default();
                gradient_sliced.set_gradient_input_batch_rm(previous_gradient_rm.iter().map(|g| split_columns_rm(g, start_col, end_col)).collect());
                gradient_sliced.set_total_valid_tokens(total_valid_tokens);
                layer.backward(&gradient_sliced).get_gradient_input_batch_rm()
            })
            .collect();

        let mut gradient = Gradient::new_default();
        gradient.set_gradient_input_batch_rm(sum_heads_rm(input_gradients));
        gradient.set_total_valid_tokens(total_valid_tokens);
        self.gradient = Some(gradient.clone());
        gradient
    }

    pub fn update_parameters(&mut self) {
        // Skip the update if a sublayer has not computed its gradient
        if self.layers.iter().any(|l| l.gradient.is_none()) {
            return;
        }

        let (mut weight_gradients, mut bias_gradients) = self.get_combined_gradients();
        let (mut combined_weights, mut combined_bias) = self.get_combined_weights();

        let mut gradient = self.gradient.take().expect("No gradients available");
        let total_valid_tokens = r(gradient.get_total_valid_tokens().max(1) as f64);

        weight_gradients = average_matrix_by_scalar_rm(&weight_gradients, total_valid_tokens);
        bias_gradients = average_vector_by_scalar(&bias_gradients, total_valid_tokens);

        clip_all_gradients_by_global_norm_rm(&mut weight_gradients, &mut bias_gradients, self.global_norm, self.max_norm);

        normalize_gradients_rm(&mut weight_gradients);
        normalize_bias(&mut bias_gradients);

        let previous_gradient = self.previous_gradient.as_ref();
        update_bias_adam_w(&mut combined_bias, &bias_gradients, previous_gradient, &mut gradient, self.learning_rate, self.time_step);
        update_weights_adam_w_rm(Gradient::WEIGHTS, &mut combined_weights, &weight_gradients, previous_gradient, &mut gradient, self.learning_rate, self.time_step);
        self.previous_gradient = Some(gradient);

        // Distribute the updated parameters back to the sublayers
        for (layer, &(start_col, end_col)) in self.layers.iter_mut().zip(self.col_ranges.iter()) {
            layer.weights = split_columns_rm(&combined_weights, start_col, end_col);
            layer.bias.copy_from_slice(&combined_bias[start_col..end_col]);
        }
    }
}

impl LayerInterface for MultiLinearLayerRm {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput {
        MultiLinearLayerRm::forward(self, layer_input)
    }
    fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        MultiLinearLayerRm::backward(self, previous_gradient)
    }
    fn update_parameters(&mut self) {
        MultiLinearLayerRm::update_parameters(self)
    }
}
