use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput},
    network_layers::{
        layer::{LayerEnum, LayerType},
        network_layers_rm::{layer_rm::LayerRm, norm_layer_rm::NormalNormLayerRm},
    },
    network_types::transformer::transformer_updater::calculate_alpha,
    utils::{
        dtype::{real_from_f64, C, Real, ZERO},
        matrix::RowMajorMatrix,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardLayerRm {
    pub layers: Vec<LayerEnum>,
    pub norm_layer: Option<LayerEnum>,
    pub gradient: Option<Gradient>,
    pub learning_rate: f64,
    pub alpha: Real,
    pub beta: Real,

    #[serde(skip)]
    pub input_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub output_batch_rm: Option<Vec<RowMajorMatrix<C>>>,
    #[serde(skip)]
    pub padding_mask_batch: Option<Vec<Vec<u32>>>,
    #[serde(skip)]
    pub time_step: usize,
    #[serde(skip)]
    pub batch_size: usize,
}

impl FeedForwardLayerRm {
    pub fn new(rows: usize, cols: usize, learning_rate: f64) -> Self {
        let epsilon: f64 = 0.00000001;

        // Mirror the existing FFN layout: SWiGLU -> (Dense back) .
        // Note: SWiGLU requires output dim doubled.
        let cols_swiglu = cols * 2;

        let mut layers: Vec<LayerEnum> = vec![];
        let dense_1 = LayerRm::new(rows, cols_swiglu, &learning_rate, &crate::neural_networks::network_layers::layer::ActivationType::SWiGLU, LayerType::DenseLayer);
        let dense_2 = LayerRm::new(cols, rows, &learning_rate, &crate::neural_networks::network_layers::layer::ActivationType::LEAKYRELU, LayerType::DenseLayer);

        layers.push(LayerEnum::DenseRm(Box::new(dense_1)));
        layers.push(LayerEnum::DenseRm(Box::new(dense_2)));

        let norm_layer = Some(LayerEnum::NormRm(Box::new(NormalNormLayerRm::new(rows, epsilon, learning_rate))));

        let alpha = calculate_alpha();
        let beta: Real = real_from_f64(1.0) / alpha;

        Self {
            layers,
            norm_layer,
            gradient: None,
            learning_rate,
            input_batch_rm: None,
            output_batch_rm: None,
            padding_mask_batch: None,
            time_step: 0,
            batch_size: 0,
            alpha,
            beta,
        }
    }

    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_rm = input
            .get_input_batch_rm_ref()
            .expect("FeedForwardLayerRm::forward expects RM input")
            .to_vec();

        self.batch_size = input.get_batch_size();
        self.time_step = input.get_time_step();
        self.input_batch_rm = Some(input_rm.clone());

        let mut padding_mask_batch = input.get_padding_mask_batch();
        if padding_mask_batch.is_empty() {
            padding_mask_batch = vec![vec![1; input_rm[0].rows]; input_rm.len()];
        }
        self.padding_mask_batch = Some(padding_mask_batch.clone());

        let mut layer_input = input.clone();
        layer_input.clear_input_batch();
        layer_input.set_input_batch_rm(input_rm.clone());
        layer_input.set_padding_mask_batch(padding_mask_batch.clone());

        let mut output_rm: Vec<RowMajorMatrix<C>> = input_rm.clone();

        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::NormRm(norm_layer) => {
                    let mut layer_output = norm_layer.forward(&layer_input);
                    output_rm = layer_output
                        .take_output_batch_rm()
                        .expect("FeedForwardLayerRm expects Norm to return RM output");
                    layer_input.set_input_batch_rm(output_rm.clone());
                }
                _ => {}
            }
        }

        for layer in self.layers.iter_mut() {
            match layer {
                LayerEnum::DenseRm(dense_layer) => {
                    layer_input.set_input_batch_rm(output_rm);
                    let mut out_dense = dense_layer.forward(&layer_input);
                    output_rm = out_dense
                        .take_output_batch_rm()
                        .expect("DenseRm forward expected output_batch_rm");
                }
                LayerEnum::Linear(linear_layer) => {
                    layer_input.set_input_batch_rm(output_rm);
                    let mut out_linear = linear_layer.forward(&layer_input);
                    output_rm = out_linear
                        .take_output_batch_rm()
                        .expect("Linear RM forward expected output_batch_rm");
                }
                _ => {}
            }
        }

        // Residual: beta * FFN(x) + x
        let mut output_final: Vec<RowMajorMatrix<C>> = Vec::with_capacity(output_rm.len());
        for (y, x) in output_rm.into_iter().zip(input_rm.iter()) {
            assert_eq!(y.rows, x.rows);
            assert_eq!(y.cols, x.cols);
            let mut out = y;
            for i in 0..out.data.len() {
                out.data[i] = out.data[i] * C::new(self.beta, ZERO) + x.data[i];
            }
            output_final.push(out);
        }

        self.output_batch_rm = Some(output_final.clone());

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch_rm(output_final);
        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let prev_rm_ref = previous_gradient
            .get_gradient_input_batch_rm_ref()
            .expect("FeedForwardLayerRm::backward expects RM gradients");

        let mut output_grad_rm: Vec<RowMajorMatrix<C>> = prev_rm_ref.to_vec();

        // Scale by beta for the FFN path
        for g in output_grad_rm.iter_mut() {
            for v in g.data.iter_mut() {
                *v = *v * C::new(self.beta, ZERO);
            }
        }

        // Backprop through inner layers in reverse
        let mut grad = Gradient::new_default();
        grad.set_gradient_input_batch_rm(output_grad_rm);

        for layer in self.layers.iter_mut().rev() {
            match layer {
                LayerEnum::DenseRm(dense_layer) => {
                    grad = dense_layer.backward(&grad);
                }
                LayerEnum::Linear(linear_layer) => {
                    grad = linear_layer.backward(&grad);
                }
                _ => {}
            }
        }

        // Backprop through norm layer (if present)
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::NormRm(norm_layer) => {
                    grad = norm_layer.backward(&grad);
                }
                _ => {}
            }
        }

        // Residual adds upstream gradient directly
        let mut out = grad;
        let mut combined = out.get_gradient_input_batch_rm();
        let upstream = prev_rm_ref;
        if combined.len() == upstream.len() {
            for (c, u) in combined.iter_mut().zip(upstream.iter()) {
                for i in 0..c.data.len() {
                    c.data[i] += u.data[i];
                }
            }
        }
        out.set_gradient_input_batch_rm(combined);

        self.gradient = Some(out.clone());
        out
    }

    pub fn update_parameters(&mut self) {
        for layer in self.layers.iter_mut() {
            match layer {
                LayerEnum::DenseRm(dense) => dense.update_parameters(),
                LayerEnum::Linear(linear) => linear.update_parameters(),
                _ => {}
            }
        }
        if let Some(norm_layer_enum) = self.norm_layer.as_mut() {
            match norm_layer_enum {
                LayerEnum::NormRm(norm) => norm.update_parameters(),
                _ => {}
            }
        }

        self.gradient = None;
    }
}
