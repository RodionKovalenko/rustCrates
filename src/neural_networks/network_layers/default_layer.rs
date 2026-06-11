use crate::neural_networks::network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

pub trait LayerInterface {
    fn forward(&mut self, layer_input: &LayerInput) -> LayerOutput;
    fn backward(&mut self,  previous_gradient: &Gradient) -> Gradient;
    fn update_parameters(&mut self);
}
