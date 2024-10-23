use crate::neural_networks::network_components::layer::Layer;

pub trait Network<const M: usize, const N: usize> {
    fn get_learning_rate(&self) -> f32;
    fn get_layers(&self) -> Vec<Box<Layer<M, N>>>;
}
