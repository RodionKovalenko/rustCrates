use crate::neural_networks::network_components::layer::Layer;

pub trait Network {
    fn get_learning_rate(&self) -> f32;
    fn get_layers(&self) -> Vec<Layer<16, 16>>;
}