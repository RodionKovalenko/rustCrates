use crate::neural_networks::network_components::layer::LayerEnum;

pub trait Network {
    fn get_learning_rate(&self) -> f32;
    fn get_layers(&self) -> Vec<LayerEnum>;
}
