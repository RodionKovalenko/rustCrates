use crate::neural_networks::network_layers::layer::LayerEnum;


pub trait Network {
    fn get_layers(&self) -> Vec<LayerEnum>;
}
