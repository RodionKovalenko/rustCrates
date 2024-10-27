#[cfg(test)]

mod tests {
    use crate::neural_networks::{
        network_components::layer::{ActivationType, LayerType},
        network_types::transformer::attention_layer::AttentionLayer,
    };

    #[test]
    fn test_attention_layer_initialization() {
        let layer = AttentionLayer::<4, 4>::create_default_attention_layer(
            ActivationType::LEAKYRELU,
            LayerType::HiddenLayer,
        );

        assert_eq!(layer.layer_type, LayerType::HiddenLayer);
        assert_eq!(layer.activation_type, ActivationType::LEAKYRELU);
        println!("{:?}", layer);
    }
}
