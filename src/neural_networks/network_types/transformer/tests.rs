#[cfg(test)]

mod tests {
    use crate::neural_networks::{
        network_components::layer::{ActivationType, LayerType},
        network_types::transformer::attention_head::AttentionHead,
    };

    #[test]
    fn test_attention_layer_initialization() {
        let rows: usize = 15;
        let cols: usize = 15;

        let layer = AttentionHead::create_default_attention_layer(
            rows, 
            cols,
            ActivationType::LEAKYRELU,
            LayerType::HiddenLayer,
        );

        assert_eq!(layer.layer_type, LayerType::HiddenLayer);
        assert_eq!(layer.activation_type, ActivationType::LEAKYRELU);
        println!("{:?}", layer);
    }
}
