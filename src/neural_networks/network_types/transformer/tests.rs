#[cfg(test)]

mod tests {
    use crate::neural_networks::{network_components::layer::LayerType, network_types::transformer::attention_head::AttentionHead};

    #[test]
    fn test_attention_layer_initialization() {
        let rows: usize = 15;
        let cols: usize = 15;

        let layer = AttentionHead::create_default_attention_layer(rows, cols, LayerType::AttentionLayer);

        assert_eq!(layer.layer_type, LayerType::AttentionLayer);
        println!("{:?}", layer);
    }
}
