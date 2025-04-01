#[cfg(test)]
mod test_serialization {
    use crate::neural_networks::network_components::{embedding_layer::EmbeddingLayer, layer::LayerEnum};

    #[test]
    fn test_serialization() {
        let vocab_size = 50254;
        let embedding_dim = 16;

        let embedding_layer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim, false);
        let layer_enum: LayerEnum =  LayerEnum::Embedding(Box::new(embedding_layer));

        println!("starting serialization");

        // ✅ Test serialization
        match bincode::serialize(&layer_enum) {
            Ok(bytes) => {
                println!("✅ Serialization successful! Bytes: {:?}", bytes);
                // ✅ Test deserialization
                match bincode::deserialize::<LayerEnum>(&bytes) {
                    Ok(_) => println!("✅ Deserialization successful!"),
                    Err(e) => eprintln!("❌ Deserialization failed: {:?}", e),
                }
            }
            Err(e) => eprintln!("❌ Serialization failed: {:?}", e),
        }
    }
}
