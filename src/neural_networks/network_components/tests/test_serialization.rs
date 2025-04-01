#[cfg(test)]
mod test_serialization {
    use crate::{
        database::sled_db::get_storage_path_transformer_db,
        neural_networks::{
            network_components::{embedding_layer::EmbeddingLayer, layer::LayerEnum},
            network_types::{
                neural_network_generic::{NeuralNetwork, OperationMode},
                transformer::transformer_builder::create_transformer,
            },
            utils::file::{derialize_bin, remove_file, serialize_bin},
        },
    };

    #[test]
    fn test_serialization() {
        let vocab_size = 50254;
        let embedding_dim = 16;

        let embedding_layer = EmbeddingLayer::get_or_create(vocab_size, embedding_dim, false);
        let layer_enum: LayerEnum = LayerEnum::Embedding(Box::new(embedding_layer));

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

    #[test]
    fn test_serialization_struct_db() {
        let mut transformer = create_transformer(OperationMode::TRAINING);
        transformer.learning_rate = 0.00001;
        let file_name: &str = "test_transformer";

        let serialized_embedding = bincode::serialize(&transformer).expect("Failed to serialize transformer model");
        let filepath_buf: std::path::PathBuf = get_storage_path_transformer_db(file_name);
        let filepath: &str = filepath_buf.to_str().unwrap();

        serialize_bin(&serialized_embedding, filepath).expect("File cannot be serialized");

        let transformer_result: Result<NeuralNetwork, std::io::Error> = derialize_bin::<NeuralNetwork>(filepath);
        let mut transformer = transformer_result.unwrap();
        
        println!("transformer learning rate: {:?}", transformer.learning_rate);

        transformer.learning_rate = 0.00002;
        let serialized_embedding = bincode::serialize(&transformer).expect("Failed to serialize transformer model");
        serialize_bin(&serialized_embedding, filepath).expect("File cannot be serialized");

        let transformer_result: Result<NeuralNetwork, std::io::Error> = derialize_bin::<NeuralNetwork>(filepath);
        let mut transformer = transformer_result.unwrap();
        println!("transformer learning rate: {:?}", transformer.learning_rate);

        transformer.learning_rate = 0.00003;
        let serialized_embedding = bincode::serialize(&transformer).expect("Failed to serialize transformer model");
        serialize_bin(&serialized_embedding, filepath).expect("File cannot be serialized");

        let transformer_result: Result<NeuralNetwork, std::io::Error> = derialize_bin::<NeuralNetwork>(filepath);
        let transformer = transformer_result.unwrap();
        println!("transformer learning rate: {:?}", transformer.learning_rate);

        remove_file(filepath);
    }
}
