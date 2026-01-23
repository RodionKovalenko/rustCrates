#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use sled;

    use crate::neural_networks::network_layers::embedding_layer::EmbeddingLayer;
    use crate::neural_networks::network_layers::network_layers_rm::embedding_layer_rm::EmbeddingLayerRm;

    fn temp_sled_db() -> (sled::Db, std::path::PathBuf) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rustcrates_test_sled_embedding_{now}"));
        let db = sled::open(&path).expect("failed to open temp sled db");
        (db, path)
    }

    #[test]
    fn embedding_layer_get_embedding_missing_returns_err_not_panic() {
        let (db, path) = temp_sled_db();

        let missing = EmbeddingLayer::get_embedding(&db, 50277u32);
        assert!(missing.is_err());

        drop(db);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn embedding_layer_rm_get_embedding_missing_returns_err() {
        let (db, path) = temp_sled_db();

        let missing = EmbeddingLayerRm::get_embedding(&db, 50277u32);
        assert!(missing.is_err());

        drop(db);
        let _ = fs::remove_dir_all(path);
    }
}
