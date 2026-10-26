//! `train-rm` with all records: 1071 training and 119 validation records, the two parts disjoint.
#[cfg(test)]
mod test_training_split {
    use crate::neural_networks::{
        network_components::input::{load_data_xquad_de_as_dataset, Dataset},
        training::train_transformer::{select_records, ALL_RECORDS},
    };

    fn sorted_pairs<'a>(batches: impl Iterator<Item = &'a Dataset<String, String>>) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = batches.flat_map(|batch| batch.iter().map(|(input, target)| (input.clone(), target.clone())).collect::<Vec<_>>()).collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn test_all_records_split_into_1071_training_and_119_validation() {
        let batch_size = 2;
        let dataset = load_data_xquad_de_as_dataset().expect("XQuAD (de) must load");
        assert_eq!(dataset.input.len(), 1190);

        let mut selected = select_records(&dataset, ALL_RECORDS, batch_size);
        assert_eq!(selected.input.len(), 1190, "ALL_RECORDS must keep every record");

        // What train() does first.
        selected.setup_splits(None);
        assert_eq!((selected.total_training_records_size, selected.total_validation_records_size), (1071, 119));

        let training = sorted_pairs(selected.split_into_batches(batch_size).iter());
        let validation = sorted_pairs(selected.get_validation_batches(batch_size).iter());
        assert_eq!((training.len(), validation.len()), (1071, 119));

        // Training and validation together are exactly the selected records: nothing twice, nothing missing.
        let mut all = training.clone();
        all.extend(validation.iter().cloned());
        all.sort();
        assert_eq!(all, sorted_pairs(std::iter::once(&selected)));
    }
}
