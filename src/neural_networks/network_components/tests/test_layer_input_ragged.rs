#[cfg(test)]
mod test_layer_input_ragged {
    use crate::neural_networks::network_components::layer_input_struct::LayerInput;
    use num::Complex;

    #[test]
    fn test_layer_input_set_input_batch_ragged_skips_rm_cache() {
        let mut layer_input = LayerInput::new_default();

        // One sample, ragged rows (2 vs 3 columns). This can happen for sparse/top-k style
        // intermediate representations during training.
        let ragged = vec![vec![
            vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
            vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0), Complex::new(5.0, 0.0)],
        ]];

        layer_input.set_input_batch(ragged);

        assert!(layer_input.get_input_batch_ref().is_some());
        assert!(
            layer_input.get_input_batch_rm_ref().is_none(),
            "RM cache must not be created for ragged inputs"
        );
    }
}
