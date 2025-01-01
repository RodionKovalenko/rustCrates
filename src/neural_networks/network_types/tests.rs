#[cfg(test)]

mod tests {
    use num::Complex;

    use crate::neural_networks::{
        network_components::layer::{create_default_layer, ActivationType, Layer, LayerType},
        network_types::neural_network_generic::{create, NeuralNetwork}        
    };

    #[test]
    fn test_feedforward_nn() {
        let number_inputs: usize = 2;
        let number_outputs = 10;
        let number_of_hidden_layers: usize = 1;
        let number_of_hidden_neurons: usize = 10;
        let minibatch_size: usize = 50;
        let learning_rate: f32 = 0.5;

        let rows: usize = 15;
        let cols: usize = 15;

        let feedforward_network: NeuralNetwork = create(
            number_inputs,
            number_outputs,
            number_of_hidden_layers,
            number_of_hidden_neurons,
            minibatch_size,
            learning_rate
        );

        // println!("{:?}", &feedforward_network);

        println!("learning rate{:?}", &feedforward_network.learning_rate);
        println!("layers len {:?}", &feedforward_network.layers.len());
        println!( "minibatch size {:?}", &feedforward_network.get_minibatch_size());

        assert_eq![feedforward_network.learning_rate, learning_rate];
        assert_eq![
            feedforward_network.layers.len(),
            0
        ];
        assert_eq![feedforward_network.get_minibatch_size(), minibatch_size];
    }

    #[test]
    fn test_layer_serialization() {
        let rows: usize = 15;
        let cols: usize = 15;

        // Create a default layer instance
        let original_layer: Layer = Layer::default(rows, cols);

        // Serialize the layer to a JSON string
        let serialized = serde_json::to_string(&original_layer).expect("Failed to serialize layer");

        // Deserialize the JSON string back into a layer instance
        let deserialized_layer: Layer =
            serde_json::from_str(&serialized).expect("Failed to deserialize layer");

        // Assert that the original layer and the deserialized layer are equal
        are_complex_arrays_equal::<15, 15>(
            &original_layer.weights,
            &deserialized_layer.weights,
            1e-10,
        );
        assert_eq!(original_layer.bias, deserialized_layer.bias);
        assert_eq!(
            original_layer.activation_type,
            deserialized_layer.activation_type
        );
        assert_eq!(original_layer.layer_type, deserialized_layer.layer_type);
        assert_eq!(
            original_layer.inactivated_output,
            deserialized_layer.inactivated_output
        );
        assert_eq!(
            original_layer.activated_output,
            deserialized_layer.activated_output
        );
        assert_eq!(original_layer.gradient, deserialized_layer.gradient);
        assert_eq!(original_layer.gradient_w, deserialized_layer.gradient_w);
        assert_eq!(original_layer.errors, deserialized_layer.errors);
        assert_eq!(
            original_layer.previous_gradient,
            deserialized_layer.previous_gradient
        );
        assert_eq!(original_layer.m1, deserialized_layer.m1);
        assert_eq!(original_layer.v1, deserialized_layer.v1);

        const M: usize = 15;
        const N: usize = 15;

        let original_layer: Layer = create_default_layer(rows, cols, &ActivationType::SIGMOID, LayerType::InputLayer);

        // Serialize the layer to a JSON string
        let serialized = serde_json::to_string(&original_layer).expect("Failed to serialize layer");

        // Deserialize the JSON string back into a layer instance
        let deserialized_layer: Layer = serde_json::from_str(&serialized).expect("Failed to deserialize layer");

        // Assert that the original layer and the deserialized layer are equal
        are_complex_arrays_equal::<M, N>(
            &original_layer.weights,
            &deserialized_layer.weights,
            1e-10,
        );
        assert_eq!(original_layer.bias, deserialized_layer.bias);
        assert_eq!(
            original_layer.activation_type,
            deserialized_layer.activation_type
        );
        assert_eq!(original_layer.layer_type, deserialized_layer.layer_type);
        assert_eq!(
            original_layer.inactivated_output,
            deserialized_layer.inactivated_output
        );
        assert_eq!(
            original_layer.activated_output,
            deserialized_layer.activated_output
        );
        assert_eq!(original_layer.gradient, deserialized_layer.gradient);
        assert_eq!(original_layer.gradient_w, deserialized_layer.gradient_w);
        assert_eq!(original_layer.errors, deserialized_layer.errors);
        assert_eq!(
            original_layer.previous_gradient,
            deserialized_layer.previous_gradient
        );
        assert_eq!(original_layer.m1, deserialized_layer.m1);
        assert_eq!(original_layer.v1, deserialized_layer.v1);
    }

    pub fn are_complex_arrays_equal<const M: usize, const N: usize>(
        left: &Vec<Vec<Complex<f64>>>,
        right: &Vec<Vec<Complex<f64>>>,
        epsilon: f64,
    ) -> bool {
        for i in 0..N {
            for j in 0..M {
                if !is_complex_equal(&left[i][j], &right[i][j], epsilon) {
                    return false;
                }
            }
        }
        true
    }

    fn is_complex_equal(a: &Complex<f64>, b: &Complex<f64>, epsilon: f64) -> bool {
        (a.re - b.re).abs() < epsilon && (a.im - b.im).abs() < epsilon
    }
}
