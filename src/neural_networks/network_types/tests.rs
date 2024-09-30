#[cfg(test)]

mod tests {
    use crate::neural_networks::network_types::{feedforward_network_generic::{create, FeedforwardNetwork}, network_trait::Network};

    #[test]
    fn test_feedforward_nn() {
        let number_inputs: usize = 2;
        let number_outputs = 10;
        let number_of_hidden_layers: usize = 1;
        let number_of_hidden_neurons:  usize = 10;
        let minibatch_size: usize = 50;
        let learning_rate: f32 = 0.5;
    
        let feedforward_network: FeedforwardNetwork = create(
            number_inputs,
            number_outputs,
            number_of_hidden_layers,
            number_of_hidden_neurons,
            minibatch_size,
            learning_rate,
        );
    
        // println!("{:?}", &feedforward_network);
    
        println!("learning rate{:?}", &feedforward_network.get_learning_rate());
        println!("layers len {:?}", &feedforward_network.get_layers());
        println!("minibatch size {:?}", &feedforward_network.get_minibatch_size());

        assert_eq![feedforward_network.get_learning_rate(), learning_rate];
        assert_eq![feedforward_network.get_layers().len(), number_of_hidden_layers + 2];
        assert_eq![feedforward_network.get_minibatch_size(), minibatch_size];
    }
}