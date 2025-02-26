#[cfg(test)]

mod tests {
    use std::time::Instant;

    use num::Complex;

    use crate::neural_networks::{
        network_components::{
            input::{DataTrait, Dataset},
            layer::LayerType,
        },
        network_types::{
            neural_network_generic::{NeuralNetwork, OperationMode},
            transformer::{attention_head::AttentionHead, self_attention_layer::SelfAttentionLayer, transformer_builder::create_transformer, transformer_network::train},
        },
    };

    #[test]
    fn test_attention_layer_initialization() {
        let rows: usize = 15;
        let cols: usize = 15;

        let layer = AttentionHead::create_default_attention_layer(rows, cols, LayerType::AttentionLayer);

        assert_eq!(layer.layer_type, LayerType::AttentionLayer);
        println!("{:?}", layer);
    }

    #[test]
    fn test_self_attention_layer() {
        let num_heads = 4;
        let rows = 2;
        let cols = 4 * 16;
        let learning_rate = 0.01;
        let mut self_attention_layer = SelfAttentionLayer::new(num_heads, rows, cols, learning_rate);

        let input_batch_1 = vec![
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        ];

        let input_batch_2 = vec![
            vec![
                Complex::new(5.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(6.0, 0.0),
                Complex::new(8.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
            vec![
                Complex::new(5.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(6.0, 0.0),
                Complex::new(8.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        ];
        let input_batch_3 = vec![
            vec![
                Complex::new(9.0, 0.0),
                Complex::new(10.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(11.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
            vec![
                Complex::new(9.0, 0.0),
                Complex::new(10.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(11.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        ];
        let input_batch_4 = vec![
            vec![
                Complex::new(12.0, 0.0),
                Complex::new(13.0, 0.0),
                Complex::new(14.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
            vec![
                Complex::new(12.0, 0.0),
                Complex::new(13.0, 0.0),
                Complex::new(14.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        ];

        let input_batch = vec![input_batch_1, input_batch_2, input_batch_3, input_batch_4];
        let padding_mask_batch = vec![vec![0, 1], vec![1, 0], vec![0, 1], vec![1, 0]];
        let output = self_attention_layer.forward(&input_batch, &padding_mask_batch);

        println!("output in self attention layer: {:?}", output);
    }

    #[test]
    #[ignore]
    fn test_train_transformer() {
        let now = Instant::now();
        let mut transformer: NeuralNetwork = create_transformer(OperationMode::TRAINING);

        let seconds_elapsed = now.elapsed();
        println!("time elapsed in seconds: {:?}", &seconds_elapsed);

        let input_str1: &str = "Hallo, wie geht es dir?";
        // let input_str2: &str = "Was ist die Hauptstadt von Deutschland? Ich möchte es wissen";
        // let input_str3: &str = "Was kommt nach Donnerstag?";
        // let input_str4: &str = "Was macht 2+3 aus?";

        // let input_str5: &str = "Was macht 2+4?";
        // let input_str6: &str = "Was macht 2+5?";
        // let input_str7: &str = "Was macht 2+6?";
        // let input_str8: &str = "Was macht 2+7?";

        let target1: &str = "Mir geht es gut";
        // let target2: &str = "Berlin ist die Hauptstadt von Deutschland";
        // let target3: &str = "Nach Donnerstag kommt Freitag.";
        // let target4: &str = "2 +3 macht 5";

        // let target5: &str = "2 + 4 macht 6";
        // let target6: &str = "2 + 5 macht 7";
        // let target7: &str = "2 + 6 macht 8";
        // let target8: &str = "2 + 7 macht 9";

        let mut input: Vec<String> = Vec::new();
        input.push(input_str1.to_string());
        // input.push(input_str2.to_string());
        // input.push(input_str3.to_string());
        // input.push(input_str4.to_string());
        // input.push(input_str5.to_string());
        // input.push(input_str6.to_string());
        // input.push(input_str7.to_string());
        // input.push(input_str8.to_string());

        let mut target: Vec<String> = Vec::new();
        target.push(target1.to_string());
        // target.push(target2.to_string());
        // target.push(target3.to_string());
        // target.push(target4.to_string());
        // target.push(target5.to_string());
        // target.push(target6.to_string());
        // target.push(target7.to_string());
        // target.push(target8.to_string());

        let dataset = Dataset::new(input, target);
        let num_epochs: usize = 200;

        train(&mut transformer, dataset, num_epochs);
        let seconds_elapsed_end = now.elapsed();

        println!("time elapsed in seconds: {:?}", seconds_elapsed_end - seconds_elapsed);
    }
}
