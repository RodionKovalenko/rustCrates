//! The row-major transformer (`create_transformer_rm`) must build the same network as `create_transformer`
//! and train the same way: same loss and same weight gradients for the same parameters and batch.
#[cfg(test)]
mod test_transformer_rm {
    use crate::{
        neural_networks::{
            network_components::{
                input::{concat_batches, extend_input_with_bos},
                layer_input_struct::LayerInput,
            },
            network_layers::{
                eml_linear_layer::EmlLinearLayer,
                layer::LayerEnum,
                network_layers_rm::{eml_linear_layer_rm::EmlLinearLayerRm, norm_layer_rm::NormalNormLayerRm},
                norm_layer::NormalNormLayer,
            },
            network_types::{
                neural_network_generic::{NeuralNetwork, OperationMode},
                transformer::{
                    transformer_builder::create_transformer,
                    transformer_builder_rm::create_transformer_rm,
                    transformer_network::{backward, cross_entropy_sum_batch, predict, TOP_K_SIZE},
                    transformer_updater::update_transformer,
                },
            },
            utils::{adam_w::WARMUP_STEPS, matrix_rm::RowMajorMatrix, tokenizer::tokenize_batch},
        },
    };

    use super::super::rm_test_utils::{
        assert_close_1d, assert_close_2d, assert_close_real_2d, assert_ctl_gradients_close, assert_dense_gradients_close, assert_input_gradients_close, assert_layer_norm_gradients_close,
        assert_norm_gradients_close, assert_sparse_head_gradients_close, copy_parameters, copy_sparse_head,
    };

    const INPUTS: [&str; 2] = ["Was ist die Hauptstadt von Frankreich?", "Wie hoch ist der Eiffelturm?"];
    const TARGETS: [&str; 2] = ["Die Hauptstadt von Frankreich ist Paris.", "Der Eiffelturm ist etwa 324 Meter hoch."];

    /// Training input prepared exactly like one batch in `transformer_network::train`.
    fn training_input(time_step: usize) -> (LayerInput, Vec<Vec<u32>>) {
        let inputs: Vec<String> = INPUTS.iter().map(|s| s.to_string()).collect();
        let targets: Vec<String> = TARGETS.iter().map(|t| format!(" <sep> {} <eos>", t)).collect();
        let (_, input_ids) = tokenize_batch(&extend_input_with_bos(&inputs), false).unwrap();
        let (_, target_ids) = tokenize_batch(&targets, false).unwrap();

        // Next-token training: the inputs drop their last token, the targets stay.
        let batch_ids: Vec<Vec<u32>> = concat_batches(&input_ids, &target_ids).iter().map(|seq| seq[..seq.len() - 1].to_vec()).collect();
        let valid_target_tokens: usize = target_ids.iter().map(|seq| seq.iter().filter(|&&id| id != 1).count()).sum();

        let mut layer_input = LayerInput::new_default();
        layer_input.set_batch_ids(batch_ids.clone());
        layer_input.set_time_step(time_step);
        layer_input.set_batch_size(batch_ids.len());
        layer_input.set_forward_only(false);
        layer_input.set_calculate_gradient(true);
        layer_input.set_target_batch_ids(target_ids.clone());
        layer_input.set_top_k_size(TOP_K_SIZE);
        layer_input.set_total_valid_tokens(valid_target_tokens.max(1));
        (layer_input, target_ids)
    }

    fn copy_norm(norm: &NormalNormLayer, norm_rm: &mut NormalNormLayerRm) {
        norm_rm.gamma = norm.gamma.clone();
        norm_rm.beta = norm.beta.clone();
        norm_rm.epsilon = norm.epsilon;
    }

    /// Gives the RM network exactly the parameters of the Vec network (layer by layer).
    fn copy_network_parameters(network: &NeuralNetwork, network_rm: &mut NeuralNetwork) {
        for (i, (layer, layer_rm)) in network.layers.iter().zip(network_rm.layers.iter_mut()).enumerate() {
            match (layer, layer_rm) {
                (LayerEnum::Embedding(embedding), LayerEnum::EmbeddingRm(embedding_rm)) => embedding_rm.weights = RowMajorMatrix::from_rows(&embedding.tied_weights),
                (LayerEnum::Norm(norm), LayerEnum::NormRm(norm_rm)) => copy_norm(norm, norm_rm),
                (LayerEnum::SparseSelfAttention(attention), LayerEnum::SparseSelfAttentionRm(attention_rm)) => {
                    for (head, head_rm) in attention.attention_heads.iter().zip(attention_rm.attention_heads.iter_mut()) {
                        copy_sparse_head(head, head_rm);
                    }
                    let Some(LayerEnum::Norm(norm)) = &attention.norm_layer else { panic!("layer {}: attention is expected to use a Norm layer", i) };
                    copy_norm(norm, &mut attention_rm.norm_layer);
                }
                (LayerEnum::FeedForward(ffn), LayerEnum::FeedForwardRm(ffn_rm)) => {
                    for (dense, dense_rm) in ffn.layers.iter().zip(ffn_rm.layers.iter_mut()) {
                        let (LayerEnum::Dense(dense), LayerEnum::DenseRm(dense_rm)) = (dense, dense_rm) else { panic!("layer {}: unexpected FFN layout", i) };
                        dense_rm.weights = RowMajorMatrix::from_rows(&dense.weights);
                        dense_rm.bias = dense.bias.clone();
                    }
                    let (Some(LayerEnum::Norm(norm)), Some(LayerEnum::NormRm(norm_rm))) = (&ffn.norm_layer, &mut ffn_rm.norm_layer) else { panic!("layer {}: FFN norm layout differs", i) };
                    copy_norm(norm, norm_rm);
                }
                (LayerEnum::ComplexToLinear(ctl), LayerEnum::ComplexToLinearRm(ctl_rm)) => {
                    ctl_rm.weights_1 = RowMajorMatrix::from_rows(&ctl.weights_1);
                    ctl_rm.weights_2 = RowMajorMatrix::from_rows(&ctl.weights_2);
                }
                (LayerEnum::Eml(eml), LayerEnum::EmlRm(eml_rm)) => **eml_rm = copy_parameters::<EmlLinearLayer, EmlLinearLayerRm>(eml),
                (layer, layer_rm) => panic!("layer {}: {} has no matching RM layer ({})", i, layer.name(), layer_rm.name()),
            }
        }
    }

    /// All weight gradients stored by the backward pass (the embedding stores the gradient it receives).
    fn assert_network_gradients_close(network: &NeuralNetwork, network_rm: &NeuralNetwork) {
        for (i, (layer, layer_rm)) in network.layers.iter().zip(network_rm.layers.iter()).enumerate() {
            let name = format!("layer {} ({})", i, layer_rm.name());
            match (layer, layer_rm) {
                (LayerEnum::Embedding(embedding), LayerEnum::EmbeddingRm(embedding_rm)) => {
                    assert_input_gradients_close(&name, embedding.gradient.as_ref().unwrap(), embedding_rm.gradient.as_ref().unwrap())
                }
                (LayerEnum::Norm(norm), LayerEnum::NormRm(norm_rm)) => assert_norm_gradients_close(&name, norm, norm_rm),
                (LayerEnum::SparseSelfAttention(attention), LayerEnum::SparseSelfAttentionRm(attention_rm)) => {
                    for (h, (head, head_rm)) in attention.attention_heads.iter().zip(attention_rm.attention_heads.iter()).enumerate() {
                        assert_sparse_head_gradients_close(&format!("{} head {}", name, h), head, head_rm);
                    }
                    assert_layer_norm_gradients_close(&name, &attention.norm_layer, &attention_rm.norm_layer);
                }
                (LayerEnum::FeedForward(ffn), LayerEnum::FeedForwardRm(ffn_rm)) => {
                    for (d, (dense, dense_rm)) in ffn.layers.iter().zip(ffn_rm.layers.iter()).enumerate() {
                        let (LayerEnum::Dense(dense), LayerEnum::DenseRm(dense_rm)) = (dense, dense_rm) else { panic!("{}: unexpected FFN layout", name) };
                        assert_dense_gradients_close(&format!("{} dense {}", name, d), dense, dense_rm);
                    }
                    let (Some(LayerEnum::Norm(norm)), Some(LayerEnum::NormRm(norm_rm))) = (&ffn.norm_layer, &ffn_rm.norm_layer) else { panic!("{}: FFN norm layout differs", name) };
                    assert_norm_gradients_close(&format!("{} norm", name), norm, norm_rm);
                }
                (LayerEnum::ComplexToLinear(ctl), LayerEnum::ComplexToLinearRm(ctl_rm)) => assert_ctl_gradients_close(&name, ctl, ctl_rm),
                (LayerEnum::Eml(eml), LayerEnum::EmlRm(eml_rm)) => {
                    assert_input_gradients_close(&name, eml.gradient.as_ref().unwrap(), eml_rm.gradient().unwrap());
                    for (which, g, g_rm) in [
                        ("w_uc", eml.grad_w_uc(), eml_rm.grad_w_uc()),
                        ("w_vc", eml.grad_w_vc(), eml_rm.grad_w_vc()),
                        ("w_uw", eml.grad_w_uw(), eml_rm.grad_w_uw()),
                        ("w_vw", eml.grad_w_vw(), eml_rm.grad_w_vw()),
                    ] {
                        assert_close_real_2d(&format!("{} {} gradient", name, which), g.unwrap(), g_rm.unwrap());
                    }
                }
                (layer, layer_rm) => panic!("layer {}: {} has no matching RM layer ({})", i, layer.name(), layer_rm.name()),
            }
        }
    }

    fn flat_losses(losses: &Vec<Vec<Vec<crate::neural_networks::utils::dtype::C>>>) -> Vec<crate::neural_networks::utils::dtype::C> {
        losses.iter().flatten().flatten().copied().collect()
    }

    #[test]
    fn test_transformer_rm_has_the_architecture_of_transformer() {
        let network = create_transformer(OperationMode::TRAINING);
        let network_rm = create_transformer_rm(OperationMode::TRAINING);

        assert!(!network.is_row_major());
        assert!(network_rm.is_row_major());
        assert_ne!(network.model_file(), network_rm.model_file(), "RM and Vec model must not share a model file");

        let names: Vec<String> = network.layers.iter().map(|layer| format!("{}Rm", layer.name())).collect();
        let names_rm: Vec<String> = network_rm.layers.iter().map(|layer| layer.name().to_string()).collect();
        assert_eq!(names, names_rm, "every layer must have its RM counterpart at the same position");
    }

    /// One training step of the whole network: same loss per position, same gradients in every layer and the same
    /// embedding table after the update.
    #[test]
    fn test_transformer_rm_training_step_matches_transformer() {
        let mut network = create_transformer(OperationMode::TRAINING);
        let mut network_rm = create_transformer_rm(OperationMode::TRAINING);
        copy_network_parameters(&network, &mut network_rm);

        let (layer_input, target_ids) = training_input(1);
        network.minibatch_size = target_ids.len();
        network_rm.minibatch_size = target_ids.len();

        let output = predict(&mut network, &layer_input);
        let output_rm = predict(&mut network_rm, &layer_input);

        let (losses, losses_rm) = (output.get_cross_entropy_loss_batch(), output_rm.get_cross_entropy_loss_batch());
        assert!(!losses.is_empty(), "training forward must produce a loss");
        assert_close_1d("cross-entropy loss per position", &flat_losses(&losses), &flat_losses(&losses_rm));

        backward(&mut network, &target_ids, false);
        backward(&mut network_rm, &target_ids, false);
        assert_network_gradients_close(&network, &network_rm);

        update_transformer(&mut network, &target_ids);
        update_transformer(&mut network_rm, &target_ids);
        let (Some(LayerEnum::Embedding(embedding)), Some(LayerEnum::EmbeddingRm(embedding_rm))) = (network.layers.first(), network_rm.layers.first()) else {
            panic!("both networks must start with the embedding")
        };
        assert_close_2d("embedding table after the update", &embedding.tied_weights, &embedding_rm.weights);
    }

    /// Validation runs the forward pass without gradients: the loss must be the same as in training (not 0), and the
    /// output head must not store a gradient.
    #[test]
    fn test_validation_forward_computes_the_training_loss() {
        for mut network in [create_transformer(OperationMode::TRAINING), create_transformer_rm(OperationMode::TRAINING)] {
            let name = if network.is_row_major() { "RM transformer" } else { "transformer" };
            let (mut layer_input, _) = training_input(1);

            layer_input.set_calculate_gradient(false);
            let losses_validation = flat_losses(&predict(&mut network, &layer_input).get_cross_entropy_loss_batch());
            let stored_gradient = match network.layers.last() {
                Some(LayerEnum::Eml(eml)) => eml.gradient.is_some(),
                Some(LayerEnum::EmlRm(eml_rm)) => eml_rm.gradient().is_some(),
                _ => panic!("{}: expected an EML head", name),
            };
            assert!(!stored_gradient, "{}: no gradient may be stored without calculate_gradient", name);

            layer_input.set_calculate_gradient(true);
            let losses_training = flat_losses(&predict(&mut network, &layer_input).get_cross_entropy_loss_batch());

            assert!(losses_validation.iter().any(|loss| loss.re > 0.0), "{}: validation loss must not be 0", name);
            assert_close_1d(&format!("{} validation vs training loss", name), &losses_training, &losses_validation);
        }
    }

    /// `save_to_sled` / `get_from_db` store the network with bincode: an RM network must survive the round trip.
    #[test]
    fn test_transformer_rm_survives_serialization() {
        let mut network_rm = create_transformer_rm(OperationMode::TRAINING);
        let bytes = bincode::serialize(&network_rm).expect("serialize RM transformer");
        let mut loaded: NeuralNetwork = bincode::deserialize(&bytes).expect("deserialize RM transformer");
        loaded.post_load_init();
        assert!(loaded.is_row_major());

        let (layer_input, _) = training_input(1);
        let losses = predict(&mut network_rm, &layer_input).get_cross_entropy_loss_batch();
        let losses_loaded = predict(&mut loaded, &layer_input).get_cross_entropy_loss_batch();
        assert_close_1d("loss after reload", &flat_losses(&losses), &flat_losses(&losses_loaded));
    }

    /// A few training steps on one batch must lower the loss. The steps start after the learning-rate warmup,
    /// which would otherwise keep the learning rate near zero.
    #[test]
    fn test_transformer_rm_learns() {
        let mut network_rm = create_transformer_rm(OperationMode::TRAINING);
        let mut losses = Vec::new();
        for time_step in WARMUP_STEPS..WARMUP_STEPS + 8 {
            let (layer_input, target_ids) = training_input(time_step);
            network_rm.minibatch_size = target_ids.len();
            network_rm.time_step = time_step;
            let output = predict(&mut network_rm, &layer_input);
            losses.push(cross_entropy_sum_batch(&output.get_cross_entropy_loss_batch(), &target_ids).re);
            backward(&mut network_rm, &target_ids, true);
        }
        println!("RM transformer losses: {:?}", losses);
        assert!(losses.iter().all(|loss| loss.is_finite()), "loss must stay finite: {:?}", losses);
        assert!(losses.last().unwrap() < losses.first().unwrap(), "loss must decrease: {:?}", losses);
    }
}
