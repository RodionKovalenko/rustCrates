use crate::neural_networks::{
    network_components::gradient_struct::Gradient, network_layers::layer::LayerEnum, network_types::{neural_network_generic::NeuralNetwork, transformer::transformer_builder::NUM_SELF_ATT_LAYERS}, utils::{dtype::{C, Real, r}, matrix::{normalize_bias, normalize_gradients, normalize_gradients_batch}}
};

pub const VERBOSE: bool = false;
pub const SHOW_MAX_PARAMS: bool = false;

fn update_by_norm(transformer: &mut NeuralNetwork) {
    let mut global_weights = Vec::new();
    let mut global_biases = Vec::new();
    let mut gradient: &mut Gradient;

    for layer in transformer.layers.iter_mut() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_avg_pooling_layer) => {
                // println!("adaptive avg pooling layer with output size: {:?}", &adaptive_avg_pooling_layer.output_size);
            }
            LayerEnum::Embedding(_embedding_layer) => {
                gradient = _embedding_layer.gradient.as_mut().expect("No gradient found");

                let mut gradient_input_batch = gradient.get_gradient_input_batch();

                normalize_gradients_batch(&mut gradient_input_batch);
                gradient.set_gradient_input_batch(gradient_input_batch);

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("embedding layer updating gradient");
                    max_weight(&gradient.get_gradient_input());
                }
            }
            LayerEnum::Norm(norm_layer) => {
                gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                global_biases.push(gradient.get_gradient_beta());
                global_biases.push(gradient.get_gradient_gamma());

                let mut gamma_grad = gradient.get_gradient_gamma();
                let mut beta_grad = gradient.get_gradient_beta();
                // Normalize gradients
                normalize_bias(&mut beta_grad);
                normalize_bias(&mut gamma_grad);

                gradient.set_gradient_gamma(gamma_grad);
                gradient.set_gradient_beta(beta_grad);

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("norm layer updating gradient");
                    max_bias(&gradient.get_gradient_beta());
                    max_bias(&gradient.get_gradient_gamma());
                }
            }
            LayerEnum::NormRm(norm_layer) => {
                gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                global_biases.push(gradient.get_gradient_beta());
                global_biases.push(gradient.get_gradient_gamma());

                let mut gamma_grad = gradient.get_gradient_gamma();
                let mut beta_grad = gradient.get_gradient_beta();
                normalize_bias(&mut beta_grad);
                normalize_bias(&mut gamma_grad);

                gradient.set_gradient_gamma(gamma_grad);
                gradient.set_gradient_beta(beta_grad);

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("norm_rm layer updating gradient");
                    max_bias(&gradient.get_gradient_beta());
                    max_bias(&gradient.get_gradient_gamma());
                }
            }
            LayerEnum::RMSNorm(norm_layer) => {
                gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                global_biases.push(gradient.get_gradient_beta());
                global_biases.push(gradient.get_gradient_gamma());

                let mut gamma_grad = gradient.get_gradient_gamma();
                let mut beta_grad = gradient.get_gradient_beta();
                // Normalize gradients
                normalize_bias(&mut beta_grad);
                normalize_bias(&mut gamma_grad);

                gradient.set_gradient_gamma(gamma_grad);
                gradient.set_gradient_beta(beta_grad);

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("RMS norm layer updating gradient");
                    max_bias(&gradient.get_gradient_beta());
                    max_bias(&gradient.get_gradient_gamma());
                }
            }

            LayerEnum::SelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    gradient = attention_head.gradient.as_mut().expect("No gradient found");

                    global_weights.push(gradient.get_gradient_weights_k());
                    global_weights.push(gradient.get_gradient_weights_q());
                    global_weights.push(gradient.get_gradient_weights_v());
                    global_weights.push(gradient.get_gradient_bias_pos());

                    let mut gradient_weight_k_batch = gradient.get_gradient_weights_k_batch();
                    let mut gradient_weight_v_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_v_batch();
                    let mut gradient_weight_q_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_q_batch();
                    let mut gradient_weight_pos_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_bias_pos_batch();

                    normalize_gradients_batch(&mut gradient_weight_k_batch);
                    normalize_gradients_batch(&mut gradient_weight_v_batch);
                    normalize_gradients_batch(&mut gradient_weight_q_batch);
                    normalize_gradients_batch(&mut gradient_weight_pos_batch);

                    gradient.set_gradient_weights_k_batch(gradient_weight_k_batch);
                    gradient.set_gradient_weights_v_batch(gradient_weight_v_batch);
                    gradient.set_gradient_weights_q_batch(gradient_weight_q_batch);
                    gradient.set_gradient_bias_pos_batch(gradient_weight_pos_batch);

                    if VERBOSE && SHOW_MAX_PARAMS {
                        println!("self attention layer updating gradient");
                        max_weight(&gradient.get_gradient_weights_k());
                        max_weight(&gradient.get_gradient_weights_q());
                        max_weight(&gradient.get_gradient_weights_v());
                        max_weight(&gradient.get_gradient_bias_pos());
                    }
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            // Normalize gradients
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("RMS norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            // Normalize gradients
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        LayerEnum::NormRm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("norm_rm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    gradient = attention_head.gradient.as_mut().expect("No gradient found");

                    global_weights.push(gradient.get_gradient_weights_k());
                    global_weights.push(gradient.get_gradient_weights_q());
                    global_weights.push(gradient.get_gradient_weights_v());
                    global_weights.push(gradient.get_gradient_bias_pos());

                    let mut gradient_weight_k_batch = gradient.get_gradient_weights_k_batch();
                    let mut gradient_weight_v_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_v_batch();
                    let mut gradient_weight_q_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_q_batch();
                    let mut gradient_weight_pos_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_bias_pos_batch();

                    normalize_gradients_batch(&mut gradient_weight_k_batch);
                    normalize_gradients_batch(&mut gradient_weight_v_batch);
                    normalize_gradients_batch(&mut gradient_weight_q_batch);
                    normalize_gradients_batch(&mut gradient_weight_pos_batch);

                    gradient.set_gradient_weights_k_batch(gradient_weight_k_batch);
                    gradient.set_gradient_weights_v_batch(gradient_weight_v_batch);
                    gradient.set_gradient_weights_q_batch(gradient_weight_q_batch);
                    gradient.set_gradient_bias_pos_batch(gradient_weight_pos_batch);

                    if let Some(ctl) = &mut attention_head.ctl_k {
                        if let Some(ctl_gradient) = ctl.gradient.as_mut() {
                            let mut weight_gradient_batch_1 = ctl_gradient.get_gradient_weight_batch();
                            let mut weight_gradient_batch_2 = ctl_gradient.get_gradient_weight_2_batch();

                            normalize_gradients_batch(&mut weight_gradient_batch_1);
                            normalize_gradients_batch(&mut weight_gradient_batch_2);

                            ctl_gradient.set_gradient_weight_batch(weight_gradient_batch_1);
                            ctl_gradient.set_gradient_weight_2_batch(weight_gradient_batch_2);
                        }
                    }
                    if let Some(ctl) = &mut attention_head.ctl_q {
                        if let Some(ctl_gradient) = ctl.gradient.as_mut() {
                            let mut weight_gradient_batch_1 = ctl_gradient.get_gradient_weight_batch();
                            let mut weight_gradient_batch_2 = ctl_gradient.get_gradient_weight_2_batch();

                            normalize_gradients_batch(&mut weight_gradient_batch_1);
                            normalize_gradients_batch(&mut weight_gradient_batch_2);

                            ctl_gradient.set_gradient_weight_batch(weight_gradient_batch_1);
                            ctl_gradient.set_gradient_weight_2_batch(weight_gradient_batch_2);
                        }
                    }

                    if VERBOSE && SHOW_MAX_PARAMS {
                        println!("self sparse attention layer updating gradient");
                        max_weight(&gradient.get_gradient_weights_k());
                        max_weight(&gradient.get_gradient_weights_q());
                        max_weight(&gradient.get_gradient_weights_v());
                        max_weight(&gradient.get_gradient_bias_pos());
                    }
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            // Normalize gradients
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("rms norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            // Normalize gradients
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttentionRm(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    gradient = attention_head.gradient.as_mut().expect("No gradient found");

                    global_weights.push(gradient.get_gradient_weights_k());
                    global_weights.push(gradient.get_gradient_weights_q());
                    global_weights.push(gradient.get_gradient_weights_v());
                    global_weights.push(gradient.get_gradient_bias_pos());

                    let mut gradient_weight_k_batch = gradient.get_gradient_weights_k_batch();
                    let mut gradient_weight_v_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_v_batch();
                    let mut gradient_weight_q_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_q_batch();
                    let mut gradient_weight_pos_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_bias_pos_batch();

                    normalize_gradients_batch(&mut gradient_weight_k_batch);
                    normalize_gradients_batch(&mut gradient_weight_v_batch);
                    normalize_gradients_batch(&mut gradient_weight_q_batch);
                    normalize_gradients_batch(&mut gradient_weight_pos_batch);

                    gradient.set_gradient_weights_k_batch(gradient_weight_k_batch);
                    gradient.set_gradient_weights_v_batch(gradient_weight_v_batch);
                    gradient.set_gradient_weights_q_batch(gradient_weight_q_batch);
                    gradient.set_gradient_bias_pos_batch(gradient_weight_pos_batch);

                    if let Some(ctl_gradient) = attention_head.ctl_k.gradient.as_mut() {
                        let mut weight_gradient_batch_1 = ctl_gradient.get_gradient_weight_batch();
                        let mut weight_gradient_batch_2 = ctl_gradient.get_gradient_weight_2_batch();
                        normalize_gradients_batch(&mut weight_gradient_batch_1);
                        normalize_gradients_batch(&mut weight_gradient_batch_2);
                        ctl_gradient.set_gradient_weight_batch(weight_gradient_batch_1);
                        ctl_gradient.set_gradient_weight_2_batch(weight_gradient_batch_2);
                    }

                    if let Some(ctl_gradient) = attention_head.ctl_q.gradient.as_mut() {
                        let mut weight_gradient_batch_1 = ctl_gradient.get_gradient_weight_batch();
                        let mut weight_gradient_batch_2 = ctl_gradient.get_gradient_weight_2_batch();
                        normalize_gradients_batch(&mut weight_gradient_batch_1);
                        normalize_gradients_batch(&mut weight_gradient_batch_2);
                        ctl_gradient.set_gradient_weight_batch(weight_gradient_batch_1);
                        ctl_gradient.set_gradient_weight_2_batch(weight_gradient_batch_2);
                    }
                }

                if let Some(norm_gradient) = self_attention_layer.norm_layer.gradient.as_mut() {
                    global_biases.push(norm_gradient.get_gradient_beta());
                    global_biases.push(norm_gradient.get_gradient_gamma());

                    let mut gamma_grad = norm_gradient.get_gradient_gamma();
                    let mut beta_grad = norm_gradient.get_gradient_beta();
                    normalize_bias(&mut beta_grad);
                    normalize_bias(&mut gamma_grad);
                    norm_gradient.set_gradient_gamma(gamma_grad);
                    norm_gradient.set_gradient_beta(beta_grad);
                }
            }
            LayerEnum::SelfAttentionApproximation(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    gradient = attention_head.gradient.as_mut().expect("No gradient found");

                    global_weights.push(gradient.get_gradient_weights_k());
                    global_weights.push(gradient.get_gradient_weights_q());
                    global_weights.push(gradient.get_gradient_weights_v());
                    global_weights.push(gradient.get_gradient_bias_pos());

                    let mut gradient_weight_k_batch = gradient.get_gradient_weights_k_batch();
                    let mut gradient_weight_v_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_v_batch();
                    let mut gradient_weight_q_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_q_batch();
                    let mut gradient_weight_pos_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_bias_pos_batch();

                    normalize_gradients_batch(&mut gradient_weight_k_batch);
                    normalize_gradients_batch(&mut gradient_weight_v_batch);
                    normalize_gradients_batch(&mut gradient_weight_q_batch);
                    normalize_gradients_batch(&mut gradient_weight_pos_batch);

                    gradient.set_gradient_weights_k_batch(gradient_weight_k_batch);
                    gradient.set_gradient_weights_v_batch(gradient_weight_v_batch);
                    gradient.set_gradient_weights_q_batch(gradient_weight_q_batch);
                    gradient.set_gradient_bias_pos_batch(gradient_weight_pos_batch);

                    if VERBOSE && SHOW_MAX_PARAMS {
                        println!("self attention approximation layer updating gradient");
                        max_weight(&gradient.get_gradient_weights_k());
                        max_weight(&gradient.get_gradient_weights_q());
                        max_weight(&gradient.get_gradient_weights_v());
                        max_weight(&gradient.get_gradient_bias_pos());
                    }
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            // Normalize gradients
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            // Normalize gradients
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SelfAttentionApproximationRm(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    gradient = attention_head.gradient.as_mut().expect("No gradient found");

                    global_weights.push(gradient.get_gradient_weights_k());
                    global_weights.push(gradient.get_gradient_weights_q());
                    global_weights.push(gradient.get_gradient_weights_v());
                    global_weights.push(gradient.get_gradient_bias_pos());

                    let mut gradient_weight_k_batch = gradient.get_gradient_weights_k_batch();
                    let mut gradient_weight_v_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_v_batch();
                    let mut gradient_weight_q_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_weights_q_batch();
                    let mut gradient_weight_pos_batch: Vec<Vec<Vec<C>>> = gradient.get_gradient_bias_pos_batch();

                    normalize_gradients_batch(&mut gradient_weight_k_batch);
                    normalize_gradients_batch(&mut gradient_weight_v_batch);
                    normalize_gradients_batch(&mut gradient_weight_q_batch);
                    normalize_gradients_batch(&mut gradient_weight_pos_batch);

                    gradient.set_gradient_weights_k_batch(gradient_weight_k_batch);
                    gradient.set_gradient_weights_v_batch(gradient_weight_v_batch);
                    gradient.set_gradient_weights_q_batch(gradient_weight_q_batch);
                    gradient.set_gradient_bias_pos_batch(gradient_weight_pos_batch);

                    if VERBOSE && SHOW_MAX_PARAMS {
                        println!("self attention approximation rm layer updating gradient");
                        max_weight(&gradient.get_gradient_weights_k());
                        max_weight(&gradient.get_gradient_weights_q());
                        max_weight(&gradient.get_gradient_weights_v());
                        max_weight(&gradient.get_gradient_bias_pos());
                    }
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        LayerEnum::NormRm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("norm_rm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        LayerEnum::Norm(_) => {
                            // Vec norm should not be used in RM approximation layer.
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForward(ffn_layer) => {
                for layer in ffn_layer.layers.iter_mut() {
                    match layer {
                        LayerEnum::Dense(dense_layer) => {
                            gradient = dense_layer.gradient.as_mut().expect("No gradient found");

                            global_weights.push(gradient.get_gradient_weights());
                            global_biases.push(gradient.get_gradient_bias());

                            let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                            let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                            normalize_gradients_batch(&mut weight_gradient_batch);
                            normalize_gradients(&mut bias_gradient_batch);

                            gradient.set_gradient_weight_batch(weight_gradient_batch);
                            gradient.set_gradient_bias_batch(bias_gradient_batch);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("dense layer updating gradient");
                                max_weight(&gradient.get_gradient_weights());
                                max_bias(&gradient.get_gradient_bias());
                            }
                        }
                        LayerEnum::Linear(linear_layer) => {
                            gradient = linear_layer.gradient.as_mut().expect("No gradient found");

                            global_weights.push(gradient.get_gradient_weights());
                            global_biases.push(gradient.get_gradient_bias());

                            let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                            let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                            normalize_gradients_batch(&mut weight_gradient_batch);
                            normalize_gradients(&mut bias_gradient_batch);

                            gradient.set_gradient_weight_batch(weight_gradient_batch);
                            gradient.set_gradient_bias_batch(bias_gradient_batch);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("linear layer updating gradient");
                                max_weight(&gradient.get_gradient_weights());
                                max_bias(&gradient.get_gradient_bias());
                            }
                        }
                        _ => {}
                    }
                    if let Some(norm_layer) = ffn_layer.norm_layer.as_mut() {
                        match norm_layer {
                            LayerEnum::RMSNorm(norm_layer) => {
                                gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                                global_biases.push(gradient.get_gradient_beta());
                                global_biases.push(gradient.get_gradient_gamma());

                                let mut gamma_grad = gradient.get_gradient_gamma();
                                let mut beta_grad = gradient.get_gradient_beta();
                                // Normalize gradients
                                normalize_bias(&mut beta_grad);
                                normalize_bias(&mut gamma_grad);

                                gradient.set_gradient_gamma(gamma_grad);
                                gradient.set_gradient_beta(beta_grad);

                                if VERBOSE && SHOW_MAX_PARAMS {
                                    println!("RMS norm layer updating gradient");
                                    max_bias(&gradient.get_gradient_beta());
                                    max_bias(&gradient.get_gradient_gamma());
                                }
                            }
                            LayerEnum::Norm(norm_layer) => {
                                gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                                global_biases.push(gradient.get_gradient_beta());
                                global_biases.push(gradient.get_gradient_gamma());

                                let mut gamma_grad = gradient.get_gradient_gamma();
                                let mut beta_grad = gradient.get_gradient_beta();
                                // Normalize gradients
                                normalize_bias(&mut beta_grad);
                                normalize_bias(&mut gamma_grad);

                                gradient.set_gradient_gamma(gamma_grad);
                                gradient.set_gradient_beta(beta_grad);

                                if VERBOSE && SHOW_MAX_PARAMS {
                                    println!("norm layer updating gradient");
                                    max_bias(&gradient.get_gradient_beta());
                                    max_bias(&gradient.get_gradient_gamma());
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if let Some(norm_layer) = ffn_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            // Normalize gradients
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("RMS norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            // Normalize gradients
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);

                            if VERBOSE && SHOW_MAX_PARAMS {
                                println!("norm layer updating gradient");
                                max_bias(&gradient.get_gradient_beta());
                                max_bias(&gradient.get_gradient_gamma());
                            }
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForwardRm(ffn_layer_rm) => {
                for layer in ffn_layer_rm.layers.iter_mut() {
                    match layer {
                        LayerEnum::DenseRm(dense_layer_rm) => {
                            gradient = dense_layer_rm.gradient.as_mut().expect("No gradient found");

                            global_weights.push(gradient.get_gradient_weights());
                            global_biases.push(gradient.get_gradient_bias());

                            let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                            let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                            normalize_gradients_batch(&mut weight_gradient_batch);
                            normalize_gradients(&mut bias_gradient_batch);

                            gradient.set_gradient_weight_batch(weight_gradient_batch);
                            gradient.set_gradient_bias_batch(bias_gradient_batch);
                        }
                        LayerEnum::Dense(dense_layer) => {
                            gradient = dense_layer.gradient.as_mut().expect("No gradient found");

                            global_weights.push(gradient.get_gradient_weights());
                            global_biases.push(gradient.get_gradient_bias());

                            let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                            let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                            normalize_gradients_batch(&mut weight_gradient_batch);
                            normalize_gradients(&mut bias_gradient_batch);

                            gradient.set_gradient_weight_batch(weight_gradient_batch);
                            gradient.set_gradient_bias_batch(bias_gradient_batch);
                        }
                        LayerEnum::Linear(linear_layer) => {
                            gradient = linear_layer.gradient.as_mut().expect("No gradient found");

                            global_weights.push(gradient.get_gradient_weights());
                            global_biases.push(gradient.get_gradient_bias());

                            let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                            let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                            normalize_gradients_batch(&mut weight_gradient_batch);
                            normalize_gradients(&mut bias_gradient_batch);

                            gradient.set_gradient_weight_batch(weight_gradient_batch);
                            gradient.set_gradient_bias_batch(bias_gradient_batch);
                        }
                        _ => {}
                    }
                }

                if let Some(norm_layer) = ffn_layer_rm.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);
                        }
                        LayerEnum::NormRm(norm_layer) => {
                            gradient = norm_layer.gradient.as_mut().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());

                            let mut gamma_grad = gradient.get_gradient_gamma();
                            let mut beta_grad = gradient.get_gradient_beta();
                            normalize_bias(&mut beta_grad);
                            normalize_bias(&mut gamma_grad);

                            gradient.set_gradient_gamma(gamma_grad);
                            gradient.set_gradient_beta(beta_grad);
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::Dense(dense_layer) => {
                gradient = dense_layer.gradient.as_mut().expect("No gradient found");

                global_weights.push(gradient.get_gradient_weights());
                global_biases.push(gradient.get_gradient_bias());

                let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                normalize_gradients_batch(&mut weight_gradient_batch);
                normalize_gradients(&mut bias_gradient_batch);

                gradient.set_gradient_weight_batch(weight_gradient_batch);
                gradient.set_gradient_bias_batch(bias_gradient_batch);

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("dense layer updating gradient");
                    max_weight(&gradient.get_gradient_weights());
                    max_bias(&gradient.get_gradient_bias());
                }
            }
            LayerEnum::DenseRm(dense_layer_rm) => {
                gradient = dense_layer_rm.gradient.as_mut().expect("No gradient found");

                global_weights.push(gradient.get_gradient_weights());
                global_biases.push(gradient.get_gradient_bias());

                let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                normalize_gradients_batch(&mut weight_gradient_batch);
                normalize_gradients(&mut bias_gradient_batch);

                gradient.set_gradient_weight_batch(weight_gradient_batch);
                gradient.set_gradient_bias_batch(bias_gradient_batch);
            }
            LayerEnum::Linear(linear_layer) => {
                gradient = linear_layer.gradient.as_mut().expect("No gradient found");

                global_weights.push(gradient.get_gradient_weights());
                global_biases.push(gradient.get_gradient_bias());

                let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                normalize_gradients_batch(&mut weight_gradient_batch);
                normalize_gradients(&mut bias_gradient_batch);

                gradient.set_gradient_weight_batch(weight_gradient_batch);
                gradient.set_gradient_bias_batch(bias_gradient_batch);

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("linear layer updating gradient");
                    max_weight(&gradient.get_gradient_weights());
                    max_bias(&gradient.get_gradient_bias());
                }
            }
            LayerEnum::SparseLinear(sparse_linear_layer) => {
                gradient = sparse_linear_layer.gradient.as_mut().expect("No gradient found");

                global_weights.push(gradient.get_gradient_weights());
                global_biases.push(gradient.get_gradient_bias());

                let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                normalize_gradients_batch(&mut weight_gradient_batch);
                normalize_gradients(&mut bias_gradient_batch);

                gradient.set_gradient_weight_batch(weight_gradient_batch);
                gradient.set_gradient_bias_batch(bias_gradient_batch);

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("linear layer updating gradient");
                    max_weight(&gradient.get_gradient_weights());
                    max_bias(&gradient.get_gradient_bias());
                }
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                gradient = sparse_linear_layer.gradient.as_mut().expect("No gradient found");

                global_weights.push(gradient.get_gradient_weights());
                global_biases.push(gradient.get_gradient_bias());

                let mut weight_gradient_batch = gradient.get_gradient_weight_batch();
                let mut bias_gradient_batch = gradient.get_gradient_bias_batch();

                normalize_gradients_batch(&mut weight_gradient_batch);
                normalize_gradients(&mut bias_gradient_batch);

                gradient.set_gradient_weight_batch(weight_gradient_batch);
                gradient.set_gradient_bias_batch(bias_gradient_batch);

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("linear_rm layer updating gradient");
                    max_weight(&gradient.get_gradient_weights());
                    max_bias(&gradient.get_gradient_bias());
                }
            }
            LayerEnum::MultiLinear(linear_layer) => {
                let (weight_grad, bias_grad) = linear_layer.get_combined_gradients();

                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("multi linear layer updating gradient");
                    max_weight(&weight_grad);
                    max_bias(&bias_grad);
                }

                global_weights.push(weight_grad);
                global_biases.push(bias_grad);
            }
            LayerEnum::Wavelet(_wavelet_layer) => {}
            LayerEnum::WaveletRm(_wavelet_layer) => {}
            LayerEnum::DiscreteWavelet(_wavelet_layer) => {}
            LayerEnum::DiscreteWaveletRm(_wavelet_layer) => {}
            LayerEnum::ComplexToLinear(ctl_layer) => {
                gradient = ctl_layer.gradient.as_mut().expect("No gradient found");
                let mut weight_gradient_batch_1 = gradient.get_gradient_weight_batch();
                let mut weight_gradient_batch_2 = gradient.get_gradient_weight_2_batch();

                normalize_gradients_batch(&mut weight_gradient_batch_1);
                normalize_gradients_batch(&mut weight_gradient_batch_2);

                gradient.set_gradient_weight_batch(weight_gradient_batch_1);
                gradient.set_gradient_weight_2_batch(weight_gradient_batch_2);
            }
            LayerEnum::Softmax(_softmax_layer) => {
                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("softmax layer - no parameters to update");
                }
            }
            LayerEnum::SoftmaxRm(_softmax_layer) => {
                if VERBOSE && SHOW_MAX_PARAMS {
                    println!("softmax_rm layer - no parameters to update");
                }
            }
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
            LayerEnum::PositionalEncodingRm(_positional_encoding_layer) => {}
        }
    }
}

pub fn update_transformer(transformer_network: &mut NeuralNetwork, target_batch_ids: &Vec<Vec<u32>>) {
    update_by_norm(transformer_network);

    for layer in transformer_network.layers.iter_mut().rev() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_pooling) => {}
            LayerEnum::Embedding(embedding_layer) => {
                embedding_layer.update_parameters(&target_batch_ids, transformer_network.learning_rate);
            }
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
            LayerEnum::PositionalEncodingRm(_positional_encoding_layer) => {}
            LayerEnum::Norm(_norm_layer) => {
                _norm_layer.update_parameters();
            }
            LayerEnum::NormRm(_norm_layer) => {
                _norm_layer.update_parameters();
            }
            LayerEnum::SelfAttention(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SparseSelfAttention(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SparseSelfAttentionRm(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SelfAttentionApproximation(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SelfAttentionApproximationRm(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::FeedForward(dense_layer) => {
                dense_layer.update_parameters();
            }
            LayerEnum::FeedForwardRm(ffn_layer_rm) => {
                ffn_layer_rm.update_parameters();
            }
            LayerEnum::Linear(linear_layer) => {
                linear_layer.update_parameters();
            }
            LayerEnum::SparseLinear(sparse_linear_layer) => {
                sparse_linear_layer.update_parameters();
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                sparse_linear_layer.update_parameters();
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                multi_linear_layer.update_parameters();
            }
            LayerEnum::Wavelet(_wavelet_layer) => {}
            LayerEnum::WaveletRm(_wavelet_layer) => {}
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                wavelet_layer.update_parameters();
            }
            LayerEnum::ComplexToLinear(ctl_layer) => {
                ctl_layer.update_parameters();
            }
            LayerEnum::Softmax(_softmax_layer) => {
                _softmax_layer.update_parameters();
            }
            LayerEnum::SoftmaxRm(_softmax_layer) => {
                _softmax_layer.update_parameters();
            }
            _ => {
                println!("Layer type not supported for backward pass");
            }
        }
    }
}

pub fn update_k_mean_clusters(transformer_network: &mut NeuralNetwork, epoch: usize) {
    for layer in transformer_network.layers.iter_mut() {
        match layer {
            LayerEnum::SparseLinear(sparse_linear_layer) => {
                sparse_linear_layer.update_centroids(epoch);
            }
            LayerEnum::SparseLinearRm(sparse_linear_layer) => {
                sparse_linear_layer.update_centroids(epoch);
            }
            _ => {}
        }
    }
}

pub fn max_weight(weights: &Vec<Vec<C>>) -> f64 {
    let mut max_weight = 0.0;
    for row in weights.iter() {
        for &weight in row.iter() {
            let abs_weight = weight.norm() as f64;
            if abs_weight > max_weight {
                max_weight = abs_weight;
            }
        }
    }

    if VERBOSE && SHOW_MAX_PARAMS {
        println!("Max weight: {}", max_weight);
    }
    max_weight
}

pub fn max_bias(bias: &[C]) -> f64 {
    let mut max_bias = 0.0;
    for &b in bias.iter() {
        let abs_b = b.norm() as f64;
        if abs_b > max_bias {
            max_bias = abs_b;
        }
    }

    if VERBOSE && SHOW_MAX_PARAMS {
        println!("Max bias: {}", max_bias);
    }
    max_bias
}

// for scaling the input
pub fn calculate_alpha() -> Real {
    r((3.0f64 * NUM_SELF_ATT_LAYERS as f64).powf(0.25))
}

// for scaling the residuals
pub fn calculate_beta() -> Real {
    let alpha = calculate_alpha();
    r(1.0) / alpha
}
