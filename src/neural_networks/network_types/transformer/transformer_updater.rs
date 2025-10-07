use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer::LayerEnum},
    network_types::{neural_network_generic::NeuralNetwork, transformer::transformer_network::EMA_SCALER},
    utils::matrix::compute_global_norm,
};

fn update_global_norm(transformer: &mut NeuralNetwork) {
    let mut global_weights = Vec::new();
    let mut global_biases = Vec::new();
    let mut gradient: &Gradient;

    for layer in transformer.layers.iter_mut() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_avg_pooling_layer) => {
                // println!("adaptive avg pooling layer with output size: {:?}", &adaptive_avg_pooling_layer.output_size);
            }
            LayerEnum::Embedding(_embedding_layer) => {}
            LayerEnum::Norm(norm_layer) => {
                gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                global_biases.push(gradient.get_gradient_beta());
                global_biases.push(gradient.get_gradient_gamma());
            }
            LayerEnum::RMSNorm(norm_layer) => {
                gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                global_biases.push(gradient.get_gradient_beta());
                global_biases.push(gradient.get_gradient_gamma());
            }

            LayerEnum::SelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    gradient = attention_head.gradient.as_ref().expect("No gradient found");

                    global_weights.push(gradient.get_gradient_weights_k());
                    global_weights.push(gradient.get_gradient_weights_q());
                    global_weights.push(gradient.get_gradient_weights_v());
                    global_weights.push(gradient.get_gradient_bias_pos());
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    gradient = attention_head.gradient.as_ref().expect("No gradient found");

                    global_weights.push(gradient.get_gradient_weights_k());
                    global_weights.push(gradient.get_gradient_weights_q());
                    global_weights.push(gradient.get_gradient_weights_v());
                    global_weights.push(gradient.get_gradient_bias_pos());
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SelfAttentionApproximation(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    gradient = attention_head.gradient.as_ref().expect("No gradient found");

                    global_weights.push(gradient.get_gradient_weights_k());
                    global_weights.push(gradient.get_gradient_weights_q());
                    global_weights.push(gradient.get_gradient_weights_v());
                    global_weights.push(gradient.get_gradient_bias_pos());
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForward(ffn_layer) => {
                for layer in ffn_layer.layers.iter_mut() {
                    match layer {
                        LayerEnum::Dense(dense_layer) => {
                            gradient = dense_layer.gradient.as_ref().expect("No gradient found");

                            global_weights.push(gradient.get_gradient_weights());
                            global_biases.push(gradient.get_gradient_bias());
                        }
                        LayerEnum::Linear(linear_layer) => {
                            gradient = linear_layer.gradient.as_ref().expect("No gradient found");

                            global_weights.push(gradient.get_gradient_weights());
                            global_biases.push(gradient.get_gradient_bias());
                        }
                        _ => {}
                    }
                }
                if let Some(norm_layer) = ffn_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());
                        }
                        LayerEnum::Norm(norm_layer) => {
                            gradient = norm_layer.gradient.as_ref().expect("No gradient found");

                            global_biases.push(gradient.get_gradient_beta());
                            global_biases.push(gradient.get_gradient_gamma());
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::Dense(dense_layer) => {
                gradient = dense_layer.gradient.as_ref().expect("No gradient found");

                global_weights.push(gradient.get_gradient_weights());
                global_biases.push(gradient.get_gradient_bias());
            }
            LayerEnum::Linear(linear_layer) => {
                gradient = linear_layer.gradient.as_ref().expect("No gradient found");

                global_weights.push(gradient.get_gradient_weights());
                global_biases.push(gradient.get_gradient_bias());
            }
            LayerEnum::MultiLinear(linear_layer) => {
                let (weight_grad, bias_grad) = linear_layer.get_combined_gradients();

                global_weights.push(weight_grad);
                global_biases.push(bias_grad);
            }
            LayerEnum::Wavelet(_wavelet_layer) => {}
            LayerEnum::DiscreteWavelet(_wavelet_layer) => {}
            LayerEnum::Softmax(_softmax_layer) => {}
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
        }
    }

    let global_norm = compute_global_norm(&global_weights, &global_biases);
    transformer.ema = transformer.smoothing * transformer.ema + (1.0 - transformer.smoothing) * global_norm;
    let max_norm = transformer.ema * EMA_SCALER;

    transformer.global_norm = global_norm;
    transformer.max_norm = max_norm;

    println!("Global norm: {}, Max norm: {}", global_norm, max_norm);

    update_layers_max_norm(transformer, global_norm, max_norm);
}

fn update_layers_max_norm(transformer: &mut NeuralNetwork, global_norm: f64, max_norm: f64) {
    for layer in transformer.layers.iter_mut() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_avg_pooling_layer) => {
                // println!("adaptive avg pooling layer with output size: {:?}", &adaptive_avg_pooling_layer.output_size);
            }
            LayerEnum::Embedding(_embedding_layer) => {}
            LayerEnum::Norm(norm_layer) => {
                norm_layer.max_norm = max_norm;
                norm_layer.global_norm = global_norm;
            }
            LayerEnum::RMSNorm(norm_layer) => {
                norm_layer.max_norm = max_norm;
                norm_layer.global_norm = global_norm;
            }

            LayerEnum::SelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    attention_head.max_norm = max_norm;
                    attention_head.global_norm = global_norm;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            norm_layer.max_norm = max_norm;
                            norm_layer.global_norm = global_norm;
                        }
                        LayerEnum::Norm(norm_layer) => {
                            norm_layer.max_norm = max_norm;
                            norm_layer.global_norm = global_norm;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SparseSelfAttention(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    attention_head.max_norm = max_norm;
                    attention_head.global_norm = global_norm;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            norm_layer.max_norm = max_norm;
                            norm_layer.global_norm = global_norm;
                        }
                        LayerEnum::Norm(norm_layer) => {
                            norm_layer.max_norm = max_norm;
                            norm_layer.global_norm = global_norm;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::SelfAttentionApproximation(self_attention_layer) => {
                for attention_head in self_attention_layer.attention_heads.iter_mut() {
                    attention_head.max_norm = max_norm;
                    attention_head.global_norm = global_norm;
                }
                if let Some(norm_layer) = self_attention_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            norm_layer.max_norm = max_norm;
                            norm_layer.global_norm = global_norm;
                        }
                        LayerEnum::Norm(norm_layer) => {
                            norm_layer.max_norm = max_norm;
                            norm_layer.global_norm = global_norm;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::FeedForward(ffn_layer) => {
                for layer in ffn_layer.layers.iter_mut() {
                    match layer {
                        LayerEnum::Dense(dense_layer) => {
                            dense_layer.max_norm = max_norm;
                            dense_layer.global_norm = global_norm;
                        }
                        LayerEnum::Linear(linear_layer) => {
                            linear_layer.max_norm = max_norm;
                            linear_layer.global_norm = global_norm;
                        }
                        _ => {}
                    }
                }
                if let Some(norm_layer) = ffn_layer.norm_layer.as_mut() {
                    match norm_layer {
                        LayerEnum::RMSNorm(norm_layer) => {
                            norm_layer.max_norm = max_norm;
                            norm_layer.global_norm = global_norm;
                        }
                        LayerEnum::Norm(norm_layer) => {
                            norm_layer.max_norm = max_norm;
                            norm_layer.global_norm = global_norm;
                        }
                        _ => {}
                    }
                }
            }
            LayerEnum::Dense(dense_layer) => {
                dense_layer.max_norm = max_norm;
                dense_layer.global_norm = global_norm;
            }
            LayerEnum::Linear(linear_layer) => {
                linear_layer.max_norm = max_norm;
                linear_layer.global_norm = global_norm;
            }
            LayerEnum::MultiLinear(linear_layer) => {
                linear_layer.max_norm = max_norm;
                linear_layer.global_norm = global_norm;
            }
            LayerEnum::Wavelet(_wavelet_layer) => {}
            LayerEnum::DiscreteWavelet(_wavelet_layer) => {}
            LayerEnum::Softmax(_softmax_layer) => {}
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
        }
    }
}

pub fn update_transformer(transformer_network: &mut NeuralNetwork, target_batch_ids: &Vec<Vec<u32>>) {
    update_global_norm(transformer_network);

    for layer in transformer_network.layers.iter_mut().rev() {
        match layer {
            LayerEnum::AdaptiveAvgPool1d(_adaptive_pooling) => {}
            LayerEnum::Embedding(embedding_layer) => {
                embedding_layer.update_parameters(&target_batch_ids, transformer_network.learning_rate);
            }
            LayerEnum::PositionalEncoding(_positional_encoding_layer) => {}
            LayerEnum::Norm(_norm_layer) => {}
            LayerEnum::SelfAttention(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SparseSelfAttention(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::SelfAttentionApproximation(attention_layer) => {
                attention_layer.update_parameters();
            }
            LayerEnum::FeedForward(dense_layer) => {
                dense_layer.update_parameters();
            }
            LayerEnum::Linear(linear_layer) => {
                linear_layer.update_parameters();
            }
            LayerEnum::MultiLinear(multi_linear_layer) => {
                multi_linear_layer.update_parameters();
            }
            LayerEnum::Wavelet(_wavelet_layer) => {}
            LayerEnum::DiscreteWavelet(wavelet_layer) => {
                wavelet_layer.update_parameters();
            }
            LayerEnum::Softmax(_softmax_layer) => {}
            _ => {
                println!("Layer type not supported for backward pass");
            }
        }
    }
}
