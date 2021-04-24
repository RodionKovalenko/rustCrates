use crate::network_components::layer::Layer;
use crate::network_components::layer::LayerType;
use crate::utils::derivative::get_derivative;

pub fn calculate_gradient(layers: &mut Vec<Layer<f64>>,
                          layer_ind: usize,
                          num_sets: usize,
                          learn_rate: f64) -> Vec<Vec<f64>> {
    let mut layer = layers[layer_ind].clone();

    // println!("input size: {}, {}, {}", layer.input_data.len(), layer.input_data[0].len(), layer.input_data[0][0].len());
    // println!("errors size: {}, {}", layer.errors.len(), layer.errors[0].len());

    // println!("errors current layer size: {}, {}", layers[layer_ind].errors.len(), layers[layer_ind].errors[0].len());
    if matches!(layer.layer_type, LayerType::OutputLayer) {
        // calculate gradients and error
        for inp_set_ind in 0..num_sets {
            for j in 0..layer.input_weights[0].len() {
                for i in 0..layer.input_weights.len() {
                    // calculate gradients and errors for output layer
                    for d in 0..layer.activated_output[inp_set_ind].len() {
                        layer.gradient[i][j] += layer.errors[inp_set_ind][j] *
                            layer.input_data[inp_set_ind][d][j];
                    }
                }
            }
        }

        // update weights and bias
        for inp_set_ind in 0..num_sets {
            for j in 0..layer.input_weights[0].len() {
                for i in 0..layer.input_weights.len() {
                    // update layer weights
                    layers[layer_ind].input_weights[i][j] -=
                        (layer.gradient[i][j] * learn_rate) / num_sets as f64;
                }
                // update bias
                layers[layer_ind].layer_bias[j] -=
                    (learn_rate * layer.errors[inp_set_ind][j]) / num_sets as f64;
            }
        }
    } else {
        for inp_set_ind in 0..num_sets {
            for j in 0..layer.input_weights[0].len() {
                // errors of the next layer
                for e in 0..layers[layer_ind + 1].errors[0].len() {
                    layer.errors[inp_set_ind][j] +=
                        layers[layer_ind + 1].errors[inp_set_ind][e]
                            * layers[layer_ind + 1].input_weights[j][e];
                }

                for d in 0..layer.activated_output[inp_set_ind].len() {
                    // multiply errors with derivative of output
                    layer.errors[inp_set_ind][j] *= get_derivative(
                        layer.activated_output[inp_set_ind][d][j],
                        layer.activation_type.clone());
                }

                for i in 0..layer.input_weights.len() {
                    // calculate gradients and errors for current layer
                    for d in 0..layer.activated_output[inp_set_ind].len() {
                        layer.gradient[i][j] += layer.errors[inp_set_ind][j] *
                            layer.input_data[inp_set_ind][d][i];
                    }
                }
            }
        }

        // update weights and bias
        for inp_set_ind in 0..num_sets {
            for j in 0..layer.input_weights[0].len() {
                for i in 0..layer.input_weights.len() {
                    // update layer weights
                    layers[layer_ind].input_weights[i][j] -=
                        (layer.gradient[i][j] * learn_rate) / num_sets as f64;
                }
                // update bias

                layers[layer_ind].layer_bias[j] -=
                    (learn_rate * layer.errors[inp_set_ind][j]) / num_sets as f64;
            }
        }
    }
    layers[layer_ind].errors = layer.errors.clone();
    layers[layer_ind].gradient = layer.gradient.clone();
    layers[layer_ind].gradient.clone()
}