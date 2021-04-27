use crate::network_components::layer::Layer;
use crate::network_components::layer::LayerType;
use crate::utils::derivative::get_derivative;
use rand::Rng;

pub fn calculate_gradient(layers: &mut Vec<Layer<f64>>,
                          layer_ind: usize,
                          num_sets: usize,
                          mut learn_rate: f64) -> Vec<Vec<f64>> {
    let mut layer = layers[layer_ind].clone();
    let mut rng = rand::thread_rng();
    let gamma = 0.6;
    let p = 0.7;
    let ada_grad_optimizer = 0.0;
    let mut delta_theta;
    let mut r_sqrt = 0.0;

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

        // ada_grad_optimizer = get_ada_grad_optimizer(&layer.gradient);
        r_sqrt = get_r_rms_prop(&layer.gradient, p, layer.rmsp_p);
        learn_rate = learn_rate / r_sqrt;
        layer.rmsp_p = r_sqrt;

        // update weights and bias
        for j in 0..layer.input_weights[0].len() {
            for i in 0..layer.input_weights.len() {
                // dropout
                if rng.gen_bool(0.05) {
                    continue;
                }
                // update layer weights
                delta_theta = ((gamma * layer.previous_gradient[i][j]) -
                    (layer.gradient[i][j] * learn_rate))
                    / num_sets as f64;
                layer.input_weights[i][j] += delta_theta;
                // momentum
                layer.previous_gradient[i][j] = delta_theta;
            }
            // update bias
            for inp_set_ind in 0..num_sets {
                layer.layer_bias[j] -=
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
        // ada_grad_optimizer = get_ada_grad_optimizer(&layer.gradient);
        r_sqrt = get_r_rms_prop(&layer.gradient, p, layer.rmsp_p);
        learn_rate = learn_rate / r_sqrt;
        layer.rmsp_p = r_sqrt;

        // update weights and bias
        for j in 0..layer.input_weights[0].len() {
            for i in 0..layer.input_weights.len() {
                // dropout
                if rng.gen_bool(0.05) {
                    continue;
                }
                // update layer weights
                // update layer weights
                delta_theta = ((gamma * layer.previous_gradient[i][j]) -
                    (layer.gradient[i][j] * learn_rate))
                    / num_sets as f64;
                layer.input_weights[i][j] += delta_theta;
                // momentum
                layer.previous_gradient[i][j] = delta_theta;
            }
            // update bias
            for inp_set_ind in 0..num_sets {
                layer.layer_bias[j] -=
                    (learn_rate * layer.errors[inp_set_ind][j]) / num_sets as f64;
            }
        }
    }
    layers[layer_ind] = layer.clone();
    layers[layer_ind].gradient.clone()
}

pub fn get_ada_grad_optimizer(gradients: &Vec<Vec<f64>>) -> f64 {
    // https://mlfromscratch.com/optimizers-explained/#/
    // sum must not be null, because of sqrt
    let mut sum = 0.0000000005;

    for i in 0..gradients.len() {
        for j in 0..gradients[0].len() {
            sum += gradients[i][j] * gradients[i][j];
        }
    }

    sum.sqrt()
}

pub fn get_r_rms_prop(gradients: &Vec<Vec<f64>>, p: f64, r: f64) -> f64 {
    // https://mlfromscratch.com/optimizers-explained/#/
    // sum must not be null, because of sqrt
    let mut sum = 0.0000005;

    for i in 0..gradients.len() {
        for j in 0..gradients[0].len() {
            sum += p * r + (1.0 - p) * gradients[i][j] * gradients[i][j];
        }
    }

    sum.sqrt()
}