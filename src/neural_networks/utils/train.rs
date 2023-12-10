use rand::Rng;
use core::num::FpCategory::Nan;
use num::abs;


// pub fn calculate_gradient(layers: &mut Vec<Layer<f64>>,
//                           layer_ind: &usize,
//                           num_sets: &usize,
//                           mut learn_rate: &f64,
//                           iter: &i32,
//                           minibatch_start: &usize, minibatch_size: &usize) -> Vec<Vec<f64>> {
//     let mut layer = layers[*layer_ind].clone();
//     let mut rng = rand::thread_rng();
//     let mut b1 = 0.9;
//     let b2 = 0.999;
//     let e = 0.00000001;
//     let mut v1 = 0.0;
//     let mut v_hat = 0.0;
//     let mut m_hat = 0.0;
//     let mut m1 = 0.0;
//     let mut delta_theta: f64;
//
//     let mut minibatch_end = minibatch_start + minibatch_size;
//
//     if (minibatch_end > *num_sets) {
//         minibatch_end = *num_sets;
//     }
//
//     // println!("input size: {}, {}, {}", layer.input_data.len(), layer.input_data[0].len(), layer.input_data[0][0].len());
//     // println!("errors size: {}, {}", layer.errors.len(), layer.errors[0].len());
//     // println!("errors current layer size: {}, {}", layers[layer_ind].errors.len(), layers[layer_ind].errors[0].len());
//     if matches!(layer.layer_type, LayerType::OutputLayer) {
//         // calculate gradients and error
//         for inp_set_ind in minibatch_start..&minibatch_end {
//             for j in 0..layer.input_weights[0].len() {
//                 for i in 0..layer.input_weights.len() {
//                     // calculate gradients and errors for output layer
//                     for d in 0..layer.activated_output[inp_set_ind].len() {
//                         layer.gradient[i][j] += (layer.errors[inp_set_ind][j] *
//                             layer.input_data[inp_set_ind][d][j]) / num_sets as f64;
//                     }
//                 }
//             }
//         }
//
//         //ada_grad_optimizer = get_ada_grad_optimizer(&layer.gradient);
//
//         // update weights and bias
//         for j in 0..layer.input_weights[0].len() {
//             for i in 0..layer.input_weights.len() {
//                 // dropout
//                 if rng.gen_bool(0.05) {
//                     continue;
//                 }
//                 m1 = get_adam_value(&layer.gradient[i][j], b1, layer.m1[i][j]);
//                 v1 = get_r_rms_value(&layer.gradient[i][j], b2, layer.v1[i][j]);
//
//                 m_hat = m1 / (1.0 - b1.powf((iter + 1) as f64));
//                 v_hat = v1 / (1.0 - b2.powf((iter + 1) as f64));
//
//                 delta_theta = ((learn_rate * m_hat) / (v_hat.sqrt() + e));
//                 layer.input_weights[i][j] -= delta_theta;
//
//                 layer.m1[i][j] = m1;
//                 layer.v1[i][j] = v1;
//             }
//             // update bias
//             for inp_set_ind in minibatch_start..minibatch_end {
//                 layer.layer_bias[j] -=
//                     (learn_rate * layer.errors[inp_set_ind][j]) / num_sets as f64;
//             }
//         }
//     } else {
//         for inp_set_ind in minibatch_start..minibatch_end {
//             for j in 0..layer.input_weights[0].len() {
//                 // errors of the next layer
//                 for e in 0..layers[layer_ind + 1].errors[0].len() {
//                     layer.errors[inp_set_ind][j] +=
//                         layers[layer_ind + 1].errors[inp_set_ind][e]
//                             * layers[layer_ind + 1].input_weights[j][e];
//                 }
//
//                 for d in 0..layer.activated_output[inp_set_ind].len() {
//                     // multiply errors with derivative of output
//                     layer.errors[inp_set_ind][j] *= get_derivative(
//                         layer.activated_output[inp_set_ind][d][j],
//                         layer.activation_type.clone());
//                 }
//
//                 for i in 0..layer.input_weights.len() {
//                     // calculate gradients and errors for current layer
//                     for d in 0..layer.activated_output[inp_set_ind].len() {
//                         layer.gradient[i][j] += (layer.errors[inp_set_ind][j] *
//                             layer.input_data[inp_set_ind][d][i]) / num_sets as f64;
//                     }
//                 }
//             }
//         }
//
//         //ada_grad_optimizer = get_ada_grad_optimizer(&layer.gradient);
//
//         // update weights and bias
//         for j in 0..layer.input_weights[0].len() {
//             for i in 0..layer.input_weights.len() {
//                 // dropout
//                 if rng.gen_bool(0.05) {
//                     continue;
//                 }
//                 // // update layer weights
//                 m1 = get_adam_value(&layer.gradient[i][j], b1, layer.m1[i][j]);
//                 v1 = get_r_rms_value(&layer.gradient[i][j], b2, layer.v1[i][j]);
//
//                 m_hat = m1 / (1.0 - b1.powf((iter + 1) as f64));
//                 v_hat = v1 / (1.0 - b2.powf((iter + 1) as f64));
//
//                 delta_theta = ((learn_rate * m_hat) / (v_hat.sqrt() + e));
//                 layer.input_weights[i][j] -= delta_theta;
//
//                 layer.m1[i][j] = m1;
//                 layer.v1[i][j] = v1;
//             }
//             // update bias
//             for inp_set_ind in minibatch_start..minibatch_end {
//                 layer.layer_bias[j] -=
//                     (learn_rate * layer.errors[inp_set_ind][j]) / num_sets as f64;
//             }
//         }
//     }
//
//     layers[layer_ind] = layer.clone();
//     layers[layer_ind].gradient.clone()
// }
//
// pub fn get_ada_grad_optimizer(gradients: &Vec<Vec<f64>>) -> f64 {
//     // https://mlfromscratch.com/optimizers-explained/#/
//     // sum must not be null, because of sqrt
//     let mut sum = 0.0000000005;
//
//     for i in 0..gradients.len() {
//         for j in 0..gradients[0].len() {
//             sum += gradients[i][j] * gradients[i][j];
//         }
//     }
//
//     sum.sqrt()
// }
//
// pub fn get_r_rms_prop(gradients: &Vec<Vec<f64>>, b1: f64, r: f64) -> f64 {
//     // https://mlfromscratch.com/optimizers-explained/#/
//     // sum must not be null, because of sqrt
//     let mut sum = 0.0000005;
//
//     for i in 0..gradients.len() {
//         for j in 0..gradients[0].len() {
//             sum += b1 * r + (1.0 - b1) * gradients[i][j] * gradients[i][j];
//         }
//     }
//
//     sum
// }
//
// pub fn get_r_rms_value(gradient: &f64, b2: f64, v1: f64) -> f64 {
//     // https://mlfromscratch.com/optimizers-explained/#/
//     // sum must not be null, because of sqrt
//     let mut sum = 0.0000005;
//
//     sum += b2 * v1 + (1.0 - b2) * gradient * gradient;
//     abs(sum)
// }
//
// pub fn get_adam(gradients: &Vec<Vec<f64>>, b1: f64, r: f64) -> f64 {
//     // https://mlfromscratch.com/optimizers-explained/#/
//     // sum must not be null, because of sqrt
//     let mut sum = 0.0000005;
//
//     for i in 0..gradients.len() {
//         for j in 0..gradients[0].len() {
//             sum += b1 * r + (1.0 - b1) * gradients[i][j];
//         }
//     }
//
//     sum
// }
//
// pub fn get_adam_value(gradient: &f64, b1: f64, m1: f64) -> f64 {
//     // https://mlfromscratch.com/optimizers-explained/#/
//     // sum must not be null, because of sqrt
//     let mut sum = 0.0000005;
//
//
//     sum += b1 * m1 + (1.0 - b1) * gradient;
//
//     sum
// }