use crate::network_components::layer::Layer;
use crate::network_components::layer::LayerType;
use std::fmt::Debug;
use std::ops::{Mul, AddAssign, Sub, Add, Div};
use crate::utils::derivative::get_derivative;

pub fn calculate_gradient<T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Sub<Output=T> + Add<Output=T> + Div<Output=T>>
(layers: &mut Vec<Layer<T>>, layer_ind: usize, num_sets: usize, learn_rate: f64) -> Vec<Vec<T>> {
    let mut layer = layers[layer_ind].clone();

    // println!("input size: {}, {}, {}", layer.input_data.len(), layer.input_data[0].len(), layer.input_data[0][0].len());
    // println!("errors size: {}, {}", layer.errors.len(), layer.errors[0].len());

    if matches!(layer.layer_type, LayerType::OutputLayer) {
        for inp_set_ind in 0..num_sets {
            for j in 0..layer.input_weights[0].len() {
                for i in 0..layer.input_weights.len() {
                    layer.gradient[i][j] = layer.gradient[i][j].clone() +
                        layer.errors[inp_set_ind][j].clone() *
                            layer.input_data[inp_set_ind][0][j].clone();

                    layers[layer_ind].input_weights[i][j] =
                        layers[layer_ind].input_weights[i][j].clone() +
                            ((layer.gradient[i][j].clone()
                                * T::from(learn_rate)) / T::from(num_sets as f64));
                }

                layers[layer_ind].layer_bias[j] = layers[layer_ind].layer_bias[j].clone() +
                    ((T::from(learn_rate) * layer.errors[inp_set_ind][j].clone())
                        / T::from(num_sets as f64));
            }
        }
    } else {
        for inp_set_ind in 0..num_sets {
            for j in 0..layer.input_weights[0].len() {
                for e in 0..layers[layer_ind + 1].errors[0].len() {
                    layer.errors[inp_set_ind][j] = layer.errors[inp_set_ind][j].clone() +
                        (layers[layer_ind + 1].errors[inp_set_ind][e].clone()
                            * layers[layer_ind + 1].input_weights[j][e].clone())
                }

                layer.errors[inp_set_ind][j] = layer.errors[inp_set_ind][j].clone() *
                    get_derivative(layer.activated_output[inp_set_ind][0][j].clone(),
                                   layer.activation_type.clone());

                for i in 0..layer.input_weights.len() {
                    layer.gradient[i][j] = layer.gradient[i][j].clone() +
                        layer.errors[inp_set_ind][j].clone() *
                            layer.input_data[inp_set_ind][0][i].clone();

                    layers[layer_ind].input_weights[i][j] =
                        layers[layer_ind].input_weights[i][j].clone() +
                            ((layer.gradient[i][j].clone()
                                * T::from(learn_rate)) / T::from(num_sets as f64));
                }

                layers[layer_ind].layer_bias[j] = layers[layer_ind].layer_bias[j].clone() +
                    ((T::from(learn_rate) * layer.errors[inp_set_ind][j].clone())
                        / T::from(num_sets as f64));
            }
        }
    }
    layers[layer_ind].errors = layer.errors.clone();
    layers[layer_ind].gradient = layer.gradient;
    layers[layer_ind].gradient.clone()
}