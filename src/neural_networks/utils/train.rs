use crate::network_components::layer::Layer;
use crate::network_components::layer::LayerType;
use crate::utils::matrix::*;
use std::fmt::Debug;
use std::ops::{Mul, AddAssign, Sub, Add, Div};
use crate::utils::derivative::get_derivative;
use num::Zero;
use crate::network_components::input::Data;
use rand::Rng;

pub fn calculate_gradient<T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Sub<Output=T> + Add<Output=T> + Div<Output=T>>
(layers: &mut Vec<Layer<T>>, ind: usize) -> Vec<Vec<T>> {
    let mut layer = layers[ind].clone();
    let layer_inact_output = layer.inactivated_output.clone();

    if matches!(layer.layer_type, LayerType::OutputLayer) {
        layer.gradient = layer.errors.clone();
    } else {
        let mut next_layer = layers[ind + 1].clone();
        layer.gradient = next_layer.gradient.clone();
    }

    // println!("gradient layer {:?}, {:?}", layer.layer_type, layer.gradient);

    if !matches!(layer.layer_type, LayerType::OutputLayer) {
        layer.gradient = multiple_generic(&layers[ind + 1].gradient,
                                          &layers[ind + 1].input_weights);

        // println!("gradient matrix i {}, {}", &layers[ind + 1].gradient.len(), &layers[ind + 1].gradient[0].len());
        //println!("weight matrix i {}, {}", &layers[ind + 1].input_weights.len(), &layers[ind + 1].input_weights[0].len());
    }

    for i in 0..layer.gradient.len() {
        for j in 0..layer.gradient[0].len() {
            layer.gradient[i][j] =
                layer.gradient[i][j].clone()
                    * get_derivative(layer_inact_output[i][j].clone(),
                                     layer.activation_type.clone());

            if matches!(layer.layer_type, LayerType::OutputLayer) {
                layer.gradient[i][j] = layer.gradient[i][j].clone() * T::from(-1.0);
            }
        }
    }

    println!("gradient layer result {:?}, {:?}", layer.layer_type, layer.gradient);
    println!("gradient layer size {:?}, {:?}", layer.gradient.len(), layer.gradient[0].len());

    layers[ind].gradient = layer.gradient;
    layers[ind].gradient.clone()
}

pub fn update_weights<T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Sub<Output=T> + Add<Output=T> + Div<Output=T>>
(layers: &mut Vec<Layer<T>>, ind: usize) {
    let mut layer = layers[ind].clone();
    let mut gradient = Vec::new();
    let mut input;
    let mut rnd = rand::thread_rng();

    if matches!(layer.layer_type, LayerType::InputLayer) {
        input = layer.input_data.clone();

        for i in 0..input.len() {
            for j in 0..input[0].len() {
                if j < layer.gradient[0].len() {
                    layer.gradient[0][j] = layer.gradient[0][j].clone()
                        + (layer.gradient[0][j].clone() * input[i][j].clone());
                }
            }
        }
    } else {
        input = layers[ind - 1].inactivated_output.clone();

        for i in 0..input.len() {
            for j in 0..input[0].len() {
                if j < layer.gradient[0].len() {
                    layer.gradient[0][j] = layer.gradient[0][j].clone()
                        + (layer.gradient[0][j].clone() * input[i][j].clone());
                }
            }
        }
    }

    // println!("------------------------------");
    if matches!(layer.layer_type, LayerType::OutputLayer) {
        gradient = transpose(&layer.gradient);
    }

    // println!("update weights: gradient matrix i {}, {}", &layers[ind].gradient.len(), &layers[ind].gradient[0].len());
    // println!("update weights: weight matrix i {}, {}", &layer.input_weights.len(), &layer.input_weights[0].len());

    for i in 0..layer.input_weights.len() {
        for j in 0..layer.input_weights[0].len() {
            if matches!(layer.layer_type, LayerType::OutputLayer) {
                layer.input_weights[i][j] = layer.input_weights[i][j].clone() + (T::from(0.01) *
                    gradient[i][0].clone());
            } else {
                layer.input_weights[i][j] = layer.input_weights[i][j].clone()
                    + (T::from(0.01) * layer.gradient[0][j].clone());
            }

            if layer.input_weights[i][j].clone().into() as f64 > 1.0 {
                let value = rnd.gen_range(-0.6, 0.6);
                layer.input_weights[i][j] = T::from(value);
            }
        }
    }

    // println!("update weights layer gradient {:?}, {:?}", layer.layer_type, layer.gradient);
    // println!("------------------------------");

    layers[ind] = layer;
}