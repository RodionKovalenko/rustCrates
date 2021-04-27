use std::fmt::Debug;
use std::f32::consts::E;
use std::ops::{Mul, AddAssign, Add, Div};

pub fn sigmoid<T: Debug + Clone + From<f64> + Into<f64> +
Mul<Output=T> + AddAssign + Add<Output=T> + Div<Output=T>>(matrix_a: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut result_matrix = matrix_a.clone();

    for j in 0..matrix_a[0].len() {
        for i in 0..matrix_a.len() {
            result_matrix[i][j] = sigmoid_value(matrix_a[i][j].clone());
        }
    }

    result_matrix
}

pub fn sigmoid_value<T: Debug + Clone + From<f64> + Into<f64> +
Mul<Output=T> + AddAssign + Add<Output=T> + Div<Output=T>>(value: T) -> T {
    T::from(1.0) / (T::from(1.0) + T::from(E.powf(-1.0 * value.into() as f32) as f64))
}

pub fn tanh(matrix_a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result_matrix = matrix_a.clone();

    for j in 0..matrix_a[0].len() {
        for i in 0..matrix_a.len() {
            result_matrix[i][j] = tanh_value(matrix_a[i][j]);
        }
    }

    result_matrix
}

pub fn tanh_value(value: f64) -> f64 {
    let output = (E.powf(value as f32) - E.powf(-value as f32)) as f64
        / (E.powf(value as f32) + E.powf(-1.0 * value as f32)) as f64;

    output
}