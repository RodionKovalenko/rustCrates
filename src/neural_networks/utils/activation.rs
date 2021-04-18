use std::fmt::Debug;
use std::f32::consts::E;
use std::ops::{Mul, AddAssign, Add, Div};

// struct StoredValue {
//     val: f32,
// }
//
// impl From<f64> for StoredValue {
//     fn from(value: f64) -> StoredValue {
//         StoredValue { val: value as f32 }
//     }
// }

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