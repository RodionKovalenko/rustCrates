use std::f64::consts::PI;
use std::fmt::Debug;
use std::f64::consts::E;
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
    T::from(1.0) / (T::from(1.0) + T::from(E.powf(-1.0 * value.into())))
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
    let output: f64 = (E.powf(value) - E.powf(-value))
        / (E.powf(value) + E.powf(-1.0 * value));

    output
}

pub fn gauss<T, V, G>(v: &T, m: &V, var: &G) -> f64
    where
        T: Debug + Clone + Into<f64> + Mul<Output=T> + Add<Output=T> + Div<Output=T>,
        V: Debug + Clone + Into<f64> + Mul<Output=V> + Add<Output=V> + Div<Output=V>,
        G: Debug + Clone + Into<f64> + Mul<Output=G> + Add<Output=G> + Div<Output=G>
{
    let value = v.clone().into();
    let mean = m.clone().into();
    let sigma = var.clone().into().sqrt();

    return (1.0 / (sigma * (2.0 * PI).sqrt()))
        * E.powf(-1.0 * (value - mean).powf(2.0) / (2.0 * var.clone().into()));

}

pub fn guass_d1<T, V, G>(v: &Vec<T>, m: &V, s: &G) -> Vec<f64>
    where
        T: Debug + Clone + Into<f64> + Mul<Output=T> + Add<Output=T> + Div<Output=T>,
        V: Debug + Clone + Into<f64> + Mul<Output=V> + Add<Output=V> + Div<Output=V>,
        G: Debug + Clone + Into<f64> + Mul<Output=G> + Add<Output=G> + Div<Output=G>
{
    let mean = m.clone().into();
    let sigma = s.clone().into();

    let mut result = vec![0.0; v.len()];

    for i in 0..v.len() {
        result[i] = gauss(&v[i], &mean, &sigma);
    }

    result
}

pub fn guass_d2<T, V, G>(v: &Vec<Vec<T>>, m: &V, s: &G) -> Vec<Vec<f64>>
    where
        T: Debug + Clone + Into<f64> + Mul<Output=T> + Add<Output=T> + Div<Output=T>,
        V: Debug + Clone + Into<f64> + Mul<Output=V> + Add<Output=V> + Div<Output=V>,
        G: Debug + Clone + Into<f64> + Mul<Output=G> + Add<Output=G> + Div<Output=G>
{
    let mean = m.clone().into();
    let sigma = s.clone().into();

    let mut result: Vec<Vec<f64>> = vec![vec![0.0; v[0].len()]; v.len()];

    for i in 0..v.len() {
        result[i] = guass_d1(&v[i], &mean, &sigma);
    }

    result
}