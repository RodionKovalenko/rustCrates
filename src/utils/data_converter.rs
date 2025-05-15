use num_complex::Complex;
use crate::utils::num_trait::{Array, ArrayType};

pub fn convert_to_f64_1d<T>(data: &T) -> Vec<f64>
    where T: ArrayType
{
    let result = data.as_f64_array();

    match result {
        Array::Array1D(vec) => vec,
        _ => vec![0.0]
    }
}

pub fn convert_to_f64_2d<T>(data: &T) -> Vec<Vec<f64>>
    where T: ArrayType
{
    let result = data.as_f64_array();

    match result {
        Array::Array2D(vec) => vec,
        _ => vec![vec![0.0]]
    }
}

pub fn convert_to_f64_3d<T>(data: &T) -> Vec<Vec<Vec<f64>>>
    where T: ArrayType
{
    let result = data.as_f64_array();

    match result {
        Array::Array3D(vec) => vec,
        _ => vec![vec![vec![0.0]]]
    }
}

pub fn convert_to_f64_4d<T>(data: &T) -> Vec<Vec<Vec<Vec<f64>>>>
    where T: ArrayType
{
    let result = data.as_f64_array();

    match result {
        Array::Array4D(vec) => vec,
        _ => vec![vec![vec![vec![0.0]]]]
    }
}

pub fn convert_to_f64_5d<T>(data: &T) -> Vec<Vec<Vec<Vec<Vec<f64>>>>>
    where T: ArrayType
{
    let result = data.as_f64_array();

    match result {
        Array::Array5D(vec) => vec,
        _ => vec![vec![vec![vec![vec![0.0]]]]]
    }
}

pub fn convert_to_f64_6d<T>(data: &T) -> Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>
    where T: ArrayType
{
    let result = data.as_f64_array();

    match result {
        Array::Array6D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![0.0]]]]]]
    }
}

pub fn convert_to_f64_7d<T>(data: &T) -> Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>
    where T: ArrayType
{
    let result = data.as_f64_array();

    match result {
        Array::Array7D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![vec![0.0]]]]]]]
    }
}

pub fn convert_to_f64_8d<T>(data: &T) -> Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>>
    where T: ArrayType
{
    let result = data.as_f64_array();

    match result {
        Array::Array8D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![vec![vec![0.0]]]]]]]]
    }
}

pub fn convert_to_c_f64_1d<T>(data: &T) -> Vec<Complex<f64>>
    where T: ArrayType
{
    let result = data.to_complex_array();

    match result {
        Array::ArrayC1D(vec) => vec,
        _ => vec![Complex::new(0.0, 0.0)]
    }
}

pub fn convert_to_c_f64_2d<T>(data: &T) -> Vec<Vec<Complex<f64>>>
    where T: ArrayType
{
    let result = data.to_complex_array();

    match result {
        Array::ArrayC2D(vec) => vec,
        _ => vec![vec![Complex::new(0.0, 0.0)]]
    }
}


pub fn convert_to_c_array_f64_1d(data: Array) -> Vec<Complex<f64>>
{
    match data {
        Array::ArrayC1D(vec) => vec,
        _ => vec![Complex::new(0.0, 0.0)]
    }
}

pub fn convert_to_c_array_f64_2d(data: Array) -> Vec<Vec<Complex<f64>>>
{
    match data {
        Array::ArrayC2D(vec) => vec,
        _ => vec![vec![Complex::new(0.0, 0.0)]]
    }
}

pub fn convert_to_c_array_f64_3d(data: Array) -> Vec<Vec<Vec<Complex<f64>>>>
{
    match data {
        Array::ArrayC3D(vec) => vec,
        _ => vec![vec![vec![Complex::new(0.0, 0.0)]]]
    }
}

pub fn convert_to_c_array_f64_4d(data: Array) -> Vec<Vec<Vec<Vec<Complex<f64>>>>>
{
    match data {
        Array::ArrayC4D(vec) => vec,
        _ => vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]
    }
}

pub fn convert_to_c_array_f64_5d(data: Array) -> Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>
{
    match data {
        Array::ArrayC5D(vec) => vec,
        _ => vec![vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]]
    }
}

pub fn convert_to_c_array_f64_6d(data: Array) -> Vec<Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>>
{
    match data {
        Array::ArrayC6D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]]]
    }
}

pub fn convert_to_c_array_f64_7d(data: Array) -> Vec<Vec<Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>>>
{
    match data {
        Array::ArrayC7D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]]]]
    }
}
pub fn convert_to_c_array_f64_8d(data: Array) -> Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>>>>
{
    match data {
        Array::ArrayC8D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]]]]]
    }
}

pub fn convert_to_c_f64_3d<T>(data: &T) -> Vec<Vec<Vec<Complex<f64>>>>
    where T: ArrayType
{
    let result = data.to_complex_array();

    match result {
        Array::ArrayC3D(vec) => vec,
        _ => vec![vec![vec![Complex::new(0.0, 0.0)]]]
    }
}

pub fn convert_to_c_f64_4d<T>(data: &T) -> Vec<Vec<Vec<Vec<Complex<f64>>>>>
    where T: ArrayType
{
    let result = data.to_complex_array();

    match result {
        Array::ArrayC4D(vec) => vec,
        _ => vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]
    }
}

pub fn convert_to_c_f64_5d<T>(data: &T) -> Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>
    where T: ArrayType
{
    let result = data.to_complex_array();

    match result {
        Array::ArrayC5D(vec) => vec,
        _ => vec![vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]]
    }
}

pub fn convert_to_c_f64_6d<T>(data: &T) -> Vec<Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>>
    where T: ArrayType
{
    let result = data.to_complex_array();

    match result {
        Array::ArrayC6D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]]]
    }
}

pub fn convert_to_c_f64_7d<T>(data: &T) -> Vec<Vec<Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>>>
    where T: ArrayType
{
    let result = data.to_complex_array();

    match result {
        Array::ArrayC7D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]]]]
    }
}

pub fn convert_to_c_f64_8d<T>(data: &T) -> Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<Complex<f64>>>>>>>>>
    where T: ArrayType
{
    let result = data.to_complex_array();

    match result {
        Array::ArrayC8D(vec) => vec,
        _ => vec![vec![vec![vec![vec![vec![vec![vec![Complex::new(0.0, 0.0)]]]]]]]]
    }
}

pub fn convert_c_to_f64_3d(data: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<f64>>>
{
    let mut result: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; data[0][0].len()]; data[0].len()]; data.len()];

    for i in 0..data.len() {
        for j in 0..data[i].len() {
            for k in 0..data[i][j].len() {
                result[i][j][k] = data[i][j][k].re;
            }
        }
    }

    result
}