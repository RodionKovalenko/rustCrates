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