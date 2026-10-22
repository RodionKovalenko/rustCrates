use crate::neural_networks::{
    network_layers::layer::ActivationType,
    utils::{
        activation::{elu_complex, gelu_complex, leaky_relu_complex, relu_complex, selu_complex, sigmoid_complex, softplus_complex, softsign_complex, swiglu, tanh_complex},
        dtype::{r, C},
        matrix::RowMajorMatrix,
    },
};

/// Applies `f` elementwise and returns a new matrix with the same shape.
#[inline]
pub fn map_rm<F: Fn(C) -> C>(data: &RowMajorMatrix<C>, f: F) -> RowMajorMatrix<C> {
    RowMajorMatrix::from_data(data.rows, data.cols, data.data.iter().map(|&x| f(x)).collect())
}

/// RM counterpart of `activation::swish`.
pub fn swish_rm(data: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    map_rm(data, |x| x * sigmoid_complex(&x))
}

/// RM counterpart of `activation::swiglu_matrix`: each row is split in half `[a | b]`, output is `a * b * sigmoid(b)`.
pub fn swiglu_matrix_rm(data: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let half = data.cols / 2;
    let mut out = RowMajorMatrix::create_empty(data.rows, half);
    for row in 0..data.rows {
        let src = data.row_range(row);
        let dst = out.row_range(row);
        for col in 0..half {
            out.data[dst.start + col] = swiglu(&data.data[src.start + col], &data.data[src.start + half + col]);
        }
    }
    out
}

/// RM counterpart of `activation::activate_output_complex`.
pub fn activate_output_complex_rm(data: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    match activation {
        ActivationType::SIGMOID => map_rm(data, |x| sigmoid_complex(&x)),
        ActivationType::TANH => map_rm(data, tanh_complex),
        ActivationType::LINEAR => data.clone(),
        ActivationType::RELU => map_rm(data, relu_complex),
        ActivationType::LEAKYRELU => map_rm(data, |x| leaky_relu_complex(x, r(0.01))),
        ActivationType::ELU => map_rm(data, |x| elu_complex(x, r(1.0))),
        ActivationType::SELU => map_rm(data, |x| selu_complex(&x)),
        ActivationType::GELU => map_rm(data, |x| gelu_complex(&x)),
        ActivationType::SOFTSIGN => map_rm(data, |x| softsign_complex(&x)),
        ActivationType::SOFTPLUS => map_rm(data, |x| softplus_complex(&x)),
        ActivationType::PROBIT => data.clone(),
        ActivationType::RANDOM => data.clone(),
        ActivationType::SWiGLU => swiglu_matrix_rm(data),
        _ => RowMajorMatrix::create_empty(0, 0),
    }
}
