use crate::neural_networks::{
    network_layers::layer::ActivationType,
    utils::{
        activation::sigmoid_complex,
        derivative::gelu_derivative_complex,
        dtype::{C, ONE, ZERO},
        matrix::RowMajorMatrix,
    },
};

pub fn swish_grad_rm(b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(b.rows, b.cols, vec![C::new(ZERO, ZERO); b.rows * b.cols]);
    let one = C::new(ONE, ZERO);
    for i in 0..out.data.len() {
        let z = b.data[i];
        let sigma = sigmoid_complex(&z);
        out.data[i] = sigma + z * sigma * (one - sigma);
    }
    out
}

pub fn activation_derivative_rm(activated: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(activated.rows, activated.cols, vec![C::new(ZERO, ZERO); activated.rows * activated.cols]);
    match activation {
        ActivationType::LINEAR => {
            for v in out.data.iter_mut() {
                *v = C::new(ONE, ZERO);
            }
        }
        ActivationType::TANH => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = C::new(ONE, ZERO) - (z * z);
            }
        }
        ActivationType::SIGMOID => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = z * (C::new(ONE, ZERO) - z);
            }
        }
        ActivationType::RELU => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = if z.re > ZERO { C::new(ONE, ZERO) } else { C::new(ZERO, ZERO) };
            }
        }
        ActivationType::LEAKYRELU => {
            for (dst, &z) in out.data.iter_mut().zip(activated.data.iter()) {
                *dst = if z.re > ZERO { C::new(ONE, ZERO) } else { C::new(0.01, ZERO) };
            }
        }
        _ => {}
    }
    out
}

pub fn activation_derivative_from_raw_rm(raw: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(raw.rows, raw.cols, vec![C::new(ZERO, ZERO); raw.rows * raw.cols]);
    match activation {
        ActivationType::GELU => {
            for (dst, &z) in out.data.iter_mut().zip(raw.data.iter()) {
                *dst = gelu_derivative_complex(z);
            }
        }
        _ => {}
    }
    out
}
