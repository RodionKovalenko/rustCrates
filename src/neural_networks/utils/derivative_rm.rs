use crate::neural_networks::{
    network_layers::layer::ActivationType,
    utils::{
        activation::sigmoid_complex,
        activation_rm::map_rm,
        derivative::{
            elu_derivative_complex, gelu_derivative_complex, leaky_relu_derivative_complex, relu_derivative_complex, selu_derivative_complex, sigmoid_derivative_complex, softplus_derivative_complex,
            softsign_derivative_complex, tanh_derivative_complex,
        },
        dtype::{C, ONE, ZERO},
        matrix::RowMajorMatrix,
    },
};

/// RM counterpart of `derivative::get_gradient_swish`.
pub fn swish_grad_rm(b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let one = C::new(ONE, ZERO);
    map_rm(b, |z| {
        let sigma = sigmoid_complex(&z);
        sigma + z * sigma * (one - sigma)
    })
}

/// RM counterpart of `derivative::get_gradient_complex`.
///
/// `activated_data` is the activation output, `input_data` the raw (pre-activation) values.
/// Like the Vec version, GELU and SOFTSIGN are derived from the raw values, all others from the activated ones.
pub fn get_gradient_complex_rm(activated_data: &RowMajorMatrix<C>, input_data: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    match activation {
        ActivationType::SIGMOID => map_rm(activated_data, sigmoid_derivative_complex),
        ActivationType::TANH => map_rm(activated_data, tanh_derivative_complex),
        ActivationType::LINEAR => map_rm(activated_data, |_| C::new(ONE, ZERO)),
        ActivationType::RELU => map_rm(activated_data, relu_derivative_complex),
        ActivationType::LEAKYRELU => map_rm(activated_data, |x| leaky_relu_derivative_complex(x, 0.01)),
        ActivationType::ELU => map_rm(activated_data, |x| elu_derivative_complex(x, 1.0)),
        ActivationType::SELU => map_rm(activated_data, selu_derivative_complex),
        ActivationType::GELU => map_rm(input_data, gelu_derivative_complex),
        ActivationType::SOFTSIGN => map_rm(input_data, softsign_derivative_complex),
        ActivationType::SOFTPLUS => map_rm(activated_data, softplus_derivative_complex),
        ActivationType::PROBIT => map_rm(activated_data, |_| C::new(ONE, ZERO)),
        ActivationType::RANDOM => activated_data.clone(),
        _ => RowMajorMatrix::create_empty(0, 0),
    }
}
