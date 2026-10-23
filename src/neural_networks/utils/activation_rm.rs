use crate::neural_networks::{
    network_layers::layer::ActivationType,
    utils::{
        activation::{
            elu_complex, gelu_complex, leaky_relu_complex, logit_indices_for_row, relu_complex, selu_complex, sigmoid_complex, softmax_ce_grad_complex_slice, softmax_row_complex, softplus_complex,
            softsign_complex, swiglu, tanh_complex, target_offset,
        },
        dtype::{r, Real, C, ZERO},
        matrix_rm::RowMajorMatrix,
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

/// RM counterpart of `activation::softmax_backward_real_with_gradient`.
///
/// - `logits` are shape (seq_len x vocab_or_k)
/// - `logit_indices[t]` (optional) maps sparse column -> original vocab index for that timestep
pub fn softmax_backward_real_with_gradient_rm(
    logits: &RowMajorMatrix<C>,
    targets: &[u32],
    padding_mask: &[u32],
    total_valid_tokens: usize,
    logit_indices: &[Vec<usize>],
) -> (Vec<Vec<C>>, RowMajorMatrix<C>) {
    let (seq_len, cols) = (logits.rows, logits.cols);

    assert_eq!(padding_mask.len(), seq_len);
    assert!(targets.len() <= seq_len);

    let offset = target_offset(padding_mask, targets.len());
    let scale: Real = r(1.0) / r(total_valid_tokens.max(1) as f64);

    let mut losses: Vec<Vec<C>> = Vec::with_capacity(seq_len);
    let mut grad_rm = RowMajorMatrix::create_empty(seq_len, cols);

    for t in 0..seq_len {
        if padding_mask[t] == 0 || t < offset {
            losses.push(vec![C::new(ZERO, ZERO)]);
            continue;
        }

        let target_token = targets[t - offset] as usize;
        let indices_for_t = logit_indices_for_row(logit_indices, t, cols);
        let row = logits.row_range(t);

        let (loss_real, grad_row) = softmax_ce_grad_complex_slice(&logits.data[row.clone()], target_token, &indices_for_t);
        losses.push(vec![C::new(loss_real * scale, ZERO)]);

        // grad_row has one entry per logit index; in sparse mode that matches cols.
        for (dst, g) in grad_rm.data[row].iter_mut().zip(grad_row.into_iter()) {
            *dst = g * scale;
        }
    }

    (losses, grad_rm)
}

/// RM counterpart of `activation::softmax_last_row`: softmax of the last row only, all other rows are zero.
pub fn softmax_last_row_rm(input: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::create_empty(input.rows, input.cols);
    if input.rows > 0 {
        let row = input.row_range(input.rows - 1);
        let softmax = softmax_row_complex(&input.data[row.clone()].to_vec());
        out.data[row].copy_from_slice(&softmax);
    }
    out
}
