use crate::neural_networks::utils::{
    adam_w::{MAX_ELEMENT, MAX_NORM},
    dtype::{r, Real, C, ZERO},
    matrix::RowMajorMatrix,
};

/// RM counterpart of `matrix::hadamard_product_2d_c`.
pub fn hadamard_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = RowMajorMatrix::from_data(a.rows, a.cols, vec![C::new(ZERO, ZERO); a.rows * a.cols]);
    for i in 0..out.data.len() {
        out.data[i] = a.data[i] * b.data[i];
    }
    out
}

/// RM counterpart of `matrix::conjugate`.
pub fn conjugate_rm(m: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(m.rows, m.cols, vec![C::new(ZERO, ZERO); m.rows * m.cols]);
    for (dst, &v) in out.data.iter_mut().zip(m.data.iter()) {
        *dst = v.conj();
    }
    out
}

/// RM counterpart of `matrix::add_matrix`.
pub fn add_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = RowMajorMatrix::from_data(a.rows, a.cols, vec![C::new(ZERO, ZERO); a.rows * a.cols]);
    for i in 0..out.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }
    out
}

pub fn split_columns_rm(matrix: &RowMajorMatrix<C>, start_col: usize, end_col: usize) -> RowMajorMatrix<C> {
    assert!(start_col <= end_col);
    assert!(end_col <= matrix.cols);
    let cols = end_col - start_col;
    let mut data = vec![C::new(ZERO, ZERO); matrix.rows * cols];
    for r_i in 0..matrix.rows {
        let src_row = matrix.row_range(r_i);
        let dst_row_start = r_i * cols;
        let src_start = src_row.start + start_col;
        let src_end = src_row.start + end_col;
        data[dst_row_start..dst_row_start + cols].copy_from_slice(&matrix.data[src_start..src_end]);
    }
    RowMajorMatrix::from_data(matrix.rows, cols, data)
}

/// Concatenates `left` and `right` column-wise: `[left | right]`. Inverse of `split_data_by_columns_rm`.
pub fn concat_columns_rm(left: &RowMajorMatrix<C>, right: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(left.rows, right.rows);
    let cols = left.cols + right.cols;
    let mut data = Vec::with_capacity(left.rows * cols);
    for r_i in 0..left.rows {
        data.extend_from_slice(&left.data[left.row_range(r_i)]);
        data.extend_from_slice(&right.data[right.row_range(r_i)]);
    }
    RowMajorMatrix::from_data(left.rows, cols, data)
}

/// Sums each column of `grad` into `dst` (bias gradient accumulation).
pub fn accumulate_bias_from_rm(dst: &mut [C], grad: &RowMajorMatrix<C>) {
    assert_eq!(dst.len(), grad.cols);
    for r_i in 0..grad.rows {
        let row = grad.row_range(r_i);
        for c in 0..grad.cols {
            dst[c] += grad.data[row.start + c];
        }
    }
}

/// RM counterpart of `Gradient::group_gradient_batch`: sums a batch of equally shaped matrices.
pub fn sum_batch_rm(batch: &[RowMajorMatrix<C>]) -> Option<RowMajorMatrix<C>> {
    let first = batch.first()?;
    let mut out = RowMajorMatrix::create_empty(first.rows, first.cols);
    for m in batch {
        assert_eq!((m.rows, m.cols), (out.rows, out.cols), "all matrices in the batch must have the same shape");
        for (dst, &v) in out.data.iter_mut().zip(m.data.iter()) {
            *dst += v;
        }
    }
    Some(out)
}

/// RM counterpart of `matrix::average_matrix_by_scalar`.
pub fn average_matrix_by_scalar_rm(matrix: &RowMajorMatrix<C>, scalar: Real) -> RowMajorMatrix<C> {
    RowMajorMatrix::from_data(matrix.rows, matrix.cols, matrix.data.iter().map(|&v| v / scalar).collect())
}

/// RM counterpart of `matrix::clip_all_gradients_by_global_norm_2d`.
pub fn clip_all_gradients_by_global_norm_rm(grads: &mut RowMajorMatrix<C>, bias: &mut Vec<C>, total_norm: f64, _max_norm: f64) {
    if total_norm > MAX_NORM {
        let scale = r(1.0 / total_norm);
        grads.data.iter_mut().for_each(|val| *val *= scale);
        bias.iter_mut().for_each(|val| *val *= scale);
    }
}

/// RM counterpart of `matrix::normalize_gradients`.
pub fn normalize_gradients_rm(gradients: &mut RowMajorMatrix<C>) {
    // Step 1: elementwise clamp
    for val in gradients.data.iter_mut() {
        let norm = val.norm() as f64;
        if norm > MAX_ELEMENT && norm > 0.0 {
            *val *= r(MAX_ELEMENT / norm);
        }
    }

    // Step 2: compute global L2 norm
    let global_norm: f64 = gradients.data.iter().map(|g| g.norm_sqr() as f64).sum::<f64>().sqrt();

    // Step 3: scale proportionally if global norm exceeds MAX_NORM
    if global_norm > MAX_NORM && global_norm > 0.0 {
        let scale = r(MAX_NORM / global_norm);
        gradients.data.iter_mut().for_each(|val| *val *= scale);
    }
}
