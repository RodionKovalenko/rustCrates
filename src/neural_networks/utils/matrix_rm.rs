use num::Complex;
use rayon::prelude::*;

use crate::neural_networks::utils::{
    adam_w::{MAX_ELEMENT, MAX_NORM},
    dtype::{c_from_f64, c_to_f64, r, Real, C, CF64, ZERO},
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use crate::neural_networks::utils::gpu_matmul::{gpu_worthwhile, try_multiply_complex_gpu};

/// Contiguous row-major matrix storage.
///
/// This is the layout expected by the GPU path (`GpuMatmul`) and can also be used
/// efficiently with BLAS by using a transpose/operand-swap trick.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowMajorMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T> RowMajorMatrix<T> {
    pub fn from_data(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must be rows*cols");
        Self { rows, cols, data }
    }

    #[inline]
    pub fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }
}

impl RowMajorMatrix<C> {
    /// Zero matrices with the same shapes as the matrices in `batch`.
    pub fn create_from_batch(batch: &[RowMajorMatrix<C>]) -> Vec<RowMajorMatrix<C>> {
        batch.iter().map(|m| RowMajorMatrix::create_empty(m.rows, m.cols)).collect()
    }
    pub fn create_empty(rows: usize, cols: usize) -> RowMajorMatrix<C> {
        RowMajorMatrix::from_data(rows, cols, vec![C::ZERO; rows * cols])
    }
}

impl<T: Copy> RowMajorMatrix<T> {
    pub fn from_rows(rows: &[Vec<T>]) -> Self {
        let m = rows.len();
        assert!(m > 0, "matrix must not be empty");
        let n = rows[0].len();
        assert!(n > 0, "matrix must not be empty");
        for r in rows {
            assert_eq!(r.len(), n, "all rows must have the same length");
        }

        let data: Vec<T> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Self { rows: m, cols: n, data }
    }

    /// Builds a matrix from row vectors, returning `None` when the input is empty or ragged.
    ///
    /// This is useful for code paths where inputs may be variable-length (e.g. top-k / sparse
    /// representations) and we want to keep the Vec-based representation without panicking.
    pub fn try_from_rows(rows: &[Vec<T>]) -> Option<Self> {
        let m = rows.len();
        if m == 0 {
            return None;
        }

        let n = rows[0].len();
        if n == 0 {
            return None;
        }

        if rows.iter().any(|r| r.len() != n) {
            return None;
        }

        let data: Vec<T> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Some(Self { rows: m, cols: n, data })
    }

    pub fn to_rows(&self) -> Vec<Vec<T>> {
        let mut out = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            out.push(self.data[i * self.cols..(i + 1) * self.cols].to_vec());
        }
        out
    }
}

impl<T> RowMajorMatrix<T> {
    #[inline]
    pub fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        let start = row * self.cols;
        start..start + self.cols
    }
}

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

/// In-place `dst += src`.
pub fn add_assign_rm(dst: &mut RowMajorMatrix<C>, src: &RowMajorMatrix<C>) {
    assert_eq!((dst.rows, dst.cols), (src.rows, src.cols));
    for (d, &s) in dst.data.iter_mut().zip(src.data.iter()) {
        *d += s;
    }
}

/// In-place `m *= scalar`.
pub fn scale_rm_in_place(m: &mut RowMajorMatrix<C>, scalar: C) {
    m.data.iter_mut().for_each(|v| *v = *v * scalar);
}

/// Converts to `Complex<f64>` (wavelet transforms run in f64).
pub fn rm_to_f64(m: &RowMajorMatrix<C>) -> RowMajorMatrix<CF64> {
    RowMajorMatrix::from_data(m.rows, m.cols, m.data.iter().copied().map(c_to_f64).collect())
}

/// Converts back from `Complex<f64>`.
pub fn rm_from_f64(m: &RowMajorMatrix<CF64>) -> RowMajorMatrix<C> {
    RowMajorMatrix::from_data(m.rows, m.cols, m.data.iter().copied().map(c_from_f64).collect())
}

/// Dense projection `X * W (+ b)` for every matrix of the batch (one GEMM for the whole batch on the GPU).
pub fn linear_forward_batch_rm(input_batch: &[RowMajorMatrix<C>], weights: &RowMajorMatrix<C>, bias: Option<&[C]>) -> Vec<RowMajorMatrix<C>> {
    let mut output_batch = multiply_batch_with_weights_rm(input_batch, weights, MatOp::None);
    if let Some(bias) = bias {
        output_batch.par_iter_mut().for_each(|output| add_vector_rm(output, bias));
    }
    output_batch
}

/// Backward of `Y = X * W + b` for every sample, given `delta_batch` = dL/dY.
/// Returns the batches of input gradients (delta * W^H), weight gradients (X^H * delta) and bias gradients.
pub fn linear_backward_batch_rm(
    input_batch: &[RowMajorMatrix<C>],
    delta_batch: &[RowMajorMatrix<C>],
    weights: &RowMajorMatrix<C>,
) -> (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>, Vec<Vec<C>>) {
    assert_eq!(input_batch.len(), delta_batch.len(), "input and gradient batch size mismatch");
    let input_gradients = multiply_batch_with_weights_rm(delta_batch, weights, MatOp::ConjTranspose);
    let (weight_gradients, bias_gradients): (Vec<RowMajorMatrix<C>>, Vec<Vec<C>>) = input_batch
        .par_iter()
        .zip(delta_batch.par_iter())
        .map(|(input, delta)| {
            let mut bias = vec![C::new(ZERO, ZERO); delta.cols];
            accumulate_bias_from_rm(&mut bias, delta);
            (multiply_complex_rm_op(input, MatOp::ConjTranspose, delta, MatOp::None), bias)
        })
        .unzip();
    (input_gradients, weight_gradients, bias_gradients)
}

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

pub fn add_vector_rm(matrix: &mut RowMajorMatrix<Complex<Real>>, bias: &[Complex<Real>]) {
    assert_eq!(matrix.cols, bias.len(), "bias length must match matrix cols");
    for r in 0..matrix.rows {
        let row = matrix.row_range(r);
        for c in 0..matrix.cols {
            matrix.data[row.start + c] += bias[c];
        }
    }
}

pub fn conjugate_transpose_rm(matrix: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let rows = matrix.rows;
    let cols = matrix.cols;

    // Output is (cols x rows)
    let mut data = vec![C::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = matrix.data[matrix.idx(r, c)].conj();
        }
    }

    RowMajorMatrix::from_data(cols, rows, data)
}

pub fn transpose_rm<T: Copy + Default>(matrix: &RowMajorMatrix<T>) -> RowMajorMatrix<T> {
    let rows = matrix.rows;
    let cols = matrix.cols;

    let mut data = vec![T::default(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = matrix.data[matrix.idx(r, c)];
        }
    }

    RowMajorMatrix::from_data(cols, rows, data)
}

pub fn transpose_rm_f64(matrix: &RowMajorMatrix<Real>) -> RowMajorMatrix<Real> {
    let rows = matrix.rows;
    let cols = matrix.cols;

    let mut data = vec![ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = matrix.data[r * cols + c];
        }
    }

    RowMajorMatrix::from_data(cols, rows, data)
}

pub fn append_rows_rm<T: Copy>(dst: &mut RowMajorMatrix<T>, src: &RowMajorMatrix<T>) {
    assert_eq!(dst.cols, src.cols, "column count mismatch");
    dst.data.extend_from_slice(&src.data);
    dst.rows += src.rows;
}

pub fn multiply_f64_complex_rm(a: &RowMajorMatrix<Real>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(a.cols, b.rows, "A's columns must match B's rows");
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut c = vec![C::new(ZERO, ZERO); m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a.data[i * k + kk];
            if a_ik == ZERO {
                continue;
            }
            let b_row = kk * n;
            let c_row = i * n;
            for j in 0..n {
                c[c_row + j] += b.data[b_row + j] * C::new(a_ik, ZERO);
            }
        }
    }
    RowMajorMatrix::from_data(m, n, c)
}

pub fn multiply_complex_with_f64_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<Real>) -> RowMajorMatrix<C> {
    assert_eq!(a.cols, b.rows, "A's columns must match B's rows");
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut c = vec![C::new(ZERO, ZERO); m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a.data[i * k + kk];
            if a_ik == C::new(ZERO, ZERO) {
                continue;
            }
            let b_row = kk * n;
            let c_row = i * n;
            for j in 0..n {
                c[c_row + j] += a_ik * b.data[b_row + j];
            }
        }
    }
    RowMajorMatrix::from_data(m, n, c)
}

pub fn conjugate_transpose_to_rm(matrix: &[Vec<C>]) -> RowMajorMatrix<C> {
    let rows = matrix.len();
    assert!(rows > 0, "matrix must not be empty");
    let cols = matrix[0].len();
    assert!(cols > 0, "matrix must not be empty");
    for r in matrix {
        assert_eq!(r.len(), cols, "all rows must have the same length");
    }

    // Output is (cols x rows)
    let mut data = vec![C::new(ZERO, ZERO); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = matrix[r][c].conj();
        }
    }

    RowMajorMatrix::from_data(cols, rows, data)
}

/// Operation applied to a matrix operand before it is multiplied (as in BLAS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatOp {
    None,
    Transpose,
    ConjTranspose,
}

impl MatOp {
    /// (rows, cols) of op(matrix).
    fn shape(self, matrix: &RowMajorMatrix<C>) -> (usize, usize) {
        match self {
            MatOp::None => (matrix.rows, matrix.cols),
            MatOp::Transpose | MatOp::ConjTranspose => (matrix.cols, matrix.rows),
        }
    }

    /// op(matrix) materialized on the CPU.
    fn apply(self, matrix: &RowMajorMatrix<C>) -> std::borrow::Cow<'_, RowMajorMatrix<C>> {
        match self {
            MatOp::None => std::borrow::Cow::Borrowed(matrix),
            MatOp::Transpose => std::borrow::Cow::Owned(transpose_rm(matrix)),
            MatOp::ConjTranspose => std::borrow::Cow::Owned(conjugate_transpose_rm(matrix)),
        }
    }
}

/// A × B. With the cuda feature every complex matmul runs on the GPU; the CPU is only the fallback.
pub fn multiply_complex_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    multiply_rm(a, MatOp::None, b, MatOp::None, false)
}

/// op_a(A) × op_b(B). On the GPU the (conjugate) transposes are done by cuBLAS, not materialized.
pub fn multiply_complex_rm_op(a: &RowMajorMatrix<C>, op_a: MatOp, b: &RowMajorMatrix<C>, op_b: MatOp) -> RowMajorMatrix<C> {
    multiply_rm(a, op_a, b, op_b, false)
}

/// X × op(W) for a weight matrix W. On the GPU, W stays in device memory and is only uploaded again
/// after it changed.
pub fn multiply_complex_rm_weights(x: &RowMajorMatrix<C>, weights: &RowMajorMatrix<C>, op_w: MatOp) -> RowMajorMatrix<C> {
    multiply_rm(x, MatOp::None, weights, op_w, true)
}

/// X_i × op(W) for every matrix of the batch. On the GPU the batch is stacked into one matrix, so a
/// single GEMM uses the (cached) weights; on the CPU the samples are multiplied in parallel.
pub fn multiply_batch_with_weights_rm(batch: &[RowMajorMatrix<C>], weights: &RowMajorMatrix<C>, op_w: MatOp) -> Vec<RowMajorMatrix<C>> {
    let (k, n) = op_w.shape(weights);
    assert!(batch.iter().all(|x| x.cols == k), "batch columns must match the weight rows");

    #[cfg(feature = "cuda")]
    {
        let rows: Vec<usize> = batch.iter().map(|x| x.rows).collect();
        let total_rows: usize = rows.iter().sum();
        if total_rows > 0 && k > 0 && n > 0 && gpu_worthwhile(total_rows, k, n) {
            let stacked: Vec<C> = batch.iter().flat_map(|x| x.data.iter().copied()).collect();
            if let Some(c) = try_multiply_complex_gpu(&stacked, MatOp::None, &weights.data, op_w, total_rows, k, n, true) {
                let mut offset = 0;
                return rows
                    .iter()
                    .map(|&r| {
                        let out = RowMajorMatrix::from_data(r, n, c[offset * n..(offset + r) * n].to_vec());
                        offset += r;
                        out
                    })
                    .collect();
            }
        }
    }

    let weights = op_w.apply(weights);
    batch.par_iter().map(|x| RowMajorMatrix::from_data(x.rows, n, multiply_complex_cpu_rm(&x.data, &weights.data, x.rows, k, n))).collect()
}

fn multiply_rm(a: &RowMajorMatrix<C>, op_a: MatOp, b: &RowMajorMatrix<C>, op_b: MatOp, cache_b: bool) -> RowMajorMatrix<C> {
    let (m, k) = op_a.shape(a);
    let (k_b, n) = op_b.shape(b);
    assert!(m > 0 && k > 0 && n > 0, "Matrices must not be empty");
    assert_eq!(k, k_b, "A's columns must match B's rows");

    #[cfg(feature = "cuda")]
    {
        if gpu_worthwhile(m, k, n) {
            if let Some(c) = try_multiply_complex_gpu(&a.data, op_a, &b.data, op_b, m, k, n, cache_b) {
                return RowMajorMatrix::from_data(m, n, c);
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = cache_b;

    let (a, b) = (op_a.apply(a), op_b.apply(b));
    RowMajorMatrix::from_data(m, n, multiply_complex_cpu_rm(&a.data, &b.data, m, k, n))
}

fn multiply_complex_cpu_rm(a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Vec<C> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);

    let mut c = vec![C::new(ZERO, ZERO); m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] = c[i * n + j] + aip * b[p * n + j];
            }
        }
    }
    c
}

pub fn split_data_by_columns_rm(matrix: &RowMajorMatrix<C>) -> (RowMajorMatrix<C>, RowMajorMatrix<C>) {
    let half_cols = matrix.cols / 2;
    let mut left_rm = vec![C::ZERO; matrix.rows * half_cols];
    let mut right_rm = vec![C::ZERO; matrix.rows * half_cols];

    for row in 0..matrix.rows {
        for col in 0..matrix.cols {
            let ind = matrix.idx(row, col);
            let value = matrix.data[ind];

            let half_ind = row * half_cols + col % half_cols;

            if col < half_cols {
                left_rm[half_ind] = value;
            } else {
                right_rm[half_ind] = value;
            }
        }
    }

    let left = RowMajorMatrix::from_data(matrix.rows, half_cols, left_rm);
    let right = RowMajorMatrix::from_data(matrix.rows, half_cols, right_rm);

    (left, right)
}

/// Concatenates column blocks: `[block][batch]` (seq x cols_block) -> `[batch]` (seq x sum of cols).
/// Used for attention heads and for the column chunks of `MultiLinearLayerRm`.
pub fn concat_heads_rm(head_outputs: &[Vec<RowMajorMatrix<C>>]) -> Vec<RowMajorMatrix<C>> {
    assert!(!head_outputs.is_empty(), "no column blocks to concatenate");
    let batch_size = head_outputs[0].len();

    (0..batch_size)
        .map(|b| {
            let rows = head_outputs[0][b].rows;
            let total_cols: usize = head_outputs.iter().map(|block| block[b].cols).sum();
            let mut data = Vec::with_capacity(rows * total_cols);
            for row in 0..rows {
                for block in head_outputs {
                    assert_eq!(block[b].rows, rows, "column blocks must have the same number of rows");
                    data.extend_from_slice(&block[b].data[block[b].row_range(row)]);
                }
            }
            RowMajorMatrix::from_data(rows, total_cols, data)
        })
        .collect()
}

/// Inverse of `concat_heads_rm`: splits every matrix of the batch into `num_heads` equal column blocks, `[head][batch]`.
pub fn split_heads_rm(batch: &[RowMajorMatrix<C>], num_heads: usize) -> Vec<Vec<RowMajorMatrix<C>>> {
    assert!(num_heads > 0);
    (0..num_heads)
        .map(|head| {
            batch
                .iter()
                .map(|m| {
                    assert!(m.cols % num_heads == 0, "dim={} must be divisible by num_heads={}", m.cols, num_heads);
                    let head_dim = m.cols / num_heads;
                    split_columns_rm(m, head * head_dim, (head + 1) * head_dim)
                })
                .collect()
        })
        .collect()
}

/// Element-wise sum over heads of `[head][batch]` matrices of equal shape -> `[batch]`.
pub fn sum_heads_rm(head_batches: Vec<Vec<RowMajorMatrix<C>>>) -> Vec<RowMajorMatrix<C>> {
    let mut heads = head_batches.into_iter();
    let mut sum = heads.next().expect("no attention head gradients");
    for head in heads {
        for (s, h) in sum.iter_mut().zip(head.iter()) {
            add_assign_rm(s, h);
        }
    }
    sum
}

/// Last `rows` rows of `m` (all rows if it has fewer).
pub fn tail_rows_rm(m: &RowMajorMatrix<C>, rows: usize) -> RowMajorMatrix<C> {
    if m.rows <= rows {
        return m.clone();
    }
    RowMajorMatrix::from_data(rows, m.cols, m.data[(m.rows - rows) * m.cols..].to_vec())
}
