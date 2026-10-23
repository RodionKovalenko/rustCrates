use num::Complex;
use rayon::prelude::*;

use crate::neural_networks::utils::{
    adam_w::{MAX_ELEMENT, MAX_NORM},
    dtype::{c_from_f64, c_to_f64, r, Real, C, CF64, ZERO},
};
use serde::{Deserialize, Serialize};

#[cfg(all(feature = "cuda", feature = "dtype-f64"))]
use crate::neural_networks::utils::matrix::GPU_MATMUL;

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

/// Dense projection `X * W (+ b)` for every matrix of the batch.
pub fn linear_forward_batch_rm(input_batch: &[RowMajorMatrix<C>], weights: &RowMajorMatrix<C>, bias: Option<&[C]>) -> Vec<RowMajorMatrix<C>> {
    input_batch
        .par_iter()
        .map(|input| {
            let mut output = multiply_complex_rm(input, weights);
            if let Some(bias) = bias {
                add_vector_rm(&mut output, bias);
            }
            output
        })
        .collect()
}

/// Gradients of `Y = X * W + b` for one sample.
pub struct LinearGradientsRm {
    /// dL/dX = delta * W^H
    pub input: RowMajorMatrix<C>,
    /// dL/dW = X^H * delta
    pub weights: RowMajorMatrix<C>,
    /// dL/db = column sums of delta
    pub bias: Vec<C>,
}

/// Backward of `Y = X * W + b` given `delta` = dL/dY and the precomputed `weights_h` = W^H.
pub fn linear_backward_rm(input: &RowMajorMatrix<C>, delta: &RowMajorMatrix<C>, weights_h: &RowMajorMatrix<C>) -> LinearGradientsRm {
    let weights = multiply_complex_rm(&conjugate_transpose_rm(input), delta);
    let mut bias = vec![C::new(ZERO, ZERO); delta.cols];
    accumulate_bias_from_rm(&mut bias, delta);
    let input = multiply_complex_rm(delta, weights_h);
    LinearGradientsRm { input, weights, bias }
}

/// `linear_backward_rm` for every sample of the batch. Returns (input, weight, bias) gradient batches.
pub fn linear_backward_batch_rm(
    input_batch: &[RowMajorMatrix<C>],
    delta_batch: &[RowMajorMatrix<C>],
    weights: &RowMajorMatrix<C>,
) -> (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>, Vec<Vec<C>>) {
    assert_eq!(input_batch.len(), delta_batch.len(), "input and gradient batch size mismatch");
    let weights_h = conjugate_transpose_rm(weights);
    let gradients: Vec<LinearGradientsRm> = input_batch.par_iter().zip(delta_batch.par_iter()).map(|(input, delta)| linear_backward_rm(input, delta, &weights_h)).collect();
    unzip_linear_gradients(gradients)
}

pub fn unzip_linear_gradients(gradients: Vec<LinearGradientsRm>) -> (Vec<RowMajorMatrix<C>>, Vec<RowMajorMatrix<C>>, Vec<Vec<C>>) {
    let mut inputs = Vec::with_capacity(gradients.len());
    let mut weights = Vec::with_capacity(gradients.len());
    let mut biases = Vec::with_capacity(gradients.len());
    for g in gradients {
        inputs.push(g.input);
        weights.push(g.weights);
        biases.push(g.bias);
    }
    (inputs, weights, biases)
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

pub fn multiply_complex_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert!(a.rows > 0 && a.cols > 0 && b.rows > 0 && b.cols > 0, "Matrices must not be empty");
    assert_eq!(a.cols, b.rows, "A's columns must match B's rows");

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    // Try GPU path first (if cuda feature enabled)
    #[cfg(feature = "cuda")]
    {
        // Keep your existing heuristic gate for now
        if n >= 50280 {
            if let Ok(c) = multiply_complex_gpu_rm(&a.data, &b.data, m, k, n) {
                return RowMajorMatrix::from_data(m, n, c);
            }
        }
    }

    let c = multiply_complex_cpu_rm(&a.data, &b.data, m, k, n);
    RowMajorMatrix::from_data(m, n, c)
}

#[cfg(all(feature = "cuda", feature = "dtype-f64"))]
fn multiply_complex_gpu_rm(a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Result<Vec<C>, Box<dyn std::error::Error>> {
    // Handle poisoned mutex gracefully
    let mut gpu_guard = match GPU_MATMUL.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("GPU mutex was poisoned, attempting recovery...");
            poisoned.into_inner()
        }
    };

    let gpu = match gpu_guard.as_mut() {
        Some(g) => g,
        None => return Err("GPU not available".into()),
    };

    let c = gpu.multiply_complex(a, b, m, k, n)?;
    Ok(c)
}

#[cfg(all(feature = "cuda", not(feature = "dtype-f64")))]
fn multiply_complex_gpu_rm(_a: &[C], _b: &[C], _m: usize, _k: usize, _n: usize) -> Result<Vec<C>, Box<dyn std::error::Error>> {
    Err("GPU matmul currently only enabled for dtype-f64".into())
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
