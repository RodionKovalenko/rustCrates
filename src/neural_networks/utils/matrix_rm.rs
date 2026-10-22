use crate::neural_networks::utils::{
    dtype::{r, C, ZERO},
    matrix::RowMajorMatrix,
};

pub fn hadamard_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut out = RowMajorMatrix::from_data(a.rows, a.cols, vec![C::new(ZERO, ZERO); a.rows * a.cols]);
    for i in 0..out.data.len() {
        out.data[i] = a.data[i] * b.data[i];
    }
    out
}

pub fn conjugate_rm(m: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(m.rows, m.cols, vec![C::new(ZERO, ZERO); m.rows * m.cols]);
    for (dst, &v) in out.data.iter_mut().zip(m.data.iter()) {
        *dst = v.conj();
    }
    out
}

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

pub fn accumulate_bias_from_rm(dst: &mut [C], grad: &RowMajorMatrix<C>) {
    assert_eq!(dst.len(), grad.cols);
    for r_i in 0..grad.rows {
        let row = grad.row_range(r_i);
        for c in 0..grad.cols {
            dst[c] += grad.data[row.start + c];
        }
    }
}
