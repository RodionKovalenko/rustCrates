use faer::Mat;
// use ndarray::Array2;
use num::Complex;
use num_traits::Float;
use num_traits::NumCast;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
// use std::ffi::c_void;
use std::fmt::Debug;
use std::ops::Div;
use std::ops::{Add, Mul, Sub};
use std::sync::{Arc, Mutex};

use crate::neural_networks::network_types::transformer::transformer_updater::VERBOSE;
use crate::neural_networks::utils::adam_w::MAX_ELEMENT;
use crate::neural_networks::utils::adam_w::MAX_NORM;
use crate::neural_networks::utils::dtype::{r, Real, C, ZERO};

#[cfg(feature = "cuda")]
use super::gpu_matmul::GpuMatmul;

#[cfg(feature = "cuda")]
use once_cell::sync::Lazy;

#[cfg(feature = "cuda")]
pub static GPU_MATMUL: Lazy<Mutex<Option<GpuMatmul>>> = Lazy::new(|| {
    // Try to initialize GPU, but don't panic if it fails
    match GpuMatmul::new(1024, 1024, 1024) {
        Ok(gpu) => Mutex::new(Some(gpu)),
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}. Will use CPU fallback.", e);
            Mutex::new(None)
        }
    }
});


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

pub fn add_vector_rm(matrix: &mut RowMajorMatrix<Complex<Real>>, bias: &[Complex<Real>]) {
    assert_eq!(matrix.cols, bias.len(), "bias length must match matrix cols");
    for r in 0..matrix.rows {
        let row = matrix.row_range(r);
        for c in 0..matrix.cols {
            matrix.data[row.start + c] += bias[c];
        }
    }
}

pub fn conjugate_transpose_rm(matrix: &RowMajorMatrix<Complex<Real>>) -> RowMajorMatrix<Complex<Real>> {
    let rows = matrix.rows;
    let cols = matrix.cols;

    // Output is (cols x rows)
    let mut data = vec![Complex::new(ZERO, ZERO); rows * cols];
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

pub fn multiply_complex(matrix_a: &[Vec<Complex<Real>>], matrix_b: &[Vec<Complex<Real>>]) -> Vec<Vec<Complex<Real>>> {
    // Try GPU path first (if cuda feature enabled)
    #[cfg(all(feature = "cuda", feature = "dtype-f64"))]
    {
        let m = matrix_a.len();
        let k = matrix_a[0].len();
        let n = matrix_b[0].len();
        // Attempt GPU acceleration for any size
        if n >= 50280 {
            if let Ok(result) = multiply_complex_gpu(matrix_a, matrix_b, m, k, n) {
                return result;
            } else {
                println!("Falling back to CPU complex matmul due to GPU error.");
            }
        }
    }

    // Fallback to CPU BLAS
    multiply_complex_cpu(matrix_a, matrix_b)
}

pub fn multiply_complex_f32(matrix_a: &[Vec<Complex<Real>>], matrix_b: &[Vec<f32>]) -> Vec<Vec<Complex<Real>>> {
    let num_rows = matrix_a.len();
    let num_columns = matrix_b[0].len();

    // Ensure that the number of columns in matrix_a is equal to the number of rows in matrix_b
    if matrix_a[0].len() != matrix_b.len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows. A: {} {}, B: {} {}", matrix_a.len(), matrix_a[0].len(), matrix_b.len(), matrix_b[0].len());
    }

    // Initialize result matrix with 0.0 values
    let mut result_matrix: Vec<Vec<Complex<Real>>> = vec![vec![Complex::new(0.0, 0.0); num_columns]; num_rows];

    // Create a custom thread pool with exactly 10 threads
    let pool = ThreadPoolBuilder::new().num_threads(8).build().unwrap();

    // Run the multiplication within this custom thread pool
    pool.install(|| {
        // Parallelize the rows of the result matrix using Rayon
        result_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..num_columns {
                row[j] = (0..matrix_b.len()).map(|k| matrix_a[i][k] * Complex::new(matrix_b[k][j] as Real, 0.0)).sum();
            }
        });
    });

    result_matrix
}

#[cfg(all(feature = "cuda", feature = "dtype-f64"))]
fn multiply_complex_gpu(matrix_a: &[Vec<Complex<Real>>], matrix_b: &[Vec<Complex<Real>>], m: usize, k: usize, n: usize) -> Result<Vec<Vec<Complex<Real>>>, Box<dyn std::error::Error>> {
    // Handle poisoned mutex gracefully
    let mut gpu_guard = match GPU_MATMUL.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("GPU mutex was poisoned, attempting recovery...");
            poisoned.into_inner()
        }
    };

    // Check if GPU is available
    let gpu = match gpu_guard.as_mut() {
        Some(g) => g,
        None => return Err("GPU not available".into()),
    };

    // Flatten 2D -> 1D (row-major)
    let a_flat: Vec<Complex<Real>> = matrix_a.iter().flatten().copied().collect();
    let b_flat: Vec<Complex<Real>> = matrix_b.iter().flatten().copied().collect();

    // GPU multiply
    let c_flat = gpu.multiply_complex(&a_flat, &b_flat, m, k, n)?;

    // Reshape 1D -> 2D
    let mut result = Vec::with_capacity(m);
    for i in 0..m {
        result.push(c_flat[i * n..(i + 1) * n].to_vec());
    }

    Ok(result)
}

fn multiply_complex_cpu(matrix_a: &[Vec<Complex<Real>>], matrix_b: &[Vec<Complex<Real>>]) -> Vec<Vec<Complex<Real>>> {
    let m = matrix_a.len();
    let k = matrix_a[0].len();
    let n = matrix_b[0].len();

    assert!(m > 0 && n > 0 && k > 0, "Matrices must not be empty");
    assert!(matrix_b.len() == k, "A's columns must match B's rows A {} {}, B {} {}", m, k, matrix_b.len(), n);
    for row in matrix_a {
        assert_eq!(row.len(), k, "All rows of A must have the same length");
    }
    for row in matrix_b {
        assert_eq!(row.len(), n, "All rows of B must have the same length");
    }

    // Build contiguous row-major buffers (cheaper than a column-major re-pack)
    let a_rm: Vec<Complex<Real>> = matrix_a.iter().flat_map(|r| r.iter().copied()).collect();
    let b_rm: Vec<Complex<Real>> = matrix_b.iter().flat_map(|r| r.iter().copied()).collect();

    let c_rm = multiply_complex_cpu_rm(&a_rm, &b_rm, m, k, n);

    let mut result = Vec::with_capacity(m);
    for i in 0..m {
        result.push(c_rm[i * n..(i + 1) * n].to_vec());
    }
    result
}

pub fn multiply<T, V>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<V>>) -> Vec<Vec<f64>>
where
    T: Into<f64> + Clone + Debug,
    V: Into<f64> + Clone + Debug,
{
    // Convert matrix_a and matrix_b to Vec<Vec<f64>> for thread safety
    let mut matrix_a_clone: Vec<Vec<f64>> = matrix_a.iter().map(|row| row.iter().map(|x| x.clone().into()).collect()).collect();
    let mut matrix_b_clone: Vec<Vec<f64>> = matrix_b.iter().map(|row| row.iter().map(|x| x.clone().into()).collect()).collect();

    let mut num_rows = matrix_a_clone.len();
    let mut num_columns = matrix_b_clone[0].len();

    if matrix_a_clone[0].len() != matrix_b.len() {
        if matrix_a_clone[0].len() == matrix_b_clone[0].len() {
            matrix_b_clone = transpose(&matrix_b_clone);
            num_columns = matrix_b_clone[0].len();
        } else if matrix_a_clone.len() == matrix_b.len() {
            matrix_a_clone = transpose(&matrix_a_clone);
            num_rows = matrix_a_clone.len();
        }
    }

    // Ensure that the number of columns in matrix_a is equal to the number of rows in matrix_b
    if matrix_a[0].len() != matrix_b.len() && matrix_a.len() != matrix_b.len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    // Initialize result matrix with 0.0 values
    let mut result_matrix: Vec<Vec<f64>> = vec![vec![0.0; num_columns]; num_rows];

    // Create a custom thread pool with exactly 10 threads
    let pool = ThreadPoolBuilder::new().num_threads(8).build().unwrap();

    // Run the multiplication within this custom thread pool
    pool.install(|| {
        // Parallelize the rows of the result matrix using Rayon
        result_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..num_columns {
                row[j] = (0..matrix_b_clone.len()).map(|k| matrix_a_clone[i][k] * matrix_b_clone[k][j]).sum();
            }
        });
    });

    result_matrix
}

// Converts Vec<Vec<Complex<f64>>> into faer::Mat<Complex<f64>> (flattened)
pub unsafe fn convert_to_faer_mat_unchecked(matrix: &[Vec<Complex<f64>>]) -> Mat<Complex<f64>> {
    let rows = matrix.len();
    if rows == 0 {
        return Mat::from_fn(0, 0, |_, _| unreachable!());
    }
    let cols = matrix[0].len();

    // Now build faer::Mat from slice
    Mat::from_fn(rows, cols, |i, j| matrix[i][j])
}

pub fn multiply_complex_fear(matrix_a: &Vec<Vec<Complex<f64>>>, matrix_b: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let a_rows = matrix_a.len();
    let a_cols = matrix_a[0].len();
    let b_rows = matrix_b.len();
    let b_cols = matrix_b[0].len();

    // Validate dimensions for matrix multiplication
    if a_cols != b_rows {
        panic!("Invalid matrix dimensions: A is {}x{}, B is {}x{}", a_rows, a_cols, b_rows, b_cols);
    }
    let mat_a = unsafe { convert_to_faer_mat_unchecked(matrix_a) };
    let mat_b = unsafe { convert_to_faer_mat_unchecked(matrix_b) };

    // Perform matrix multiplication using faer
    let mat_c = &mat_a * &mat_b;

    // Convert the result matrix back to Vec<Vec<Complex<f64>>>
    // Convert result back to Vec<Vec<Complex<f64>>>
    let mut result = vec![vec![Complex::new(0.0, 0.0); mat_c.ncols()]; mat_c.nrows()];
    for i in 0..mat_c.nrows() {
        for j in 0..mat_c.ncols() {
            result[i][j] = mat_c[(i, j)];
        }
    }

    result

    // let num_rows = matrix_a.len();
    // let num_columns = matrix_b[0].len();
    // let matrix_a_clone = matrix_a.clone();
    // let matrix_b_clone = matrix_b.clone();

    // // Ensure that the number of columns in matrix_a is equal to the number of rows in matrix_b
    // if matrix_a[0].len() != matrix_b.len() {
    //     panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    // }

    // // Initialize result matrix with 0.0 values
    // let mut result_matrix: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); num_columns]; num_rows];

    // // println!("anzahl cput {}", num_cpus::get());

    // let pool = ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap();

    // pool.install(|| {
    //     result_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
    //         for j in 0..num_columns {
    //             row[j] = (0..matrix_b_clone.len()).map(|k| matrix_a_clone[i][k] * matrix_b_clone[k][j]).sum();
    //             //row[j] = (0..matrix_b_clone.len()).map(|k| Complex::new(matrix_a_clone[i][k].re * matrix_b_clone[k][j].re, 0.0)).sum();
    //         }
    //     });
    // });

    // result_matrix
}

pub fn multiply_complex_with_f64(matrix_a: &[Vec<C>], matrix_b: &[Vec<Real>]) -> Vec<Vec<C>> {
    let num_rows = matrix_a.len();
    let num_columns = matrix_b[0].len();

    // Ensure that the number of columns in matrix_a is equal to the number of rows in matrix_b
    if matrix_a[0].len() != matrix_b.len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    // Initialize result matrix with 0.0 values
    let mut result_matrix: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); num_columns]; num_rows];

    // println!("anzahl cput {}", num_cpus::get());

    let pool = ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap();

    pool.install(|| {
        result_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..num_columns {
                row[j] = (0..matrix_b.len()).map(|k| matrix_a[i][k] * matrix_b[k][j]).sum();
            }
        });
    });

    result_matrix
}

pub fn multiply_complex_with_f32(matrix_a: &[Vec<C>], matrix_b: &[Vec<Real>]) -> Vec<Vec<C>> {
    multiply_complex_with_f64(matrix_a, matrix_b)
}

pub fn multiply_f64_complex(matrix_a: &[Vec<Real>], matrix_b: &[Vec<C>]) -> Vec<Vec<C>> {
    let num_rows = matrix_a.len();
    let num_columns = matrix_b[0].len();

    // println!("matrix a dim : {} x {}", matrix_a.len(), matrix_a[0].len());
    // println!("matrix b dim : {} x {}", matrix_b.len(), matrix_b[0].len());
    // Ensure that the number of columns in matrix_a is equal to the number of rows in matrix_b
    if matrix_a[0].len() != matrix_b.len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    // Initialize result matrix with 0.0 values
    let mut result_matrix: Vec<Vec<C>> = vec![vec![C::new(ZERO, ZERO); num_columns]; num_rows];

    let pool = ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap();

    pool.install(|| {
        result_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..num_columns {
                row[j] = (0..matrix_b.len()).map(|k| matrix_a[i][k] * matrix_b[k][j]).sum();
            }
        });
    });

    result_matrix
}

pub fn conjugate_transpose<T: Float>(matrix: &Vec<Vec<Complex<T>>>) -> Vec<Vec<Complex<T>>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![Complex::new(T::zero(), T::zero()); rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j].conj();
        }
    }
    result
}

pub fn conjugate<T: Float>(matrix: &Vec<Vec<Complex<T>>>) -> Vec<Vec<Complex<T>>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![Complex::new(T::zero(), T::zero()); cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = matrix[i][j].conj();
        }
    }
    result
}

pub fn conjugate_1d<T: Float>(matrix: &Vec<Complex<T>>) -> Vec<Complex<T>> {
    let mut result = vec![Complex::new(T::zero(), T::zero()); matrix.len()];

    for i in 0..matrix.len() {
        result[i] = matrix[i].conj();
    }
    result
}

pub fn conjugate_2d<T: Float>(matrix: &Vec<Vec<Complex<T>>>) -> Vec<Vec<Complex<T>>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![Complex::new(T::zero(), T::zero()); cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = matrix[i][j].conj();
        }
    }
    result
}

pub fn transpose<T: Debug + Clone + Sync + Send>(matrix_a: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let num_rows = matrix_a.len();
    let num_cols = matrix_a[0].len();

    // Use Arc and Mutex to wrap the result matrix
    let matrix_result = Arc::new(Mutex::new(vec![Vec::with_capacity(num_rows); num_cols]));

    // Create a custom thread pool with a specific number of threads (e.g., 10 threads)
    let pool = ThreadPoolBuilder::new().num_threads(3).build().unwrap();

    // Parallelize the column processing using the custom thread pool
    pool.install(|| {
        (0..num_cols).into_par_iter().for_each(|j| {
            let mut row = Vec::with_capacity(num_rows); // Create a local row for the result matrix
            for i in 0..num_rows {
                row.push(matrix_a[i][j].clone()); // Collect the elements for the j-th column
            }

            // Lock the Mutex to safely modify matrix_result
            let mut result_lock = matrix_result.lock().unwrap();
            result_lock[j] = row; // Assign the row to the transposed matrix
        });
    });

    // Return the result after unlocking
    let result_lock = matrix_result.lock().unwrap();
    (*result_lock).clone()
}

pub fn hadamard_product_2d_c(input_1: &Vec<Vec<C>>, input_2: &Vec<Vec<C>>) -> Vec<Vec<C>> {
    let rows = input_1.len();
    let cols = input_1[0].len();

    assert!(rows == input_2.len() && cols == input_2[0].len(), "Input matrices must have the same dimensions");

    // Initialize result matrix with zeros
    let mut result = vec![vec![C::new(ZERO, ZERO); cols]; rows];

    // println!("input 1 dim: {} x {}", input_1.len(), input_1[0].len());
    // println!("input 2 dim: {} x {}", input_2.len(), input_2[0].len());

    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = input_1[i][j] * input_2[i][j];
        }
    }

    result
}

pub fn dot_product_complex(input_1: &Vec<Complex<f64>>, input_2: &Vec<Complex<f64>>) -> Complex<f64> {
    assert_eq!(input_1.len(), input_2.len(), "Vectors must be the same length");
    let mut sum = Complex::<f64>::new(0.0, 0.0);
    for i in 0..input_1.len() {
        sum += input_1[i] * input_2[i];
    }
    sum
}
pub fn convert_3d_to_2d<T: Clone>(array_3d: &Vec<Vec<Vec<T>>>) -> Vec<Vec<T>> {
    let mut array_2d = Vec::new();

    // Assuming all layers have the same number of rows and columns
    let depth = array_3d.len();
    let rows = array_3d[0].len();

    // Iterate over each layer
    for layer in 0..depth {
        let mut flat_layer = Vec::new();
        for row in 0..rows {
            // Extend the flat_layer by appending each row from the current layer
            flat_layer.extend(array_3d[layer][row].clone());
        }
        array_2d.push(flat_layer);
    }

    array_2d
}

pub fn create_generic<T>(num_rows: usize) -> Vec<Vec<T>> {
    let mut matrix_result: Vec<Vec<T>> = Vec::new();

    for _i in 0..num_rows {
        matrix_result.push(Vec::new());
    }

    matrix_result
}

pub fn flatten_2d<T: NumCast + Copy>(array: &Vec<Vec<T>>) -> Vec<f64> {
    let mut matrix_result: Vec<f64> = Vec::new();

    for i in 0..array.len() {
        for j in 0..array.len() {
            if let Some(num) = NumCast::from(array[i][j]) {
                matrix_result.push(num);
            } else {
                // Handle the error or panic if you expect all conversions to succeed
                panic!("Failed to cast an element to f64");
            }
        }
    }

    matrix_result
}

pub fn create_2d(num_rows: usize, num_columns: usize) -> Vec<Vec<f64>> {
    let mut matrix_result: Vec<Vec<f64>> = Vec::new();

    for i in 0..num_rows {
        matrix_result.push(Vec::new());
        for j in 0..num_columns {
            if j >= matrix_result[i].len() {
                matrix_result[i].push(0.0);
            }
            matrix_result[i][j] = 0.0;
        }
    }

    matrix_result
}

pub fn create_generic_3d<T>(num_rows: usize, num_dim: usize) -> Vec<Vec<Vec<T>>> {
    let mut matrix_result: Vec<Vec<Vec<T>>> = Vec::new();

    for _d in 0..num_dim {
        matrix_result.push(create_generic(num_rows));
    }

    matrix_result
}

pub fn create_generic_one_dim<T: Default + Clone>(size: usize) -> Vec<T> {
    vec![T::default(); size]
}

pub fn parse_3_dim_to_float(matrix: &Vec<Vec<Vec<i32>>>) -> Vec<Vec<Vec<f64>>> {
    let num_dim = matrix.len();
    let num_rows = matrix[0].len();
    let num_columns = matrix[0][0].len();
    let mut matrix_result = vec![vec![vec![0f64; num_columns]; num_rows]; num_dim];

    for i in 0..matrix.len() {
        matrix_result[i] = parse_2dim_to_float(&matrix[i]);
    }

    //println!("dim1: {}, dim2: {}, dim3: {}", matrix_result.len(), matrix_result[0].len(), matrix_result[0][0].len());
    //println!("3 dim matrix:{:?}", matrix_result);

    matrix_result
}

pub fn parse_2dim_to_float(matrix: &Vec<Vec<i32>>) -> Vec<Vec<f64>> {
    let num_rows = matrix.len();
    let num_columns = matrix[0].len();
    let mut matrix_result = vec![vec![0f64; num_columns]; num_rows];

    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            matrix_result[i][j] = matrix[i][j] as f64;
        }
    }

    matrix_result
}

pub fn subtract<T: Debug + Clone + Sub<Output = T>>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut matrix_result: Vec<Vec<T>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[0].len() {
            matrix_result[i][j] = matrix_a[i][j].clone() - matrix_b[i][j].clone();
        }
    }

    // println!("created new matrix is {:?}", matrix_result);

    matrix_result
}

pub fn add_matrix<T: std::ops::Add<Output = T> + Copy>(a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
    assert!(
        a.len() == b.len() && a[0].len() == b[0].len(),
        "Input matrices must not be empty, a: {} x {} and b: {} x {} must have the same dimensions",
        a.len(),
        a[0].len(),
        b.len(),
        b[0].len(),
    );
    a.iter().zip(b.iter()).map(|(row_a, row_b)| row_a.iter().zip(row_b.iter()).map(|(&x, &y)| x + y).collect()).collect()
}

pub fn add_matrix_3d<T: Debug + Clone + Add<Output = T>>(matrix_a: &Vec<Vec<Vec<T>>>, matrix_b: &Vec<Vec<Vec<T>>>) -> Vec<Vec<Vec<T>>> {
    let mut matrix_result: Vec<Vec<Vec<T>>> = matrix_a.clone();

    assert!(
        matrix_a.len() == matrix_b.len() && matrix_a[0].len() == matrix_b[0].len() && matrix_a[0][0].len() == matrix_b[0][0].len(),
        "Input matrices must not be empty, a: {} x {} x {} and b: {} x {} x {} must have the same dimensions",
        matrix_a.len(),
        matrix_a[0].len(),
        matrix_a[0][0].len(),
        matrix_b.len(),
        matrix_b[0].len(),
        matrix_b[0][0].len(),
    );
    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            for k in 0..matrix_a[i][j].len() {
                matrix_result[i][j][k] = matrix_result[i][j][k].clone() + matrix_b[i % matrix_b.len()][j % matrix_b[0].len()][k % matrix_b[0][0].len()].clone();
            }
        }
    }

    matrix_result
}

pub fn add_matrix_3d_in_place<T: Debug + Clone + Add<Output = T>>(matrix_a: &mut Vec<Vec<Vec<T>>>, matrix_b: &Vec<Vec<Vec<T>>>) {
    assert!(
        matrix_a.len() == matrix_b.len() && matrix_a[0].len() == matrix_b[0].len() && matrix_a[0][0].len() == matrix_b[0][0].len(),
        "Input matrices must not be empty, a: {} x {} x {} and b: {} x {} x {} must have the same dimensions",
        matrix_a.len(),
        matrix_a[0].len(),
        matrix_a[0][0].len(),
        matrix_b.len(),
        matrix_b[0].len(),
        matrix_b[0][0].len(),
    );
    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            for k in 0..matrix_a[i][j].len() {
                matrix_a[i][j][k] = matrix_a[i][j][k].clone() + matrix_b[i % matrix_b.len()][j % matrix_b[0].len()][k % matrix_b[0][0].len()].clone();
            }
        }
    }
}

pub fn add_matrix_1d_c(matrix_a: &Vec<Complex<f64>>, matrix_b: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let mut matrix_result: Vec<Complex<f64>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        let val = matrix_result[i].clone() + matrix_b[i % matrix_b.len()].clone();
        matrix_result[i] = Complex::new(val.re, 0.0);
    }

    matrix_result
}

pub fn add_matrix_2d_c(matrix_a: &Vec<Vec<C>>, matrix_b: &Vec<Vec<C>>) -> Vec<Vec<C>> {
    let mut matrix_result: Vec<Vec<C>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            let val = matrix_result[i][j].clone() + matrix_b[i % matrix_b.len()][j % matrix_b[0].len()].clone();
            matrix_result[i][j] = val;
        }
    }

    matrix_result
}

pub fn add_matrix_3d_c(matrix_a: &Vec<Vec<Vec<C>>>, matrix_b: &Vec<Vec<Vec<C>>>) -> Vec<Vec<Vec<C>>> {
    let mut matrix_result: Vec<Vec<Vec<C>>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            for k in 0..matrix_a[i][j].len() {
                let val = matrix_result[i][j][k].clone() + matrix_b[i % matrix_b.len()][j % matrix_b[0].len()][k % matrix_b[0][0].len()].clone();
                matrix_result[i][j][k] = Complex::new(val.re, ZERO);
            }
        }
    }

    matrix_result
}

pub fn add_vector(matrix_a: &mut Vec<Vec<C>>, matrix_b: &Vec<C>) {
    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            matrix_a[i][j] = matrix_a[i][j].clone() + matrix_b[j].clone();
        }
    }
}

pub fn add_vectors<T: Debug + Clone + Add<Output = T>>(matrix_a: &Vec<T>, matrix_b: &Vec<T>) -> Vec<T> {
    let mut matrix_result: Vec<T> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        matrix_result[i] = matrix_result[i].clone() + matrix_b[i].clone();
    }

    matrix_result
}

pub fn average_vector_by_scalar<T>(matrix_a: &Vec<T>, scalar: Real) -> Vec<T>
where
    T: PolarConvertible + Debug + Clone + Div<Real, Output = T>,
{
    matrix_a.iter().map(|val| val.clone() / scalar).collect()
    //average_gradient_polar_1d_generic(matrix_a, scalar)
}

pub fn average_matrix_by_scalar<T>(matrix_a: &Vec<Vec<T>>, scalar: Real) -> Vec<Vec<T>>
where
    T: PolarConvertible + Debug + Clone + Div<Real, Output = T>,
{
    matrix_a.iter().map(|val| average_vector_by_scalar(val, scalar)).collect()
    //average_gradient_polar_generic(matrix_a, scalar)
}

pub fn average_matrix_3d_by_scalar<T>(matrix_a: &Vec<Vec<Vec<T>>>, scalar: Real) -> Vec<Vec<Vec<T>>>
where
    T: PolarConvertible + Debug + Clone + Div<Real, Output = T>,
{
    matrix_a.iter().map(|val| average_matrix_by_scalar(val, scalar)).collect()
}

pub fn scale_matrix_3d_by_scalar_in_place<T: Debug + Clone + Mul<Real, Output = T>>(matrix_a: &mut Vec<Vec<Vec<T>>>, scalar: Real) {
    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            for k in 0..matrix_a[i][j].len() {
                matrix_a[i][j][k] = matrix_a[i][j][k].clone() * scalar;
            }
        }
    }
}

pub fn scale_matrix_3d_by_scalar<T>(matrix_a: &Vec<Vec<Vec<T>>>, scalar: Real) -> Vec<Vec<Vec<T>>>
where
    T: PolarConvertible + Debug + Clone + Mul<Real, Output = T>,
{
    matrix_a.iter().map(|val| scale_matrix_by_scalar(val, scalar)).collect()
}

pub fn scale_matrix_by_scalar<T>(matrix_a: &Vec<Vec<T>>, scalar: Real) -> Vec<Vec<T>>
where
    T: PolarConvertible + Debug + Clone + Mul<Real, Output = T>,
{
    matrix_a.iter().map(|val| scale_vector_by_scalar(val, scalar)).collect()
}

pub fn scale_vector_by_scalar<T>(matrix_a: &Vec<T>, scalar: Real) -> Vec<T>
where
    T: PolarConvertible + Debug + Clone + Mul<Real, Output = T>,
{
    matrix_a.iter().map(|val| val.clone() * scalar).collect()
    //average_gradient_polar_1d_generic(matrix_a, scalar)
}

pub trait PolarConvertible: Clone {
    fn re(&self) -> f64;
    fn im(&self) -> f64;
    fn from_polar(magnitude: f64, angle: f64) -> Self;
}

impl PolarConvertible for Complex<f64> {
    fn re(&self) -> f64 {
        self.re
    }

    fn im(&self) -> f64 {
        self.im
    }

    fn from_polar(magnitude: f64, angle: f64) -> Self {
        Complex::from_polar(magnitude, angle)
    }
}

impl PolarConvertible for Complex<f32> {
    fn re(&self) -> f64 {
        self.re as f64
    }

    fn im(&self) -> f64 {
        self.im as f64
    }

    fn from_polar(magnitude: f64, angle: f64) -> Self {
        Complex::from_polar(magnitude as f32, angle as f32)
    }
}

impl PolarConvertible for f64 {
    fn re(&self) -> f64 {
        *self
    }

    fn im(&self) -> f64 {
        0.0
    }

    fn from_polar(magnitude: f64, angle: f64) -> Self {
        Complex::from_polar(magnitude, angle).re
    }
}

impl PolarConvertible for f32 {
    fn re(&self) -> f64 {
        *self as f64
    }

    fn im(&self) -> f64 {
        0.0
    }

    fn from_polar(magnitude: f64, angle: f64) -> Self {
        Complex::from_polar(magnitude as f32, angle as f32).re
    }
}

pub fn average_gradient_polar_generic<T: PolarConvertible>(sums: &[Vec<T>], batch_size: f64) -> Vec<Vec<T>> {
    let rows = sums.len();
    let cols = sums[0].len();
    let mut out = vec![vec![]; rows];

    for i in 0..rows {
        for j in 0..cols {
            let s = &sums[i][j];
            let mean_re = s.re() / batch_size;
            let mean_im = s.im() / batch_size;
            let mag = (mean_re.powi(2) + mean_im.powi(2)).sqrt();
            let ang = mean_im.atan2(mean_re);
            out[i].push(T::from_polar(mag, ang));
        }
    }

    out
}

pub fn average_gradient_polar_1d_generic<T: PolarConvertible>(sums: &[T], batch_size: f64) -> Vec<T> {
    sums.iter()
        .map(|s| {
            let mean_re = s.re() / batch_size;
            let mean_im = s.im() / batch_size;
            let mag = (mean_re.powi(2) + mean_im.powi(2)).sqrt();
            let ang = mean_im.atan2(mean_re);
            T::from_polar(mag, ang)
        })
        .collect()
}

pub fn multiply_scalar_with_matrix<T>(scalar: T, matrix: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Mul<Output = T> + Clone, // T must implement multiplication and cloning
{
    matrix.iter().map(|row| row.iter().map(|x| scalar.clone() * x.clone()).collect()).collect()
}

// Assuming this is the method you've defined for finding the highest index in the last row:
pub fn find_highest_index_last_row(input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Option<Vec<u32>> {
    let mut max_index_batch: Vec<u32> = vec![];

    for input in input_batch {
        // Get the last row from the input matrix
        let last_row = &input[input.len() - 1];

        // Initialize variables to track the index of the highest magnitude
        let mut max_index = 0;
        let mut max_magnitude = 0.0;

        // Iterate through the last row to find the highest magnitude
        for (i, value) in last_row.iter().enumerate() {
            let magnitude = value.norm(); // norm() gives the magnitude (absolute value) of the complex number
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                max_index = i;
            }
        }
        max_index_batch.push(max_index as u32);
    }

    Some(max_index_batch) // Return the index of the token with the highest probability
}
pub fn find_highest_index_batch(predicted_softmax_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<u32>> {
    predicted_softmax_batch
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|token_probs| {
                    // Find index with highest probability (argmax)
                    token_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
                        .map(|(index, _)| index as u32) // Convert index to u32
                        .unwrap_or(0) // Default to 0 if something goes wrong
                })
                .collect()
        })
        .collect()
}

pub fn apply_padding_mask_batch(input_batch: &mut Vec<Vec<Vec<Complex<f64>>>>, padding_mask_batch: &Vec<Vec<u32>>) {
    for (batch_ind, input) in input_batch.iter_mut().enumerate() {
        apply_padding_mask(input, &padding_mask_batch[batch_ind]);
    }
}

pub fn apply_padding_mask(input: &mut Vec<Vec<Complex<f64>>>, padding_mask: &Vec<u32>) {
    for (seq_ind, seq) in input.iter_mut().enumerate() {
        if padding_mask[seq_ind] == 0 {
            for value in seq.iter_mut() {
                *value = Complex::new(0.0, 0.0);
            }
        }
    }
}

pub fn get_reduced_matrix(matrix: &Vec<Vec<Complex<f64>>>, num_rows: usize, num_cols: usize) -> Vec<Vec<Complex<f64>>> {
    matrix
        .iter()
        .take(num_rows) // take first 5 rows
        .map(|row| row.iter().take(num_cols).cloned().collect()) // take first 5 columns from each row
        .collect()
}

pub fn compute_global_norm(grads: &Vec<Vec<Vec<Complex<f64>>>>, bias: &Vec<Vec<Complex<f64>>>) -> f64 {
    let mut total_norm = 0.0;

    let mut total_real = 0.0;
    let mut total_imag = 0.0;

    for g in grads {
        for row in g.iter() {
            for val in row.iter() {
                total_norm += val.norm_sqr();
                total_real += val.re * val.re;
                total_imag += val.im * val.im;
            }
        }
    }

    for row in bias.iter() {
        for val in row.iter() {
            total_norm += val.norm_sqr();
            total_real += val.re * val.re;
            total_imag += val.im * val.im;
        }
    }

    if VERBOSE {
        println!("Total norm: {:8e}", total_norm.sqrt());
        println!(
            "Real norm: {:}, Imag norm: {:}, ratio: {:}",
            total_real.sqrt(),
            total_imag.sqrt(),
            total_imag.sqrt() / total_real.sqrt()
        );
    }

    total_norm.sqrt()
}

pub fn clip_all_gradients_by_global_norm_3d(grads: &mut Vec<Vec<Vec<Complex<Real>>>>, bias: &mut Vec<Complex<Real>>, total_norm: f64, max_norm: f64) {
    if total_norm > max_norm {
        let scale = r(1.0 / total_norm);
        for g in grads {
            for row in g.iter_mut() {
                for val in row.iter_mut() {
                    *val *= scale;
                }
            }
        }

        bias.iter_mut().for_each(|val| *val *= scale);
    }
}

pub fn clip_all_gradients_by_global_norm_2d(grads: &mut Vec<Vec<Complex<Real>>>, bias: &mut Vec<Complex<Real>>, total_norm: f64, _max_norm: f64) {
    if total_norm > MAX_NORM {
        let scale = r(1.0 / total_norm);
        for row in grads.iter_mut() {
            for val in row.iter_mut() {
                *val *= scale;
            }
        }

        bias.iter_mut().for_each(|val| *val *= scale);
    }
}

pub fn normalize_gradients(gradients: &mut Vec<Vec<Complex<Real>>>) {
    // Step 1: elementwise clamp
    for row in gradients.iter_mut() {
        for val in row.iter_mut() {
            let norm = val.norm() as f64;
            if norm > MAX_ELEMENT && norm > 0.0 {
                *val *= r(MAX_ELEMENT / norm);
            }
        }
    }

    // Step 2: compute global L2 norm
    let global_norm: f64 = gradients.iter().flat_map(|row| row.iter()).map(|g| g.norm_sqr() as f64).sum::<f64>().sqrt();

    // Step 3: scale proportionally if global norm exceeds MAX_NORM
    if global_norm > MAX_NORM && global_norm > 0.0 {
        let scale = r(MAX_NORM / global_norm);
        for row in gradients.iter_mut() {
            for val in row.iter_mut() {
                *val *= scale;
            }
        }
    }
}

pub fn normalize_bias(bias: &mut Vec<Complex<Real>>) {
    // Step 1: elementwise clamp
    for val in bias.iter_mut() {
        let norm = val.norm() as f64;
        if norm > MAX_ELEMENT && norm > 0.0 {
            *val *= r(MAX_ELEMENT / norm);
        }
    }

    // Step 2: compute global L2 norm
    let global_norm: f64 = bias.iter().map(|g| g.norm_sqr() as f64).sum::<f64>().sqrt();

    // Step 3: scale proportionally
    if global_norm > MAX_NORM && global_norm > 0.0 {
        let scale = r(MAX_NORM / global_norm);
        for val in bias.iter_mut() {
            *val *= scale;
        }
    }
}

pub fn normalize_gradients_batch(gradients_batch: &mut Vec<Vec<Vec<Complex<Real>>>>) {
    for gradients in gradients_batch.iter_mut() {
        normalize_gradients(gradients);
    }
}

pub fn is_nan_or_inf<T: Float>(z: &Complex<T>) -> bool {
    z.re.is_nan() || z.re.is_infinite() || z.im.is_nan() || z.im.is_infinite()
}

pub fn contains_nan_or_inf<T: Float>(matrix: &mut Vec<Vec<Complex<T>>>) -> bool {
    let mut found = false;

    for row in matrix.iter_mut() {
        for z in row.iter_mut() {
            if is_nan_or_inf(z) {
                found = true;
            }
        }
    }

    found
}

pub fn check_nan_or_inf_3d<T: Float>(matrix_batch: &mut Vec<Vec<Vec<Complex<T>>>>, message: &str) {
    for matrix in matrix_batch.iter_mut() {
        if contains_nan_or_inf(matrix) {
            panic!("{:?}: The value is Not Valid", message);
        }
    }
}

pub fn check_nan_or_inf<T: Float>(matrix: &mut Vec<Vec<Complex<T>>>, message: &str) -> bool {
    if contains_nan_or_inf(matrix) {
        panic!("{:?}: The value is Not Valid", message);
    } else {
        false
    }
}

pub fn split_data_by_columns(data: &Vec<Vec<C>>) -> (Vec<Vec<C>>, Vec<Vec<C>>) {
    let (left, right): (Vec<Vec<C>>, Vec<Vec<C>>) = data
        .iter()
        .map(|row| {
            let column_splt = row.len() / 2;
            let (l, r) = row.split_at(column_splt);
            (l.to_vec(), r.to_vec()) // clone the slices
        })
        .unzip();

    (left, right)
}
