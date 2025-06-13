use faer::Mat;
use num::Complex;
use num_traits::NumCast;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};
use std::sync::{Arc, Mutex};
// use std::ffi::c_char;

// extern "C" {
//     fn zgemm_(transa: *const c_char, transb: *const c_char, m: *const i64, n: *const i64, k: *const i64, alpha: *const Complex<f64>, a: *const Complex<f64>, lda: *const i64, b: *const Complex<f64>, ldb: *const i64, beta: *const Complex<f64>, c: *mut Complex<f64>, ldc: *const i64);
// }
// pub fn multiply_complex(matrix_a: &[Vec<Complex<f64>>], matrix_b: &[Vec<Complex<f64>>]) -> Vec<Vec<Complex<f64>>> {
//     let m = matrix_a.len() as i64;
//     let k = matrix_a[0].len() as i64;
//     let n = matrix_b[0].len() as i64;

//     assert!(m > 0 && n > 0 && k > 0, "Matrices must not be empty");
//     assert!(matrix_b.len() as i64 == k, "A's columns must match B's rows");
//     for row in matrix_a {
//         assert_eq!(row.len(), k as usize, "All rows of A must have the same length");
//     }
//     for row in matrix_b {
//         assert_eq!(row.len(), n as usize, "All rows of B must have the same length");
//     }

//     fn flatten_col_major(mat: &[Vec<Complex<f64>>], rows: i64, cols: i64) -> Vec<Complex<f64>> {
//         let mut v = Vec::with_capacity((rows * cols) as usize);
//         for col in 0..cols {
//             for row in 0..rows {
//                 v.push(mat[row as usize][col as usize]);
//             }
//         }
//         v
//     }

//     let a = flatten_col_major(matrix_a, m, k);
//     let b = flatten_col_major(matrix_b, k, n);
//     let mut c = vec![Complex::<f64>::new(0.0, 0.0); (m * n) as usize];

//     let transa = b'N';
//     let transb = b'N';
//     let lda = m;
//     let ldb = k;
//     let ldc = m;
//     let alpha = Complex::<f64>::new(1.0, 0.0);
//     let beta = Complex::<f64>::new(0.0, 0.0);

//     //println!("Calling zgemm_ with m={}, n={}, k={}, lda={}, ldb={}, ldc={}", m, n, k, lda, ldb, ldc);

//     unsafe {
//         zgemm_(&transa as *const u8 as *const c_char, &transb as *const u8 as *const c_char, &m, &n, &k, &alpha, a.as_ptr(), &lda, b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc);
//     }

//     let mut result = vec![vec![Complex::<f64>::new(0.0, 0.0); n as usize]; m as usize];
//     for col in 0..n as usize {
//         for row in 0..m as usize {
//             result[row][col] = c[col * m as usize + row];
//         }
//     }
//     result
// }

// pub fn multiply_complex(matrix_a: &[Vec<Complex<f64>>], matrix_b: &[Vec<Complex<f64>>]) -> Vec<Vec<Complex<f64>>> {
//     let m = matrix_a.len();
//     let k = matrix_a[0].len();
//     let n = matrix_b[0].len();

//     // Convert matrix_a into Array2 (shape m x k)
//     let mut a = Array2::<Complex<f64>>::zeros((m, k));
//     for (i, row) in matrix_a.iter().enumerate() {
//         for (j, &val) in row.iter().enumerate() {
//             a[[i, j]] = val;
//         }
//     }

//     // Convert matrix_b into Array2 (shape k x n)
//     let mut b = Array2::<Complex<f64>>::zeros((k, n));
//     for (i, row) in matrix_b.iter().enumerate() {
//         for (j, &val) in row.iter().enumerate() {
//             b[[i, j]] = val;
//         }
//     }

//     // Multiply using ndarray dot (will use BLAS if enabled)
//     let c = a.dot(&b);

//     // Convert result back to Vec<Vec<Complex<f64>>>
//     let mut result = vec![vec![Complex::new(0.0, 0.0); n]; m];
//     for i in 0..m {
//         for j in 0..n {
//             result[i][j] = c[[i, j]];
//         }
//     }

//     result
// }

// #[repr(C)]
// #[derive(Clone, Copy, Debug)]
// pub struct CuDoubleComplex {
//     pub x: f64,
//     pub y: f64,
// }

// impl From<Complex<f64>> for CuDoubleComplex {
//     fn from(c: Complex<f64>) -> Self {
//         Self { x: c.re, y: c.im }
//     }
// }

// #[link(name = "cublas")]
// extern "system" {
//     pub fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
//     pub fn cublasDestroy_v2(handle: *mut c_void) -> i32;

//     pub fn cublasZgemm3m(handle: *mut c_void, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: *const CuDoubleComplex, A: *const CuDoubleComplex, lda: i32, B: *const CuDoubleComplex, ldb: i32, beta: *const CuDoubleComplex, C: *mut CuDoubleComplex, ldc: i32) -> i32;

//     pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
//     pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
//     pub fn cudaFree(devPtr: *mut c_void) -> i32;
// }

// const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
// const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
// const CUBLAS_STATUS_SUCCESS: i32 = 0;
// const CUBLAS_OP_N: i32 = 0;

// fn check_cuda(status: i32) {
//     if status != 0 {
//         panic!("CUDA call failed with status: {}", status);
//     }
// }

// pub fn multiply_complex_cuda(matrix_a: &Vec<Vec<Complex<f64>>>, matrix_b: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
//     let a_rows = matrix_a.len();
//     let a_cols = matrix_a[0].len();
//     let b_rows = matrix_b.len();
//     let b_cols = matrix_b[0].len();

//     if a_cols != b_rows {
//         panic!("Invalid matrix dimensions: A is {}x{}, B is {}x{}", a_rows, a_cols, b_rows, b_cols);
//     }

//     // Flatten in column-major order
//     let a_flat: Vec<CuDoubleComplex> = (0..a_cols).flat_map(|j| (0..a_rows).map(move |i| CuDoubleComplex::from(matrix_a[i][j]))).collect();
//     let b_flat: Vec<CuDoubleComplex> = (0..b_cols).flat_map(|j| (0..b_rows).map(move |i| CuDoubleComplex::from(matrix_b[i][j]))).collect();
//     let mut c_flat: Vec<CuDoubleComplex> = vec![CuDoubleComplex { x: 0.0, y: 0.0 }; a_rows * b_cols];

//     let m = a_rows as i32;
//     let n = b_cols as i32;
//     let k = a_cols as i32;

//     let lda = m;
//     let ldb = k;
//     let ldc = m;

//     let alpha = CuDoubleComplex { x: 1.0, y: 0.0 };
//     let beta = CuDoubleComplex { x: 0.0, y: 0.0 };

//     unsafe {
//         let mut handle: *mut c_void = std::ptr::null_mut();
//         let status = cublasCreate_v2(&mut handle);
//         if status != CUBLAS_STATUS_SUCCESS || handle.is_null() {
//             panic!("Failed to create cuBLAS handle: status = {}, handle = {:?}", status, handle);
//         }

//         let size_a = a_flat.len() * std::mem::size_of::<CuDoubleComplex>();
//         let size_b = b_flat.len() * std::mem::size_of::<CuDoubleComplex>();
//         let size_c = c_flat.len() * std::mem::size_of::<CuDoubleComplex>();

//         let mut d_a: *mut c_void = std::ptr::null_mut();
//         let mut d_b: *mut c_void = std::ptr::null_mut();
//         let mut d_c: *mut c_void = std::ptr::null_mut();

//         check_cuda(cudaMalloc(&mut d_a, size_a));
//         check_cuda(cudaMalloc(&mut d_b, size_b));
//         check_cuda(cudaMalloc(&mut d_c, size_c));

//         check_cuda(cudaMemcpy(d_a, a_flat.as_ptr() as *const c_void, size_a, CUDA_MEMCPY_HOST_TO_DEVICE));
//         check_cuda(cudaMemcpy(d_b, b_flat.as_ptr() as *const c_void, size_b, CUDA_MEMCPY_HOST_TO_DEVICE));

//         let status = cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a as *const CuDoubleComplex, lda, d_b as *const CuDoubleComplex, ldb, &beta, d_c as *mut CuDoubleComplex, ldc);

//         if status != CUBLAS_STATUS_SUCCESS {
//             cublasDestroy_v2(handle);
//             panic!("cublasZgemm3m failed with status: {}", status);
//         }

//         check_cuda(cudaMemcpy(c_flat.as_mut_ptr() as *mut c_void, d_c, size_c, CUDA_MEMCPY_DEVICE_TO_HOST));

//         cudaFree(d_a);
//         cudaFree(d_b);
//         cudaFree(d_c);

//         cublasDestroy_v2(handle);
//     }

//     // Convert back to row-major
//     let mut result = vec![vec![Complex::new(0.0, 0.0); b_cols]; a_rows];
//     for j in 0..b_cols {
//         for i in 0..a_rows {
//             let z = c_flat[j * a_rows + i];
//             result[i][j] = Complex::new(z.x, z.y);
//         }
//     }

//     result
// }

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

pub fn multiply_complex(matrix_a: &Vec<Vec<Complex<f64>>>, matrix_b: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
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

pub fn multiply_complex_with_f64(matrix_a: &Vec<Vec<Complex<f64>>>, matrix_b: &Vec<Vec<f64>>) -> Vec<Vec<Complex<f64>>> {
    let num_rows = matrix_a.len();
    let num_columns = matrix_b[0].len();

    // Ensure that the number of columns in matrix_a is equal to the number of rows in matrix_b
    if matrix_a[0].len() != matrix_b.len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    // Initialize result matrix with 0.0 values
    let mut result_matrix: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); num_columns]; num_rows];

    // println!("anzahl cput {}", num_cpus::get());

    let pool = ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap();

    pool.install(|| {
        result_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..num_columns {
                //row[j] = (0..matrix_b.len()).map(|k| matrix_a[i][k] * Complex::new(matrix_b[k][j], 0.0)).sum();
                row[j] = (0..matrix_b.len()).map(|k| matrix_a[i][k] * matrix_b[k][j]).sum();
            }
        });
    });

    result_matrix
}

pub fn multiply_f64_complex(matrix_a: &Vec<Vec<f64>>, matrix_b: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let num_rows = matrix_a.len();
    let num_columns = matrix_b[0].len();

    // Ensure that the number of columns in matrix_a is equal to the number of rows in matrix_b
    if matrix_a[0].len() != matrix_b.len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    // Initialize result matrix with 0.0 values
    let mut result_matrix: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); num_columns]; num_rows];

    let pool = ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap();

    pool.install(|| {
        result_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..num_columns {
                //row[j] = (0..matrix_b.len()).map(|k| Complex::new(matrix_a[i][k], 0.0) * matrix_b[k][j]).sum();
                row[j] = (0..matrix_b.len()).map(|k| matrix_a[i][k] * matrix_b[k][j]).sum();
            }
        });
    });

    result_matrix
}

pub fn conjugate_transpose(matrix: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![Complex::new(0.0, 0.0); rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j].conj();
        }
    }
    result
}

pub fn conjugate(matrix: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

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

pub fn hadamard_product_2d_c(input_1: &Vec<Vec<Complex<f64>>>, input_2: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let rows = input_1.len();
    let cols = input_1[0].len();

    // Initialize result matrix with zeros
    let mut result = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

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
    a.iter().zip(b.iter()).map(|(row_a, row_b)| row_a.iter().zip(row_b.iter()).map(|(&x, &y)| x + y).collect()).collect()
}

pub fn add_matrix_3d<T: Debug + Clone + Add<Output = T>>(matrix_a: &Vec<Vec<Vec<T>>>, matrix_b: &Vec<Vec<Vec<T>>>) -> Vec<Vec<Vec<T>>> {
    let mut matrix_result: Vec<Vec<Vec<T>>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            for k in 0..matrix_a[i][j].len() {
                matrix_result[i][j][k] = matrix_result[i][j][k].clone() + matrix_b[i % matrix_b.len()][j % matrix_b[0].len()][k % matrix_b[0][0].len()].clone();
            }
        }
    }

    matrix_result
}

pub fn add_matrix_3d_c(matrix_a: &Vec<Vec<Vec<Complex<f64>>>>, matrix_b: &Vec<Vec<Vec<Complex<f64>>>>) -> Vec<Vec<Vec<Complex<f64>>>> {
    let mut matrix_result: Vec<Vec<Vec<Complex<f64>>>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            for k in 0..matrix_a[i][j].len() {
                let val = matrix_result[i][j][k].clone() + matrix_b[i % matrix_b.len()][j % matrix_b[0].len()][k % matrix_b[0][0].len()].clone();
                matrix_result[i][j][k] = Complex::new(val.re, 0.0);
            }
        }
    }

    matrix_result
}

pub fn add_vector<T: Debug + Clone + Add<Output = T>>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<T>) -> Vec<Vec<T>> {
    let mut matrix_result: Vec<Vec<T>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            matrix_result[i][j] = matrix_result[i][j].clone() + matrix_b[j].clone();
        }
    }

    matrix_result
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

pub fn clip_gradients(gradients: &mut Vec<Vec<Complex<f64>>>, threshold: f64) {
    for row in gradients.iter_mut() {
        clip_gradient_1d(row, threshold);
    }
}

pub fn clip_gradient_1d(gradients: &mut Vec<Complex<f64>>, threshold: f64) {
    for val in gradients.iter_mut() {
        let norm = val.norm();

        if norm > threshold {
            *val *= threshold / norm; // Scale down to threshold
        }
    }
}
pub fn is_nan_or_inf(z: &Complex<f64>) -> bool {
    z.re.is_nan() || z.re.is_infinite() || z.im.is_nan() || z.im.is_infinite()
}

pub fn contains_nan_or_inf(matrix: &mut Vec<Vec<Complex<f64>>>) -> bool {
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

pub fn check_nan_or_inf_3d(matrix_batch: &mut Vec<Vec<Vec<Complex<f64>>>>, message: &str) {
    for matrix in matrix_batch.iter_mut() {
        if contains_nan_or_inf(matrix) {
            panic!("{:?}: The value is Not Valid", message);
        }
    }
}

pub fn check_nan_or_inf(matrix: &mut Vec<Vec<Complex<f64>>>, message: &str) -> bool {
    if contains_nan_or_inf(matrix) {
        panic!("{:?}: The value is Not Valid", message);
    } else {
        false
    }
}
