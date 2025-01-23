use num::Complex;
use num_traits::NumCast;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};
use std::sync::{Arc, Mutex};

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

pub fn multiply_complex<T, V>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<V>>) -> Vec<Vec<Complex<f64>>>
where
    T: Into<Complex<f64>> + Clone + Debug,
    V: Into<Complex<f64>> + Clone + Debug,
{
    // Convert matrices to Complex<f64>
    let mut matrix_a_clone = convert_to_complex(matrix_a);
    let mut matrix_b_clone = convert_to_complex(matrix_b);

    let mut num_rows = matrix_a_clone.len();
    let mut num_columns = matrix_b_clone[0].len();

    // Handle mismatched dimensions for matrix multiplication
    if matrix_a_clone[0].len() != matrix_b_clone.len() {
        if matrix_a_clone[0].len() == matrix_b_clone[0].len() {
            matrix_b_clone = transpose(&matrix_b_clone);
            num_columns = matrix_b_clone[0].len();
        } else if matrix_a_clone.len() == matrix_b_clone.len() {
            matrix_a_clone = transpose(&matrix_a_clone);
            num_rows = matrix_a_clone.len();
        }
    }

    if matrix_a_clone[0].len() != matrix_b_clone.len() && matrix_a_clone.len() != matrix_b_clone.len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    let mut result_matrix: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); num_columns]; num_rows];

    // Create a custom thread pool with exactly 8 threads
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

// A helper function to convert matrix elements to Complex<f64>
fn convert_to_complex<T: Into<Complex<f64>> + Clone + Debug>(matrix: &Vec<Vec<T>>) -> Vec<Vec<Complex<f64>>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|v| {
                    // Convert the element to Complex<f64>
                    v.clone().into() // This uses the Into trait to convert any type to Complex<f64>
                })
                .collect()
        })
        .collect()
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

pub fn get_error<T: Debug + Clone + Sub<Output = T> + Add<Output = T> + Mul<Output = T> + From<f64> + Into<f64>>(target_m: &Vec<Vec<T>>, output_m: &Vec<Vec<T>>) -> Vec<T> {
    // convert output into one dimensional array
    let mut output_one_dim: Vec<T> = Vec::new();
    let mut target_one_dim: Vec<T> = Vec::new();
    let mut matrix_result: Vec<T> = Vec::new();

    // println!("target matrix size: {}, {}", target_m.len(), target_m[0].len());
    // println!("output matrix size: {}, {}", output_m.len(), output_m[0].len());
    // println!("created new matrix error is {}", matrix_result.len());

    for j in 0..output_m[0].len() {
        for i in 0..output_m.len() {
            output_one_dim[j] = output_one_dim[j].clone() + output_m[i][j].clone();
        }
    }

    for j in 0..target_m[0].len() {
        for i in 0..target_m.len() {
            target_one_dim[j] = target_one_dim[j].clone() + target_m[i][j].clone();
        }
    }

    for j in 0..target_m[0].len() {
        matrix_result[j] = output_one_dim[j].clone() - target_one_dim[j].clone();
    }

    matrix_result
}

pub fn add_matrix<T: Debug + Clone + Add<Output = T>>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut matrix_result: Vec<Vec<T>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[i].len() {
            matrix_result[i][j] = matrix_result[i][j].clone() + matrix_b[i][j].clone();
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
pub fn find_highest_index(input_batch: &Vec<Vec<Vec<Complex<f64>>>>) -> Option<Vec<u32>> {
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
