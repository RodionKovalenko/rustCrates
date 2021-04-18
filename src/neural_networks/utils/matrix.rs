use std::fmt::Debug;
use std::ops::{Mul, AddAssign};

pub fn multiple(matrix_a: &Vec<Vec<f64>>, matrix_b: &Vec<Vec<f64>>)
                -> Vec<Vec<f64>> {
    let num_rows = matrix_a.len();
    let mut num_columns = matrix_b[0].len();
    let matrix_a_clone = matrix_a.clone();
    let mut matrix_b_clone = matrix_b.clone();

    if matrix_a[0].len() != matrix_b.len() && matrix_a[0].len() != matrix_b[0].len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    if matrix_a[0].len() != matrix_b.len() {
        matrix_b_clone = transpose(matrix_b);
        num_columns = matrix_b_clone[0].len();
    }

    let mut result_matrix = vec![
        vec![0.0f64; num_columns];
        num_rows
    ];

    // println!("matrix a rows {}, matrix a columns {}", matrix_a_clone.len(), matrix_a_clone[0].len());
    // println!("matrix a rows {}, matrix a columns {}", matrix_b_clone.len(), matrix_b_clone[0].len());
    println!("result matrix rows {}, columns {}", result_matrix.len(), result_matrix[0].len());
    // println!("matrix A {:?}", matrix_a_clone);
    // println!("");
    // println!("matrix B {:?}", matrix_b_clone);
    // println!("");

    for i in 0..num_rows {
        for j in 0..num_columns {
            for k in 0..matrix_b_clone.len() {
                result_matrix[i][j] += matrix_a_clone[i][k] * matrix_b_clone[k][j];
            }
        }
    }

    result_matrix
}

pub fn multiple_generic<T: Debug + Clone + Mul<Output=T> + AddAssign + From<f64>>
(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let num_rows = matrix_a.len();
    let mut num_columns = matrix_b[0].len();
    let matrix_a_clone: Vec<Vec<T>> = matrix_a.clone();
    let mut matrix_b_clone: Vec<Vec<T>> = matrix_b.clone();

    if matrix_a[0].len() != matrix_b.len() && matrix_a[0].len() != matrix_b[0].len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    if matrix_a[0].len() != matrix_b.len() {
        matrix_b_clone = transpose(matrix_b);
        num_columns = matrix_b_clone[0].len();
    }

    let mut result_matrix: Vec<Vec<T>> = create_generic(num_rows, num_columns);

    // println!("matrix a rows {}, matrix a columns {}", matrix_a_clone.len(), matrix_a_clone[0].len());
    // println!("matrix b rows {}, matrix b columns {}", matrix_b_clone.len(), matrix_b_clone[0].len());

    // println!("matrix A {:?}", matrix_a_clone);
    // println!("");
    // println!("matrix B {:?}", matrix_b_clone);
    // println!("");

    for i in 0..num_rows {
        for j in 0..num_columns {
            for k in 0..matrix_b_clone.len() {
                if j >= result_matrix[i].len() {
                    result_matrix[i].push(matrix_a_clone[i][k].clone() * matrix_b_clone[k][j].clone());
                } else {
                    result_matrix[i][j] += matrix_a_clone[i][k].clone() * matrix_b_clone[k][j].clone();
                }
            }
        }
    }

    // println!("result matrix rows {}, columns {}", result_matrix.len(), result_matrix[0].len());
    // println!("");

    result_matrix
}


pub fn transpose<T: Debug + Clone>(matrix_a: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut matrix_result = Vec::new();
    let mut row_vector;

    for j in 0..matrix_a[0].len() {
        row_vector = Vec::new();
        for i in 0..matrix_a.len() {
            row_vector.push(matrix_a[i][j].clone());
        }
        matrix_result.push(row_vector);
    }

    // println!("matrix to transpose rows: {}, columns: {}", matrix_a.len(), matrix_a[0].len());
    // println!("matrix generic transposed rows: {:?}, columns: {}", matrix_b_clone.len(), matrix_b_clone[0].len());

    matrix_result
}

pub fn create_generic<T: Debug + Clone>(num_rows: usize, num_columns: usize) -> Vec<Vec<T>> {
    let mut matrix_result: Vec<Vec<T>> = Vec::new();

    for _i in 0..num_rows {
        matrix_result.push(Vec::with_capacity(num_columns));
    }

    matrix_result
}

pub fn create_generic_one_dim<T: Debug + Clone + From<f64>>
(num_rows: usize) -> Vec<T> {
    let mut matrix_result: Vec<T> = Vec::new();

    for _i in 0..num_rows {
        matrix_result.push(T::from(0.0));
    }

    matrix_result
}

pub fn parse_to_float(matrix: &Vec<Vec<i32>>) -> Vec<Vec<f64>> {
    let num_rows = matrix.len();
    let num_columns = matrix[0].len();
    let mut matrix_result = vec![
        vec![0f64; num_columns];
        num_rows
    ];

    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            matrix_result[i][j] = matrix[i][j] as f64;
        }
    }

    matrix_result
}