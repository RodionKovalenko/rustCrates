use std::fmt::Debug;
use std::ops::{Mul, AddAssign, Sub, Add};

pub fn multiple(matrix_a: &Vec<Vec<f64>>, matrix_b: &Vec<Vec<f64>>)
                -> Vec<Vec<f64>> {
    let num_rows = matrix_a.len();
    let mut num_columns = matrix_b[0].len();
    let matrix_a_clone = matrix_a.clone();
    let mut matrix_b_clone = matrix_b.clone();

    println!("matrix a : {}, {}", matrix_a.len(), matrix_a[0].len());
    println!("matrix b : {}, {}", matrix_b.len(), matrix_b[0].len());

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

    println!("result matrix rows {}, columns {}", result_matrix.len(), result_matrix[0].len());

    for i in 0..num_rows {
        for j in 0..num_columns {
            for k in 0..matrix_b_clone.len() {
                result_matrix[i][j] += matrix_a_clone[i][k] * matrix_b_clone[k][j];
            }
        }
    }

    result_matrix
}

pub fn multiple_generic_2d<T: Debug + Clone + Mul<Output=T> + AddAssign + From<f64>>
(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let num_rows = matrix_a.len() as i32;
    let mut num_columns = matrix_b[0].len() as i32;
    let matrix_a_clone: Vec<Vec<T>> = matrix_a.clone();
    let mut matrix_b_clone: Vec<Vec<T>> = matrix_b.clone();

    // println!("matrix a : {}, {}", matrix_a.len(), matrix_a[0].len());
    // println!("matrix b : {}, {}", matrix_b.len(), matrix_b[0].len());

    if matrix_a[0].len() != matrix_b.len() && matrix_a[0].len() != matrix_b[0].len() {
        panic!("Matrix A does not have the same number of columns as Matrix B rows.");
    }

    if matrix_a[0].len() != matrix_b.len() {
        matrix_b_clone = transpose(matrix_b);
        num_columns = matrix_b_clone[0].len() as i32;
    }

    let mut result_matrix: Vec<Vec<T>> = create_generic(num_rows);

    for i in 0..num_rows as usize {
        for j in 0..num_columns as usize {
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

pub fn create_generic<T>(num_rows: i32) -> Vec<Vec<T>> {
    let mut matrix_result: Vec<Vec<T>> = Vec::new();

    for _i in 0..num_rows as usize {
        matrix_result.push(Vec::new());
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

pub fn create_generic_3d<T>(num_rows: i32, num_dim: i32) -> Vec<Vec<Vec<T>>> {
    let mut matrix_result: Vec<Vec<Vec<T>>> = Vec::new();

    for _d in 0..num_dim {
        matrix_result.push(create_generic(num_rows));
    }

    matrix_result
}

pub fn create_generic_one_dim<T> () -> Vec<T> {
    let mut matrix_result: Vec<T> = Vec::new();

    matrix_result
}

pub fn parse_3_dim_to_float(matrix: &Vec<Vec<Vec<i32>>>) -> Vec<Vec<Vec<f64>>> {
    let num_dim = matrix.len();
    let num_rows = matrix[0].len();
    let num_columns = matrix[0][0].len();
    let mut matrix_result = vec![
        vec![
            vec![0f64; num_columns];
            num_rows
        ];
        num_dim
    ];

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

pub fn subtract<T: Debug + Clone + Sub<Output=T>>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut matrix_result: Vec<Vec<T>> = matrix_a.clone();

    for i in 0..matrix_a.len() {
        for j in 0..matrix_a[0].len() {
            matrix_result[i][j] = matrix_a[i][j].clone() - matrix_b[i][j].clone();
        }
    }

    // println!("created new matrix is {:?}", matrix_result);

    matrix_result
}


pub fn get_error<T: Debug + Clone + Sub<Output=T> + Add<Output=T> + Mul<Output=T> + From<f64> + Into<f64>>
(target_m: &Vec<Vec<T>>, output_m: &Vec<Vec<T>>) -> Vec<T> {
    // convert output into one dimensional array
    let mut output_one_dim: Vec<T> = create_generic_one_dim();
    let mut target_one_dim: Vec<T> = create_generic_one_dim();
    let mut matrix_result: Vec<T> = create_generic_one_dim();

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

pub fn add<T: Debug + Clone + Add<Output=T>>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<T>) -> Vec<Vec<T>> {
    let mut matrix_result: Vec<Vec<T>> = matrix_a.clone();

    for j in 0..matrix_a[0].len() {
        matrix_result[0][j] = matrix_result[0][j].clone() + matrix_b[j].clone();
    }

    // println!("created new matrix is {:?}", matrix_result);

    matrix_result
}