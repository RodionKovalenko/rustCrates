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

pub fn transpose(matrix_a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut matrix_b_clone = vec![
        vec![0.0f64; matrix_a.len()];
        matrix_a[0].len()
    ];

    for i in 0..matrix_b_clone.len() {
        for j in 0..matrix_b_clone[0].len() {
            matrix_b_clone[i][j] = matrix_a[j][i];
        }
    }

    matrix_b_clone
}