
pub fn sigmoid<T: Debug + Clone>(matrix_a: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut matrix_b_clone = Vec::new();
    let mut row_vector;

    for j in 0..matrix_a[0].len() {
        row_vector = Vec::new();
        for i in 0..matrix_a.len() {
            row_vector.push(matrix_a[i][j].clone());
        }
        matrix_b_clone.push(row_vector);
    }

    println!("matrix to transpose rows: {}, columns: {}", matrix_a.len(), matrix_a[0].len());
    println!("matrix generic transposed rows: {:?}, columns: {}", matrix_b_clone.len(), matrix_b_clone[0].len());

    matrix_b_clone
}