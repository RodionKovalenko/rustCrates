
pub fn transpose(matrix_a: &Vec<Vec<T>>) -> Vec<Vec<T>> {
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