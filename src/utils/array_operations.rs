

pub fn sum_arrays(array1: &Vec<Vec<f64>>, array2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut sum_array: Vec<Vec<f64>> = vec![vec![0.0; array1[0].len()]; array1.len()];

    for i in 0..array1.len() {
        for j in 0..array1[i].len() {
            sum_array[i][j] = &array1[i][j] + &array2[i][j];
        }
    }

    sum_array
}