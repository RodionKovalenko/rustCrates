use crate::neural_networks::utils::matrix::{create_2d, create_generic, create_generic_one_dim, transpose};

pub fn standardize(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut standardized_mat = create_generic(matrix.len());
    let mean = get_mean_2d(&matrix);
    let variance = get_variance_2d(&matrix, mean);

    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            standardized_mat[i][j] = (matrix[i][j] - mean) / variance;
        }
    }

    standardized_mat
}

pub fn standardize_mean(matrix: &Vec<Vec<f64>>, mean: f64, variance: f64) -> Vec<Vec<f64>> {
    let mut standardized_mat = create_generic(matrix.len());

    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            standardized_mat[i][j] = (matrix[i][j] - mean) / variance.sqrt();
        }
    }

    standardized_mat
}

pub fn standardize_median(matrix: &Vec<Vec<f64>>, mean: f64, variance: f64) -> Vec<Vec<f64>> {
    let mut standardized_mat = create_generic(matrix.len());

    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            standardized_mat[i][j] = (matrix[i][j] - mean) / variance;
        }
    }

    standardized_mat
}

pub fn get_variance_2d(matrix: &Vec<Vec<f64>>, mean: f64) -> f64 {
    let mut variance = 0.0;

    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            variance += (matrix[i][j] - mean).powf(2.0) / (matrix.len() * matrix[0].len()) as f64;
        }
    }
    variance
}

pub fn get_variance_3d(matrix: &Vec<Vec<Vec<f64>>>, mean: f64) -> f64 {
    let mut variance = 0.0;

    println!("mean : {}", mean);

    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            for k in 0..matrix[0][0].len() {
                variance += (matrix[i][j][k] - mean).powf(2.0) /
                    (matrix.len() * matrix[0].len() * matrix[0][0].len()) as f64;
            }
        }
    }
    variance
}


pub fn get_mean_2d(matx: &Vec<Vec<f64>>) -> f64 {
    let mut mean = 0.0;

    for i in 0..matx.len() {
        for j in 0..matx[0].len() {
            mean += matx[i][j] / (matx.len() * matx[0].len()) as f64;
        }
    }

    mean
}

pub fn get_mean_3d(matx: &Vec<Vec<Vec<f64>>>) -> f64 {
    let mut mean = 0.0;

    for i in 0..matx.len() {
        for j in 0..matx[0].len() {
            for k in 0..matx[0][0].len() {
                mean += matx[i][j][k] / (matx.len() * matx[0].len() * matx[0][0].len()) as f64;
            }
        }
    }

    mean
}

pub fn get_median_2d(matx: &Vec<Vec<f64>>) -> f64 {
    let result_sorted = unfold_2d(&matx);
    let size = result_sorted.len();

    if result_sorted.len() % 2 == 0 {
        (result_sorted[size / 2] + result_sorted[(size / 2) + 1]) / 2.0
    } else {
        (result_sorted[(size + 1) / 2]) / 2.0
    }
}

pub fn get_median_3d(matx: &Vec<Vec<Vec<f64>>>) -> f64 {
    let result_sorted = unfold_3d(&matx);
    let size = result_sorted.len();

    if result_sorted.len() % 2 == 0 {
        (result_sorted[size / 2] + result_sorted[(size / 2) + 1]) / 2.0
    } else {
        (result_sorted[(size + 1) / 2]) / 2.0
    }
}

pub fn unfold_2d(matx: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();

    let mut ind = 0;
    for i in 0..matx.len() {
        for j in 0..matx[0].len() {
            result[ind] = matx[i][j];
            ind += 1;
        }
    }

    result.sort_by(|a, b| a.partial_cmp(b).unwrap());

    //println!("matrix sorted: {:?} ", result);
    result
}

pub fn unfold_3d(matx: &Vec<Vec<Vec<f64>>>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();

    let mut ind = 0;
    for i in 0..matx.len() {
        for j in 0..matx[0].len() {
            for k in 0..matx[0][0].len() {
                result[ind] = matx[i][j][k];
                ind += 1;
            }
        }
    }
    result.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // println!("unfolded: {:?}", result);

    result
}


pub fn normalize_max_mean(matr: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let tranposed = transpose(matr);
    let mut normalized_m = create_2d(matr.len(), matr[0].len());

    for i in 0..tranposed.len() {
        let mut column: Vec<f64> = tranposed[i].clone();
        column.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let max = column[column.len() - 1];
        let min = column[0];

        for j in 0..tranposed[0].len() {
            normalized_m[j][i] = (tranposed[i][j] - min) / (max - min);

            if normalized_m[j][i] > 1.0 {
                println!("initial value: {}, max: {}, min: {}", matr[i][j], max, min);
                panic!("normalization failed because the value is larger than 1");
            }
        }
    }

    normalized_m
}