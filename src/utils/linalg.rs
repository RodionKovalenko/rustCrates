
use rand::Rng;

pub fn multiply_matrices(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let size = a.len();
    let mut result = vec![vec![0.0; size]; size];

    for i in 0..size {
        for j in 0..size {
            result[i][j] = (0..size).fold(0.0, |sum, k| sum + a[i][k] * b[k][j]);
        }
    }
    result
}

pub fn is_identity_matrix(matrix: &Vec<Vec<f64>>, tolerance: f64) -> bool {
    let size = matrix.len();
    for i in 0..size {
        for j in 0..size {
            if (i == j && (matrix[i][j] - 1.0).abs() > tolerance)
                || (i != j && matrix[i][j].abs() > tolerance)
            {
                return false;
            }
        }
    }
    true
}

pub fn get_determinant(matrix: &Vec<Vec<f64>>) -> f64 {
    let size = matrix.len();

    // Ensure the matrix is square
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
        panic!("Matrix must be square and non-empty");
    }

    // Perform LU decomposition
    match lu_decomposition(matrix) {
        Some((_, upper, parity)) => {
            // Calculate determinant as the product of the diagonal elements of U
            let determinant = (0..size).fold(1.0, |prod, i| prod * upper[i][i]);

            // Adjust for the parity of row swaps
            determinant * parity
        }
        None => 0.0, // Singular matrix
    }
}

fn lu_decomposition(matrix: &Vec<Vec<f64>>) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, f64)> {
    let size = matrix.len();
    let mut lower = vec![vec![0.0; size]; size];
    let mut upper = matrix.clone();
    let mut parity = 1.0;

    for i in 0..size {
        // Find pivot (partial pivoting)
        let mut max_row = i;
        for k in i + 1..size {
            if upper[k][i].abs() > upper[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows if necessary and adjust parity
        if i != max_row {
            upper.swap(i, max_row);
            parity *= -1.0;
        }

        if upper[i][i].abs() < 1e-9 {
            return None; // Singular matrix
        }

        // Elimination step: make upper triangular
        for j in i + 1..size {
            let factor = upper[j][i] / upper[i][i];
            for k in i..size {
                upper[j][k] -= factor * upper[i][k];
            }
            lower[j][i] = factor;
        }

        // Set the diagonal of lower to 1
        lower[i][i] = 1.0;
    }

    Some((lower, upper, parity))
}

pub fn gaussian_elimination_inverse(matrix: &Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let size = matrix.len();

    // Create an augmented matrix (matrix | identity)
    let mut augmented = matrix.clone();
    for i in 0..size {
        for j in size..size * 2 {
            if j - size == i {
                augmented[i].push(1.0); // Add identity matrix part
            } else {
                augmented[i].push(0.0);
            }
        }
    }

    // Perform Gaussian elimination
    for i in 0..size {
        // Find the row with the largest value in column i (partial pivoting)
        let mut max_row = i;
        for j in i + 1..size {
            if augmented[j][i].abs() > augmented[max_row][i].abs() {
                max_row = j;
            }
        }

        // Swap rows if necessary
        if i != max_row {
            augmented.swap(i, max_row);
        }

        // Ensure pivot is not too small (avoid division by zero or numerical instability)
        if augmented[i][i].abs() < 1e-12 {
            println!("matrix is singular, no inverse");
            return None; // Matrix is singular, no inverse
        }

        // Normalize the pivot row
        let pivot = augmented[i][i];
        for j in i..size * 2 {
            // Ensure `j` is within bounds of augmented matrix
            if j < size * 2 {
                augmented[i][j] /= pivot;
            }
        }

        // Eliminate the current column from all other rows
        for j in 0..size {
            if j != i {
                let factor = augmented[j][i];
                for k in i..size * 2 {
                    // Ensure `k` is within bounds of augmented matrix
                    if k < size * 2 {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    let mut inverse = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            inverse[i][j] = augmented[i][j + size]; // The second half of the augmented matrix is the inverse
        }
    }

    Some(inverse)
}

pub fn verify_lu(lower: &Vec<Vec<f64>>, upper: &Vec<Vec<f64>>, original: &Vec<Vec<f64>>) -> bool {
    let size = original.len();
    let mut reconstructed = vec![vec![0.0; size]; size];

    for i in 0..size {
        for j in 0..size {
            reconstructed[i][j] = (0..size).fold(0.0, |sum, k| sum + lower[i][k] * upper[k][j]);
        }
    }

    // Compare reconstructed matrix with the original
    for i in 0..size {
        for j in 0..size {
            if (reconstructed[i][j] - original[i][j]).abs() > 1e-9 {
                return false; // Mismatch
            }
        }
    }
    true
}

pub fn generate_random_matrix(size: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![vec![0.0; size]; size];

    for i in 0..size {
        for j in 0..size {
            matrix[i][j] = rng.gen_range(-10.0..10.0); // Random values between -10 and 10
        }
    }
    
    matrix
}

