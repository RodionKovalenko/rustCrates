use nalgebra::{Matrix3, Matrix4};

use crate::neural_networks::utils::matrix::transpose;

pub fn inverseMatrixIdentity(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    matrix
}

fn fill_in_array(m: &mut Vec<Vec<f64>>) {
    for row in m.iter_mut() {
        for elem in row.iter_mut() {
            *elem = f64::INFINITY;
        }
    }
}

fn initialize_identity(a: &mut Vec<Vec<f64>>) {
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            a[i][j] = if i == j { 1.0 } else { 0.0 };
        }
    }
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
    let mut parity = 1.0; // Tracks row swaps for determinant sign adjustment

    for i in 0..size {
        // Partial pivoting: Find the row with the largest pivot and swap
        let mut max_row = i;
        for k in i + 1..size {
            if upper[k][i].abs() > upper[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows in upper if needed and adjust parity
        if i != max_row {
            upper.swap(i, max_row);
            parity *= -1.0; // Row swap changes determinant sign
        }

        // Check for singularity
        if upper[i][i].abs() < 1e-9 {
            return None; // Singular matrix
        }

        // Update the upper and lower matrices
        for j in i + 1..size {
            let factor = upper[j][i] / upper[i][i];
            upper[j][i] = 0.0; // Eliminate the element below the pivot

            for k in i + 1..size {
                upper[j][k] -= factor * upper[i][k];
            }

            lower[j][i] = factor; // Store the factor in the lower matrix
        }

        lower[i][i] = 1.0; // Set the diagonal of L to 1
    }

    Some((lower, upper, parity))
}


pub fn invert_matrix(matrix: &Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let size = matrix.len();

    // Ensure the matrix is square
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
        panic!("Matrix must be square and non-empty");
    }

    // Perform LU decomposition
    let (lower, upper, _) = lu_decomposition(matrix)?;

    // Initialize the inverse matrix with zeros
    let mut inverse = vec![vec![0.0; size]; size];

    // Solve for each column of the identity matrix
    for col in 0..size {
        // Create the identity column vector
        let mut identity_column = vec![0.0; size];
        identity_column[col] = 1.0;

        // Solve LY = b using forward substitution
        let mut y = vec![0.0; size];
        for i in 0..size {
            y[i] = identity_column[i]
                - (0..i).fold(0.0, |sum, j| sum + lower[i][j] * y[j]);
        }

        // Solve UX = Y using back substitution
        let mut x = vec![0.0; size];
        for i in (0..size).rev() {
            x[i] = (y[i]
                - (i + 1..size).fold(0.0, |sum, j| sum + upper[i][j] * x[j]))
                / upper[i][i];
        }

        // Copy the solution vector into the corresponding column of the inverse matrix
        for row in 0..size {
            inverse[row][col] = x[row];
        }
    }

    Some(inverse)
}

fn verify_lu(lower: &Vec<Vec<f64>>, upper: &Vec<Vec<f64>>, original: &Vec<Vec<f64>>) -> bool {
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


pub fn test_inverse_matrix_lib() {
    let matrix = Matrix4::new(
        1.0, 2.0, 3.0, 4.0, 4.0, 1.0, 4.0, 5.0, 1.0, 6.0, 3.0, 6.0, 2.0, 4.0, 7.0, 1.0,
    );

    // Attempt to invert the matrix
    match matrix.try_inverse() {
        Some(inverse) => {
            println!("The inverse is: \n{}", inverse);
        }
        None => {
            println!("Matrix is not invertible!");
        }
    }

    let matrix = Matrix3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);

    // Attempt to invert the matrix
    match matrix.try_inverse() {
        Some(inverse) => {
            println!("The inverse is: \n{}", inverse);
        }
        None => {
            println!("Matrix is not invertible!");
        }
    }

    let matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
    ];
    let determinant = get_determinant(&matrix);

    println!("determinant: {}", determinant);

    let matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4.0, 1.0, 4.0, 5.0],
        vec![1.0, 6.0, 3.0, 6.0],
        vec![2.0, 4.0, 7.0, 1.0],
    ];

    let inverse_matrix = invert_matrix(&matrix).unwrap();

    println!("inverse matrix: {:?}", inverse_matrix);

    let matrix = vec![
        vec![4.0, 7.0, 2.0],
        vec![3.0, 6.0, 1.0],
        vec![2.0, 5.0, 3.0],
    ];

    let determinant = get_determinant(&matrix);
    println!("Determinant: {}", determinant);

    let matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 1.0, 4.0],
        vec![1.0, 6.0, 3.0],
    ];
    let determinant = get_determinant(&matrix);
    println!("determinat 3x3: {}", &determinant);

    let matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4.0, 1.0, 4.0, 5.0],
        vec![1.0, 6.0, 3.0, 6.0],
        vec![2.0, 4.0, 7.0, 1.0],
    ];
    let determinant = get_determinant(&matrix);
    println!("determinat 4x4: {}", &determinant);
}
