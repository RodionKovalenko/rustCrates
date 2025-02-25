use num::Complex;
use rand::Rng;
use std::fmt::Debug;

// Define a trait for matrix access (both for Vec<Vec<T>> and arrays [[T; M]; N])
pub trait MatrixAccess<T> {
    fn get_element(&self, i: usize, j: usize) -> Option<&T>;
    fn set_element(&mut self, i: usize, j: usize, value: T);
}

// Implement MatrixAccess for Vec<Vec<T>>
impl<T: Debug + Clone> MatrixAccess<T> for Vec<Vec<T>> {
    fn get_element(&self, i: usize, j: usize) -> Option<&T> {
        self.get(i).and_then(|row| row.get(j))
    }

    fn set_element(&mut self, i: usize, j: usize, value: T) {
        if j >= self[i].len() {
            self[i].push(value);
        } else {
            self[i][j] = value;
        }
    }
}

// Implement MatrixAccess for fixed-size arrays [[T; M]; N]
impl<T: Debug + Clone, const N: usize, const M: usize> MatrixAccess<T> for [[T; M]; N] {
    fn get_element(&self, i: usize, j: usize) -> Option<&T> {
        if i < N && j < M {
            Some(&self[i][j])
        } else {
            None
        }
    }

    fn set_element(&mut self, i: usize, j: usize, value: T) {
        if i < N && j < M {
            self[i][j] = value;
        } else {
            panic!("Index out of bounds for fixed-size array");
        }
    }
}

// Generic function to set weights in both Vec<Vec<T>> and [[T; M]; N]]
pub fn set_weights<M, T>(weight_matrix: &mut M, i: usize, j: usize, value: T)
where
    M: MatrixAccess<T>,
    T: Debug + Clone,
{
    weight_matrix.set_element(i, j, value);
}

// Initialize weights for Vec<Vec<f64>>
pub fn initialize_weights(
    num_layer_inputs_dim2: usize,
    number_hidden_neurons: usize,
    weight_matrix: &mut Vec<Vec<f64>>,
) {
    let mut rng = rand::rng();

    for _i in 0..num_layer_inputs_dim2 {
        weight_matrix.push(Vec::new());
    }

    for i in 0..num_layer_inputs_dim2 {
        for j in 0..number_hidden_neurons {
            let random_value = rng.random_range(-0.6..0.6);
            set_weights(weight_matrix, i, j, random_value);
        }
    }
}

// Initialize weights for fixed-size array with Complex<f64>
pub fn initialize_weights_complex(
    rows: usize, 
    cols: usize,
    weight_matrix: &mut Vec<Vec<Complex<f64>>>,
) {
    let fan_in = rows as f64;
    let fan_out = cols as f64;

    for i in 0..rows {
        for j in 0..cols {
            let random_value = Complex::new(xavier_init(fan_in, fan_out),  xavier_init(fan_in, fan_out));
            set_weights(weight_matrix, i, j, random_value);
        }
    }
}

fn xavier_init(fan_in: f64, fan_out: f64) -> f64 {
    let limit = (6.0 / (fan_in + fan_out)).sqrt();
    rand::rng().random_range(-limit..limit)
}