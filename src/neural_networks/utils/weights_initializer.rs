use num::Complex;
use rand::{RngExt, rngs::ThreadRng};
use std::fmt::Debug;

use crate::neural_networks::utils::dtype::{r, C};

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

pub fn initialize_weights_f32(rows: usize, cols: usize, weight_matrix: &mut Vec<Vec<f32>>) {
    let mut rng = rand::rng(); // Use the thread-local RNG
    let fan_in = rows as f64;
    let fan_out = cols as f64;

    for i in 0..rows {
        for j in 0..cols {
            let random_value: f32 = xavier_init_f32(fan_in, fan_out, &mut rng); // Use gen_range for sampling
            set_weights(weight_matrix, i, j, random_value);
        }
    }
}

// Initialize weights for Vec<Vec<f64>>
// Correcting the RNG method usage to rand::thread_rng
pub fn initialize_weights(rows: usize, cols: usize, weight_matrix: &mut Vec<Vec<f64>>) {
    let mut rng = rand::rng(); // Use the thread-local RNG
    let fan_in = rows as f64;
    let fan_out = cols as f64;

    for i in 0..rows {
        for j in 0..cols {
            let random_value = xavier_init(fan_in, fan_out, &mut rng); // Use gen_range for sampling
            set_weights(weight_matrix, i, j, random_value);
        }
    }
}

// Initialize weights for Vec<Vec<Complex<f64>>>
pub fn initialize_weights_complex_only_real(rows: usize, cols: usize, weight_matrix: &mut Vec<Vec<C>>) {
    let fan_in = rows as f64;
    let fan_out = cols as f64;
    let mut rng = rand::rng(); // Use the thread-local RNG

    for i in 0..rows {
        for j in 0..cols {
            let random_value = Complex::new(r(xavier_init(fan_in, fan_out, &mut rng)), r(0.0));
            set_weights(weight_matrix, i, j, random_value);
        }
    }
}

// Initialize weights for Vec<Vec<Complex<f64>>>
pub fn initialize_weights_complex(rows: usize, cols: usize, weight_matrix: &mut Vec<Vec<C>>) {
    let fan_in = rows as f64;
    let fan_out = cols as f64;
    let mut rng = rand::rng(); // Use the thread-local RNG

    for i in 0..rows {
        for j in 0..cols {
            let random_value = Complex::new(r(xavier_init(fan_in, fan_out, &mut rng)), r(xavier_init(fan_in, fan_out, &mut rng)));
            set_weights(weight_matrix, i, j, random_value);
        }
    }
}

// Initialize weights for Vec<Vec<Complex<f64>>>
pub fn initialize_bias(rows: usize, weight_matrix: &mut Vec<C>) {
    let fan_in = rows as f64;
    let mut rng = rand::rng();

    for i in 0..rows {
        let random_value = Complex::new(r(xavier_init(fan_in, fan_in, &mut rng)), r(xavier_init(fan_in, fan_in, &mut rng)));
        weight_matrix[i] = random_value;
    }
}

fn xavier_init(fan_in: f64, fan_out: f64, rng: &mut ThreadRng) -> f64 {
    let limit = (6.0 / (fan_in + fan_out)).sqrt();
    rng.random_range(-limit..limit)
}

fn xavier_init_f32(fan_in: f64, fan_out: f64, rng: &mut ThreadRng) -> f32 {
    let limit: f32 = (6.0 / (fan_in + fan_out)).sqrt() as f32;
    rng.random_range(-limit..limit)
}
