use num::Complex;
use rand::Rng;

pub fn generate_random_complex_3d(batch_size: usize, rows: usize, cols: usize) -> Vec<Vec<Vec<Complex<f64>>>> {
    let mut rng = rand::rng();

    (0..batch_size)
        .map(|_| {
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| {
                            let real = rng.random_range(-1.0..1.0); // Random f64 in range [-1.0, 1.0)
                            let imag = rng.random_range(-1.0..1.0);
                            Complex::new(real, imag)
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn generate_random_f64_3d(batch_size: usize, rows: usize, cols: usize) -> Vec<Vec<Vec<f64>>> {
    let mut rng = rand::rng();

    (0..batch_size)
        .map(|_| {
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| {
                            let real = rng.random_range(-1.0..1.0); // Random f64 in range [-1.0, 1.0)
                            real
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn generate_random_complex_2d(rows: usize, cols: usize) -> Vec<Vec<Complex<f64>>> {
    let mut rng = rand::rng();

    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let real = rng.random_range(-1.0..1.0); // Random f64 in range [-1.0, 1.0)
                    let imag = rng.random_range(-1.0..1.0);
                    Complex::new(real, imag)
                })
                .collect()
        })
        .collect()
}

pub fn generate_random_u32_batch(batch_size: usize, output_dim: usize, max_value: u32) -> Vec<Vec<u32>> {
    let mut rng = rand::rng();

    (0..batch_size)
        .map(|_| {
            (0..output_dim)
                .map(|_| rng.random_range(2..max_value)) // random u32 in 0..max_value
                .collect()
        })
        .collect()
}

pub fn generate_u32_batch_from_indices(batch_size: usize, output_dim: usize) -> Vec<Vec<u32>> {
    (0..batch_size)
        .map(|_| {
            (0..output_dim)
                .map(|output_index| output_index as u32) // Just using the index of the last dimension
                .collect()
        })
        .collect()
}
