#[cfg(test)]
mod test_gpu_matmul {
    use std::time::Instant;

    use crate::neural_networks::utils::{
        dtype::C,
        gpu_matmul::GpuMatmul,
        matrix::RowMajorMatrix,
        matrix_rm::multiply_complex_rm,
        random_arrays::generate_random_complex_2d,
    };

    const TOLERANCE: f64 = 1e-9;

    /// Returns false (and prints why) if the CUDA driver or cuBLAS DLLs cannot be loaded, so the tests skip instead of panicking.
    fn gpu_available() -> bool {
        let present = unsafe { cudarc::driver::sys::is_culib_present() && cudarc::cublas::sys::is_culib_present() };
        if !present {
            println!("CUDA driver / cuBLAS not found (check PATH for nvcuda.dll and cublas64_12.dll) - skipping GPU test");
        }
        present
    }

    fn reference_matmul(a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Vec<C> {
        let mut c = vec![C::new(0.0, 0.0); m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aip * b[p * n + j];
                }
            }
        }
        c
    }

    fn assert_close(expected: &[C], actual: &[C]) {
        assert_eq!(expected.len(), actual.len());
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).norm();
            assert!(diff <= TOLERANCE * e.norm().max(1.0), "mismatch at {}: expected {:?}, gpu {:?}", i, e, a);
        }
    }

    fn flat(rows: usize, cols: usize) -> Vec<C> {
        generate_random_complex_2d(rows, cols).into_iter().flatten().collect()
    }

    #[test]
    fn test_gpu_matmul_complex_matches_cpu() {
        if !gpu_available() {
            return;
        }
        let (m, k, n) = (5, 32, 1000);
        let a = flat(m, k);
        let b = flat(k, n);

        let mut gpu = GpuMatmul::new(m, k, n).expect("Failed to initialize GPU");
        let result = gpu.multiply_complex(&a, &b, m, k, n).expect("GPU matmul failed");

        assert_close(&reference_matmul(&a, &b, m, k, n), &result);
    }

    #[test]
    fn test_gpu_matmul_vocab_size_through_multiply_complex_rm() {
        if !gpu_available() {
            return;
        }
        // n >= 50280 is the threshold at which multiply_complex_rm switches to the GPU.
        // NOTE: for this size GpuMatmul::multiply_complex currently only multiplies the real parts,
        // so the inputs are kept real here to make the result comparable with the CPU reference.
        let (m, k, n) = (16, 64, 50280);
        let real = |v: Vec<C>| v.into_iter().map(|z| C::new(z.re, 0.0)).collect::<Vec<C>>();
        let a = RowMajorMatrix::from_data(m, k, real(flat(m, k)));
        let b = RowMajorMatrix::from_data(k, n, real(flat(k, n)));

        let start = Instant::now();
        let expected = reference_matmul(&a.data, &b.data, m, k, n);
        let cpu_time = start.elapsed();

        // First call includes CUDA / cuBLAS initialization.
        let start = Instant::now();
        let _ = multiply_complex_rm(&a, &b);
        let gpu_first_time = start.elapsed();

        let start = Instant::now();
        let result = multiply_complex_rm(&a, &b);
        let gpu_time = start.elapsed();

        println!("[{}x{}] * [{}x{}]: cpu {:?}, gpu first call {:?}, gpu {:?}", m, k, k, n, cpu_time, gpu_first_time, gpu_time);
        assert_close(&expected, &result.data);
    }
}
