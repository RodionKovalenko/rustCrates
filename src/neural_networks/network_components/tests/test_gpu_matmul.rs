#[cfg(test)]
mod test_gpu_matmul {
    use std::time::Instant;

    use crate::neural_networks::utils::{
        dtype::C,
        gpu_matmul::GpuMatmul,
        matrix::multiply_complex,
        matrix_rm::RowMajorMatrix,
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

        let mut gpu = GpuMatmul::new().expect("Failed to initialize GPU");
        let result = gpu.multiply_complex(&a, &b, m, k, n).expect("GPU matmul failed");

        assert_close(&reference_matmul(&a, &b, m, k, n), &result);
    }

    #[test]
    fn test_gpu_matmul_vocab_size_through_multiply_complex_rm() {
        if !gpu_available() {
            return;
        }
        // Output projection to the vocabulary, full complex inputs.
        let (m, k, n) = (16, 64, 50280);
        let a = RowMajorMatrix::from_data(m, k, flat(m, k));
        let b = RowMajorMatrix::from_data(k, n, flat(k, n));

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

    /// Every size goes through the GPU (no size gate); the result must equal the CPU reference.
    #[test]
    fn test_gpu_matmul_small_sizes_through_multiply_complex_rm_and_vec() {
        if !gpu_available() {
            return;
        }
        for (m, k, n) in [(1, 1, 1), (3, 5, 2), (7, 16, 8), (32, 64, 64)] {
            let a = flat(m, k);
            let b = flat(k, n);
            let expected = reference_matmul(&a, &b, m, k, n);

            let result_rm = multiply_complex_rm(&RowMajorMatrix::from_data(m, k, a.clone()), &RowMajorMatrix::from_data(k, n, b.clone()));
            assert_close(&expected, &result_rm.data);

            let to_rows = |v: &[C], cols: usize| v.chunks(cols).map(|row| row.to_vec()).collect::<Vec<Vec<C>>>();
            let result = multiply_complex(&to_rows(&a, k), &to_rows(&b, n));
            assert_close(&expected, &result.into_iter().flatten().collect::<Vec<C>>());
        }
    }
}
