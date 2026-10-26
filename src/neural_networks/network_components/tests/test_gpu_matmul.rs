#[cfg(test)]
mod test_gpu_matmul {
    use std::time::Instant;

    use crate::neural_networks::utils::{
        dtype::{r, C},
        gpu_matmul::GpuMatmul,
        matrix::multiply_complex,
        matrix_rm::{conjugate_transpose_rm, multiply_batch_with_weights_rm, multiply_complex_rm, multiply_complex_rm_op, multiply_complex_rm_weights, transpose_rm, MatOp, RowMajorMatrix},
        random_arrays::generate_random_complex_2d,
    };

    /// Relative tolerance against the CPU reference: cuBLAS sums in a different order, which f32 shows more.
    #[cfg(feature = "dtype-f64")]
    const TOLERANCE: f64 = 1e-9;
    #[cfg(not(feature = "dtype-f64"))]
    const TOLERANCE: f64 = 1e-4;

    /// Returns false (and prints why) if the CUDA driver or cuBLAS DLLs cannot be loaded, so the tests skip instead of panicking.
    fn gpu_available() -> bool {
        let present = unsafe { cudarc::driver::sys::is_culib_present() && cudarc::cublas::sys::is_culib_present() };
        if !present {
            println!("CUDA driver / cuBLAS not found (check PATH for nvcuda.dll and cublas64_12.dll) - skipping GPU test");
        }
        present
    }

    fn reference_matmul(a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Vec<C> {
        let mut c = vec![C::new(r(0.0), r(0.0)); m * n];
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

    /// op(matrix) computed explicitly on the CPU.
    fn apply(op: MatOp, matrix: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
        match op {
            MatOp::None => matrix.clone(),
            MatOp::Transpose => transpose_rm(matrix),
            MatOp::ConjTranspose => conjugate_transpose_rm(matrix),
        }
    }

    fn reference_rm(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> Vec<C> {
        reference_matmul(&a.data, &b.data, a.rows, a.cols, b.cols)
    }

    fn assert_close(expected: &[C], actual: &[C]) {
        assert_eq!(expected.len(), actual.len());
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).norm();
            assert!(diff <= r(TOLERANCE) * e.norm().max(r(1.0)), "mismatch at {}: expected {:?}, gpu {:?}", i, e, a);
        }
    }

    fn flat(rows: usize, cols: usize) -> Vec<C> {
        generate_random_complex_2d(rows, cols).into_iter().flatten().collect()
    }

    fn random_rm(rows: usize, cols: usize) -> RowMajorMatrix<C> {
        RowMajorMatrix::from_data(rows, cols, flat(rows, cols))
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

    /// All combinations of None / Transpose / ConjTranspose, done by cuBLAS, against explicit CPU transposes.
    #[test]
    fn test_gpu_matmul_ops_match_cpu() {
        if !gpu_available() {
            return;
        }
        let (m, k, n) = (7, 5, 3);
        let ops = [MatOp::None, MatOp::Transpose, MatOp::ConjTranspose];
        for op_a in ops {
            for op_b in ops {
                // Stored shapes such that op_a(A) is m x k and op_b(B) is k x n.
                let a = if op_a == MatOp::None { random_rm(m, k) } else { random_rm(k, m) };
                let b = if op_b == MatOp::None { random_rm(k, n) } else { random_rm(n, k) };
                let expected = reference_rm(&apply(op_a, &a), &apply(op_b, &b));
                let result = multiply_complex_rm_op(&a, op_a, &b, op_b);
                assert_eq!((result.rows, result.cols), (m, n), "{:?} x {:?}: shape", op_a, op_b);
                assert_close(&expected, &result.data);
            }
        }
    }

    /// Cached weights must be uploaded again when they change (e.g. after the optimizer step).
    #[test]
    fn test_gpu_weight_cache_detects_changed_weights() {
        if !gpu_available() {
            return;
        }
        let x = random_rm(6, 8);
        let mut weights = random_rm(8, 4);

        for step in 0..3 {
            assert_close(&reference_rm(&x, &weights), &multiply_complex_rm_weights(&x, &weights, MatOp::None).data);
            // Same cached weights, used conjugate-transposed (input gradient delta * W^H).
            let delta = random_rm(6, 4);
            assert_close(&reference_rm(&delta, &conjugate_transpose_rm(&weights)), &multiply_complex_rm_weights(&delta, &weights, MatOp::ConjTranspose).data);

            // Simulates an update: change the weights in place (same allocation, so the same cache key).
            for (i, w) in weights.data.iter_mut().enumerate() {
                *w += C::new(r(0.01 * (step + 1) as f64), r(-0.02 * i as f64));
            }
        }
    }

    /// One stacked GEMM for the whole batch (different row counts) must equal the per-sample products.
    #[test]
    fn test_gpu_batch_with_weights_matches_cpu() {
        if !gpu_available() {
            return;
        }
        let weights = random_rm(8, 6);
        let batch: Vec<RowMajorMatrix<C>> = [3, 1, 5, 2].iter().map(|&rows| random_rm(rows, 8)).collect();
        for (i, out) in multiply_batch_with_weights_rm(&batch, &weights, MatOp::None).iter().enumerate() {
            assert_eq!((out.rows, out.cols), (batch[i].rows, 6));
            assert_close(&reference_rm(&batch[i], &weights), &out.data);
        }

        let delta_batch: Vec<RowMajorMatrix<C>> = [3, 1, 5, 2].iter().map(|&rows| random_rm(rows, 6)).collect();
        let weights_h = conjugate_transpose_rm(&weights);
        for (i, out) in multiply_batch_with_weights_rm(&delta_batch, &weights, MatOp::ConjTranspose).iter().enumerate() {
            assert_eq!((out.rows, out.cols), (delta_batch[i].rows, 8));
            assert_close(&reference_rm(&delta_batch[i], &weights_h), &out.data);
        }
    }

    #[test]
    fn test_gpu_matmul_vocab_size_through_multiply_complex_rm() {
        if !gpu_available() {
            return;
        }
        // Output projection to the vocabulary, full complex inputs.
        let (m, k, n) = (16, 64, 50280);
        let a = random_rm(m, k);
        let b = random_rm(k, n);

        let start = Instant::now();
        let expected = reference_rm(&a, &b);
        let cpu_time = start.elapsed();

        // First call includes CUDA / cuBLAS initialization.
        let start = Instant::now();
        let _ = multiply_complex_rm(&a, &b);
        let gpu_first_time = start.elapsed();

        let start = Instant::now();
        let result = multiply_complex_rm(&a, &b);
        let gpu_time = start.elapsed();
        assert_close(&expected, &result.data);

        // As weights: uploaded once, then only A and C travel.
        let start = Instant::now();
        let _ = multiply_complex_rm_weights(&a, &b, MatOp::None);
        let gpu_weights_upload_time = start.elapsed();

        let start = Instant::now();
        let result_cached = multiply_complex_rm_weights(&a, &b, MatOp::None);
        let gpu_weights_cached_time = start.elapsed();
        assert_close(&expected, &result_cached.data);

        println!(
            "[{}x{}] * [{}x{}]: cpu {:?}, gpu first call {:?}, gpu {:?}, gpu weights first upload {:?}, gpu weights cached {:?}",
            m, k, k, n, cpu_time, gpu_first_time, gpu_time, gpu_weights_upload_time, gpu_weights_cached_time
        );
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
