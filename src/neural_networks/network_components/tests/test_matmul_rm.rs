//! The CPU matrix product of the RM layers against a plain triple loop, for every combination of plain, transposed
//! and conjugate-transposed operands.
#[cfg(test)]
mod test_matmul_rm {
    use rand::{rngs::SmallRng, RngExt, SeedableRng};

    use crate::neural_networks::utils::{
        dtype::{Real, C},
        matrix_rm::{multiply_batch_with_weights_rm, multiply_complex_rm_op, MatOp, RowMajorMatrix},
    };

    #[cfg(feature = "dtype-f64")]
    const TOLERANCE: f64 = 1e-12;
    #[cfg(not(feature = "dtype-f64"))]
    const TOLERANCE: f64 = 1e-4;

    fn random(rows: usize, cols: usize, rng: &mut SmallRng) -> RowMajorMatrix<C> {
        RowMajorMatrix::from_data(rows, cols, (0..rows * cols).map(|_| C::new(rng.random_range(-1.0f64..1.0) as Real, rng.random_range(-1.0f64..1.0) as Real)).collect())
    }

    /// op(matrix) as a new row-major matrix.
    fn apply(op: MatOp, m: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
        match op {
            MatOp::None => m.clone(),
            MatOp::Transpose | MatOp::ConjTranspose => {
                let mut data = Vec::with_capacity(m.data.len());
                for c in 0..m.cols {
                    for r in 0..m.rows {
                        let v = m.data[r * m.cols + c];
                        data.push(if op == MatOp::ConjTranspose { v.conj() } else { v });
                    }
                }
                RowMajorMatrix::from_data(m.cols, m.rows, data)
            }
        }
    }

    /// Reference: a plain triple loop.
    fn naive(a: &RowMajorMatrix<C>, b: &RowMajorMatrix<C>) -> Vec<C> {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        let mut c = vec![C::new(0.0 as Real, 0.0 as Real); m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a.data[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aip * b.data[p * n + j];
                }
            }
        }
        c
    }

    fn assert_close(expected: &[C], actual: &[C], label: &str) {
        assert_eq!(expected.len(), actual.len(), "{label}: length");
        let scale = expected.iter().map(|v| v.norm() as f64).fold(1.0, f64::max);
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(((*e - *a).norm() as f64) <= TOLERANCE * scale, "{label}: element {i}: {e} vs {a}");
        }
    }

    #[test]
    fn test_cpu_matmul_matches_triple_loop_for_all_operand_ops() {
        let mut rng = SmallRng::seed_from_u64(7);
        let ops = [MatOp::None, MatOp::Transpose, MatOp::ConjTranspose];
        for (m, k, n) in [(1, 1, 1), (3, 5, 2), (17, 9, 33), (300, 128, 96)] {
            for op_a in ops {
                for op_b in ops {
                    let a = if op_a == MatOp::None { random(m, k, &mut rng) } else { random(k, m, &mut rng) };
                    let b = if op_b == MatOp::None { random(k, n, &mut rng) } else { random(n, k, &mut rng) };
                    let expected = naive(&apply(op_a, &a), &apply(op_b, &b));
                    let result = multiply_complex_rm_op(&a, op_a, &b, op_b);
                    assert_eq!((result.rows, result.cols), (m, n));
                    assert_close(&expected, &result.data, &format!("{m}x{k}x{n} {op_a:?} {op_b:?}"));
                }
            }
        }
    }

    #[test]
    fn test_cpu_batch_with_weights_matches_triple_loop() {
        let mut rng = SmallRng::seed_from_u64(8);
        let weights = random(64, 48, &mut rng);
        let batch: Vec<RowMajorMatrix<C>> = [5, 1, 40, 12].iter().map(|&rows| random(rows, 64, &mut rng)).collect();
        for (x, out) in batch.iter().zip(multiply_batch_with_weights_rm(&batch, &weights, MatOp::None)) {
            assert_close(&naive(x, &weights), &out.data, "batch x W");
        }
        let deltas: Vec<RowMajorMatrix<C>> = [5, 1, 40, 12].iter().map(|&rows| random(rows, 48, &mut rng)).collect();
        for (d, out) in deltas.iter().zip(multiply_batch_with_weights_rm(&deltas, &weights, MatOp::ConjTranspose)) {
            assert_close(&naive(d, &apply(MatOp::ConjTranspose, &weights)), &out.data, "delta x W^H");
        }
    }
}
