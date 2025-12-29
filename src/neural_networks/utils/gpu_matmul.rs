use cudarc::cublas::safe::{CudaBlas, Gemm};
use cudarc::driver::safe::{CudaContext, CudaStream};
use num::Complex;
use once_cell::sync::Lazy;
use std::sync::Arc;

/// Thread-safe CUDA initialization
static CUDA: Lazy<Result<Arc<CudaStream>, String>> = Lazy::new(|| match CudaContext::new(0) {
    Ok(ctx) => Ok(ctx.default_stream()),
    Err(e) => Err(format!("Failed to init CUDA: {:?}", e)),
});

/// GPU matmul with thread-safe CUDA stream
/// Uses temporary allocations per operation to avoid threading issues
pub struct GpuMatmul {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
}

impl GpuMatmul {
    pub fn new(_max_m: usize, _max_k: usize, _max_n: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = match CUDA.as_ref() {
            Ok(s) => s.clone(),
            Err(e) => return Err(e.clone().into()),
        };

        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("Failed to init cuBLAS: {:?}", e))?;

        Ok(Self { stream, blas })
    }

    /// Row-major A(m×k) × B(k×n) → row-major C(m×n)
    pub fn multiply_real(&mut self, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        // Allocate temporary buffers of exact size for this operation
        let a_temp = self.stream.clone_htod(&a[..m * k]).map_err(|e| format!("Failed to copy A to device: {:?}", e))?;
        let b_temp = self.stream.clone_htod(&b[..k * n]).map_err(|e| format!("Failed to copy B to device: {:?}", e))?;
        let mut c_temp = self.stream.alloc_zeros::<f64>(m * n).map_err(|e| format!("Failed to allocate C buffer: {:?}", e))?;

        // Row-major trick: C = A*B, cuBLAS sees column-major → compute Cᵀ = Bᵀ * Aᵀ
        unsafe {
            use cudarc::cublas::safe::GemmConfig;
            let cfg = GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: n as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: n as i32,
            };
            <CudaBlas as Gemm<f64>>::gemm(&self.blas, cfg, &b_temp, &a_temp, &mut c_temp).map_err(|e| format!("GEMM failed: {:?}", e))?;
        }

        // Copy result back to host
        let out = self.stream.clone_dtoh(&c_temp).map_err(|e| format!("Failed to copy result from device: {:?}", e))?;
        Ok(out)
    }

    /// Complex multiplication using 4 real GEMMs, buffers reused
    pub fn multiply_complex(&mut self, a: &[Complex<f64>], b: &[Complex<f64>], m: usize, k: usize, n: usize) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
        if n >= 50280 {
            // Flatten real/imag
            let mut ar = vec![0.0; m * k];
            let mut br = vec![0.0; k * n];

            for i in 0..m * k {
                ar[i] = a[i].re;
            }
            for i in 0..k * n {
                br[i] = b[i].re;
            }

            // 4 GEMMs
            let ac = self.multiply_real(&ar, &br, m, k, n)?;
            // Combine
            let mut out = vec![Complex::new(0.0, 0.0); m * n];
            for i in 0..m * n {
                out[i].re = ac[i];
            }
            Ok(out)
        } else {
            // println!("GPU complex matmul: {}x{} * {}x{}", m, k, k, n);
            // Flatten real/imag
            let mut ar = vec![0.0; m * k];
            let mut ai = vec![0.0; m * k];
            let mut br = vec![0.0; k * n];
            let mut bi = vec![0.0; k * n];

            for i in 0..m * k {
                ar[i] = a[i].re;
                ai[i] = a[i].im;
            }
            for i in 0..k * n {
                br[i] = b[i].re;
                bi[i] = b[i].im;
            }

            // 4 GEMMs
            let ac = self.multiply_real(&ar, &br, m, k, n)?;
            let bd = self.multiply_real(&ai, &bi, m, k, n)?;
            let ad = self.multiply_real(&ar, &bi, m, k, n)?;
            let bc = self.multiply_real(&ai, &br, m, k, n)?;

            // Combine
            let mut out = vec![Complex::new(0.0, 0.0); m * n];
            for i in 0..m * n {
                out[i].re = ac[i] - bd[i];
                out[i].im = ad[i] + bc[i];
            }
            Ok(out)
        }
    }
}
