use cudarc::cublas::safe::{CudaBlas, Gemm};
use cudarc::driver::safe::{CudaContext, CudaStream};
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};

use crate::neural_networks::utils::dtype::{Real, C};

/// Thread-safe CUDA initialization
static CUDA: Lazy<Result<Arc<CudaStream>, String>> = Lazy::new(|| {
    // cudarc panics on the first call into a library it cannot load, so check that the driver and
    // cuBLAS libraries exist before touching them; without them every matmul uses the CPU.
    let libraries_present = unsafe { cudarc::driver::sys::is_culib_present() && cudarc::cublas::sys::is_culib_present() };
    if !libraries_present {
        return Err("CUDA driver / cuBLAS library not found".to_string());
    }
    match CudaContext::new(0) {
        Ok(ctx) => Ok(ctx.default_stream()),
        Err(e) => Err(format!("Failed to init CUDA: {:?}", e)),
    }
});

/// Process-wide GPU matmul; `None` when CUDA / cuBLAS cannot be initialized (CPU fallback).
pub static GPU_MATMUL: Lazy<Mutex<Option<GpuMatmul>>> = Lazy::new(|| match GpuMatmul::new() {
    Ok(gpu) => Mutex::new(Some(gpu)),
    Err(e) => {
        eprintln!("Failed to initialize GPU: {}. Will use CPU fallback.", e);
        Mutex::new(None)
    }
});

/// Row-major complex C(m×n) = A(m×k) × B(k×n) on the GPU.
///
/// Returns `None` when no GPU is available or the GPU call fails; the caller then multiplies on the CPU.
/// Used by every complex matmul (Vec and RM) whenever the `cuda` feature is enabled.
pub fn try_multiply_complex_gpu(a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Option<Vec<C>> {
    // Handle poisoned mutex gracefully
    let mut gpu_guard = match GPU_MATMUL.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("GPU mutex was poisoned, attempting recovery...");
            poisoned.into_inner()
        }
    };

    let gpu = gpu_guard.as_mut()?;
    match gpu.multiply_complex(a, b, m, k, n) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("GPU complex matmul {}x{} * {}x{} failed: {}. Falling back to CPU.", m, k, k, n, e);
            None
        }
    }
}

/// GPU matmul with thread-safe CUDA stream
/// Uses temporary allocations per operation to avoid threading issues
pub struct GpuMatmul {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
}

impl GpuMatmul {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let stream = match CUDA.as_ref() {
            Ok(s) => s.clone(),
            Err(e) => return Err(e.clone().into()),
        };

        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("Failed to init cuBLAS: {:?}", e))?;

        Ok(Self { stream, blas })
    }

    /// Row-major A(m×k) × B(k×n) → row-major C(m×n), in the precision of the build (`Real`).
    pub fn multiply_real(&mut self, a: &[Real], b: &[Real], m: usize, k: usize, n: usize) -> Result<Vec<Real>, Box<dyn std::error::Error>> {
        // Allocate temporary buffers of exact size for this operation
        let a_temp = self.stream.clone_htod(&a[..m * k]).map_err(|e| format!("Failed to copy A to device: {:?}", e))?;
        let b_temp = self.stream.clone_htod(&b[..k * n]).map_err(|e| format!("Failed to copy B to device: {:?}", e))?;
        let mut c_temp = self.stream.alloc_zeros::<Real>(m * n).map_err(|e| format!("Failed to allocate C buffer: {:?}", e))?;

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
            <CudaBlas as Gemm<Real>>::gemm(&self.blas, cfg, &b_temp, &a_temp, &mut c_temp).map_err(|e| format!("GEMM failed: {:?}", e))?;
        }

        // Copy result back to host
        let out = self.stream.clone_dtoh(&c_temp).map_err(|e| format!("Failed to copy result from device: {:?}", e))?;
        Ok(out)
    }

    /// Full complex multiplication (real and imaginary parts) with 3 real GEMMs (Gauss trick).
    pub fn multiply_complex(&mut self, a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Result<Vec<C>, Box<dyn std::error::Error>> {
        assert_eq!(a.len(), m * k, "A must be m x k");
        assert_eq!(b.len(), k * n, "B must be k x n");

        // Flatten real/imag
        let ar: Vec<Real> = a.iter().map(|z| z.re).collect();
        let ai: Vec<Real> = a.iter().map(|z| z.im).collect();
        let br: Vec<Real> = b.iter().map(|z| z.re).collect();
        let bi: Vec<Real> = b.iter().map(|z| z.im).collect();

        // Gauss trick: 3 GEMMs instead of 4
        // k1 = ar * br, k2 = ai * bi, k3 = (ar+ai) * (br+bi)
        // real = k1 - k2, imag = k3 - k1 - k2
        let k1 = self.multiply_real(&ar, &br, m, k, n)?;
        let k2 = self.multiply_real(&ai, &bi, m, k, n)?;

        let ar_plus_ai: Vec<Real> = ar.iter().zip(ai.iter()).map(|(r, i)| r + i).collect();
        let br_plus_bi: Vec<Real> = br.iter().zip(bi.iter()).map(|(r, i)| r + i).collect();
        let k3 = self.multiply_real(&ar_plus_ai, &br_plus_bi, m, k, n)?;

        // Combine
        Ok((0..m * n).map(|i| C::new(k1[i] - k2[i], k3[i] - k1[i] - k2[i])).collect())
    }
}
