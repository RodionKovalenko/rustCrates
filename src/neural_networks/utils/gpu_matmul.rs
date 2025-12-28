use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::cublas::Gemm;
use once_cell::sync::Lazy;
use num::Complex;
use std::sync::Mutex;

/// Global CUDA device
static CUDA: Lazy<Arc<CudaDevice>> = Lazy::new(|| Arc::new(CudaDevice::new(0).expect("Failed to init CUDA")));

/// GPU matmul with persistent buffers
pub struct GpuMatmul {
    dev: Arc<CudaDevice>,

    // Persistent device buffers
    a_buf: CudaSlice<f64>,
    b_buf: CudaSlice<f64>,
    c_buf: CudaSlice<f64>,

    cap_m: usize,
    cap_k: usize,
    cap_n: usize,
}

impl GpuMatmul {
    pub fn new(max_m: usize, max_k: usize, max_n: usize) -> Self {
        let dev = CUDA.clone();

        let a_buf = dev.alloc_zeros::<f64>(max_m * max_k).unwrap();
        let b_buf = dev.alloc_zeros::<f64>(max_k * max_n).unwrap();
        let c_buf = dev.alloc_zeros::<f64>(max_m * max_n).unwrap();

        Self {
            dev,
            a_buf,
            b_buf,
            c_buf,
            cap_m: max_m,
            cap_k: max_k,
            cap_n: max_n,
        }
    }

    /// Row-major A(m×k) × B(k×n) → row-major C(m×n)
    pub fn multiply_real(&mut self, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        assert!(m <= self.cap_m && k <= self.cap_k && n <= self.cap_n);

        // Copy inputs to persistent buffers (fast, device-local)
        self.dev.htod_copy_into(&a[..m*k], &mut self.a_buf).unwrap();
        self.dev.htod_copy_into(&b[..k*n], &mut self.b_buf).unwrap();

        // Row-major trick: C = A*B, cuBLAS sees column-major → compute Cᵀ = Bᵀ * Aᵀ
        self.dev.gemm(
            Gemm::new(n as i32, m as i32, k as i32)
                .transa(false)
                .transb(false),
            &self.b_buf,
            &self.a_buf,
            &mut self.c_buf,
        ).unwrap();

        // Copy result back to host (only once per call)
        let mut out = vec![0.0f64; m * n];
        self.dev.dtoh_sync_copy_into(&self.c_buf, &mut out).unwrap();
        out
    }

    /// Complex multiplication using 4 real GEMMs, buffers reused
    pub fn multiply_complex(&mut self, a: &[Complex<f64>], b: &[Complex<f64>], m: usize, k: usize, n: usize) -> Vec<Complex<f64>> {
        // Flatten real/imag
        let mut ar = vec![0.0; m * k];
        let mut ai = vec![0.0; m * k];
        let mut br = vec![0.0; k * n];
        let mut bi = vec![0.0; k * n];

        for i in 0..m*k { ar[i] = a[i].re; ai[i] = a[i].im; }
        for i in 0..k*n { br[i] = b[i].re; bi[i] = b[i].im; }

        // 4 GEMMs
        let ac = self.multiply_real(&ar, &br, m, k, n);
        let bd = self.multiply_real(&ai, &bi, m, k, n);
        let ad = self.multiply_real(&ar, &bi, m, k, n);
        let bc = self.multiply_real(&ai, &br, m, k, n);

        // Combine
        let mut out = vec![Complex::new(0.0,0.0); m*n];
        for i in 0..m*n {
            out[i].re = ac[i] - bd[i];
            out[i].im = ad[i] + bc[i];
        }
        out
    }
}

// Example global lazy GPU instance for easy reuse
use std::sync::Arc;
use std::sync::Mutex;

static GPU_MATMUL: Lazy<Mutex<GpuMatmul>> = Lazy::new(|| {
    Mutex::new(GpuMatmul::new(1024, 1024, 1024))
});
