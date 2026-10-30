use cudarc::cublas::safe::CudaBlas;
use cudarc::cublas::sys::{self as cublas_sys, cublasOperation_t, cublasStatus_t};
use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::sys::CUdeviceptr;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::neural_networks::utils::dtype::{Real, C};
use crate::neural_networks::utils::matrix_rm::MatOp;

/// Device memory the weight cache may use before it is emptied (the RTX 3060 has 6 GB).
const MAX_CACHED_BYTES: usize = 1 << 30;

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

/// Smallest complex GEMM (`m*k*n` multiply-adds) the transformer layers send to the GPU. Below it, copying the
/// operands to the GPU and back costs more than the product on the CPU: at embedding width 16 sending every
/// product to the GPU made training ~1.5x slower than the CPU build. A layer of width 128 (`batch * tokens x 128 x 128`,
/// tens of millions) and the vocabulary projection go to the GPU.
///
/// Override with the environment variable `GPU_MIN_MACS` (0 = every product on the GPU); the test
/// `test_gpu_crossover` prints CPU and GPU times per size to choose it for a machine.
pub const DEFAULT_GPU_MIN_MACS: usize = 1 << 20;

static GPU_MIN_MACS: Lazy<AtomicUsize> =
    Lazy::new(|| AtomicUsize::new(std::env::var("GPU_MIN_MACS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(DEFAULT_GPU_MIN_MACS)));

pub fn gpu_min_macs() -> usize {
    GPU_MIN_MACS.load(Ordering::Relaxed)
}

pub fn set_gpu_min_macs(macs: usize) {
    GPU_MIN_MACS.store(macs, Ordering::Relaxed);
}

/// Whether an `m x k` by `k x n` product is large enough for the GPU (`gpu_min_macs`).
pub fn gpu_worthwhile(m: usize, k: usize, n: usize) -> bool {
    m.saturating_mul(k).saturating_mul(n) >= gpu_min_macs()
}

/// Row-major complex C(m×n) = op_a(A) × op_b(B) on the GPU, with op_a(A) of shape m×k and op_b(B) of shape k×n.
///
/// `cache_b` keeps B on the GPU between calls (for weight matrices, see `GpuMatmul::gemm`).
/// Returns `None` when no GPU is available or the GPU call fails; the caller then multiplies on the CPU.
pub fn try_multiply_complex_gpu(a: &[C], op_a: MatOp, b: &[C], op_b: MatOp, m: usize, k: usize, n: usize, cache_b: bool) -> Option<Vec<C>> {
    // Handle poisoned mutex gracefully
    let mut gpu_guard = match GPU_MATMUL.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("GPU mutex was poisoned, attempting recovery...");
            poisoned.into_inner()
        }
    };

    let gpu = gpu_guard.as_mut()?;
    match gpu.gemm(a, op_a, b, op_b, m, k, n, cache_b) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("GPU complex matmul {}x{} * {}x{} failed: {}. Falling back to CPU.", m, k, k, n, e);
            None
        }
    }
}

/// A weight matrix kept on the GPU, with the host copy it was uploaded from.
struct CachedMatrix {
    host: Vec<C>,
    device: CudaSlice<Real>,
}

/// GPU matmul with thread-safe CUDA stream.
///
/// Device buffers for A, B and C are reused between calls (they only grow), and weight matrices are cached
/// on the device, so a call only copies what changed.
pub struct GpuMatmul {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    buffer_a: Option<CudaSlice<Real>>,
    buffer_b: Option<CudaSlice<Real>>,
    buffer_c: Option<CudaSlice<Real>>,
    /// Keyed by (host address, length) of the matrix data.
    weight_cache: HashMap<(usize, usize), CachedMatrix>,
    cached_bytes: usize,
}

impl GpuMatmul {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let stream = match CUDA.as_ref() {
            Ok(s) => s.clone(),
            Err(e) => return Err(e.clone().into()),
        };

        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("Failed to init cuBLAS: {:?}", e))?;

        Ok(Self { stream, blas, buffer_a: None, buffer_b: None, buffer_c: None, weight_cache: HashMap::new(), cached_bytes: 0 })
    }

    /// Row-major complex C(m×n) = A(m×k) × B(k×n), nothing cached.
    pub fn multiply_complex(&mut self, a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Result<Vec<C>, Box<dyn std::error::Error>> {
        self.gemm(a, MatOp::None, b, MatOp::None, m, k, n, false)
    }

    /// Row-major complex C(m×n) = op_a(A) × op_b(B) with one complex GEMM (Zgemm for f64, Cgemm for f32).
    ///
    /// The complex data is copied as-is (`Complex<Real>` has the layout of cuBLAS' complex type), and the
    /// (conjugate) transposes are done by cuBLAS. With `cache_b` the matrix B stays on the GPU: it is only
    /// uploaded again when its content changed (checked against the host copy of the last upload).
    pub fn gemm(&mut self, a: &[C], op_a: MatOp, b: &[C], op_b: MatOp, m: usize, k: usize, n: usize, cache_b: bool) -> Result<Vec<C>, Box<dyn std::error::Error>> {
        assert_eq!(a.len(), m * k, "A must have m*k elements");
        assert_eq!(b.len(), k * n, "B must have k*n elements");

        let stream = self.stream.clone();
        upload(&stream, &mut self.buffer_a, a)?;
        let key = (b.as_ptr() as usize, b.len());
        if cache_b {
            self.cache(key, b)?;
        } else {
            upload(&stream, &mut self.buffer_b, b)?;
        }
        ensure_len(&stream, &mut self.buffer_c, 2 * m * n)?;

        let a_dev = self.buffer_a.as_ref().expect("A buffer");
        let b_dev = if cache_b { &self.weight_cache[&key].device } else { self.buffer_b.as_ref().expect("B buffer") };
        let c_dev = self.buffer_c.as_mut().expect("C buffer");

        // Row-major trick: cuBLAS is column-major and sees every row-major buffer transposed, so it computes
        // Cᵀ = op_b(B)ᵀ × op_a(A)ᵀ. The ops pass through unchanged; leading dimensions are the stored column counts.
        let (stored_cols_a, stored_cols_b) = (if op_a == MatOp::None { k } else { m }, if op_b == MatOp::None { n } else { k });
        {
            let (a_ptr, _record_a) = a_dev.device_ptr(&stream);
            let (b_ptr, _record_b) = b_dev.device_ptr(&stream);
            let (c_ptr, _record_c) = c_dev.device_ptr_mut(&stream);
            let status = unsafe {
                complex_gemm(*self.blas.handle(), cublas_op(op_b), cublas_op(op_a), n as i32, m as i32, k as i32, b_ptr, stored_cols_b as i32, a_ptr, stored_cols_a as i32, c_ptr, n as i32)
            };
            if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(format!("complex GEMM failed: {:?}", status).into());
            }
        }

        let mut out = vec![C::new(0.0, 0.0); m * n];
        stream.memcpy_dtoh(&c_dev.slice(0..2 * m * n), as_real_mut(&mut out))?;
        stream.synchronize()?;
        Ok(out)
    }

    /// Makes sure the device copy of `b` is current; uploads it only if it is new or its content changed.
    fn cache(&mut self, key: (usize, usize), b: &[C]) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(entry) = self.weight_cache.get(&key) {
            if as_bytes(&entry.host) == as_bytes(b) {
                return Ok(());
            }
        }

        let bytes = 2 * b.len() * std::mem::size_of::<Real>();
        if !self.weight_cache.contains_key(&key) && self.cached_bytes + bytes > MAX_CACHED_BYTES {
            self.weight_cache.clear();
            self.cached_bytes = 0;
        }

        match self.weight_cache.get_mut(&key) {
            Some(entry) => {
                self.stream.memcpy_htod(as_real(b), &mut entry.device)?;
                entry.host.copy_from_slice(b);
            }
            None => {
                let device = self.stream.clone_htod(as_real(b))?;
                self.weight_cache.insert(key, CachedMatrix { host: b.to_vec(), device });
                self.cached_bytes += bytes;
            }
        }
        Ok(())
    }

    /// Number of weight matrices currently kept on the GPU.
    pub fn cached_matrices(&self) -> usize {
        self.weight_cache.len()
    }
}

/// Copies `data` into `buffer`, growing it when it is too small.
fn upload(stream: &Arc<CudaStream>, buffer: &mut Option<CudaSlice<Real>>, data: &[C]) -> Result<(), Box<dyn std::error::Error>> {
    ensure_len(stream, buffer, 2 * data.len())?;
    stream.memcpy_htod(as_real(data), buffer.as_mut().expect("buffer"))?;
    Ok(())
}

/// Makes `buffer` hold at least `len` values (reallocated only when it has to grow).
fn ensure_len(stream: &Arc<CudaStream>, buffer: &mut Option<CudaSlice<Real>>, len: usize) -> Result<(), Box<dyn std::error::Error>> {
    if buffer.as_ref().is_none_or(|b| b.len() < len) {
        *buffer = Some(stream.alloc_zeros::<Real>(len.max(1))?);
    }
    Ok(())
}

fn cublas_op(op: MatOp) -> cublasOperation_t {
    match op {
        MatOp::None => cublasOperation_t::CUBLAS_OP_N,
        MatOp::Transpose => cublasOperation_t::CUBLAS_OP_T,
        MatOp::ConjTranspose => cublasOperation_t::CUBLAS_OP_C,
    }
}

/// `Complex<Real>` is `#[repr(C)] { re, im }`, so a complex slice is a real slice of twice the length.
fn as_real(data: &[C]) -> &[Real] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const Real, 2 * data.len()) }
}

fn as_real_mut(data: &mut [C]) -> &mut [Real] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut Real, 2 * data.len()) }
}

/// Bitwise view, so the cache compares exact bits (NaN-safe, no float comparison).
fn as_bytes(data: &[C]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

#[cfg(feature = "dtype-f64")]
#[allow(clippy::too_many_arguments)]
unsafe fn complex_gemm(
    handle: cublas_sys::cublasHandle_t,
    op_a: cublasOperation_t,
    op_b: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    a: CUdeviceptr,
    lda: i32,
    b: CUdeviceptr,
    ldb: i32,
    c: CUdeviceptr,
    ldc: i32,
) -> cublasStatus_t {
    let alpha = cublas_sys::cuDoubleComplex { x: 1.0, y: 0.0 };
    let beta = cublas_sys::cuDoubleComplex { x: 0.0, y: 0.0 };
    unsafe { cublas_sys::cublasZgemm_v2(handle, op_a, op_b, m, n, k, &alpha, a as *const _, lda, b as *const _, ldb, &beta, c as *mut _, ldc) }
}

#[cfg(not(feature = "dtype-f64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn complex_gemm(
    handle: cublas_sys::cublasHandle_t,
    op_a: cublasOperation_t,
    op_b: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    a: CUdeviceptr,
    lda: i32,
    b: CUdeviceptr,
    ldb: i32,
    c: CUdeviceptr,
    ldc: i32,
) -> cublasStatus_t {
    let alpha = cublas_sys::cuComplex { x: 1.0, y: 0.0 };
    let beta = cublas_sys::cuComplex { x: 0.0, y: 0.0 };
    unsafe { cublas_sys::cublasCgemm_v2(handle, op_a, op_b, m, n, k, &alpha, a as *const _, lda, b as *const _, ldb, &beta, c as *mut _, ldc) }
}
