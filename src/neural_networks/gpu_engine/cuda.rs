//! CUDA implementation of the engine operations: cuBLAS GEMMs (f32, or f16 inputs with f32 accumulation on the
//! tensor cores) and the kernels of `kernels.cu`, compiled with NVRTC when the backend is created (no nvcc at build
//! time). Buffers stay on the GPU; only token ids go up and scalars (loss, norms) come down.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::cublas::result::{gemm_ex, gemm_strided_batched_ex};
use cudarc::cublas::safe::CudaBlas;
use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType};
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::backend::{Backend, Gemm};

pub const KERNELS: &str = include_str!("kernels.cu");

const KERNEL_NAMES: &[&str] = &[
    "fill", "copy", "axpy", "scale", "sum_squares", "to_half", "build_block", "fold_block_grad", "add_row_vector", "column_sums", "layer_norm",
    "layer_norm_backward", "rope_to_heads", "rope_from_heads", "softmax_window", "softmax_window_backward", "attention_residual", "real_part_scaled",
    "gated", "gated_backward", "leaky_residual", "leaky_residual_backward", "cross_entropy", "embed", "embed_backward", "adamw",
];

const THREADS: u32 = 256;
const PARTIAL_BLOCKS: u32 = 1024;

/// Compiles the kernels to PTX (also usable without a GPU, e.g. to check the source).
pub fn compile_kernels() -> Result<cudarc::nvrtc::Ptx, String> {
    compile_ptx_with_opts(KERNELS, CompileOptions { use_fast_math: Some(false), ..Default::default() }).map_err(|e| format!("{e:?}"))
}

/// GEMM precision: f32, or f16 inputs (converted on the fly) with f32 accumulation and f32 output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16,
}

struct Scratch {
    floats: CudaSlice<f32>,
    halves_a: CudaSlice<u16>,
    halves_b: CudaSlice<u16>,
    partial: CudaSlice<f32>,
    flag: CudaSlice<i32>,
}

pub struct CudaBackend {
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    functions: HashMap<&'static str, CudaFunction>,
    pub precision: Precision,
    scratch: RefCell<Scratch>,
}

fn grid(n: usize) -> LaunchConfig {
    let blocks = n.div_ceil(THREADS as usize).clamp(1, 65_535) as u32;
    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
}

fn per_row(rows: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
}

fn op(transposed: bool) -> cublasOperation_t {
    if transposed {
        cublasOperation_t::CUBLAS_OP_T
    } else {
        cublasOperation_t::CUBLAS_OP_N
    }
}

impl CudaBackend {
    pub fn new(device: usize, precision: Precision) -> Result<Self, String> {
        // cudarc panics on the first call into a library it cannot load: check the driver and cuBLAS first.
        if !unsafe { cudarc::driver::sys::is_culib_present() && cudarc::cublas::sys::is_culib_present() } {
            return Err("CUDA driver / cuBLAS library not found".into());
        }
        let ctx = CudaContext::new(device).map_err(|e| format!("CUDA init: {e:?}"))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cuBLAS init: {e:?}"))?;
        let ptx = compile_kernels()?;
        let module = ctx.load_module(ptx).map_err(|e| format!("loading the kernels: {e:?}"))?;
        let mut functions = HashMap::new();
        for name in KERNEL_NAMES {
            functions.insert(*name, module.load_function(name).map_err(|e| format!("kernel {name}: {e:?}"))?);
        }
        let scratch = Scratch {
            floats: stream.alloc_zeros(1).map_err(|e| format!("{e:?}"))?,
            halves_a: stream.alloc_zeros(1).map_err(|e| format!("{e:?}"))?,
            halves_b: stream.alloc_zeros(1).map_err(|e| format!("{e:?}"))?,
            partial: stream.alloc_zeros(PARTIAL_BLOCKS as usize).map_err(|e| format!("{e:?}"))?,
            flag: stream.alloc_zeros(1).map_err(|e| format!("{e:?}"))?,
        };
        Ok(Self { stream, blas, functions, precision, scratch: RefCell::new(scratch) })
    }

    pub fn device_name(&self) -> String {
        self.stream.context().name().unwrap_or_else(|_| "?".into())
    }

    fn f(&self, name: &str) -> &CudaFunction {
        &self.functions[name]
    }

    pub fn synchronize(&self) {
        self.stream.synchronize().expect("CUDA synchronize");
    }

    fn ensure<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(&self, buf: &mut CudaSlice<T>, len: usize) {
        if buf.len() < len {
            *buf = self.stream.alloc_zeros(len).expect("CUDA alloc");
        }
    }

    /// Converts `src[off .. off + len]` to f16 into `dst`.
    fn to_half(&self, src: &CudaSlice<f32>, off: usize, len: usize, dst: &mut CudaSlice<u16>) {
        self.ensure(dst, len);
        let (off, n) = (off as i64, len as i64);
        unsafe { self.stream.launch_builder(self.f("to_half")).arg(src).arg(&off).arg(dst).arg(&n).launch(grid(len)) }.expect("to_half");
    }
}

/// First and one-past-last float an operand of a GEMM touches.
fn extent(off: usize, ld: usize, rows: usize, cols: usize, transposed: bool, batch: usize, stride: usize) -> (usize, usize) {
    let (stored_rows, stored_cols) = if transposed { (cols, rows) } else { (rows, cols) };
    (off, off + (batch - 1) * stride + (stored_rows - 1) * ld + stored_cols)
}

impl Backend for CudaBackend {
    type Buf = CudaSlice<f32>;
    type Ids = CudaSlice<u32>;

    fn zeros(&self, len: usize) -> CudaSlice<f32> {
        self.stream.alloc_zeros(len.max(1)).expect("CUDA alloc")
    }
    fn upload(&self, data: &[f32]) -> CudaSlice<f32> {
        self.stream.clone_htod(data).expect("upload")
    }
    fn download(&self, buf: &CudaSlice<f32>) -> Vec<f32> {
        let mut out = vec![0.0f32; buf.len()];
        self.stream.memcpy_dtoh(buf, &mut out).expect("download");
        out
    }
    fn upload_ids(&self, ids: &[u32]) -> CudaSlice<u32> {
        self.stream.clone_htod(ids).expect("upload ids")
    }
    fn len(&self, buf: &CudaSlice<f32>) -> usize {
        buf.len()
    }

    fn fill(&self, buf: &mut CudaSlice<f32>, off: usize, len: usize, value: f32) {
        let (off, n) = (off as i64, len as i64);
        unsafe { self.stream.launch_builder(self.f("fill")).arg(buf).arg(&off).arg(&n).arg(&value).launch(grid(len)) }.expect("fill");
    }
    fn copy(&self, src: &CudaSlice<f32>, src_off: usize, dst: &mut CudaSlice<f32>, dst_off: usize, len: usize) {
        let (so, d_o, n) = (src_off as i64, dst_off as i64, len as i64);
        unsafe { self.stream.launch_builder(self.f("copy")).arg(src).arg(&so).arg(dst).arg(&d_o).arg(&n).launch(grid(len)) }.expect("copy");
    }
    fn axpy(&self, alpha: f32, src: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, len: usize) {
        let n = len as i64;
        unsafe { self.stream.launch_builder(self.f("axpy")).arg(&alpha).arg(src).arg(dst).arg(&n).launch(grid(len)) }.expect("axpy");
    }
    fn scale(&self, buf: &mut CudaSlice<f32>, len: usize, alpha: f32) {
        let n = len as i64;
        unsafe { self.stream.launch_builder(self.f("scale")).arg(buf).arg(&n).arg(&alpha).launch(grid(len)) }.expect("scale");
    }

    fn sum_squares(&self, buf: &CudaSlice<f32>, len: usize) -> f64 {
        self.squares_and_finite(buf, len).0
    }
    fn all_finite(&self, buf: &CudaSlice<f32>, len: usize) -> bool {
        self.squares_and_finite(buf, len).1
    }

    fn gemm(&self, g: &Gemm, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>) {
        // Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T: swap the operands, keep the flags.
        let (alpha, beta) = (g.alpha, g.beta);
        let s = &self.stream;
        let mut scratch = self.scratch.borrow_mut();
        let (a_lo, a_hi) = extent(g.a_off, g.lda, g.m, g.k, g.trans_a, g.batch, g.stride_a);
        let (b_lo, b_hi) = extent(g.b_off, g.ldb, g.k, g.n, g.trans_b, g.batch, g.stride_b);
        assert!(a_hi <= a.len() && b_hi <= b.len(), "gemm operand out of bounds: {g:?}");
        let half = self.precision == Precision::F16;
        if half {
            let Scratch { halves_a, halves_b, .. } = &mut *scratch;
            self.to_half(a, a_lo, a_hi - a_lo, halves_a);
            self.to_half(b, b_lo, b_hi - b_lo, halves_b);
        }
        let (data_type, elem) = if half { (cudaDataType::CUDA_R_16F, 2u64) } else { (cudaDataType::CUDA_R_32F, 4u64) };
        let (a_rec, b_rec);
        let (a_ptr, b_ptr) = if half {
            let (pa, ra) = scratch.halves_a.device_ptr(s);
            let (pb, rb) = scratch.halves_b.device_ptr(s);
            (a_rec, b_rec) = (Some(ra), Some(rb));
            (pa + (g.a_off - a_lo) as u64 * elem, pb + (g.b_off - b_lo) as u64 * elem)
        } else {
            let (pa, ra) = a.device_ptr(s);
            let (pb, rb) = b.device_ptr(s);
            (a_rec, b_rec) = (Some(ra), Some(rb));
            (pa + g.a_off as u64 * elem, pb + g.b_off as u64 * elem)
        };
        let (c_base, c_rec) = c.device_ptr_mut(s);
        let c_ptr = c_base + g.c_off as u64 * 4;
        let handle = *self.blas.handle();
        let (m, n, k) = (g.m as i32, g.n as i32, g.k as i32);
        unsafe {
            if g.batch == 1 {
                gemm_ex(
                    handle,
                    op(g.trans_b),
                    op(g.trans_a),
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const c_void,
                    b_ptr as *const c_void,
                    data_type,
                    g.ldb as i32,
                    a_ptr as *const c_void,
                    data_type,
                    g.lda as i32,
                    &beta as *const f32 as *const c_void,
                    c_ptr as *mut c_void,
                    cudaDataType::CUDA_R_32F,
                    g.ldc as i32,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            } else {
                gemm_strided_batched_ex(
                    handle,
                    op(g.trans_b),
                    op(g.trans_a),
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const c_void,
                    b_ptr as *const c_void,
                    data_type,
                    g.ldb as i32,
                    g.stride_b as i64,
                    a_ptr as *const c_void,
                    data_type,
                    g.lda as i32,
                    g.stride_a as i64,
                    &beta as *const f32 as *const c_void,
                    c_ptr as *mut c_void,
                    cudaDataType::CUDA_R_32F,
                    g.ldc as i32,
                    g.stride_c as i64,
                    g.batch as i32,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
        }
        .unwrap_or_else(|e| panic!("cuBLAS GEMM failed ({e:?}): {g:?}"));
        drop((a_rec, b_rec, c_rec));
    }

    fn build_block(&self, params: &CudaSlice<f32>, re_off: usize, im_off: usize, k: usize, n: usize, out: &mut CudaSlice<f32>) {
        let (r, i, kk, nn) = (re_off as i64, im_off as i64, k as i64, n as i64);
        unsafe { self.stream.launch_builder(self.f("build_block")).arg(params).arg(&r).arg(&i).arg(&kk).arg(&nn).arg(out).launch(grid(4 * k * n)) }.expect("build_block");
    }

    fn fold_block_grad(&self, block_grad: &CudaSlice<f32>, k: usize, n: usize, grads: &mut CudaSlice<f32>, re_off: usize, im_off: usize) {
        let (kk, nn, r, i) = (k as i64, n as i64, re_off as i64, im_off as i64);
        unsafe { self.stream.launch_builder(self.f("fold_block_grad")).arg(block_grad).arg(&kk).arg(&nn).arg(grads).arg(&r).arg(&i).launch(grid(k * n)) }.expect("fold_block_grad");
    }

    fn add_row_vector(&self, y: &mut CudaSlice<f32>, rows: usize, cols: usize, params: &CudaSlice<f32>, off: usize) {
        let (r, c, o) = (rows as i64, cols as i64, off as i64);
        unsafe { self.stream.launch_builder(self.f("add_row_vector")).arg(y).arg(&r).arg(&c).arg(params).arg(&o).launch(grid(rows * cols)) }.expect("add_row_vector");
    }

    fn column_sums(&self, dy: &CudaSlice<f32>, rows: usize, cols: usize, grads: &mut CudaSlice<f32>, off: usize) {
        let (r, c, o) = (rows as i64, cols as i64, off as i64);
        unsafe { self.stream.launch_builder(self.f("column_sums")).arg(dy).arg(&r).arg(&c).arg(grads).arg(&o).launch(grid(cols)) }.expect("column_sums");
    }

    fn layer_norm(&self, x: &CudaSlice<f32>, rows: usize, w: usize, eps: f32, params: &CudaSlice<f32>, gamma_off: usize, beta_off: usize, y: &mut CudaSlice<f32>, stats: &mut CudaSlice<f32>) {
        let (ww, g, b) = (w as i64, gamma_off as i64, beta_off as i64);
        unsafe { self.stream.launch_builder(self.f("layer_norm")).arg(x).arg(&ww).arg(&eps).arg(params).arg(&g).arg(&b).arg(y).arg(stats).launch(per_row(rows)) }.expect("layer_norm");
    }

    fn layer_norm_backward(&self, x: &CudaSlice<f32>, stats: &CudaSlice<f32>, dy: &CudaSlice<f32>, rows: usize, w: usize, params: &CudaSlice<f32>, gamma_off: usize, dx: &mut CudaSlice<f32>, grads: &mut CudaSlice<f32>, dgamma_off: usize, dbeta_off: usize) {
        assert_eq!(dbeta_off, dgamma_off + 2 * w, "gamma and beta must be adjacent");
        let mut scratch = self.scratch.borrow_mut();
        self.ensure(&mut scratch.floats, rows * 4 * w);
        let (ww, g) = (w as i64, gamma_off as i64);
        unsafe { self.stream.launch_builder(self.f("layer_norm_backward")).arg(x).arg(stats).arg(dy).arg(&ww).arg(params).arg(&g).arg(dx).arg(&mut scratch.floats).launch(per_row(rows)) }.expect("layer_norm_backward");
        self.column_sums(&scratch.floats, rows, 4 * w, grads, dgamma_off);
    }

    fn rope_to_heads(&self, src: &CudaSlice<f32>, src_ld: usize, re_off: usize, im_off: usize, rows: usize, seq_len: usize, heads: usize, head_dim: usize, dst: &mut CudaSlice<f32>) {
        let (ld, r, i, n, t, h, d) = (src_ld as i64, re_off as i64, im_off as i64, rows as i64, seq_len as i64, heads as i64, head_dim as i64);
        let base = super::ROPE_BASE as f32;
        unsafe { self.stream.launch_builder(self.f("rope_to_heads")).arg(src).arg(&ld).arg(&r).arg(&i).arg(&n).arg(&t).arg(&h).arg(&d).arg(&base).arg(dst).launch(grid(rows * heads * head_dim)) }.expect("rope_to_heads");
    }

    fn rope_from_heads(&self, src: &CudaSlice<f32>, rows: usize, seq_len: usize, heads: usize, head_dim: usize, dst: &mut CudaSlice<f32>, dst_ld: usize, re_off: usize, im_off: usize) {
        let (n, t, h, d, ld, r, i) = (rows as i64, seq_len as i64, heads as i64, head_dim as i64, dst_ld as i64, re_off as i64, im_off as i64);
        let base = super::ROPE_BASE as f32;
        unsafe { self.stream.launch_builder(self.f("rope_from_heads")).arg(src).arg(&n).arg(&t).arg(&h).arg(&d).arg(&base).arg(dst).arg(&ld).arg(&r).arg(&i).launch(grid(rows * heads * head_dim)) }.expect("rope_from_heads");
    }

    fn softmax_window(&self, s: &mut CudaSlice<f32>, groups: usize, t: usize, window: usize) {
        let (tt, ww) = (t as i64, window as i64);
        unsafe { self.stream.launch_builder(self.f("softmax_window")).arg(s).arg(&tt).arg(&ww).launch(per_row(groups * t)) }.expect("softmax_window");
    }

    fn softmax_window_backward(&self, p: &CudaSlice<f32>, dp: &mut CudaSlice<f32>, groups: usize, t: usize) {
        let tt = t as i64;
        unsafe { self.stream.launch_builder(self.f("softmax_window_backward")).arg(p).arg(dp).arg(&tt).launch(per_row(groups * t)) }.expect("softmax_window_backward");
    }

    fn attention_residual(&self, x: &CudaSlice<f32>, att: &CudaSlice<f32>, rows: usize, w: usize, beta: f32, y: &mut CudaSlice<f32>) {
        let (r, ww) = (rows as i64, w as i64);
        unsafe { self.stream.launch_builder(self.f("attention_residual")).arg(x).arg(att).arg(&r).arg(&ww).arg(&beta).arg(y).launch(grid(rows * 2 * w)) }.expect("attention_residual");
    }

    fn real_part_scaled(&self, dy: &CudaSlice<f32>, rows: usize, w: usize, beta: f32, out: &mut CudaSlice<f32>) {
        let (r, ww) = (rows as i64, w as i64);
        unsafe { self.stream.launch_builder(self.f("real_part_scaled")).arg(dy).arg(&r).arg(&ww).arg(&beta).arg(out).launch(grid(rows * w)) }.expect("real_part_scaled");
    }

    fn gated(&self, h: &CudaSlice<f32>, rows: usize, hidden: usize, g: &mut CudaSlice<f32>) {
        let (r, hh) = (rows as i64, hidden as i64);
        unsafe { self.stream.launch_builder(self.f("gated")).arg(h).arg(&r).arg(&hh).arg(g).launch(grid(rows * hidden)) }.expect("gated");
    }

    fn gated_backward(&self, h: &CudaSlice<f32>, dg: &CudaSlice<f32>, rows: usize, hidden: usize, dh: &mut CudaSlice<f32>) {
        let (r, hh) = (rows as i64, hidden as i64);
        unsafe { self.stream.launch_builder(self.f("gated_backward")).arg(h).arg(dg).arg(&r).arg(&hh).arg(dh).launch(grid(rows * hidden)) }.expect("gated_backward");
    }

    fn leaky_residual(&self, o: &CudaSlice<f32>, x: &CudaSlice<f32>, rows: usize, w: usize, beta: f32, slope: f32, y: &mut CudaSlice<f32>) {
        let (r, ww) = (rows as i64, w as i64);
        unsafe { self.stream.launch_builder(self.f("leaky_residual")).arg(o).arg(x).arg(&r).arg(&ww).arg(&beta).arg(&slope).arg(y).launch(grid(rows * w)) }.expect("leaky_residual");
    }

    fn leaky_residual_backward(&self, o: &CudaSlice<f32>, dy: &CudaSlice<f32>, rows: usize, w: usize, beta: f32, slope: f32, d_o: &mut CudaSlice<f32>) {
        let (r, ww) = (rows as i64, w as i64);
        unsafe { self.stream.launch_builder(self.f("leaky_residual_backward")).arg(o).arg(dy).arg(&r).arg(&ww).arg(&beta).arg(&slope).arg(d_o).launch(grid(rows * w)) }.expect("leaky_residual_backward");
    }

    fn cross_entropy(&self, logits: &mut CudaSlice<f32>, rows: usize, vocab: usize, targets: &CudaSlice<u32>, t_off: usize, scale: f32) -> f64 {
        let mut scratch = self.scratch.borrow_mut();
        self.ensure(&mut scratch.floats, rows);
        let (v, t) = (vocab as i64, t_off as i64);
        unsafe { self.stream.launch_builder(self.f("cross_entropy")).arg(logits).arg(&v).arg(targets).arg(&t).arg(&scale).arg(&mut scratch.floats).launch(per_row(rows)) }.expect("cross_entropy");
        let mut losses = vec![0.0f32; rows];
        self.stream.memcpy_dtoh(&scratch.floats.slice(0..rows), &mut losses).expect("download losses");
        losses.iter().map(|l| *l as f64).sum()
    }

    fn embed(&self, params: &CudaSlice<f32>, re_off: usize, im_off: usize, ids: &CudaSlice<u32>, ids_off: usize, rows: usize, w: usize, out: &mut CudaSlice<f32>) {
        let (r, i, io, n, ww) = (re_off as i64, im_off as i64, ids_off as i64, rows as i64, w as i64);
        unsafe { self.stream.launch_builder(self.f("embed")).arg(params).arg(&r).arg(&i).arg(ids).arg(&io).arg(&n).arg(&ww).arg(out).launch(grid(rows * w)) }.expect("embed");
    }

    fn embed_backward(&self, dx: &CudaSlice<f32>, ids: &CudaSlice<u32>, ids_off: usize, rows: usize, w: usize, grads: &mut CudaSlice<f32>, re_off: usize, im_off: usize) {
        let (io, n, ww, r, i) = (ids_off as i64, rows as i64, w as i64, re_off as i64, im_off as i64);
        unsafe { self.stream.launch_builder(self.f("embed_backward")).arg(dx).arg(ids).arg(&io).arg(&n).arg(&ww).arg(grads).arg(&r).arg(&i).launch(grid(rows * w)) }.expect("embed_backward");
    }

    fn adamw(&self, params: &mut CudaSlice<f32>, grads: &CudaSlice<f32>, m: &mut CudaSlice<f32>, v: &mut CudaSlice<f32>, len: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, grad_scale: f32, step: u64) {
        let (c1, c2) = (1.0 - beta1.powi(step as i32), 1.0 - beta2.powi(step as i32));
        let n = len as i64;
        unsafe {
            self.stream
                .launch_builder(self.f("adamw"))
                .arg(params)
                .arg(grads)
                .arg(m)
                .arg(v)
                .arg(&n)
                .arg(&lr)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&eps)
                .arg(&weight_decay)
                .arg(&grad_scale)
                .arg(&c1)
                .arg(&c2)
                .launch(grid(len))
        }
        .expect("adamw");
    }
}

impl CudaBackend {
    /// (sum of squares, all finite) of `buf[..len]`.
    fn squares_and_finite(&self, buf: &CudaSlice<f32>, len: usize) -> (f64, bool) {
        let mut scratch = self.scratch.borrow_mut();
        let Scratch { partial, flag, .. } = &mut *scratch;
        self.stream.memcpy_htod(&[0i32], flag).expect("reset flag");
        let n = len as i64;
        let cfg = LaunchConfig { grid_dim: (PARTIAL_BLOCKS, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 };
        unsafe { self.stream.launch_builder(self.f("sum_squares")).arg(buf).arg(&n).arg(&mut *partial).arg(&mut *flag).launch(cfg) }.expect("sum_squares");
        let mut sums = vec![0.0f32; PARTIAL_BLOCKS as usize];
        self.stream.memcpy_dtoh(&*partial, &mut sums).expect("download partial sums");
        let mut bad = [0i32];
        self.stream.memcpy_dtoh(&*flag, &mut bad).expect("download flag");
        (sums.iter().map(|v| *v as f64).sum(), bad[0] == 0)
    }
}
