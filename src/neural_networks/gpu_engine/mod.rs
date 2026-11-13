//! Own training engine for the complex transformer: hand-written forward and backward passes on top of a small set
//! of operations (`backend::Backend`), implemented on the CPU (`cpu`, reference) and with cuBLAS and own CUDA kernels
//! (`cuda`, feature `gpu-engine`). Same architecture as the candle model (which stays as an independent reference
//! for the gradients in the tests):
//!
//!   token ids -> complex embedding -> complex layer norm
//!     -> layers x (attention: norm, Q/K/V, RoPE, ComplexToLinear, local-window softmax, Re(V); feed-forward: norm,
//!        dense, gate a * split SiLU(b), dense, leaky ReLU; both with scaled residuals)
//!     -> ComplexToLinear -> linear head -> softmax cross-entropy

pub mod backend;
pub mod cpu;
#[cfg(feature = "gpu-engine")]
pub mod cuda;
pub mod model;

#[cfg(test)]
mod tests;

/// RoPE base of the RM model (`INITIAL_BASE * SCALING_FAKTOR`).
pub const ROPE_BASE: f64 = 1250.0;
