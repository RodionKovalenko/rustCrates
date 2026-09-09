//! # Spectral Flow
//!
//! A generative model that runs the flow entirely in the Fourier domain, on
//! the polar coordinates `(asinh(|c|/eps), arg c)` of the whitened spectrum,
//! with a per-band time schedule so low frequencies are settled before high
//! ones start moving.
//!
//! The full algorithm is written up in `documentation/spectral_transformer.txt`;
//! this module is the implementation, and the section labels in the code
//! (`A3`, `C4b`, ...) refer to that document.
//!
//! Module map:
//!
//! | Module | Blueprint section |
//! |---|---|
//! | [`fft`] | H1 — radix-2 FFT, `rfft2`/`irfft2` and their adjoints |
//! | [`bands`] | A2, B7, B8 — band table, permutation, time schedule |
//! | [`data`] | A1 — MNIST loading and 28->32 padding |
//! | [`spectral`] | A3, B4, B5, B9, B10 — whitening, polar state, targets |
//! | [`gemm`] | G — cache-blocked complex GEMM |
//! | [`params`] | A4, B14 — weights, initialisation, AdamW |
//! | [`model`] | C — forward pass and its adjoint |
//! | [`loss`] | B12 — energy-weighted flow-matching loss |
//! | [`train`] | B — training loop and checkpoints |
//! | [`sample`] | D — integration and PNG output |

pub mod bands;
pub mod bench;
pub mod data;
pub mod fft;
pub mod gemm;
pub mod loss;
pub mod model;
pub mod params;
pub mod sample;
pub mod spectral;
pub mod train;
pub mod upscale;

#[cfg(test)]
pub mod tests;
