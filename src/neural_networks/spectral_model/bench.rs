//! Component timing for the forward pass.
//!
//! Exists to answer one question: when a change costs `x%` more time for `x%`
//! more parameters, is the model arithmetic-bound or bandwidth-bound? The
//! answer decides whether future capacity is better spent on **width** (more
//! arithmetic per byte moved, scales well) or on **structure** (more traffic
//! per flop, does not).
//!
//! Run with `cargo run --release -- spectral-bench`.

use crate::neural_networks::spectral_model::bands::{BandTable, D, D_MERGE, NBANDS};
use crate::neural_networks::spectral_model::fft::Rfft2Plan;
use crate::neural_networks::spectral_model::gemm::gemm;
use crate::neural_networks::spectral_model::model::stencil_pass_for_test;
use crate::neural_networks::spectral_model::upscale::{
    spectral_upsample, spectral_upsample_apodized, spectral_upsample_apodized_with_plans,
    spectral_upsample_with_plans, DEFAULT_APODISATION,
};
use crate::neural_networks::utils::dtype::{Real, C};
use num::Complex;
use std::time::Instant;

fn bench<F: FnMut()>(label: &str, iters: usize, macs: f64, bytes: f64, mut f: F) {
    // Warm the caches so the first iteration's misses do not dominate.
    for _ in 0..3 {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let secs = t.elapsed().as_secs_f64() / iters as f64;
    // A complex MAC is 4 real multiplies + 4 real adds.
    let gflops = (macs * 8.0) / secs / 1e9;
    let gbps = bytes / secs / 1e9;
    println!(
        "  {label:<28} {:>9.3} ms   {:>7.2} GFLOP/s   {:>6.2} GB/s   {:>6.1} flop/byte",
        secs * 1000.0,
        gflops,
        gbps,
        (macs * 8.0) / bytes
    );
}

/// Time the three inner kernels at the shapes the model actually uses.
pub fn run_bench() {
    let bands = BandTable::new();
    let b = 16usize; // one time group
    let csz = std::mem::size_of::<C>() as f64;

    println!("[spectral] component benchmark, group size {b}");
    println!(
        "  {:<28} {:>9}      {:>7}       {:>6}       {:>6}",
        "kernel", "time", "compute", "traffic", "intensity"
    );

    for bd in 0..NBANDS {
        let (n, d) = (bands.counts[bd], D[bd]);
        let rows = b * n;
        let off = bands.offsets[bd];

        let h: Vec<C> = (0..rows * d).map(|i| Complex::new(i as Real * 1e-4, 0.5)).collect();
        let r: Vec<C> = (0..d * d).map(|i| Complex::new(0.01, i as Real * 1e-4)).collect();
        let mut out = vec![C::new(0.0, 0.0); rows * d];

        // S-block GEMM: O(rows * d * d) MACs, O(rows * d) traffic.
        let macs = (rows * d * d) as f64;
        let bytes = (2.0 * (rows * d) as f64 + (d * d) as f64) * csz;
        bench(&format!("band {bd} GEMM ({n}x{d})"), 20, macs, bytes, || {
            gemm(&h, &r, &mut out, rows, d, d);
        });

        // Stencil: O(rows * d) MACs but 3 strided reads per output.
        let nbrs = &bands.neighbours[off..off + n];
        let taps: Vec<C> = (0..3 * d).map(|i| Complex::new(0.3, i as Real * 1e-3)).collect();
        let macs = (rows * d * 3) as f64;
        let bytes = (4.0 * (rows * d) as f64) * csz;
        bench(&format!("band {bd} stencil k1"), 20, macs, bytes, || {
            stencil_pass_for_test(&h, &mut out, nbrs, &taps, b, n, d, 0);
        });
    }

    // R-block round trip: the FFT pair, per channel.
    for ng in [8usize, 16, 32] {
        let plan = Rfft2Plan::new(ng);
        let img: Vec<Real> = (0..ng * ng).map(|i| i as Real * 1e-3).collect();
        let spec = plan.rfft2(&img);
        let calls = b * D_MERGE;
        // ~5 n^2 log2(n) real flops for a 2-D transform of side n.
        let flops = 5.0 * (ng * ng) as f64 * (ng as f64).log2() * 2.0 * calls as f64;
        let bytes = (ng * ng) as f64 * 4.0 * 2.0 * calls as f64;
        bench(&format!("R-block FFT pair {ng}x{ng}"), 5, flops / 8.0, bytes, || {
            for _ in 0..calls {
                let s = plan.rfft2(&img);
                std::hint::black_box(&s);
                let x = plan.irfft2(&spec);
                std::hint::black_box(&x);
            }
        });
    }

    println!();
    println!("  Reading this: a kernel whose GFLOP/s is far below the GEMM's, at a");
    println!("  low flop/byte intensity, is bandwidth-bound — extra width there is");
    println!("  nearly free, extra structure is not.");
}

/// Time [`spectral_upsample`]/[`spectral_upsample_apodized`] at the sizes the
/// CLI actually uses (32->512 for `spectral-compare`/`spectral-compare3`,
/// 32->1024 for `spectral-hd`), plus the `_with_plans` variants at the same
/// sizes to isolate how much of the cost is FFT-plan construction versus the
/// transform itself.
///
/// Run with `cargo run --release -- spectral-bench-upscale`.
pub fn run_upscale_bench() {
    println!("[spectral] upscale benchmark");
    println!("  {:<38} {:>9}", "case", "time");

    for &(n, m) in &[(32usize, 512usize), (32, 1024)] {
        let img: Vec<Real> = (0..n * n).map(|i| (i as Real * 0.01).sin()).collect();

        bench(&format!("spectral_upsample {n}->{m}"), 5, 0.0, 1.0, || {
            let out = spectral_upsample(&img, n, m);
            std::hint::black_box(&out);
        });
        bench(&format!("spectral_upsample_apodized {n}->{m}"), 5, 0.0, 1.0, || {
            let out = spectral_upsample_apodized(&img, n, m, DEFAULT_APODISATION);
            std::hint::black_box(&out);
        });

        let src = Rfft2Plan::new(n);
        let dst = Rfft2Plan::new(m);
        bench(&format!("spectral_upsample_with_plans {n}->{m}"), 5, 0.0, 1.0, || {
            let out = spectral_upsample_with_plans(&img, &src, &dst);
            std::hint::black_box(&out);
        });
        bench(&format!("spectral_upsample_apodized_with_plans {n}->{m}"), 5, 0.0, 1.0, || {
            let out = spectral_upsample_apodized_with_plans(&img, &src, &dst, DEFAULT_APODISATION);
            std::hint::black_box(&out);
        });
    }

    println!();
    println!("  Reading this: the `_with_plans` variants exclude FFT-plan");
    println!("  construction, so their gap versus the plain variants is the cost of");
    println!("  rebuilding twiddle/bit-reversal tables on every call — worth caching");
    println!("  when upscaling many images at the same (n, m).");
}
