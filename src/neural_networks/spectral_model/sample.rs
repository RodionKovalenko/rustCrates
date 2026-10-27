//! Sampling. Blueprint section D, with two corrections.
//!
//! 1. The blueprint integrates `u += dt * v_u` in *global* time, but the target
//!    velocity is `du/dtau`, and `tau` advances `1/WINDOW` times faster than
//!    `t`. Integrating in per-band `dtau` is the consistent choice and is what
//!    this sampler does.
//! 2. Advancing `tau` by the telescoping difference `tau(t1) - tau(t0)` makes
//!    every band traverse exactly `0 -> 1` regardless of how the step grid
//!    lines up with the window boundaries. Evaluating the network at the
//!    window midpoint also turns plain Euler into a midpoint rule.

use crate::neural_networks::spectral_model::bands::{FlowPath, Schedule, N, NBANDS};
use crate::neural_networks::spectral_model::data::{crop_content, IMG_SIDE};
use crate::neural_networks::spectral_model::model::{GroupInput, SpectralNet};
use crate::neural_networks::spectral_model::params::SpectralParams;
use crate::neural_networks::spectral_model::spectral::{
    integrate_step, sample_prior_n, SpectralPrep,
};
use crate::neural_networks::utils::dtype::Real;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Default integration step count for sampling. 16 steps is a coarse solve;
/// this default is visibly cleaner and is shared by the CLI's sampling
/// commands and training's periodic preview PNGs so the two can't silently
/// diverge (as they did when the preview path hardcoded 16 while the CLI
/// defaulted to 64).
pub const DEFAULT_SAMPLE_STEPS: usize = 64;

/// How generated images are written: grid size, colour channels and the centred content region
/// (MNIST digits fill 28 of 32 pixels; the padding is cropped away).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImageLayout {
    pub size: usize,
    pub channels: usize,
    pub content_side: usize,
}

impl ImageLayout {
    /// 28x28 MNIST digits in a 32x32 grid.
    pub const MNIST: ImageLayout = ImageLayout { size: N, channels: 1, content_side: IMG_SIDE };
}

/// Draw `count` images. Returns `count` images of `channels * n * n` floats in `0..1`-ish
/// (the model is unconstrained, so values are clamped only when written out).
pub fn sample_batch(
    net: &SpectralNet,
    prep: &SpectralPrep,
    params: &SpectralParams,
    schedule: &Schedule,
    path: FlowPath,
    count: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<Real>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ncoef = net.bands.ncoef;
    let mut u = vec![0.0 as Real; count * ncoef];
    let mut phi = vec![0.0 as Real; count * ncoef];
    for i in 0..count {
        let (u0, p0) = sample_prior_n(ncoef, &mut rng);
        u[i * ncoef..(i + 1) * ncoef].copy_from_slice(&u0);
        phi[i * ncoef..(i + 1) * ncoef].copy_from_slice(&p0);
    }

    let end = schedule.end_time();
    let dt = end / n_steps as Real;

    for step in 0..n_steps {
        let t0 = step as Real * dt;
        let t1 = t0 + dt;
        let a = schedule.tau(t0);
        let b = schedule.tau(t1);

        let mut tau_mid = [0.0 as Real; NBANDS];
        let mut dtau = [0.0 as Real; NBANDS];
        let mut active = [false; NBANDS];
        for bd in 0..NBANDS {
            dtau[bd] = b[bd] - a[bd];
            tau_mid[bd] = 0.5 * (a[bd] + b[bd]);
            active[bd] = dtau[bd] > 0.0 && bd <= schedule.max_band;
        }
        if !active.iter().any(|&x| x) {
            continue;
        }

        let input =
            GroupInput { b: count, tau: tau_mid, active, ut: u.clone(), phit: phi.clone() };
        let cache = net.forward(params, &input);

        for i in 0..count {
            for p in 0..ncoef {
                let bd = net.bands.band_of_pos[p] as usize;
                if !active[bd] {
                    continue;
                }
                let k = i * ncoef + p;
                let (nu, nphi) =
                    integrate_step(path, u[k], phi[k], cache.v_u[k], cache.v_phi[k], dtau[bd]);
                u[k] = nu;
                phi[k] = nphi;
            }
        }
    }

    // Bands the model does not represent are dropped rather than left at the
    // prior, which would sprinkle unstructured high-frequency noise over an
    // otherwise clean low-pass reconstruction.
    for i in 0..count {
        for p in 0..ncoef {
            if net.bands.band_of_pos[p] as usize > schedule.max_band {
                u[i * ncoef + p] = 0.0;
                phi[i * ncoef + p] = 0.0;
            }
        }
    }

    (0..count)
        .map(|i| prep.polar_to_image(&u[i * ncoef..(i + 1) * ncoef], &phi[i * ncoef..(i + 1) * ncoef]))
        .collect()
}

/// [`sample_grid_to_png_with`] for MNIST digits.
#[allow(clippy::too_many_arguments)]
pub fn sample_grid_to_png(
    net: &SpectralNet,
    prep: &SpectralPrep,
    params: &SpectralParams,
    schedule: &Schedule,
    flow_path: FlowPath,
    rows: usize,
    cols: usize,
    n_steps: usize,
    seed: u64,
    path: &str,
    scale: usize,
) -> std::io::Result<()> {
    sample_grid_to_png_with(net, prep, params, schedule, flow_path, rows, cols, n_steps, seed, path, scale, ImageLayout::MNIST)
}

/// Sample a `rows x cols` grid and write it as a single PNG (grayscale or RGB, per `layout`).
#[allow(clippy::too_many_arguments)]
pub fn sample_grid_to_png_with(
    net: &SpectralNet,
    prep: &SpectralPrep,
    params: &SpectralParams,
    schedule: &Schedule,
    flow_path: FlowPath,
    rows: usize,
    cols: usize,
    n_steps: usize,
    seed: u64,
    path: &str,
    scale: usize,
    layout: ImageLayout,
) -> std::io::Result<()> {
    let imgs = sample_batch(net, prep, params, schedule, flow_path, rows * cols, n_steps, seed);
    write_grid_png_layout(&imgs, rows, cols, path, scale, layout)
}

/// Smallest side, in pixels, a written grid should have.
///
/// A single 28x28 sample is essentially invisible at 1:1 in any image viewer,
/// which makes inspecting one digit useless without a separate upscale step.
/// Small grids are therefore enlarged by an integer factor on write.
const MIN_OUTPUT_SIDE: usize = 448;

/// Write a set of 32x32 grids as one tiled PNG, cropped back to 28x28.
///
/// Upscaling is nearest-neighbour and integer-factor only: these are 28x28
/// images being inspected for per-pixel noise, and any smoothing filter would
/// hide exactly the artefact the picture is meant to reveal.
pub fn write_grid_png(
    images: &[Vec<Real>],
    rows: usize,
    cols: usize,
    path: &str,
) -> std::io::Result<()> {
    write_grid_png_scaled(images, rows, cols, path, 0)
}

/// As [`write_grid_png`], but with an explicit integer upscale.
///
/// `scale = 0` picks the smallest factor that reaches [`MIN_OUTPUT_SIDE`].
pub fn write_grid_png_scaled(
    images: &[Vec<Real>],
    rows: usize,
    cols: usize,
    path: &str,
    scale: usize,
) -> std::io::Result<()> {
    write_grid_png_layout(images, rows, cols, path, scale, ImageLayout::MNIST)
}

/// Writes `images` (laid out as `layout`) as one tiled PNG: grayscale for one channel, RGB for three.
/// Only the content region of each image is written; `scale = 0` picks the smallest factor that reaches
/// [`MIN_OUTPUT_SIDE`] (nearest neighbour, so per-pixel noise stays visible).
pub fn write_grid_png_layout(
    images: &[Vec<Real>],
    rows: usize,
    cols: usize,
    path: &str,
    scale: usize,
    layout: ImageLayout,
) -> std::io::Result<()> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }

    let side = layout.content_side;
    let (tiles_w, tiles_h) = (cols * side, rows * side);
    let scale = if scale > 0 {
        scale
    } else {
        (MIN_OUTPUT_SIDE.div_ceil(tiles_w.min(tiles_h))).max(1)
    };
    let (w, h) = ((tiles_w * scale) as u32, (tiles_h * scale) as u32);

    let mut buf = image::RgbImage::new(w, h);
    let plane = side * side;
    for (idx, img) in images.iter().enumerate().take(rows * cols) {
        let (r, c) = (idx / cols, idx % cols);
        let tile = crop_content(img, layout.size, layout.channels, side);
        for y in 0..side {
            for x in 0..side {
                let value = |ch: usize| (tile[ch * plane + y * side + x].clamp(0.0, 1.0) * 255.0).round() as u8;
                let rgb = if layout.channels == 1 { [value(0); 3] } else { [value(0), value(1), value(2)] };
                let (px, py) = ((c * side + x) * scale, (r * side + y) * scale);
                for dy in 0..scale {
                    for dx in 0..scale {
                        buf.put_pixel((px + dx) as u32, (py + dy) as u32, image::Rgb(rgb));
                    }
                }
            }
        }
    }
    let result = if layout.channels == 1 { image::DynamicImage::ImageRgb8(buf).to_luma8().save(path) } else { buf.save(path) };
    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
}

/// Write real images band-limited to `max_band`.
///
/// This is the ceiling for a given `Schedule`: it is exactly what a *perfect*
/// model restricted to those bands would produce. Compare generated samples
/// against this, not against the raw dataset — at `max_band = 1` even a flawless
/// model can only make blurry blobs, and mistaking that for a failure sends you
/// hunting a bug that is not there.
pub fn lowpass_reference_png(
    net: &SpectralNet,
    prep: &SpectralPrep,
    images: &[Vec<Real>],
    max_band: usize,
    rows: usize,
    cols: usize,
    path: &str,
) -> std::io::Result<()> {
    let out: Vec<Vec<Real>> = images
        .iter()
        .take(rows * cols)
        .map(|img| {
            let (mut u, mut phi) = prep.image_to_polar(img);
            for p in 0..prep.ncoef() {
                if net.bands.band_of_pos[p] as usize > max_band {
                    u[p] = 0.0;
                    phi[p] = 0.0;
                }
            }
            prep.polar_to_image(&u, &phi)
        })
        .collect();
    write_grid_png(&out, rows, cols, path)
}

/// Per-band mean of the magnitude coordinate `u`, for data, prior and samples.
///
/// After whitening every coefficient has RMS magnitude 1, so the *data* profile
/// is roughly flat and the prior matches it by construction. If the generated
/// profile drifts above the data in the outer bands, the model is manufacturing
/// energy where the data has none and the image will look like texture no
/// matter how good the phases are — a magnitude problem masquerading as a
/// structure problem.
pub fn report_band_profile(
    net: &SpectralNet,
    prep: &SpectralPrep,
    params: &SpectralParams,
    schedule: &Schedule,
    path: FlowPath,
    images: &[Vec<Real>],
    n_samples: usize,
    n_steps: usize,
) {
    let mut rng = SmallRng::seed_from_u64(1234);

    let mean_u = |rows: &[Vec<Real>]| -> [Real; NBANDS] {
        let mut acc = [0.0f64; NBANDS];
        let mut cnt = [0usize; NBANDS];
        for u in rows {
            for p in 0..u.len() {
                let bd = net.bands.band_of_pos[p] as usize;
                acc[bd] += u[p] as f64;
                cnt[bd] += 1;
            }
        }
        let mut out = [0.0 as Real; NBANDS];
        for bd in 0..NBANDS {
            out[bd] = (acc[bd] / cnt[bd].max(1) as f64) as Real;
        }
        out
    };

    let data: Vec<Vec<Real>> =
        images.iter().take(n_samples).map(|img| prep.image_to_polar(img).0).collect();
    let prior: Vec<Vec<Real>> = (0..n_samples).map(|_| sample_prior_n(prep.ncoef(), &mut rng).0).collect();

    // Re-derive `u` from the generated images so the comparison is like for like.
    let gen_imgs = sample_batch(net, prep, params, schedule, path, n_samples, n_steps, 4321);
    let gen: Vec<Vec<Real>> =
        gen_imgs.iter().map(|img| prep.image_to_polar(img).0).collect();

    let (d, p0, g) = (mean_u(&data), mean_u(&prior), mean_u(&gen));
    println!("band |  data u  | prior u  |  gen u   | gen/data");
    for bd in 0..NBANDS {
        println!(
            "  {bd}  |  {:7.4} |  {:7.4} |  {:7.4} |  {:6.3}",
            d[bd],
            p0[bd],
            g[bd],
            g[bd] / d[bd].max(1e-6)
        );
    }
}

/// Render the prior itself: un-whitened noise with the dataset's power
/// spectrum. This is the "before" picture — soft cloudy blobs, not white
/// static — and the baseline any sample must visibly improve on.
pub fn prior_reference_png(
    prep: &SpectralPrep,
    rows: usize,
    cols: usize,
    seed: u64,
    path: &str,
) -> std::io::Result<()> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let out: Vec<Vec<Real>> = (0..rows * cols)
        .map(|_| {
            let (u, phi) = sample_prior_n(prep.ncoef(), &mut rng);
            prep.polar_to_image(&u, &phi)
        })
        .collect();
    write_grid_png(&out, rows, cols, path)
}

/// Sample from a saved checkpoint (any size and colour depth).
pub fn sample_from_checkpoint(
    ckpt_path: &str,
    rows: usize,
    cols: usize,
    n_steps: usize,
    seed: u64,
    out_path: &str,
    scale: usize,
) -> std::io::Result<()> {
    use crate::neural_networks::spectral_model::train::Checkpoint;

    let ckpt = Checkpoint::load(ckpt_path)?;
    let (net, prep) = ckpt.net_and_prep();
    sample_grid_to_png_with(
        &net,
        &prep,
        &ckpt.ema,
        &ckpt.schedule,
        ckpt.path,
        rows,
        cols,
        n_steps,
        seed,
        out_path,
        scale,
        ckpt.layout,
    )
}
