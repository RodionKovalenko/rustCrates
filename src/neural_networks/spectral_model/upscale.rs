//! Resolution changes in the frequency domain, and the 1920x1080 composite.
//!
//! Upsampling an image is *exactly* zero-padding its spectrum: a real `n x n`
//! image's coefficients sit in the low-frequency corner of a larger grid and
//! everything outside is zero. Transforming back at the larger size gives the
//! unique band-limited interpolant through the original samples — not an
//! approximation of it, the thing itself. Sampling the result every `m/n`
//! pixels returns the input bit for bit, which is what
//! [`crate::neural_networks::spectral_model::tests`] checks.
//!
//! That property is why the upscaler only *learns* the first factor of four.
//! Above roughly 128x128 a 28x28 digit has no detail left to recover — its
//! strokes are smooth curves — so a learned model there would be an expensive
//! way to reproduce what this module computes exactly.
//!
//! What zero-padding cannot do is invent detail. It produces a smooth,
//! correctly band-limited enlargement of whatever it is given, including the
//! artefacts. Repair is the learned stage's job.

use crate::neural_networks::spectral_model::fft::Rfft2Plan;
use crate::neural_networks::utils::dtype::{Real, C};

/// Ideal band-limited upsample of a square real image, `n x n -> m x m`.
///
/// Both sides must be powers of two with `m >= n`.
pub fn spectral_upsample(img: &[Real], n: usize, m: usize) -> Vec<Real> {
    assert!(m >= n, "spectral_upsample only enlarges: {n} -> {m}");
    assert_eq!(img.len(), n * n);
    if m == n {
        return img.to_vec();
    }

    let src = Rfft2Plan::new(n);
    let dst = Rfft2Plan::new(m);
    let x = src.rfft2(img);

    // irfft2 carries a 1/side^2, so preserving pixel amplitude needs (m/n)^2.
    let gain = (m as Real / n as Real).powi(2);
    let mut out = vec![C::new(0.0, 0.0); m * dst.nh];

    for k1 in 0..n {
        // Rows above n/2 hold negative vertical frequencies and must land at
        // the *negative* end of the larger grid, not be copied positionally.
        let k1s = if k1 <= n / 2 { k1 as i32 } else { k1 as i32 - n as i32 };
        for k2 in 0..src.nh {
            let mut v = x[k1 * src.nh + k2] * gain;

            // The input's Nyquist row and column are each their own mirror: one
            // stored coefficient stands for a conjugate pair that becomes two
            // distinct frequencies on the larger grid, so each must be split.
            //
            // Both splits are needed, and the reason differs per axis. The row
            // split places half at +n/2 and half at -n/2 explicitly. The column
            // split places half at +n/2 and leaves the other half to Hermitian
            // symmetry, which reconstructs it at -n/2 on the way out. Dropping
            // the column halving looks defensible — "symmetry already covers
            // it" — but it double-counts, and the enlarged image then fails to
            // reproduce the samples it was built from.
            let row_nyq = k1s.unsigned_abs() as usize == n / 2;
            if row_nyq {
                v = v * 0.5;
            }
            if k2 == n / 2 {
                v = v * 0.5;
            }

            let mut place = |r: i32| {
                let rr = r.rem_euclid(m as i32) as usize;
                out[rr * dst.nh + k2] += v;
            };
            place(k1s);
            if row_nyq {
                place(-k1s);
            }
        }
    }

    dst.irfft2(&out)
}

/// Ideal band-limited downsample, `n x n -> m x m` with `m <= n`.
///
/// Truncating the spectrum is the adjoint of zero-padding it, and unlike
/// pixel averaging it cannot alias: frequencies above the new Nyquist are
/// removed rather than folded back on top of the ones that remain.
pub fn spectral_downsample(img: &[Real], n: usize, m: usize) -> Vec<Real> {
    assert!(m <= n, "spectral_downsample only shrinks: {n} -> {m}");
    assert_eq!(img.len(), n * n);
    if m == n {
        return img.to_vec();
    }

    let src = Rfft2Plan::new(n);
    let dst = Rfft2Plan::new(m);
    let x = src.rfft2(img);

    let gain = (m as Real / n as Real).powi(2);
    let mut out = vec![C::new(0.0, 0.0); m * dst.nh];

    // Accumulate rather than assign, and iterate over the *source* rows. Both
    // +m/2 and -m/2 fold onto the output's Nyquist row under `rem_euclid`, so
    // summing recombines exactly the halves that `spectral_upsample` split —
    // which is what makes the two operations inverse on band-limited content.
    for j1 in 0..n {
        let j1s = if j1 <= n / 2 { j1 as i32 } else { j1 as i32 - n as i32 };
        if j1s.unsigned_abs() as usize > m / 2 {
            continue;
        }
        let row = j1s.rem_euclid(m as i32) as usize;
        for k2 in 0..dst.nh.min(src.nh) {
            // The new Nyquist column carries only the half that
            // `spectral_upsample` placed explicitly; the other half lives in
            // the Hermitian mirror, which truncation discards. Doubling
            // restores it and makes the two operations exact inverses.
            let restore = if k2 == m / 2 { 2.0 as Real } else { 1.0 as Real };
            out[row * dst.nh + k2] += x[j1 * src.nh + k2] * gain * restore;
        }
    }

    dst.irfft2(&out)
}

/// Full-HD canvas dimensions.
pub const HD_W: usize = 1920;
pub const HD_H: usize = 1080;

/// Place a square image centred on a 1920x1080 canvas.
///
/// The digit stays square. 1920x1080 is 16:9 and a digit is 1:1, so filling
/// the frame would mean stretching it out of proportion; the surrounding area
/// is background, and saying so is more honest than distorting the subject to
/// hide it.
pub fn centre_on_hd(square: &[Real], side: usize) -> Vec<Real> {
    assert_eq!(square.len(), side * side);
    let mut canvas = vec![0.0 as Real; HD_W * HD_H];
    let x0 = (HD_W - side.min(HD_W)) / 2;
    let y0 = (HD_H - side.min(HD_H)) / 2;
    let copy = side.min(HD_H);
    for y in 0..copy {
        for x in 0..copy {
            canvas[(y0 + y) * HD_W + (x0 + x)] = square[y * side + x];
        }
    }
    canvas
}

/// Write a real buffer as an 8-bit grayscale PNG.
pub fn write_gray_png(px: &[Real], w: usize, h: usize, path: &str) -> std::io::Result<()> {
    assert_eq!(px.len(), w * h);
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut buf = image::GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let v = (px[y * w + x].clamp(0.0, 1.0) * 255.0).round() as u8;
            buf.put_pixel(x as u32, y as u32, image::Luma([v]));
        }
    }
    buf.save(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
}

/// Nearest-neighbour enlargement — the raw pixels, made visible.
///
/// Not a competitor to [`spectral_upsample`]: it is the *control*. It shows
/// exactly the 32x32 samples the model produced, with no interpolation of any
/// kind, so a side-by-side makes clear which features come from the generator
/// and which the upscaler introduced. (The answer should be: none. The
/// upscaler is provably information-preserving; anything that appears in one
/// panel and not the other is interpolation, not invention.)
pub fn nearest_upsample(img: &[Real], n: usize, factor: usize) -> Vec<Real> {
    assert_eq!(img.len(), n * n);
    let m = n * factor;
    let mut out = vec![0.0 as Real; m * m];
    for y in 0..m {
        for x in 0..m {
            out[y * m + x] = img[(y / factor) * n + (x / factor)];
        }
    }
    out
}

/// Write two equally sized square panels side by side, separated by a divider.
pub fn write_side_by_side(
    left: &[Real],
    right: &[Real],
    side: usize,
    gap: usize,
    path: &str,
) -> std::io::Result<()> {
    assert_eq!(left.len(), side * side);
    assert_eq!(right.len(), side * side);

    let w = side * 2 + gap;
    let mut canvas = vec![0.0 as Real; w * side];
    for y in 0..side {
        for x in 0..side {
            canvas[y * w + x] = left[y * side + x];
            canvas[y * w + side + gap + x] = right[y * side + x];
        }
        // A mid-grey divider, so the boundary is unambiguous against black.
        for g in 0..gap {
            canvas[y * w + side + g] = 0.35;
        }
    }
    write_gray_png(&canvas, w, side, path)
}

/// Default apodisation strength: taper the top 45% of the frequency range.
///
/// Chosen from a measured sweep, not by eye. Overshoot and edge sharpness fall
/// together across the whole range — there is no knee, so no setting removes
/// ringing for free:
///
/// | alpha | overshoot | sharpness |
/// |---|---|---|
/// | 0.0 | 0.326 | 0.167 |
/// | 0.4 | 0.223 | 0.115 |
/// | 0.8 | 0.115 | 0.082 |
/// | 1.0 | 0.022 | 0.066 |
///
/// 0.8 buys a 65% reduction in ringing for a 50% loss of edge sharpness, which
/// is the best practical compromise on this data. The deeper point is that
/// *every linear resampler* sits on this frontier; escaping it requires a
/// nonlinear method, which is the real argument for the learned repair stage.
pub const DEFAULT_APODISATION: Real = 0.8;

/// Band-limited upsample with a raised-cosine taper on the high frequencies.
///
/// [`spectral_upsample`] is *exact*: it reproduces every original sample and
/// adds no information. That turns out to be the wrong objective here, and the
/// reason is worth stating because the tests do not catch it.
///
/// Ideal interpolation is ideal for **band-limited** signals. A digit is not
/// one — it has sharp edges — so a 32x32 sample of it is a *truncated*
/// spectrum, and reconstructing that truncation with a sinc kernel produces
/// **Gibbs ringing**: the interpolant passes exactly through every sample and
/// oscillates between them, which shows up as dark voids inside strokes and
/// bright halos around them. The enlargement is provably faithful and
/// perceptibly worse than the raw pixels.
///
/// Tapering the band edge instead of cutting it off is the standard remedy.
/// `alpha` is the fraction of the frequency range that gets tapered: `0.0`
/// reproduces [`spectral_upsample`] exactly, `1.0` tapers all the way from DC.
/// The trade is explicit — ringing falls, fine detail softens — and it costs
/// the exact-sample-reproduction property, which is the right thing to give up
/// since that property was measuring fidelity to a truncation artefact.
pub fn spectral_upsample_apodized(img: &[Real], n: usize, m: usize, alpha: Real) -> Vec<Real> {
    assert!(m >= n, "spectral_upsample_apodized only enlarges: {n} -> {m}");
    assert_eq!(img.len(), n * n);
    if alpha <= 0.0 {
        return spectral_upsample(img, n, m);
    }

    let src = Rfft2Plan::new(n);
    let dst = Rfft2Plan::new(m);
    let x = src.rfft2(img);

    let gain = (m as Real / n as Real).powi(2);
    let rho_max = (n / 2) as Real;
    let rho_flat = rho_max * (1.0 - alpha.clamp(0.0, 1.0));
    let pi = std::f64::consts::PI as Real;

    let mut out = vec![C::new(0.0, 0.0); m * dst.nh];
    for k1 in 0..n {
        let k1s = if k1 <= n / 2 { k1 as i32 } else { k1 as i32 - n as i32 };
        for k2 in 0..src.nh {
            let rho = (((k1s * k1s) as Real) + ((k2 * k2) as Real)).sqrt();
            let w = if rho <= rho_flat {
                1.0 as Real
            } else if rho >= rho_max {
                0.0 as Real
            } else {
                0.5 * (1.0 + (pi * (rho - rho_flat) / (rho_max - rho_flat)).cos())
            };
            if w == 0.0 {
                continue;
            }

            let mut v = x[k1 * src.nh + k2] * (gain * w);
            let row_nyq = k1s.unsigned_abs() as usize == n / 2;
            if row_nyq {
                v = v * 0.5;
            }
            if k2 == n / 2 {
                v = v * 0.5;
            }

            let mut place = |r: i32| {
                let rr = r.rem_euclid(m as i32) as usize;
                out[rr * dst.nh + k2] += v;
            };
            place(k1s);
            if row_nyq {
                place(-k1s);
            }
        }
    }

    dst.irfft2(&out)
}

/// Write three equally sized square panels side by side.
pub fn write_triptych(
    panels: [&[Real]; 3],
    side: usize,
    gap: usize,
    path: &str,
) -> std::io::Result<()> {
    for p in panels {
        assert_eq!(p.len(), side * side);
    }
    let w = side * 3 + gap * 2;
    let mut canvas = vec![0.0 as Real; w * side];
    for y in 0..side {
        for (i, p) in panels.iter().enumerate() {
            let x0 = i * (side + gap);
            for x in 0..side {
                canvas[y * w + x0 + x] = p[y * side + x];
            }
        }
        for i in 0..2 {
            let x0 = side + i * (side + gap);
            for g in 0..gap {
                canvas[y * w + x0 + g] = 0.35;
            }
        }
    }
    write_gray_png(&canvas, w, side, path)
}
