//! Training loop, checkpointing and the build-order gates from blueprint
//! section H.
//!
//! One optimiser step processes `n_groups` independent time groups of
//! `group_size` images each. Groups are the unit of parallelism: they share no
//! state, so they run on rayon threads and their gradients are summed.

use crate::neural_networks::spectral_model::bands::{FlowPath, Schedule, D, NBANDS, NCOEF};
use crate::neural_networks::spectral_model::loss::{
    calibrate_for_path, flow_loss, PhaseWeights, DEFAULT_PHASE_ALPHA,
};
use crate::neural_networks::spectral_model::model::{Diagnostics, GroupInput, SpectralNet};
use crate::neural_networks::spectral_model::params::{AdamW, NormTracker, SpectralParams, N_STAGES};
use crate::neural_networks::spectral_model::spectral::{
    build_sample, sample_prior, SpectralPrep, WhiteningTable,
};
use crate::neural_networks::utils::dtype::Real;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub steps: usize,
    pub group_size: usize,
    pub n_groups: usize,
    pub lr: Real,
    pub ema_rate: Real,
    pub schedule: Schedule,
    /// How coefficients travel from noise to data; see [`FlowPath`].
    pub path: FlowPath,
    pub adaptive_roundtrip: bool,
    /// Exponent on the whitening scale in the phase energy weight; see
    /// [`crate::neural_networks::spectral_model::loss`].
    pub phase_alpha: Real,
    pub log_every: usize,
    pub ckpt_every: usize,
    pub sample_every: usize,
    pub ckpt_path: String,
    pub sample_dir: String,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 100_000,
            group_size: 16,
            n_groups: 4,
            lr: 2e-4,
            ema_rate: 0.001,
            schedule: Schedule::full(),
            path: FlowPath::default(),
            adaptive_roundtrip: true,
            phase_alpha: DEFAULT_PHASE_ALPHA,
            log_every: 50,
            ckpt_every: 2_000,
            sample_every: 2_000,
            ckpt_path: "STORAGE/spectral_flow/checkpoint.bin".to_string(),
            sample_dir: "STORAGE/spectral_flow/samples".to_string(),
            seed: 0xC0FFEE,
        }
    }
}

impl TrainConfig {
    /// Build-order step 7: bands 0-1, no band schedule, small and fast.
    ///
    /// Judge it against `spectral-lowpass 1`, not against raw MNIST — bands 0-1
    /// already carry 99.95% of the image energy, so the ceiling here is a
    /// near-perfect digit, not a blob. Structure appears somewhere past 20k
    /// steps; 4k is far too early to conclude anything, in either direction.
    pub fn warmup_gate() -> Self {
        Self {
            steps: 40_000,
            schedule: Schedule::warmup(1),
            log_every: 20,
            ckpt_every: 1_000,
            sample_every: 500,
            ckpt_path: "STORAGE/spectral_flow/warmup.bin".to_string(),
            sample_dir: "STORAGE/spectral_flow/warmup_samples".to_string(),
            ..Self::default()
        }
    }

    pub fn batch_size(&self) -> usize {
        self.group_size * self.n_groups
    }
}

/// Everything needed to resume training or to sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub params: SpectralParams,
    /// Exponential moving average of `params`; this is what sampling uses.
    pub ema: SpectralParams,
    pub opt: AdamW,
    pub whitening: WhiteningTable,
    pub schedule: Schedule,
    pub path: FlowPath,
    pub adaptive_roundtrip: bool,
    pub lambda_phi: Real,
    pub phase_alpha: Real,
    pub step: usize,
}

/// Scalar metadata, stored as JSON so the format can gain fields without
/// invalidating existing checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMeta {
    schedule: Schedule,
    #[serde(default)]
    path: FlowPath,
    adaptive_roundtrip: bool,
    lambda_phi: Real,
    #[serde(default = "default_phase_alpha")]
    phase_alpha: Real,
    step: usize,
    /// Per-band channel widths the checkpoint was trained with. Checkpoints
    /// written before this field existed predate the band-3 capacity
    /// widening, so they default to that earlier shape rather than the
    /// live `bands::D` — letting `load` detect the mismatch instead of
    /// silently deserializing arrays at the wrong length.
    #[serde(default = "default_d_bands")]
    d_bands: [usize; NBANDS],
    #[serde(default = "default_n_stages")]
    n_stages: usize,
}

fn default_phase_alpha() -> Real {
    DEFAULT_PHASE_ALPHA
}

fn default_d_bands() -> [usize; NBANDS] {
    [64, 48, 32, 16]
}

fn default_n_stages() -> usize {
    3
}

/// A checkpoint's `d_bands`/`n_stages` are baked into the length of every
/// tensor in its `SpectralParams`. Loading one trained under a different
/// architecture into today's binary would not fail here — bincode happily
/// deserializes a `Vec` at whatever length was serialized — it would fail
/// later, deep inside `model.rs`/`gemm.rs`, as an opaque index-out-of-bounds
/// panic once a forward/backward pass iterates `0..D[bd]` against the wrong
/// live constant. Catching the mismatch here, with the shapes named, turns
/// that into an actionable error instead.
pub(crate) fn check_architecture_shape(
    path: &str,
    d_bands: [usize; NBANDS],
    n_stages: usize,
) -> std::io::Result<()> {
    if d_bands != D || n_stages != N_STAGES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "checkpoint {path:?} was trained with D={d_bands:?}, N_STAGES={n_stages}, \
                 but this binary is built with D={D:?}, N_STAGES={N_STAGES} — this checkpoint \
                 is incompatible after the capacity change and cannot be resumed or sampled \
                 from; retrain from scratch."
            ),
        ));
    }
    Ok(())
}

/// The bulk arrays. Positional bincode, but this half does not change shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointBody {
    params: SpectralParams,
    ema: SpectralParams,
    opt: AdamW,
    whitening: WhiteningTable,
}

/// The pre-envelope layout, kept only so older checkpoints still load.
#[derive(Debug, Clone, Deserialize)]
struct CheckpointV1 {
    params: SpectralParams,
    ema: SpectralParams,
    opt: AdamW,
    whitening: WhiteningTable,
    schedule: Schedule,
    adaptive_roundtrip: bool,
    lambda_phi: Real,
    step: usize,
}

const CKPT_MAGIC: &[u8; 8] = b"SPECFLW1";

/// Checkpoint file layout: `magic | meta_len: u32 | meta_json | bincode(body)`.
///
/// The scalars live in JSON because bincode is positional and not
/// self-describing: appending one field to a flat bincode struct silently
/// invalidates every checkpoint ever written, which is exactly how the
/// `phase_alpha` field bricked a 44k-step run. Splitting the small,
/// frequently-extended half into a self-describing format means new fields
/// simply take their `serde(default)`.
impl Checkpoint {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        if let Some(dir) = Path::new(path).parent() {
            std::fs::create_dir_all(dir)?;
        }
        let meta = CheckpointMeta {
            schedule: self.schedule,
            path: self.path,
            adaptive_roundtrip: self.adaptive_roundtrip,
            lambda_phi: self.lambda_phi,
            phase_alpha: self.phase_alpha,
            step: self.step,
            d_bands: D,
            n_stages: N_STAGES,
        };
        let meta_json = serde_json::to_vec(&meta)?;
        let body = CheckpointBody {
            params: self.params.clone(),
            ema: self.ema.clone(),
            opt: self.opt.clone(),
            whitening: self.whitening.clone(),
        };
        let body_bin = bincode::serialize(&body)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let mut out = Vec::with_capacity(12 + meta_json.len() + body_bin.len());
        out.extend_from_slice(CKPT_MAGIC);
        out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&meta_json);
        out.extend_from_slice(&body_bin);

        // Write to a temp file and rename, so an interrupted save cannot leave
        // a truncated checkpoint where a good one used to be.
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, &out)?;
        std::fs::rename(&tmp, path)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        // A bare NotFound here surfaces as "the system cannot find the file
        // specified" with no indication of *which* file, which is a poor
        // experience when the path came from a mistyped command line.
        let bytes = std::fs::read(path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "could not read checkpoint {path:?}: {e}. \
                     Usage: spectral-sample <checkpoint.bin> <out.png> [count] [seed]"
                ),
            )
        })?;

        if bytes.len() > 12 && &bytes[..8] == CKPT_MAGIC {
            let meta_len =
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            let meta_end = 12 + meta_len;
            if meta_end > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "checkpoint metadata length exceeds file size",
                ));
            }
            let meta: CheckpointMeta = serde_json::from_slice(&bytes[12..meta_end])?;
            check_architecture_shape(path, meta.d_bands, meta.n_stages)?;
            let body: CheckpointBody = bincode::deserialize(&bytes[meta_end..])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            return Ok(Self {
                params: body.params,
                ema: body.ema,
                opt: body.opt,
                whitening: body.whitening,
                schedule: meta.schedule,
                path: meta.path,
                adaptive_roundtrip: meta.adaptive_roundtrip,
                lambda_phi: meta.lambda_phi,
                phase_alpha: meta.phase_alpha,
                step: meta.step,
            });
        }

        let old: CheckpointV1 = bincode::deserialize(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unrecognised checkpoint format at {path}: {e}"),
            )
        })?;
        // Pre-envelope checkpoints predate the metadata format entirely, so
        // they always carry the original, pre-widening architecture.
        check_architecture_shape(path, default_d_bands(), default_n_stages())?;
        println!("[spectral] migrating pre-envelope checkpoint {path}");
        Ok(Self {
            params: old.params,
            ema: old.ema,
            opt: old.opt,
            whitening: old.whitening,
            schedule: old.schedule,
            // Pre-envelope checkpoints predate the complex path entirely.
            path: FlowPath::Polar,
            adaptive_roundtrip: old.adaptive_roundtrip,
            lambda_phi: old.lambda_phi,
            phase_alpha: DEFAULT_PHASE_ALPHA,
            step: old.step,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepStats {
    pub loss: Real,
    pub mag_term: Real,
    pub phase_term: Real,
    pub grad_norm: Real,
    pub n_active: usize,
    pub diag: Diagnostics,
}

pub struct Trainer {
    pub net: SpectralNet,
    pub prep: SpectralPrep,
    pub params: SpectralParams,
    pub ema: SpectralParams,
    pub opt: AdamW,
    pub cfg: TrainConfig,
    pub lambda_phi: Real,
    pub phase_weights: PhaseWeights,
    pub norms: NormTracker,
    pub step: usize,
}

impl Trainer {
    /// Fresh model: measures the whitening table on `images`, then initialises
    /// weights. Blueprint A3-A4.
    pub fn new(images: &[Vec<Real>], cfg: TrainConfig) -> Self {
        let net = SpectralNet::new(cfg.adaptive_roundtrip);
        println!("[spectral] measuring whitening statistics over {} images", images.len());
        let prep = SpectralPrep::from_dataset(images);
        let mut rng = SmallRng::seed_from_u64(cfg.seed);
        let params = SpectralParams::init(&net.bands, &mut rng);
        let ema = params.clone();
        let opt = AdamW::new(&params, cfg.lr);
        println!("[spectral] {} real parameters", params.num_scalars());
        let phase_weights = PhaseWeights::new(&prep.s, cfg.phase_alpha);
        Self {
            net,
            prep,
            params,
            ema,
            opt,
            lambda_phi: 1.0,
            phase_weights,
            norms: NormTracker::default(),
            cfg,
            step: 0,
        }
    }

    pub fn from_checkpoint(ckpt: Checkpoint, cfg: TrainConfig) -> Self {
        let net = SpectralNet::new(ckpt.adaptive_roundtrip);
        let prep = SpectralPrep::with_table(ckpt.whitening);
        let opt = ckpt.opt;
        let phase_weights = PhaseWeights::new(&prep.s, cfg.phase_alpha);
        Self {
            net,
            prep,
            params: ckpt.params,
            ema: ckpt.ema,
            opt,
            lambda_phi: ckpt.lambda_phi,
            phase_weights,
            norms: NormTracker::default(),
            cfg,
            step: ckpt.step,
        }
    }

    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            params: self.params.clone(),
            ema: self.ema.clone(),
            opt: self.opt.clone(),
            whitening: self.prep.table(),
            schedule: self.cfg.schedule,
            path: self.cfg.path,
            adaptive_roundtrip: self.cfg.adaptive_roundtrip,
            lambda_phi: self.lambda_phi,
            phase_alpha: self.cfg.phase_alpha,
            step: self.step,
        }
    }

    /// Forward + backward for one time group. Returns its gradients and loss.
    fn group_pass(
        &self,
        images: &[Vec<Real>],
        seed: u64,
        accumulate_grads: bool,
    ) -> (Option<SpectralParams>, Real, Real, Real, usize, Diagnostics) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let b = self.cfg.group_size;
        let sched = &self.cfg.schedule;

        // One `t` per group: the whole group then shares a band schedule, so
        // every tensor stays rectangular and the inactive bands are skipped
        // wholesale rather than masked element by element.
        let t = rng.random_range(0.0f64..sched.end_time() as f64) as Real;

        let mut ut = vec![0.0 as Real; b * NCOEF];
        let mut phit = vec![0.0 as Real; b * NCOEF];
        let mut tgt_u = vec![0.0 as Real; b * NCOEF];
        let mut tgt_phi = vec![0.0 as Real; b * NCOEF];
        let mut mask = vec![false; b * NCOEF];
        let mut tau = [0.0 as Real; NBANDS];
        let mut active = [false; NBANDS];

        for bi in 0..b {
            let idx = rng.random_range(0..images.len());
            let (u1, phi1) = self.prep.image_to_polar(&images[idx]);
            let (u0, phi0) = sample_prior(&mut rng);
            let s = build_sample(&self.net.bands, sched, self.cfg.path, &u0, &phi0, &u1, &phi1, t);
            tau = s.tau;
            active = s.active;
            let lo = bi * NCOEF;
            ut[lo..lo + NCOEF].copy_from_slice(&s.ut);
            phit[lo..lo + NCOEF].copy_from_slice(&s.phit);
            tgt_u[lo..lo + NCOEF].copy_from_slice(&s.v_u);
            tgt_phi[lo..lo + NCOEF].copy_from_slice(&s.v_phi);
            mask[lo..lo + NCOEF].copy_from_slice(&s.mask);
        }

        if !active.iter().any(|&a| a) {
            return (None, 0.0, 0.0, 0.0, 0, Diagnostics::default());
        }

        let input = GroupInput { b, tau, active, ut: ut.clone(), phit: phit.clone() };
        let cache = self.net.forward(&self.params, &input);
        let l = flow_loss(
            &cache.v_u,
            &cache.v_phi,
            &tgt_u,
            &tgt_phi,
            &ut,
            &mask,
            self.lambda_phi,
            &self.phase_weights.scale,
            self.cfg.path,
        );

        let grads = if accumulate_grads {
            let mut g = self.params.zeros_like();
            self.net.backward(&self.params, &cache, &l.d_v_u, &l.d_v_phi, &mut g);
            Some(g)
        } else {
            None
        };

        (grads, l.total, l.mag_term, l.phase_term, l.n_active, cache.diag)
    }

    /// One optimiser step over `n_groups` time groups.
    pub fn train_step(&mut self, images: &[Vec<Real>]) -> StepStats {
        let base = self.cfg.seed ^ ((self.step as u64) << 20);
        let results: Vec<_> = (0..self.cfg.n_groups)
            .into_par_iter()
            .map(|g| self.group_pass(images, base.wrapping_add(g as u64 * 7919), true))
            .collect();

        let mut grads = self.params.zeros_like();
        let (mut loss, mut mag, mut phase, mut n_act, mut used) = (0.0, 0.0, 0.0, 0usize, 0usize);
        let mut diag = Diagnostics::default();
        for (g, l, m, p, na, d) in results {
            let Some(g) = g else { continue };
            add_into(&mut grads, &g);
            diag.accumulate(&d);
            loss += l;
            mag += m;
            phase += p;
            n_act += na;
            used += 1;
        }

        if used == 0 {
            return StepStats {
                loss: 0.0,
                mag_term: 0.0,
                phase_term: 0.0,
                grad_norm: 0.0,
                n_active: 0,
                diag: Diagnostics::default(),
            };
        }
        grads.scale(1.0 / used as Real);

        let grad_norm = self.opt.update(&mut self.params, &mut grads);
        self.norms.record(grad_norm, self.opt.clip_norm);
        self.ema.lerp_from(&self.params, self.cfg.ema_rate);
        self.step += 1;

        StepStats {
            loss: loss / used as Real,
            mag_term: mag / used as Real,
            phase_term: phase / used as Real,
            grad_norm,
            n_active: n_act / used,
            diag,
        }
    }

    /// Measure both loss terms at the current weights and rescale
    /// `lambda_phi` so they start comparable. Blueprint B12.
    pub fn calibrate(&mut self, images: &[Vec<Real>]) {
        let mut mag = 0.0 as Real;
        let mut phase = 0.0 as Real;
        let probes = 8;
        for i in 0..probes {
            let (_, _, m, p, _, _) =
                self.group_pass(images, self.cfg.seed ^ (0xABCD + i as u64), false);
            mag += m;
            phase += p;
        }
        self.lambda_phi =
            calibrate_for_path(self.cfg.path, mag / probes as Real, phase / probes as Real);
        println!("[spectral] lambda_phi calibrated to {:.4}", self.lambda_phi);
    }

    pub fn run(&mut self, images: &[Vec<Real>]) -> std::io::Result<()> {
        let cfg = self.cfg.clone();
        println!(
            "[spectral] training {} steps, batch {} ({} groups x {}), lr {:.1e}",
            cfg.steps,
            cfg.batch_size(),
            cfg.n_groups,
            cfg.group_size,
            cfg.lr
        );

        let start = std::time::Instant::now();
        let mut running = 0.0 as Real;
        let mut counted = 0usize;

        for local in 0..cfg.steps {
            let stats = self.train_step(images);
            running += stats.loss;
            counted += 1;

            if self.params.has_non_finite() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("weights went non-finite at step {}", self.step),
                ));
            }

            if cfg.log_every > 0 && (local + 1) % cfg.log_every == 0 {
                let per_step = start.elapsed().as_secs_f64() / (local + 1) as f64;
                let (p50, p90, _p99, clip_frac) = self.norms.percentiles();
                println!(
                    "[spectral] step {:>7} loss {:.5} (mag {:.5} phase {:.5}) |g| {:.2} \
                     p50/p90 {:.1}/{:.1} clip {:.0}% active {:>4} {:.3}s/step",
                    self.step,
                    running / counted as Real,
                    stats.mag_term,
                    stats.phase_term,
                    stats.grad_norm,
                    p50,
                    p90,
                    100.0 * clip_frac,
                    stats.n_active,
                    per_step
                );
                println!(
                    "[spectral]        mixing: stencil {:.4} of matmul, summary {:.4} of residual",
                    stats.diag.stencil_ratio(),
                    stats.diag.summary_ratio()
                );
                running = 0.0;
                counted = 0;
            }

            if cfg.ckpt_every > 0 && (local + 1) % cfg.ckpt_every == 0 {
                self.checkpoint().save(&cfg.ckpt_path)?;
                println!("[spectral] checkpoint written to {}", cfg.ckpt_path);
            }

            if cfg.sample_every > 0 && (local + 1) % cfg.sample_every == 0 {
                let path = format!("{}/step_{:07}.png", cfg.sample_dir, self.step);
                crate::neural_networks::spectral_model::sample::sample_grid_to_png(
                    &self.net,
                    &self.prep,
                    &self.ema,
                    &cfg.schedule,
                    cfg.path,
                    8,
                    8,
                    crate::neural_networks::spectral_model::sample::DEFAULT_SAMPLE_STEPS,
                    self.step as u64,
                    &path,
                    0,
                )?;
                println!("[spectral] samples written to {path}");
            }
        }

        self.checkpoint().save(&cfg.ckpt_path)?;
        Ok(())
    }
}

fn add_into(dst: &mut SpectralParams, src: &SpectralParams) {
    for (a, b) in dst.cplx.iter_mut().zip(src.cplx.iter()) {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x += *y;
        }
    }
    for (a, b) in dst.real.iter_mut().zip(src.real.iter()) {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x += *y;
        }
    }
}

/// Convenience entry point: load MNIST, resume if a checkpoint exists, train.
pub fn train_spectral_flow(cfg: TrainConfig) -> std::io::Result<()> {
    let data = crate::neural_networks::spectral_model::data::load_train()?;
    println!("[spectral] loaded {} MNIST images", data.len());

    let mut trainer = if Path::new(&cfg.ckpt_path).exists() {
        let ckpt = Checkpoint::load(&cfg.ckpt_path)?;
        println!("[spectral] resuming from step {}", ckpt.step);
        Trainer::from_checkpoint(ckpt, cfg)
    } else {
        let mut t = Trainer::new(&data.images, cfg);
        t.calibrate(&data.images);
        t
    };

    trainer.run(&data.images)
}
