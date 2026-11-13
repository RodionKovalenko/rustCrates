//! The engine (hand-written backward pass) against the candle model (autograd) with identical parameters: same loss,
//! same gradient for every parameter. Plus a training smoke test of the CPU backend.

use super::{
    cpu::CpuBackend,
    model::{EngineConfig, Layout, Model},
};

fn tiny() -> EngineConfig {
    EngineConfig { vocab_size: 40, width: 32, layers: 2, heads: 2, ffn_hidden: 24, window: 3, seq_len: 8 }
}

fn windows(config: &EngineConfig, batch: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..batch * (config.seq_len + 1))
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % config.vocab_size as u64) as u32
        })
        .collect()
}

#[test]
fn layout_matches_parameter_count_of_the_default_model() {
    let layout = Layout::new(&EngineConfig::default());
    let (total, layers) = layout.parameter_counts();
    // Same count as the candle model (63.3 M, 24.5 M in the layers).
    assert!((63_000_000..63_600_000).contains(&total), "{total}");
    assert!((24_000_000..25_000_000).contains(&layers), "{layers}");
}

#[test]
fn training_steps_reduce_the_loss() {
    let config = tiny();
    let mut model = Model::new(CpuBackend, config.clone(), 7);
    let batch = windows(&config, 4, 3);
    let first = model.forward_backward(&batch, false, 1.0);
    assert!((first - (config.vocab_size as f64).ln()).abs() < 1.5, "initial loss {first} far from ln(vocab)");
    let (mut m, mut v) = (vec![0.0; model.layout.total], vec![0.0; model.layout.total]);
    use super::backend::Backend;
    for step in 1..=30 {
        model.zero_grads();
        model.forward_backward(&batch, true, 1.0);
        let len = model.layout.total;
        let backend = CpuBackend;
        backend.adamw(&mut model.params, &model.grads, &mut m, &mut v, len, 3e-3, 0.9, 0.95, 1e-8, 0.01, 1.0, step);
        model.refresh_blocks();
    }
    let last = model.forward_backward(&batch, false, 1.0);
    assert!(last < first - 1.0, "loss {first} -> {last}");
}

#[test]
fn pretrain_learns_saves_and_resumes() {
    use super::train::{load_checkpoint, pretrain, EngineTrainSettings};
    let dir = std::env::temp_dir().join(format!("engine_pretrain_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // A learnable stream: token i is followed by token (i + 1) mod vocab.
    let config = tiny();
    let write = |name: &str, n: usize, offset: u16| {
        let bytes: Vec<u8> = (0..n).flat_map(|i| ((i as u16 + offset) % config.vocab_size as u16).to_le_bytes()).collect();
        std::fs::write(dir.join(name), bytes).unwrap();
    };
    write("train_00000.bin", 4000, 0);
    write("val_00000.bin", 600, 3);
    let mut settings = EngineTrainSettings::new(&dir);
    settings.checkpoint = dir.join("model");
    settings.config = config.clone();
    (settings.steps, settings.batch_size, settings.micro_batch, settings.learning_rate) = (60, 4, 2, 3e-3);
    (settings.warmup_steps, settings.eval_every, settings.eval_batches, settings.log_every, settings.max_hours) = (5, 30, 2, 30, None);
    pretrain(CpuBackend, &settings).unwrap();
    let (_, params, adam, state) = load_checkpoint(&settings.checkpoint).unwrap();
    assert_eq!(state.step, 60);
    assert!(adam.is_some_and(|m| m.len() == 2 * params.len()), "AdamW moments saved");
    let best = state.best_val_loss.unwrap();
    assert!(best < (config.vocab_size as f64).ln() - 1.0, "validation loss {best} did not drop");
    settings.resume = true;
    settings.steps = 30;
    pretrain(CpuBackend, &settings).unwrap();
    let (_, _, _, resumed) = load_checkpoint(&settings.checkpoint).unwrap();
    assert!(resumed.best_val_loss.unwrap() <= best);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "candle")]
mod against_candle {
    use super::*;
    use crate::neural_networks::candle_model::model::{Model as CandleModel, ModelConfig};
    use candle_core::{DType, Device, Tensor};

    fn candle_config(c: &EngineConfig) -> ModelConfig {
        ModelConfig { vocab_size: c.vocab_size, width: c.width, layers: c.layers, heads: c.heads, ffn_hidden: c.ffn_hidden, window: c.window, seq_len: c.seq_len }
    }

    /// Values of a candle parameter, read from an engine buffer with the engine layout (`values`) — the mapping
    /// used both for the parameters and for the gradients.
    fn engine_values(name: &str, config: &EngineConfig, layout: &Layout, values: &[f32]) -> Vec<f64> {
        let (w, hd) = (config.width, config.head_dim());
        let get = |seg: &str| {
            let s = layout.segment(seg).unwrap_or_else(|| panic!("no segment {seg}"));
            values[s.offset..s.offset + s.len()].iter().map(|v| *v as f64).collect::<Vec<_>>()
        };
        if let Some(rest) = name.strip_prefix("layer").filter(|r| r.contains(".attention.head")) {
            // layer{l}.attention.head{h}.{w_q|w_k|w_v}.{re|im} or .ctl_{q|k}_{1|2}
            let l: usize = rest[..rest.find('.').unwrap()].parse().unwrap();
            let after = &rest[rest.find(".attention.head").unwrap() + ".attention.head".len()..];
            let h: usize = after[..after.find('.').unwrap()].parse().unwrap();
            let field = &after[after.find('.').unwrap() + 1..];
            if let Some(part) = field.strip_prefix("ctl_") {
                let (which, half) = (&part[..1], &part[2..]);
                let block = get(&format!("layer{l}.attention.ctl_{which}"));
                let start = h * 2 * hd * hd + if half == "1" { 0 } else { hd * hd };
                return block[start..start + hd * hd].to_vec();
            }
            let (proj, comp) = (&field[..3], &field[4..]);
            let column = match proj { "w_q" => 0, "w_k" => w, _ => 2 * w } + h * hd;
            let full = get(&format!("layer{l}.attention.wqkv.{comp}"));
            return (0..w).flat_map(|r| full[r * 3 * w + column..r * 3 * w + column + hd].to_vec()).collect();
        }
        match name {
            "ctl_1" => get("ctl")[..w * w].to_vec(),
            "ctl_2" => get("ctl")[w * w..].to_vec(),
            other => get(other),
        }
    }

    /// Inverse of `engine_values` for the parameters: the engine parameter vector of a candle model.
    fn engine_parameters(model: &CandleModel, config: &EngineConfig, layout: &Layout) -> Vec<f32> {
        let mut params = vec![f32::NAN; layout.total];
        // Every engine float is reached by exactly one candle value: mark the positions through engine_values
        // applied to an index vector.
        let index: Vec<f32> = (0..layout.total).map(|i| i as f32).collect();
        for (name, var) in model.named_vars() {
            let positions = engine_values(&name, config, layout, &index);
            let values = var.as_tensor().flatten_all().unwrap().to_vec1::<f64>().unwrap();
            assert_eq!(positions.len(), values.len(), "{name}");
            for (p, v) in positions.iter().zip(values) {
                params[*p as usize] = v as f32;
            }
        }
        assert!(params.iter().all(|v| !v.is_nan()), "an engine parameter has no candle counterpart");
        params
    }

    #[test]
    fn loss_and_all_gradients_match_candle_autograd() {
        let config = tiny();
        assert!(Layout::new(&config).total < (1 << 24), "f32 indices must be exact");
        let device = Device::Cpu;
        let candle = CandleModel::new(candle_config(&config), DType::F64, &device, 11).unwrap();
        // Non-trivial norms and head bias (the defaults 1 / 0 would hide errors in their gradients).
        let mut s = 99u64;
        let mut random = |n: usize| -> Vec<f64> {
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
                })
                .collect()
        };
        for (name, var) in candle.named_vars() {
            if name.contains("norm.") || name == "head_bias" {
                let n = var.as_tensor().elem_count();
                let base = if name.contains("gamma.re") { 1.0 } else { 0.0 };
                let values: Vec<f64> = random(n).into_iter().map(|v| base + v).collect();
                var.set(&Tensor::from_vec(values, var.as_tensor().shape().clone(), &device).unwrap()).unwrap();
            }
        }
        let layout = Layout::new(&config);
        let params = engine_parameters(&candle, &config, &layout);
        let mut engine = Model::from_parameters(CpuBackend, config.clone(), &params);

        let ids = windows(&config, 3, 5);
        let batch = Tensor::from_vec(ids.clone(), (3, config.seq_len + 1), &device).unwrap();
        let candle_loss = candle.loss(&batch).unwrap();
        let grads = candle_loss.backward().unwrap();
        let candle_loss = candle_loss.to_scalar::<f64>().unwrap();

        engine.zero_grads();
        let engine_loss = engine.forward_backward(&ids, true, 1.0);
        println!("loss: candle {candle_loss:.8}, engine {engine_loss:.8}");
        assert!((candle_loss - engine_loss).abs() < 1e-5 * candle_loss.abs().max(1.0), "loss: candle {candle_loss}, engine {engine_loss}");

        let engine_grads = engine.grads.clone();
        let mut worst = (0.0f64, String::new());
        for (name, var) in candle.named_vars() {
            let expected = grads.get(var.as_tensor()).unwrap_or_else(|| panic!("{name}: no candle gradient")).flatten_all().unwrap().to_vec1::<f64>().unwrap();
            let actual = engine_values(&name, &config, &layout, &engine_grads);
            // Relative to the largest gradient of the tensor (f32 engine against the f64 reference).
            let norm = expected.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-6);
            let diff = expected.iter().zip(&actual).map(|(e, a)| (e - a).abs()).fold(0.0f64, f64::max) / norm;
            if diff > worst.0 {
                worst = (diff, name.clone());
            }
            assert!(diff < 1e-3, "{name}: engine gradient differs from candle by {diff:.2e} (relative to the largest entry {norm:.2e})");
        }
        println!("largest relative gradient difference: {:.2e} ({})", worst.0, worst.1);
    }
}

#[cfg(feature = "gpu-engine")]
mod cuda_backend {
    use super::*;
    use crate::neural_networks::gpu_engine::cuda::{compile_kernels, CudaBackend, Precision};

    /// Needs only the NVRTC library, no GPU.
    #[test]
    fn kernels_compile_with_nvrtc() {
        if let Err(e) = compile_kernels() {
            panic!("NVRTC: {e}");
        }
    }

    fn gpu(precision: Precision) -> Option<CudaBackend> {
        match CudaBackend::new(0, precision) {
            Ok(b) => Some(b),
            Err(e) => {
                println!("no GPU ({e}): skipped");
                None
            }
        }
    }

    /// Loss and every gradient of the CUDA backend against the CPU backend (same parameters, same batch).
    fn compare(precision: Precision, tolerance: f64) {
        let Some(backend) = gpu(precision) else { return };
        println!("GPU: {}", backend.device_name());
        let config = EngineConfig { vocab_size: 300, width: 64, layers: 2, heads: 4, ffn_hidden: 96, window: 5, seq_len: 32 };
        let mut cpu = Model::new(CpuBackend, config.clone(), 3);
        let params = cpu.params.clone();
        let mut cuda = Model::from_parameters(backend, config.clone(), &params);
        let ids = windows(&config, 3, 9);
        cpu.zero_grads();
        cuda.zero_grads();
        let (l_cpu, l_gpu) = (cpu.forward_backward(&ids, true, 1.0), cuda.forward_backward(&ids, true, 1.0));
        println!("{precision:?}: loss cpu {l_cpu:.6}, gpu {l_gpu:.6}");
        assert!((l_cpu - l_gpu).abs() < tolerance * l_cpu.abs(), "loss cpu {l_cpu} gpu {l_gpu}");
        use super::super::backend::Backend;
        let g_gpu = cuda.backend.download(&cuda.grads);
        let mut worst = (0.0f64, String::new());
        for seg in &cpu.layout.segments {
            let (a, b) = (&cpu.grads[seg.offset..seg.offset + seg.len()], &g_gpu[seg.offset..seg.offset + seg.len()]);
            let norm = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6) as f64;
            let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / norm;
            if diff > worst.0 {
                worst = (diff, seg.name.clone());
            }
            assert!(diff < tolerance, "{precision:?} {}: GPU gradient differs by {diff:.2e}", seg.name);
        }
        println!("{precision:?}: largest relative gradient difference {:.2e} ({})", worst.0, worst.1);
    }

    #[test]
    fn gpu_f32_matches_cpu() {
        compare(Precision::F32, 1e-4);
    }

    #[test]
    fn gpu_f16_matches_cpu_to_half_precision() {
        compare(Precision::F16, 2e-2);
    }
}
