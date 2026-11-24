//! The engine (hand-written backward pass) against the candle model (autograd) with identical parameters: same loss,
//! same gradient for every parameter. Plus a training smoke test of the CPU backend.

use super::{
    cpu::CpuBackend,
    model::{EngineConfig, Layout, Model},
};

fn tiny() -> EngineConfig {
    EngineConfig { vocab_size: 40, width: 32, layers: 2, heads: 2, ffn_hidden: 24, window: 3, seq_len: 8, complex_values: false, standard_residual: false, standard_rope: false }
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
fn next_token_logits_give_the_training_loss_and_ignore_later_tokens() {
    let config = tiny();
    let mut model = Model::new(CpuBackend, config.clone(), 11);
    let window = windows(&config, 1, 5);
    let t = config.seq_len;
    let loss = model.forward_backward(&window, false, 1.0);
    // Mean cross-entropy of the next-token logits at every position = the loss of the training forward pass.
    let mut total = 0.0;
    for p in 0..t {
        let logits = model.next_token_logits(&window[..=p]);
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v)) as f64;
        let log_sum = logits.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>().ln() + max;
        total += log_sum - logits[window[p + 1] as usize] as f64;
    }
    assert!((total / t as f64 - loss).abs() < 1e-4, "next-token logits {} vs loss {loss}", total / t as f64);
    // A context longer than seq_len uses its last seq_len tokens.
    let longer: Vec<u32> = [3, 9].iter().copied().chain(window[..t].iter().copied()).collect();
    assert_eq!(model.next_token_logits(&longer), model.next_token_logits(&window[..t]));
}

#[test]
fn sampling_is_greedy_at_temperature_zero_and_respects_top_k() {
    use super::generate::{apply_repetition_penalty, sample, SamplingSettings};
    use rand::{rngs::StdRng, SeedableRng};
    let logits = [0.1, 2.0, 1.9, -1.0, 0.5];
    let mut rng = StdRng::seed_from_u64(1);
    let greedy = SamplingSettings { temperature: 0.0, ..SamplingSettings::default() };
    assert_eq!(sample(&logits, &greedy, &mut rng), 1);
    let top2 = SamplingSettings { temperature: 1.0, top_k: 2, ..SamplingSettings::default() };
    let mut counts = [0; 5];
    for _ in 0..2000 {
        counts[sample(&logits, &top2, &mut rng) as usize] += 1;
    }
    assert_eq!(counts[0] + counts[3] + counts[4], 0, "sampled outside the top 2: {counts:?}");
    assert!(counts[1] > 800 && counts[2] > 800, "{counts:?}");
    let mut penalized = logits;
    apply_repetition_penalty(&mut penalized, &[1, 3], 2.0);
    assert_eq!(penalized, [0.1, 1.0, 1.9, -2.0, 0.5]);
}

#[test]
fn generation_stops_at_eos_and_is_reproducible_with_a_seed() {
    use super::generate::{generate, SamplingSettings};
    let config = tiny();
    let mut model = Model::new(CpuBackend, config.clone(), 3);
    let settings = SamplingSettings { seed: 42, ..SamplingSettings::default() };
    let mut seen = Vec::new();
    let first = generate(&mut model, &[1, 2, 3], 20, None, &settings, |id| seen.push(id));
    assert_eq!(first.len(), 20);
    assert_eq!(seen, first);
    assert_eq!(generate(&mut model, &[1, 2, 3], 20, None, &settings, |_| {}), first);
    let eos = first[5];
    let stopped = generate(&mut model, &[1, 2, 3], 20, Some(eos), &settings, |_| {});
    assert!(!stopped.contains(&eos) && stopped.len() <= 5, "{stopped:?}");
}

#[test]
fn masked_loss_counts_only_the_targets_and_its_gradient_matches_finite_differences() {
    use super::backend::{Backend, IGNORE_TARGET};
    let config = tiny();
    let t = config.seq_len;
    let mut model = Model::new(CpuBackend, config.clone(), 13);
    let window = windows(&config, 2, 9);
    let inputs: Vec<u32> = window.chunks(t + 1).flat_map(|w| w[..t].to_vec()).collect();
    let all: Vec<u32> = window.chunks(t + 1).flat_map(|w| w[1..].to_vec()).collect();
    // Without ignored targets the masked loss is the ordinary loss.
    let plain = model.forward_backward(&window, false, 1.0);
    assert!((model.forward_backward_targets(&inputs, &all, false, 1.0) - plain).abs() < 1e-9);
    // Only positions 5..8 of the first sequence and 2 of the second count.
    let mut targets = vec![IGNORE_TARGET; 2 * t];
    for i in [5, 6, 7, t + 2] {
        targets[i] = all[i];
    }
    let loss = model.forward_backward_targets(&inputs, &targets, false, 1.0);
    let mut expected = 0.0;
    for i in [5, 6, 7, t + 2] {
        let (s, p) = (i / t, i % t);
        let logits = model.next_token_logits(&window[s * (t + 1)..s * (t + 1) + p + 1]);
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v)) as f64;
        let log_sum = logits.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>().ln() + max;
        expected += log_sum - logits[all[i] as usize] as f64;
    }
    assert!((loss - expected / 4.0).abs() < 1e-4, "masked loss {loss} vs {}", expected / 4.0);
    assert_eq!(model.forward_backward_targets(&inputs, &vec![IGNORE_TARGET; 2 * t], true, 1.0), 0.0);
    // Gradient of the masked loss against central differences for a few parameters.
    model.zero_grads();
    model.forward_backward_targets(&inputs, &targets, true, 1.0);
    let grads = CpuBackend.download(&model.grads);
    let params = CpuBackend.download(&model.params);
    let h = 1e-2f32;
    let mut checked = 0;
    for &i in &[3usize, 777, 1500, 2600, model.layout.total - 5] {
        let mut eval = |delta: f32| {
            let mut p = params.clone();
            p[i] += delta;
            let mut m = Model::from_parameters(CpuBackend, config.clone(), &p);
            m.forward_backward_targets(&inputs, &targets, false, 1.0)
        };
        let numeric = (eval(h) - eval(-h)) / (2.0 * h as f64);
        let analytic = grads[i] as f64;
        if numeric.abs().max(analytic.abs()) > 1e-3 {
            checked += 1;
            assert!((numeric - analytic).abs() <= 0.05 * numeric.abs().max(analytic.abs()) + 1e-4, "parameter {i}: numeric {numeric} vs analytic {analytic}");
        }
    }
    assert!(checked >= 2, "too few parameters with a visible gradient");
}

#[test]
fn qa_sequences_train_only_the_answer_and_keep_it_in_the_window() {
    use super::backend::IGNORE_TARGET;
    use super::finetune::{evaluation_windows, exact_match_and_f1, sequence, training_sequences, EncodedQa};
    use rand::{rngs::StdRng, SeedableRng};
    let (inputs, targets) = sequence(&[1, 2], &[10, 11, 12], &[3, 4], &[20, 21], 99, 12);
    assert_eq!(inputs, vec![1, 2, 10, 11, 12, 3, 4, 20, 21, 99, 99, 99]);
    let i = IGNORE_TARGET;
    assert_eq!(targets, vec![i, i, i, i, i, i, 20, 21, 99, i, i, i]);
    // A long context: the answer (tokens 300..303) is always inside the window.
    let e = EncodedQa { prefix: vec![1; 10], suffix: vec![2; 10], context: (1000..1500).collect(), answer_span: (300, 303), answer: vec![7, 8, 9] };
    let mut rng = StdRng::seed_from_u64(1);
    let (mut negatives, mut positives) = (0, 0);
    for _ in 0..50 {
        for (inputs, targets) in training_sequences(&e, 99, 128, &mut rng) {
            let answer: Vec<u32> = targets.iter().copied().filter(|&t| t != IGNORE_TARGET).collect();
            if answer == vec![99] {
                negatives += 1;
                assert!(!(300..303).any(|t| inputs.contains(&(1000 + t))), "negative window contains the answer");
            } else {
                positives += 1;
                assert_eq!(answer, vec![7, 8, 9, 99]);
                assert!((300..303).all(|t| inputs.contains(&(1000 + t))), "answer not in the window");
            }
        }
    }
    assert_eq!(positives, 50);
    assert!(negatives > 10, "{negatives} negative windows");
    let w = evaluation_windows(500, 100);
    assert_eq!(w.first(), Some(&(0, 100)));
    assert_eq!(w.last().unwrap().1, 500);
    assert!(w.windows(2).all(|p| p[1].0 < p[0].1), "windows must overlap");
    assert_eq!(exact_match_and_f1("Der Rhein.", "der rhein"), (1.0, 1.0));
    let (em, f1) = exact_match_and_f1("im Jahr 1990", "1990");
    assert_eq!(em, 0.0);
    assert!((f1 - 0.5).abs() < 1e-9);
}

#[test]
fn finetuning_runs_end_to_end_on_a_small_model() {
    use super::finetune::{finetune, FinetuneSettings};
    use super::train::save_checkpoint;
    let dir = std::env::temp_dir().join(format!("engine_finetune_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut paragraphs = Vec::new();
    for i in 0..24 {
        let city = ["Berlin", "Hamburg", "München", "Köln"][i % 4];
        let context = format!("Die Stadt {city} liegt in Deutschland. Im Jahr {} wurde dort ein Museum eröffnet.", 1900 + i);
        let start = context.find(city).unwrap();
        paragraphs.push(serde_json::json!({"context": context, "qas": [{"question": "Welche Stadt liegt in Deutschland?", "answers": [{"text": city, "answer_start": context[..start].chars().count()}]}]}));
    }
    let squad = serde_json::json!({"version": "test", "data": [{"title": "t", "paragraphs": paragraphs}]});
    for name in ["qa_train_001.json", "qa_test_001.json"] {
        std::fs::write(dir.join(name), squad.to_string()).unwrap();
    }
    let config = EngineConfig { vocab_size: 50282, width: 16, layers: 1, heads: 2, ffn_hidden: 16, window: 64, seq_len: 64, ..EngineConfig::default() };
    let base = dir.join("base");
    let model = Model::new(CpuBackend, config, 5);
    save_checkpoint(&model, &base, None, &Default::default()).unwrap();
    let mut settings = FinetuneSettings::new(&dir);
    settings.train_prefix = "qa_train".into();
    settings.test_prefix = "qa_test".into();
    settings.base = base;
    settings.checkpoint = dir.join("qa");
    settings.epochs = 2;
    settings.batch_size = 4;
    settings.micro_batch = 2;
    settings.learning_rate = 3e-3;
    settings.warmup_steps = 2;
    settings.eval_every = 6;
    settings.eval_questions = 4;
    finetune(CpuBackend, &settings).unwrap();
    let state: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("qa.state.json")).unwrap()).unwrap();
    assert!(state["best_val_loss"].as_f64().is_some_and(|l| l.is_finite()), "{state}");
    assert!(dir.join("qa.params").exists());
    std::fs::remove_dir_all(&dir).ok();
}

/// Loss and gradient of `config` against central differences (f64 accumulation of the CPU backend in f32: h = 1e-2),
/// checking parameters in every segment kind; plus the next-token logits against the training loss.
fn check_architecture(config: EngineConfig) {
    use super::backend::Backend;
    let mut model = Model::new(CpuBackend, config.clone(), 21);
    let window = windows(&config, 2, 17);
    let t = config.seq_len;
    let loss = model.forward_backward(&window, false, 1.0);
    let mut total = 0.0;
    for p in 0..t {
        let logits = model.next_token_logits(&window[..=p]);
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v)) as f64;
        let log_sum = logits.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>().ln() + max;
        total += log_sum - logits[window[p + 1] as usize] as f64;
    }
    // Only the first sequence was walked: compare with its own loss.
    let first = model.forward_backward(&window[..t + 1], false, 1.0);
    assert!((total / t as f64 - first).abs() < 1e-4, "{config:?}: next-token logits {} vs loss {first}", total / t as f64);
    assert!(loss.is_finite());
    model.zero_grads();
    model.forward_backward(&window, true, 1.0);
    let grads = CpuBackend.download(&model.grads);
    let params = CpuBackend.download(&model.params);
    let mut checked = 0;
    for segment in &model.layout.segments {
        // Small step: 1e-2 already crosses kinks of the loss (leaky ReLU, window edges) for some parameters.
        let h = 1e-3f32;
        // Two parameters of every segment (first and middle).
        for i in [segment.offset, segment.offset + segment.len() / 2] {
            let mut eval = |delta: f32| {
                let mut p = params.clone();
                p[i] += delta;
                Model::from_parameters(CpuBackend, config.clone(), &p).forward_backward(&window, false, 1.0)
            };
            let numeric = (eval(h) - eval(-h)) / (2.0 * h as f64);
            let half = (eval(h / 2.0) - eval(-h / 2.0)) / (h as f64);
            let analytic = grads[i] as f64;
            let scale = numeric.abs().max(analytic.abs());
            // The RM residual is discontinuous (its leaky ReLU switches both parts on the sign of the real part):
            // a step across such a jump shows as two central differences that disagree; skip those.
            let jump = (numeric - half).abs() > 0.1 * numeric.abs().max(half.abs()) + 1e-3;
            if jump {
                assert!(!config.standard_residual, "{config:?}: {} [{i}]: the loss jumps ({numeric} vs {half})", segment.name);
            } else if scale > 2e-3 {
                checked += 1;
                assert!((numeric - analytic).abs() <= 0.05 * scale + 2e-4, "{config:?}: {} [{i}]: numeric {numeric} vs analytic {analytic}", segment.name);
            }
        }
    }
    assert!(checked >= 10, "{config:?}: only {checked} parameters with a visible gradient");
}

#[test]
fn every_architecture_variant_has_exact_gradients() {
    for complex_values in [false, true] {
        for standard_residual in [false, true] {
            for (window, standard_rope) in [(3, false), (8, true)] {
                check_architecture(EngineConfig { window, complex_values, standard_residual, standard_rope, ..tiny() });
            }
        }
    }
}

#[test]
fn old_checkpoint_configs_load_as_the_rm_architecture() {
    let old = r#"{"vocab_size": 50282, "width": 256, "layers": 12, "heads": 4, "ffn_hidden": 1024, "window": 256, "seq_len": 512}"#;
    let config: EngineConfig = serde_json::from_str(old).unwrap();
    assert_eq!(config, EngineConfig::rm());
    assert!(Layout::new(&config).final_norm.is_none());
    let new = EngineConfig::default();
    assert!(new.complex_values && new.standard_residual && new.standard_rope && new.window >= new.seq_len);
}

#[test]
fn window_order_takes_every_window_exactly_once_and_differs_per_shard() {
    use super::train::window_order;
    let order = window_order(3, 1000);
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(sorted, (0..1000).collect::<Vec<_>>(), "not a permutation");
    assert_ne!(order, (0..1000).collect::<Vec<_>>(), "not shuffled");
    assert_ne!(order, window_order(4, 1000), "same order for another shard / pass");
    assert_eq!(order, window_order(3, 1000), "not reproducible");
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
    // 4000 tokens = 444 windows of 9 tokens; 60 steps x 4 windows = 240 of them, each once, still in shard 0.
    assert_eq!((state.shard, state.window), (0, Some(240)));
    assert!(adam.is_some_and(|m| m.len() == 2 * params.len()), "AdamW moments saved");
    let best = state.best_val_loss.unwrap();
    assert!(best < (config.vocab_size as f64).ln() - 1.0, "validation loss {best} did not drop");
    settings.resume = true;
    settings.steps = 30;
    pretrain(CpuBackend, &settings).unwrap();
    let (_, _, _, resumed) = load_checkpoint(&settings.checkpoint).unwrap();
    assert!(resumed.best_val_loss.unwrap() <= best);
    // 30 more steps continue at window 240 of the same shard order: 240 + 120 = 360 <= 444.
    assert_eq!((resumed.shard, resumed.window), (0, Some(360)));

    // A checkpoint of the earlier random sampling (no window position) continues with the next shard.
    let state_file = super::train::file(&settings.checkpoint, ".state.json");
    let mut old: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&state_file).unwrap()).unwrap();
    old.as_object_mut().unwrap().remove("window");
    std::fs::write(&state_file, old.to_string()).unwrap();
    settings.steps = 2;
    pretrain(CpuBackend, &settings).unwrap();
    let (_, _, _, migrated) = load_checkpoint(&settings.checkpoint).unwrap();
    assert_eq!((migrated.shard, migrated.window), (1, Some(8)));
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
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            println!("NVRTC library not found (set LD_LIBRARY_PATH): skipped");
            return;
        }
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
    /// `grad_scale` is the loss scale of the GPU pass (as in training with f16 GEMMs); its gradients are divided by it.
    fn compare(precision: Precision, tolerance: f64, grad_scale: f32) {
        let Some(backend) = gpu(precision) else { return };
        println!("GPU: {}", backend.device_name());
        let config = EngineConfig { vocab_size: 300, width: 64, layers: 2, heads: 4, ffn_hidden: 96, window: 5, seq_len: 32, complex_values: false, standard_residual: false, standard_rope: false };
        let mut cpu = Model::new(CpuBackend, config.clone(), 3);
        let params = cpu.params.clone();
        let mut cuda = Model::from_parameters(backend, config.clone(), &params);
        let ids = windows(&config, 3, 9);
        cpu.zero_grads();
        cuda.zero_grads();
        let (l_cpu, l_gpu) = (cpu.forward_backward(&ids, true, 1.0), cuda.forward_backward(&ids, true, grad_scale));
        println!("{precision:?}: loss cpu {l_cpu:.6}, gpu {l_gpu:.6}");
        assert!((l_cpu - l_gpu).abs() < tolerance * l_cpu.abs(), "loss cpu {l_cpu} gpu {l_gpu}");
        use super::super::backend::Backend;
        let g_gpu: Vec<f32> = cuda.backend.download(&cuda.grads).iter().map(|g| g / grad_scale).collect();
        let mut relative: Vec<f64> = Vec::new();
        let mut worst = (0.0f64, String::new());
        for seg in &cpu.layout.segments {
            let (a, b) = (&cpu.grads[seg.offset..seg.offset + seg.len()], &g_gpu[seg.offset..seg.offset + seg.len()]);
            let norm = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6) as f64;
            let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / norm;
            relative.extend(a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64 / norm));
            if diff > worst.0 {
                worst = (diff, seg.name.clone());
            }
            assert!(diff < tolerance, "{precision:?} {}: GPU gradient differs by {diff:.2e}", seg.name);
        }
        relative.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{precision:?}: largest relative gradient difference {:.2e} ({}), median {:.2e}",
            worst.0,
            worst.1,
            relative[relative.len() / 2]
        );
    }

    #[test]
    fn gpu_f32_matches_cpu() {
        compare(Precision::F32, 1e-4, 1.0);
    }

    #[test]
    fn gpu_f16_matches_cpu_to_half_precision() {
        // f16 keeps ~3 decimal digits per GEMM input; measured without loss scaling: loss to 6e-5, gradients to 2.3e-2.
        compare(Precision::F16, 5e-2, 1024.0);
    }
}

/// Speed of the default model on the CPU backend (forward + backward of one 512-token sequence).
/// `cargo test --release --lib cpu_speed -- --ignored --nocapture`
#[test]
#[ignore]
fn cpu_speed_of_the_default_model() {
    let config = EngineConfig::default();
    let started = std::time::Instant::now();
    let mut model = Model::new(CpuBackend, config.clone(), 1);
    println!("model created in {:.1} s", started.elapsed().as_secs_f64());
    let ids = windows(&config, 1, 7);
    model.zero_grads();
    model.forward_backward(&ids, true, 1.0); // warm-up (allocations)
    let started = std::time::Instant::now();
    let steps = 2;
    for _ in 0..steps {
        model.zero_grads();
        model.forward_backward(&ids, true, 1.0);
    }
    let seconds = started.elapsed().as_secs_f64() / steps as f64;
    println!("CPU ({} threads): {:.2} s per 512 tokens = {:.0} tokens/s", rayon::current_num_threads(), seconds, 512.0 / seconds);
}

/// Data statistics of GermanQuAD for the fine-tuning (cargo test --release germanquad_sequences -- --ignored --nocapture).
#[test]
#[ignore]
fn germanquad_sequences() {
    use super::finetune::{encode_example, load_squad, squad_files, training_sequences};
    use crate::neural_networks::utils::tokenizer::{detokenize, token_id};
    use rand::{rngs::StdRng, SeedableRng};
    let eos = token_id("<eos>").unwrap();
    for prefix in ["germanquad_train", "germanquad_test"] {
        let examples = load_squad(&squad_files(std::path::Path::new("datasets/qa"), prefix).unwrap()).unwrap();
        let encoded: Vec<_> = examples.iter().filter_map(|e| encode_example(e).unwrap()).collect();
        let mut rng = StdRng::seed_from_u64(1);
        let sequences: Vec<_> = encoded.iter().flat_map(|e| training_sequences(e, eos, 512, &mut rng)).collect();
        let negatives = sequences.iter().filter(|s| s.1.iter().filter(|&&t| t != u32::MAX).count() == 1).count();
        println!("{prefix}: {} questions, {} encoded, {} sequences per epoch ({} without the answer)", examples.len(), encoded.len(), sequences.len(), negatives);
        let (inputs, targets) = &sequences[0];
        let answer: Vec<u32> = targets.iter().copied().filter(|&t| t != u32::MAX).collect();
        let end = inputs.iter().rposition(|&t| t != eos).unwrap();
        println!("  example input end: {:?}\n  target: {:?}", detokenize(&inputs[end.saturating_sub(40)..=end].to_vec(), false).unwrap(), detokenize(&answer, false).unwrap());
    }
}


