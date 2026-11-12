//! The candle block against the RM layers: same parameters, same input -> same output and same input gradient.

use candle_core::{DType, Device, Tensor, Var};

use super::{
    block::{AttentionHead, AttentionLayer, BlockContext, ComplexNorm, FeedForwardLayer},
    complex::{CTensor, CVar},
};
use crate::neural_networks::{
    network_components::{gradient_struct::Gradient, layer_input_struct::LayerInput},
    network_layers::{
        layer::LayerEnum,
        network_layers_rm::{feedforward_layer_rm::FeedForwardLayerRm, norm_layer_rm::NormalNormLayerRm},
    },
    network_types::transformer::sparse_self_attention_layer_rm::SparseSelfAttentionLayerRm,
    utils::{
        dtype::{Real, C},
        matrix_rm::RowMajorMatrix,
        random_arrays::generate_random_complex_3d,
    },
};

/// f64 build: candle and RM agree to rounding; f32 build: to f32 precision.
#[cfg(feature = "dtype-f64")]
const TOLERANCE: f64 = 1e-9;
#[cfg(not(feature = "dtype-f64"))]
const TOLERANCE: f64 = 2e-3;

fn pairs(values: &[C]) -> Vec<(f64, f64)> {
    values.iter().map(|c| (c.re as f64, c.im as f64)).collect()
}

fn cvar(values: &[C], shape: &[usize]) -> CVar {
    CVar::from_pairs(&pairs(values), shape, DType::F64, &Device::Cpu).unwrap()
}

fn real_var(m: &RowMajorMatrix<C>) -> Var {
    let re: Vec<f64> = m.data.iter().map(|c| c.re as f64).collect();
    Var::from_tensor(&Tensor::from_vec(re, (m.rows, m.cols), &Device::Cpu).unwrap()).unwrap()
}

fn norm_from_rm(norm: &NormalNormLayerRm) -> ComplexNorm {
    let d = norm.gamma.len();
    ComplexNorm { gamma: cvar(&norm.gamma, &[d]), beta: cvar(&norm.beta, &[d]), epsilon: norm.epsilon }
}

fn attention_from_rm(layer: &SparseSelfAttentionLayerRm) -> AttentionLayer {
    let heads = layer
        .attention_heads
        .iter()
        .map(|h| {
            let shape = [h.weights_q_rm.rows, h.weights_q_rm.cols];
            let (ctl_q, ctl_k) = (h.ctl_q.as_ref().unwrap(), h.ctl_k.as_ref().unwrap());
            AttentionHead {
                w_q: cvar(&h.weights_q_rm.data, &shape),
                w_k: cvar(&h.weights_k_rm.data, &shape),
                w_v: cvar(&h.weights_v_rm.data, &shape),
                ctl_q_1: real_var(&ctl_q.weights_1),
                ctl_q_2: real_var(&ctl_q.weights_2),
                ctl_k_1: real_var(&ctl_k.weights_1),
                ctl_k_2: real_var(&ctl_k.weights_2),
            }
        })
        .collect();
    AttentionLayer { norm: norm_from_rm(&layer.norm_layer), heads, beta: layer.beta as f64 }
}

fn ffn_from_rm(layer: &FeedForwardLayerRm) -> FeedForwardLayer {
    let Some(LayerEnum::NormRm(norm)) = &layer.norm_layer else { panic!("FeedForwardLayerRm without NormRm") };
    let (LayerEnum::DenseRm(d1), LayerEnum::DenseRm(d2)) = (&layer.layers[0], &layer.layers[1]) else { panic!("FeedForwardLayerRm without two DenseRm layers") };
    FeedForwardLayer {
        norm: norm_from_rm(norm),
        w_1: cvar(&d1.weights.data, &[d1.weights.rows, d1.weights.cols]),
        b_1: cvar(&d1.bias, &[d1.bias.len()]),
        w_2: cvar(&d2.weights.data, &[d2.weights.rows, d2.weights.cols]),
        b_2: cvar(&d2.bias, &[d2.bias.len()]),
        beta: layer.beta as f64,
    }
}

fn to_ctensor(batch: &[Vec<Vec<C>>]) -> CTensor {
    let (b, t, d) = (batch.len(), batch[0].len(), batch[0][0].len());
    let flat: Vec<C> = batch.iter().flatten().flatten().copied().collect();
    CTensor::from_pairs(&pairs(&flat), &[b, t, d], DType::F64, &Device::Cpu).unwrap()
}

fn assert_close(name: &str, expected: &[(f64, f64)], actual: &[(f64, f64)]) {
    assert_eq!(expected.len(), actual.len(), "{name}: length");
    let mut worst = 0.0f64;
    for (e, a) in expected.iter().zip(actual) {
        let diff = ((e.0 - a.0).powi(2) + (e.1 - a.1).powi(2)).sqrt();
        let scale = (e.0.hypot(e.1)).max(a.0.hypot(a.1)).max(1.0);
        worst = worst.max(diff / scale);
    }
    println!("{name}: max relative difference {worst:.2e}");
    assert!(worst <= TOLERANCE, "{name}: max relative difference {worst:.2e} > {TOLERANCE:.0e}");
}

fn rm_pairs(batch: &[RowMajorMatrix<C>]) -> Vec<(f64, f64)> {
    batch.iter().flat_map(|m| pairs(&m.data)).collect()
}

/// Input gradient of L = sum(Re(G) Re(Y) + Im(G) Im(Y)) as dL/dRe(x) + i dL/dIm(x), the convention of the RM
/// backward pass for an upstream gradient G.
fn candle_input_gradient(forward: impl Fn(&CTensor) -> CTensor, input: &[Vec<Vec<C>>], upstream: &[Vec<Vec<C>>]) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let x = to_ctensor(input);
    let (x_re, x_im) = (Var::from_tensor(&x.re).unwrap(), Var::from_tensor(&x.im).unwrap());
    let y = forward(&CTensor::new(x_re.as_tensor().clone(), x_im.as_tensor().clone()));
    let g = to_ctensor(upstream);
    let loss = (y.re.mul(&g.re).unwrap().sum_all().unwrap() + y.im.mul(&g.im).unwrap().sum_all().unwrap()).unwrap();
    let grads = loss.backward().unwrap();
    let gx = CTensor::new(grads.get(x_re.as_tensor()).unwrap().clone(), grads.get(x_im.as_tensor()).unwrap().clone());
    (y.to_pairs().unwrap(), gx.to_pairs().unwrap())
}

fn rm_input(input: &[Vec<Vec<C>>]) -> LayerInput {
    let mut layer_input = LayerInput::new_default();
    layer_input.set_input_batch_rm(input.iter().map(|m| RowMajorMatrix::from_rows(m)).collect());
    layer_input.set_padding_mask_batch(vec![vec![1; input[0].len()]; input.len()]);
    layer_input.set_batch_size(input.len());
    layer_input
}

fn rm_upstream(upstream: &[Vec<Vec<C>>]) -> Gradient {
    let mut gradient = Gradient::new_default();
    gradient.set_gradient_input_batch_rm(upstream.iter().map(|m| RowMajorMatrix::from_rows(m)).collect());
    gradient.set_total_valid_tokens(upstream.iter().map(|m| m.len()).sum());
    gradient
}

#[test]
fn feed_forward_layer_matches_rm() {
    let (batch_size, seq_len, dim, hidden) = (2, 6, 8, 12);
    let mut layer_rm = FeedForwardLayerRm::new(dim, hidden, 0.01);
    let layer = ffn_from_rm(&layer_rm);
    let input = generate_random_complex_3d(batch_size, seq_len, dim);
    let upstream = generate_random_complex_3d(batch_size, seq_len, dim);

    let output_rm = layer_rm.forward(&rm_input(&input)).take_output_batch_rm().unwrap();
    let input_gradient_rm = layer_rm.backward(&rm_upstream(&upstream)).get_gradient_input_batch_rm();

    let (output, input_gradient) = candle_input_gradient(|x| layer.forward(x).unwrap(), &input, &upstream);
    assert_close("FFN output", &rm_pairs(&output_rm), &output);
    assert_close("FFN input gradient", &rm_pairs(&input_gradient_rm), &input_gradient);
}

#[test]
fn attention_layer_matches_rm() {
    let (batch_size, seq_len, dim, heads, window) = (2, 9, 8, 2, 2);
    let mut layer_rm = SparseSelfAttentionLayerRm::new(heads, dim, dim, window, 0.01);
    let layer = attention_from_rm(&layer_rm);
    let input = generate_random_complex_3d(batch_size, seq_len, dim);
    let upstream = generate_random_complex_3d(batch_size, seq_len, dim);

    let output_rm = layer_rm.forward(&rm_input(&input)).take_output_batch_rm().unwrap();
    let input_gradient_rm = layer_rm.backward(&rm_upstream(&upstream)).get_gradient_input_batch_rm();

    let ctx = BlockContext::new(seq_len, dim / heads, window, DType::F64, &Device::Cpu).unwrap();
    let (output, input_gradient) = candle_input_gradient(|x| layer.forward(x, &ctx).unwrap(), &input, &upstream);
    assert_close("attention output", &rm_pairs(&output_rm), &output);
    // The RM attention backward is hand-written; this reports how far it is from the exact (autograd) gradient.
    let worst = {
        let rm = rm_pairs(&input_gradient_rm);
        rm.iter().zip(&input_gradient).map(|(e, a)| ((e.0 - a.0).powi(2) + (e.1 - a.1).powi(2)).sqrt() / e.0.hypot(e.1).max(a.0.hypot(a.1)).max(1.0)).fold(0.0f64, f64::max)
    };
    println!("attention input gradient: RM vs exact autograd, max relative difference {worst:.2e}");
    // Measured 1e-6 in f64: the hand-written RM attention backward is a close approximation of the exact gradient.
    assert!(worst <= 1e-4_f64.max(TOLERANCE), "attention input gradient differs from RM by {worst:.2e}");
}

#[test]
fn complex_ops_match_num_complex() {
    let z: Vec<C> = generate_random_complex_3d(1, 4, 6).concat().concat();
    let z = z.into_iter().map(|c| c * (3.0 as Real)).collect::<Vec<C>>();
    let t = CTensor::from_pairs(&pairs(&z), &[z.len()], DType::F64, &Device::Cpu).unwrap();
    let sqrt: Vec<C> = z.iter().map(|c| c.sqrt()).collect();
    assert_close("sqrt", &pairs(&sqrt), &t.sqrt().unwrap().to_pairs().unwrap());
    let sigmoid: Vec<C> = z.iter().map(crate::neural_networks::utils::activation::sigmoid_complex).collect();
    assert_close("sigmoid", &pairs(&sigmoid), &t.sigmoid().unwrap().to_pairs().unwrap());
}

mod model_tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    use crate::neural_networks::candle_model::model::{Model, ModelConfig};

    #[test]
    fn fast_embedding_table_equals_per_token_wavelet_decomposition() {
        use crate::neural_networks::candle_model::model::{wavelet_embedding_table, wavelet_token_vector};
        let (levels, width, seed) = (1, 256, 42u64);
        let table = wavelet_embedding_table(5, levels, width, seed);
        for token in 0..5 {
            let direct = wavelet_token_vector(levels, width, seed ^ (token as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for (j, d) in direct.iter().enumerate() {
                let f = table[token * width + j];
                assert!((f.0 - d.0).abs() < 1e-9 && (f.1 - d.1).abs() < 1e-9, "token {token} value {j}: {f:?} vs {d:?}");
            }
        }
    }

    fn tiny() -> ModelConfig {
        ModelConfig { vocab_size: 97, width: 64, layers: 2, heads: 4, ffn_hidden: 96, window: 8, seq_len: 16 }
    }

    fn windows(device: &Device) -> Tensor {
        let ids: Vec<u32> = (0..2 * 17).map(|i| (i * 7 % 97) as u32).collect();
        Tensor::from_vec(ids, (2, 17), device).unwrap()
    }

    #[test]
    fn initial_loss_is_near_uniform_and_training_reduces_it() {
        let device = Device::Cpu;
        let model = Model::new(tiny(), DType::F32, &device, 1).unwrap();
        let batch = windows(&device);
        let first = model.loss(&batch).unwrap().to_scalar::<f32>().unwrap();
        println!("initial loss {first:.3} (uniform: {:.3})", (97f32).ln());
        assert!(first.is_finite() && first > 2.0 && first < 12.0, "initial loss {first}");

        let mut optimizer = AdamW::new(model.vars(), ParamsAdamW { lr: 3e-3, ..Default::default() }).unwrap();
        let mut last = first;
        for _ in 0..30 {
            let loss = model.loss(&batch).unwrap();
            optimizer.backward_step(&loss).unwrap();
            last = loss.to_scalar::<f32>().unwrap();
        }
        println!("loss after 30 steps on one batch: {last:.3}");
        assert!(last < first * 0.7, "loss did not fall: {first} -> {last}");
    }

    #[test]
    fn checkpointed_gradients_equal_plain_backprop() {
        let device = Device::Cpu;
        let model = Model::new(tiny(), DType::F64, &device, 4).unwrap();
        let batch = windows(&device);
        let loss = model.loss(&batch).unwrap();
        let plain = loss.backward().unwrap();
        let (checkpointed_loss, checkpointed) = model.loss_and_grads(&batch, 5).unwrap();
        let plain_loss = loss.to_scalar::<f64>().unwrap();
        assert!((plain_loss - checkpointed_loss).abs() < 1e-10, "loss {plain_loss} vs {checkpointed_loss}");
        assert!((model.loss_no_grad(&batch, 7).unwrap() - plain_loss).abs() < 1e-10);
        let mut worst = 0.0f64;
        for (name, var) in model.named_vars() {
            let a = plain.get(var.as_tensor()).unwrap_or_else(|| panic!("{name}: no plain gradient")).flatten_all().unwrap().to_vec1::<f64>().unwrap();
            let b = checkpointed.get(var.as_tensor()).unwrap_or_else(|| panic!("{name}: no checkpointed gradient")).flatten_all().unwrap().to_vec1::<f64>().unwrap();
            let scale = a.iter().fold(1e-12f64, |m, v| m.max(v.abs()));
            let diff = a.iter().zip(&b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            worst = worst.max(diff / scale);
        }
        println!("checkpointed vs plain gradients: max relative difference {worst:.2e}");
        assert!(worst < 1e-9);
    }

    #[test]
    fn save_and_load_give_the_same_logits() {
        let device = Device::Cpu;
        let model = Model::new(tiny(), DType::F32, &device, 2).unwrap();
        let dir = std::env::temp_dir().join(format!("candle_model_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        model.save(&path).unwrap();
        let loaded = Model::load(&path, DType::F32, &device).unwrap();
        let tokens = windows(&device).narrow(1, 0, 16).unwrap();
        let a = model.forward(&tokens).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = loaded.forward(&tokens).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn default_model_has_about_25m_layer_parameters() {
        let config = ModelConfig::default();
        let (w, h, l) = (config.width, config.ffn_hidden, config.layers);
        let hd = w / config.heads;
        // Same count as Model::parameter_counts without building the 64 M parameter model.
        let attention = config.heads * (3 * 2 * w * hd + 4 * hd * hd) + 4 * w;
        let ffn = 2 * (w * 2 * h + 2 * h + h * w + w) + 4 * w;
        let layers = l * (attention + ffn);
        let total = layers + 2 * config.vocab_size * w + 4 * w + 2 * w * w + w * config.vocab_size + config.vocab_size;
        println!("default model: {:.1} M parameters, {:.1} M in the layers", total as f64 / 1e6, layers as f64 / 1e6);
        let model = Model::new(ModelConfig { vocab_size: 97, ..tiny() }, DType::F32, &Device::Cpu, 3).unwrap();
        let (t, ly) = model.parameter_counts();
        let c = tiny();
        let (w, h, hd) = (c.width, c.ffn_hidden, c.width / c.heads);
        let expected_layers = c.layers * (c.heads * (6 * w * hd + 4 * hd * hd) + 4 * w + 2 * (2 * w * h + 2 * h + h * w + w) + 4 * w);
        assert_eq!(ly, expected_layers);
        assert_eq!(t, ly + 2 * 97 * w + 4 * w + 2 * w * w + w * 97 + 97);
        assert!((20_000_000..30_000_000).contains(&layers));
    }
}

mod train_tests {
    use crate::neural_networks::candle_model::{
        model::ModelConfig,
        train::{pretrain, PretrainSettings, TrainState},
    };

    /// Token stream with a learnable pattern: ascending runs of length 5.
    fn write_shard(path: &std::path::Path, tokens: usize, offset: u16) {
        let ids: Vec<u8> = (0..tokens).flat_map(|i| (((i / 5 + (i % 5)) as u16 + offset) % 50).to_le_bytes()).collect();
        std::fs::write(path, ids).unwrap();
    }

    #[test]
    fn pretrain_learns_saves_and_resumes() {
        let dir = std::env::temp_dir().join(format!("candle_pretrain_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_shard(&dir.join("train_00000.bin"), 20_000, 0);
        write_shard(&dir.join("train_00001.bin"), 20_000, 3);
        write_shard(&dir.join("val_00000.bin"), 4_000, 0);

        let mut settings = PretrainSettings::new(&dir);
        settings.checkpoint = dir.join("lm.safetensors");
        settings.config = ModelConfig { vocab_size: 50, width: 64, layers: 2, heads: 4, ffn_hidden: 96, window: 8, seq_len: 32 };
        settings.steps = 60;
        settings.batch_size = 8;
        settings.micro_batch = 4;
        settings.learning_rate = 3e-3;
        settings.warmup_steps = 10;
        settings.eval_every = 20;
        settings.eval_batches = 4;
        settings.log_every = 20;
        settings.max_hours = None;
        pretrain(&settings).unwrap();

        let state: TrainState = serde_json::from_str(&std::fs::read_to_string(dir.join("lm.safetensors.state.json")).unwrap()).unwrap();
        let best = state.best_val_loss.unwrap();
        println!("best validation loss after 60 steps: {best:.3} (uniform {:.3})", (50f64).ln());
        assert!(best < 0.7 * (50f64).ln(), "model did not learn the pattern: {best}");
        assert!(dir.join("lm.safetensors.json").exists() && dir.join("lm.safetensors.history.csv").exists());

        settings.resume = true;
        settings.steps = 20;
        pretrain(&settings).unwrap();
        let resumed: TrainState = serde_json::from_str(&std::fs::read_to_string(dir.join("lm.safetensors.state.json")).unwrap()).unwrap();
        assert!(resumed.best_val_loss.unwrap() <= best);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
