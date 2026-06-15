// Gradient-check tests for the AdaptiveLinearLayer output head.
//
// Goal: the user reports that predictions stay wrong after training and suspects a
// gradient bug. These tests verify the analytical gradients produced by the
// training forward pass (grad w.r.t. the input H, the routing weights, and the
// cluster word weights) against finite-difference numerical gradients, exactly
// like `test_linear_layer.rs` does for the dense head.
//
// The layer is built with a fully controlled, deterministic cluster layout so the
// candidate set is stable under perturbation (a precondition for a valid numerical
// gradient check): a 2-token head cluster plus two 2-token tail clusters, with
// `top_k` chosen so no truncation or NEG_INFINITY padding ever occurs.
#[cfg(test)]
mod test_adaptive_linear_layer {
    use crate::neural_networks::{
        network_components::layer_input_struct::LayerInput,
        network_layers::adaptive_linear_layer::AdaptiveLinearLayer,
        utils::dtype::C,
    };

    const HIDDEN: usize = 4;
    const VOCAB: usize = 6;
    const TOP_K: usize = 4;

    // Deterministic small weight value so logits stay moderate and reproducible.
    fn wv(seed: usize) -> f32 {
        (((seed * 37 + 11) % 23) as f32 - 11.0) * 0.03
    }

    /// Build an AdaptiveLinearLayer with a hand-specified cluster layout:
    ///   cluster 0 (head)  -> tokens [0, 1]
    ///   cluster 1 (tail)  -> tokens [2, 3]
    ///   cluster 2 (tail)  -> tokens [4, 5]
    fn build_controlled_layer() -> AdaptiveLinearLayer {
        let mut layer = AdaptiveLinearLayer::new(0.01, HIDDEN, VOCAB);

        layer.frequency_clusters = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
        layer.head_size = 2;
        layer.tail_cluster_count = 2; // clusters 1..=2
        layer.token_cluster_by_id = vec![0, 0, 1, 1, 2, 2];
        layer.token_index_in_cluster = vec![0, 1, 0, 1, 0, 1];

        // Cluster word weights/biases: 3 clusters, each 2 words x HIDDEN.
        let mut cww: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut cwb: Vec<Vec<f32>> = Vec::new();
        for c in 0..3 {
            let mut w = vec![vec![0.0f32; HIDDEN]; 2];
            let mut b = vec![0.0f32; 2];
            for i in 0..2 {
                for d in 0..HIDDEN {
                    w[i][d] = wv(c * 100 + i * 10 + d);
                }
                b[i] = wv(c * 100 + i * 10 + 7) * 0.5;
            }
            cww.push(w);
            cwb.push(b);
        }
        layer.cluster_word_weights = cww;
        layer.cluster_word_bias = cwb;

        // Routing weights/biases: one row per tail cluster (2 rows) x HIDDEN.
        let mut rw = vec![vec![0.0f32; HIDDEN]; 2];
        let mut rb = vec![0.0f32; 2];
        for rc in 0..2 {
            for d in 0..HIDDEN {
                rw[rc][d] = wv(rc * 50 + d + 3);
            }
            rb[rc] = wv(rc * 50 + 9) * 0.5;
        }
        layer.weights = rw.clone();
        layer.previous_weights = rw;
        layer.bias = rb;

        layer
    }

    fn base_input() -> Vec<Vec<Vec<C>>> {
        // batch = 1, seq = 2, hidden = HIDDEN. Real-only (the layer ignores im).
        let mut input = vec![vec![vec![C::new(0.0, 0.0); HIDDEN]; 2]; 1];
        for s in 0..2 {
            for d in 0..HIDDEN {
                input[0][s][d] = C::new(0.12 * (s * HIDDEN + d) as f64 - 0.25, 0.0);
            }
        }
        input
    }

    // Targets chosen so every sequence row has exactly TOP_K=4 candidates
    // (2 head + 2 tail from the target's tail cluster), i.e. no padding/truncation.
    //   row 0 target = 2 (cluster 1) -> candidates [0,1,2,3]
    //   row 1 target = 4 (cluster 2) -> candidates [0,1,4,5]
    fn targets() -> Vec<Vec<u32>> {
        vec![vec![2, 4]]
    }

    fn padding() -> Vec<Vec<u32>> {
        vec![vec![1, 1]]
    }

    const TOTAL_VALID: usize = 2;

    /// Run the training forward pass and return the scalar (summed) cross-entropy loss.
    fn forward_loss(layer: &mut AdaptiveLinearLayer, input: &Vec<Vec<Vec<C>>>) -> f64 {
        let mut li = LayerInput::new_default();
        li.set_input_batch(input.clone());
        li.set_padding_mask_batch(padding());
        li.set_target_batch_ids(targets());
        li.set_top_k_size(TOP_K);
        li.set_total_valid_tokens(TOTAL_VALID);
        li.set_calculate_gradient(true);
        li.set_batch_size(1);

        let out = layer.forward(&li);
        out.get_cross_entropy_loss_batch()
            .iter()
            .flatten()
            .flatten()
            .map(|c| c.re)
            .sum()
    }

    fn rel_err(a: f64, n: f64) -> f64 {
        (a - n).abs() / (a.abs().max(n.abs()).max(1e-8))
    }

    #[test]
    fn test_adaptive_input_gradient() {
        let mut layer = build_controlled_layer();
        let input = base_input();

        // Analytical grad w.r.t. input H (computed inside the training forward).
        let loss = forward_loss(&mut layer, &input);
        assert!(loss.is_finite() && loss > 0.0, "loss must be finite and positive, got {loss}");
        let grad = layer.gradient.as_ref().expect("forward must store a gradient in training mode");
        let analytical = grad.get_gradient_input_batch();

        // Numerical grad: central differences on each real component of H.
        let eps = 1e-6;
        let mut max_rel: f64 = 0.0;
        for s in 0..2 {
            for d in 0..HIDDEN {
                let mut ip = input.clone();
                ip[0][s][d].re += eps;
                let lp = forward_loss(&mut layer, &ip);

                let mut im = input.clone();
                im[0][s][d].re -= eps;
                let lm = forward_loss(&mut layer, &im);

                let num = (lp - lm) / (2.0 * eps);
                let ana = analytical[0][s][d].re;
                let e = rel_err(ana, num);
                println!("dH[{s}][{d}] analytical={ana:+.6} numerical={num:+.6} rel_err={e:.2e}");
                max_rel = max_rel.max(e);
            }
        }
        assert!(max_rel < 1e-3, "input gradient mismatch, max rel err = {max_rel:.3e}");
    }

    #[test]
    fn test_adaptive_routing_weight_gradient() {
        let mut layer = build_controlled_layer();
        let input = base_input();

        forward_loss(&mut layer, &input);
        let grad = layer.gradient.as_ref().expect("gradient stored");
        // Grouped over the batch dimension -> [routing_count][hidden].
        let analytical = grad.get_gradient_weights();
        let routing_bias_grad = grad.get_gradient_bias();

        let routing_count = layer.weights.len();
        let eps = 1e-3f32;
        let mut max_rel: f64 = 0.0;
        for rc in 0..routing_count {
            for d in 0..HIDDEN {
                let saved = layer.weights[rc][d];

                layer.weights[rc][d] = saved + eps;
                let lp = forward_loss(&mut layer, &input);

                layer.weights[rc][d] = saved - eps;
                let lm = forward_loss(&mut layer, &input);

                layer.weights[rc][d] = saved;

                let num = (lp as f64 - lm as f64) / (2.0 * eps as f64);
                let ana = analytical[rc][d].re;
                let e = rel_err(ana, num);
                println!("dWr[{rc}][{d}] analytical={ana:+.6} numerical={num:+.6} rel_err={e:.2e}");
                max_rel = max_rel.max(e);
            }
        }

        // Routing bias too.
        for rc in 0..routing_count {
            let saved = layer.bias[rc];
            layer.bias[rc] = saved + eps;
            let lp = forward_loss(&mut layer, &input);
            layer.bias[rc] = saved - eps;
            let lm = forward_loss(&mut layer, &input);
            layer.bias[rc] = saved;

            let num = (lp as f64 - lm as f64) / (2.0 * eps as f64);
            let ana = routing_bias_grad[rc].re;
            let e = rel_err(ana, num);
            println!("dBr[{rc}] analytical={ana:+.6} numerical={num:+.6} rel_err={e:.2e}");
            max_rel = max_rel.max(e);
        }

        assert!(max_rel < 5e-2, "routing weight/bias gradient mismatch, max rel err = {max_rel:.3e}");
    }

    #[test]
    fn test_adaptive_cluster_word_weight_gradient() {
        let mut layer = build_controlled_layer();
        let input = base_input();

        forward_loss(&mut layer, &input);
        let analytical = layer
            .cluster_word_weight_gradients()
            .expect("word weight gradients stored in training")
            .clone();
        let analytical_bias = layer
            .cluster_word_bias_gradients()
            .expect("word bias gradients stored in training")
            .clone();

        let eps = 1e-3f32;
        let mut max_rel: f64 = 0.0;
        for c in 0..layer.cluster_word_weights.len() {
            for i in 0..layer.cluster_word_weights[c].len() {
                for d in 0..HIDDEN {
                    let saved = layer.cluster_word_weights[c][i][d];

                    layer.cluster_word_weights[c][i][d] = saved + eps;
                    let lp = forward_loss(&mut layer, &input);

                    layer.cluster_word_weights[c][i][d] = saved - eps;
                    let lm = forward_loss(&mut layer, &input);

                    layer.cluster_word_weights[c][i][d] = saved;

                    let num = (lp - lm) / (2.0 * eps as f64);
                    let ana = analytical[c][i][d].re;
                    let e = rel_err(ana, num);
                    if ana.abs() > 1e-9 || num.abs() > 1e-9 {
                        println!("dWc[{c}][{i}][{d}] analytical={ana:+.6} numerical={num:+.6} rel_err={e:.2e}");
                        max_rel = max_rel.max(e);
                    }
                }

                // Word bias.
                let saved = layer.cluster_word_bias[c][i];
                layer.cluster_word_bias[c][i] = saved + eps;
                let lp = forward_loss(&mut layer, &input);
                layer.cluster_word_bias[c][i] = saved - eps;
                let lm = forward_loss(&mut layer, &input);
                layer.cluster_word_bias[c][i] = saved;

                let num = (lp - lm) / (2.0 * eps as f64);
                let ana = analytical_bias[c][i].re;
                if ana.abs() > 1e-9 || num.abs() > 1e-9 {
                    let e = rel_err(ana, num);
                    println!("dBc[{c}][{i}] analytical={ana:+.6} numerical={num:+.6} rel_err={e:.2e}");
                    max_rel = max_rel.max(e);
                }
            }
        }

        assert!(max_rel < 5e-2, "cluster word weight/bias gradient mismatch, max rel err = {max_rel:.3e}");
    }

    /// Regression test for the train/inference scoring inconsistency.
    ///
    /// The bug: the routing score was added to tail-token logits only during training,
    /// so train and inference optimised/used different scoring functions and the argmax
    /// flipped between them. The fix adds the routing score in both paths.
    ///
    /// To compare fairly we need the *same candidate set* in both modes. A single tail
    /// cluster guarantees that: the router has only one cluster to pick, so candidates =
    /// head + that cluster in both training and inference. We then assert the per-token
    /// logits are identical, and the top-1 prediction matches.
    fn build_single_tail_layer() -> AdaptiveLinearLayer {
        let mut layer = AdaptiveLinearLayer::new(0.01, HIDDEN, 4);

        layer.frequency_clusters = vec![vec![0, 1], vec![2, 3]];
        layer.head_size = 2;
        layer.tail_cluster_count = 1; // only cluster 1
        layer.token_cluster_by_id = vec![0, 0, 1, 1];
        layer.token_index_in_cluster = vec![0, 1, 0, 1];

        let mut cww: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut cwb: Vec<Vec<f32>> = Vec::new();
        for c in 0..2 {
            let mut w = vec![vec![0.0f32; HIDDEN]; 2];
            let mut b = vec![0.0f32; 2];
            for i in 0..2 {
                for d in 0..HIDDEN {
                    w[i][d] = wv(c * 100 + i * 10 + d);
                }
                b[i] = wv(c * 100 + i * 10 + 7) * 0.5;
            }
            cww.push(w);
            cwb.push(b);
        }
        layer.cluster_word_weights = cww;
        layer.cluster_word_bias = cwb;

        let mut rw = vec![vec![0.0f32; HIDDEN]; 1];
        let mut rb = vec![0.0f32; 1];
        for d in 0..HIDDEN {
            rw[0][d] = wv(d + 3);
        }
        rb[0] = wv(9) * 0.5;
        layer.weights = rw.clone();
        layer.previous_weights = rw;
        layer.bias = rb;

        layer
    }

    /// Exercises the update path: after one training forward + `update_parameters`, the
    /// cluster word table (now on sparse AdamW, like the router) must actually move for the
    /// touched tokens and stay finite. Guards against NaN/Inf panics and dead updates.
    #[test]
    fn test_adaptive_update_parameters_moves_word_table() {
        let mut layer = build_controlled_layer();
        let input = base_input();

        let before = layer.cluster_word_weights.clone();
        let routing_before = layer.weights.clone();

        forward_loss(&mut layer, &input);
        layer.time_step = 1;
        layer.update_parameters();

        let after = &layer.cluster_word_weights;

        // All weights finite.
        for cluster in after {
            for row in cluster {
                for &w in row {
                    assert!(w.is_finite(), "word weight became non-finite: {w}");
                }
            }
        }

        // At least one word weight moved (the touched clusters received gradients).
        let mut max_word_delta = 0.0f32;
        for (cb, ca) in before.iter().zip(after.iter()) {
            for (rb, ra) in cb.iter().zip(ca.iter()) {
                for (&wb, &wa) in rb.iter().zip(ra.iter()) {
                    max_word_delta = max_word_delta.max((wa - wb).abs());
                }
            }
        }
        assert!(max_word_delta > 1e-7, "word table did not move after update (max delta {max_word_delta:.2e})");

        // Routing weights also moved (sanity that both groups update).
        let mut max_routing_delta = 0.0f32;
        for (rb, ra) in routing_before.iter().zip(layer.weights.iter()) {
            for (&wb, &wa) in rb.iter().zip(ra.iter()) {
                max_routing_delta = max_routing_delta.max((wa - wb).abs());
            }
        }
        println!("max word delta = {max_word_delta:.3e}, max routing delta = {max_routing_delta:.3e}");
        assert!(max_routing_delta > 1e-7, "routing weights did not move after update");
    }

    #[test]
    fn test_adaptive_train_vs_inference_logit_consistency() {
        let mut layer = build_single_tail_layer();
        // batch = 1, seq = 1.
        let mut input = vec![vec![vec![C::new(0.0, 0.0); HIDDEN]; 1]; 1];
        for d in 0..HIDDEN {
            input[0][0][d] = C::new(0.12 * d as f64 - 0.25, 0.0);
        }
        let pad = vec![vec![1u32]];
        let tgt = vec![vec![2u32]]; // token 2 -> tail cluster 1

        // Training forward.
        let mut li_train = LayerInput::new_default();
        li_train.set_input_batch(input.clone());
        li_train.set_padding_mask_batch(pad.clone());
        li_train.set_target_batch_ids(tgt.clone());
        li_train.set_top_k_size(4);
        li_train.set_total_valid_tokens(1);
        li_train.set_calculate_gradient(true);
        li_train.set_batch_size(1);
        let train_out = layer.forward(&li_train);
        let train_vals = train_out.get_output_batch();
        let train_idx = train_out.get_output_indices();

        // Inference forward (no targets, no gradient).
        let mut li_inf = LayerInput::new_default();
        li_inf.set_input_batch(input.clone());
        li_inf.set_padding_mask_batch(pad.clone());
        li_inf.set_top_k_size(4);
        li_inf.set_total_valid_tokens(1);
        li_inf.set_calculate_gradient(false);
        li_inf.set_batch_size(1);
        let inf_out = layer.forward(&li_inf);
        let inf_vals = inf_out.get_output_batch();
        let inf_idx = inf_out.get_output_indices();

        // Map token_id -> logit for both modes.
        let to_map = |vals: &Vec<Vec<Vec<C>>>, idx: &Vec<Vec<Vec<usize>>>| {
            let mut m = std::collections::HashMap::new();
            for (id, v) in idx[0][0].iter().zip(vals[0][0].iter()) {
                if v.re.is_finite() {
                    m.insert(*id, v.re);
                }
            }
            m
        };
        let tm = to_map(&train_vals, &train_idx);
        let im = to_map(&inf_vals, &inf_idx);

        println!("train logits: {tm:?}");
        println!("infer logits: {im:?}");

        assert_eq!(tm.len(), im.len(), "candidate sets differ between train and inference");
        for (id, tv) in &tm {
            let iv = im.get(id).unwrap_or_else(|| panic!("token {id} missing at inference"));
            assert!(
                (tv - iv).abs() < 1e-9,
                "token {id}: train logit {tv:+.6} != inference logit {iv:+.6} (scoring inconsistency)"
            );
        }

        // Top-1 must agree now that the scoring functions match.
        let train_top1 = train_idx[0][0].first().copied();
        let inf_top1 = inf_idx[0][0].first().copied();
        assert_eq!(train_top1, inf_top1, "train top-1 {train_top1:?} != inference top-1 {inf_top1:?}");
    }
}
