// Gradient-check tests for the EmlLinearLayer output head.
//
// Goal: verify that the analytical gradients produced by the teacher-forced training
// forward pass (grad w.r.t. the hidden input H, the four projection matrices, and the
// four temperature scalars) match finite-difference numerical gradients — the same
// methodology `test_linear_layer.rs` / `test_adaptive_linear_layer.rs` use for the
// existing heads.
//
// The layer is built small and fully deterministic. The teacher-forced path gathers a
// fixed cluster slab per target, so the active candidate set never changes under a small
// perturbation — a precondition for a valid numerical gradient check.
#[cfg(test)]
mod test_eml_linear_layer {
    use crate::neural_networks::{
        network_components::layer_input_struct::LayerInput,
        network_layers::eml_linear_layer::EmlLinearLayer,
        utils::dtype::{Real, C},
    };

    const D_MODEL: usize = 3;
    const VOCAB: usize = 6; // -> M = ceil(sqrt(6)) = 3, K = ceil(6/3) = 2, all slots valid

    /// batch = 1, seq = 2, hidden = D_MODEL. Real-only (the layer reads `.re`).
    fn base_input() -> Vec<Vec<Vec<C>>> {
        let mut input = vec![vec![vec![C::new(0.0, 0.0); D_MODEL]; 2]; 1];
        for s in 0..2 {
            for d in 0..D_MODEL {
                input[0][s][d] = C::new(0.17 * (s * D_MODEL + d) as f64 - 0.3, 0.0);
            }
        }
        input
    }

    // Two targets in distinct clusters so both word-slabs and both clusters are exercised.
    //   row 0 target = 1 -> cluster 0, word 1
    //   row 1 target = 4 -> cluster 1, word 1
    fn targets() -> Vec<Vec<u32>> {
        vec![vec![1, 4]]
    }

    fn padding() -> Vec<Vec<u32>> {
        vec![vec![1, 1]]
    }

    const TOTAL_VALID: usize = 2;

    fn forward_loss(layer: &mut EmlLinearLayer, input: &Vec<Vec<Vec<C>>>) -> f64 {
        let mut li = LayerInput::new_default();
        li.set_input_batch(input.clone());
        li.set_padding_mask_batch(padding());
        li.set_target_batch_ids(targets());
        li.set_total_valid_tokens(TOTAL_VALID);
        li.set_calculate_gradient(true);
        li.set_batch_size(1);

        let out = layer.forward(&li);
        out.get_cross_entropy_loss_batch().iter().flatten().flatten().map(|c| c.re as f64).sum()
    }

    fn rel_err(a: f64, n: f64) -> f64 {
        (a - n).abs() / (a.abs().max(n.abs()).max(1e-8))
    }

    #[test]
    fn test_eml_input_gradient() {
        let mut layer = EmlLinearLayer::new(0.01, D_MODEL, VOCAB);
        let input = base_input();

        let loss = forward_loss(&mut layer, &input);
        assert!(loss.is_finite() && loss > 0.0, "loss must be finite and positive, got {loss}");
        let analytical = layer.gradient.as_ref().expect("training forward stores a gradient").get_gradient_input_batch();

        let eps = 1e-5;
        let mut max_rel: f64 = 0.0;
        for s in 0..2 {
            for d in 0..D_MODEL {
                let mut ip = input.clone();
                ip[0][s][d].re += eps as Real;
                let lp = forward_loss(&mut layer, &ip);

                let mut im = input.clone();
                im[0][s][d].re -= eps as Real;
                let lm = forward_loss(&mut layer, &im);

                let num = (lp - lm) / (2.0 * eps);
                let ana = analytical[0][s][d].re as f64;
                let e = rel_err(ana, num);
                println!("dH[{s}][{d}] analytical={ana:+.6} numerical={num:+.6} rel_err={e:.2e}");
                max_rel = max_rel.max(e);
            }
        }
        assert!(max_rel < 1e-3, "input gradient mismatch, max rel err = {max_rel:.3e}");
    }

    #[test]
    fn test_eml_projection_gradients() {
        let mut layer = EmlLinearLayer::new(0.01, D_MODEL, VOCAB);
        let input = base_input();
        forward_loss(&mut layer, &input);

        // Snapshot analytical projection grads (immutable borrow released before perturbing).
        let g_uc = layer.grad_w_uc().expect("grads stored").clone();
        let g_vc = layer.grad_w_vc().expect("grads stored").clone();
        let g_uw = layer.grad_w_uw().expect("grads stored").clone();
        let g_vw = layer.grad_w_vw().expect("grads stored").clone();

        let eps = 1e-4f64;
        let mut max_rel: f64 = 0.0;

        // Closure-free perturbation over each of the four matrices by name.
        for which in 0..4 {
            let (rows, cols) = (D_MODEL, layer.d_coord);
            for i in 0..rows {
                for j in 0..cols {
                    let saved = match which {
                        0 => layer.w_uc[i][j],
                        1 => layer.w_vc[i][j],
                        2 => layer.w_uw[i][j],
                        _ => layer.w_vw[i][j],
                    };
                    let set = |layer: &mut EmlLinearLayer, val: Real| match which {
                        0 => layer.w_uc[i][j] = val,
                        1 => layer.w_vc[i][j] = val,
                        2 => layer.w_uw[i][j] = val,
                        _ => layer.w_vw[i][j] = val,
                    };

                    set(&mut layer, saved + eps as Real);
                    let lp = forward_loss(&mut layer, &input);
                    set(&mut layer, saved - eps as Real);
                    let lm = forward_loss(&mut layer, &input);
                    set(&mut layer, saved);

                    let num = (lp - lm) / (2.0 * eps);
                    let ana = match which {
                        0 => g_uc[i][j],
                        1 => g_vc[i][j],
                        2 => g_uw[i][j],
                        _ => g_vw[i][j],
                    } as f64;
                    let e = rel_err(ana, num);
                    if ana.abs() > 1e-9 || num.abs() > 1e-9 {
                        let name = ["W_uc", "W_vc", "W_uw", "W_vw"][which];
                        println!("d{name}[{i}][{j}] analytical={ana:+.6} numerical={num:+.6} rel_err={e:.2e}");
                        max_rel = max_rel.max(e);
                    }
                }
            }
        }
        assert!(max_rel < 5e-3, "projection gradient mismatch, max rel err = {max_rel:.3e}");
    }

    #[test]
    fn test_eml_scalar_gradients() {
        let mut layer = EmlLinearLayer::new(0.01, D_MODEL, VOCAB);
        let input = base_input();
        forward_loss(&mut layer, &input);
        let (g_ac, g_bc, g_aw, g_bw) = layer.grad_scalars().expect("grads stored");

        let eps = 1e-4f64;
        let mut max_rel: f64 = 0.0;
        for which in 0..4 {
            let saved = match which {
                0 => layer.alpha_c,
                1 => layer.beta_c,
                2 => layer.alpha_w,
                _ => layer.beta_w,
            };
            let set = |layer: &mut EmlLinearLayer, val: Real| match which {
                0 => layer.alpha_c = val,
                1 => layer.beta_c = val,
                2 => layer.alpha_w = val,
                _ => layer.beta_w = val,
            };
            set(&mut layer, saved + eps as Real);
            let lp = forward_loss(&mut layer, &input);
            set(&mut layer, saved - eps as Real);
            let lm = forward_loss(&mut layer, &input);
            set(&mut layer, saved);

            let num = (lp - lm) / (2.0 * eps);
            let ana = [g_ac, g_bc, g_aw, g_bw][which] as f64;
            let name = ["alpha_c", "beta_c", "alpha_w", "beta_w"][which];
            let e = rel_err(ana, num);
            println!("d{name} analytical={ana:+.6} numerical={num:+.6} rel_err={e:.2e}");
            max_rel = max_rel.max(e);
        }
        assert!(max_rel < 5e-3, "scalar gradient mismatch, max rel err = {max_rel:.3e}");
    }
}
