//! Complex layer norm and the split SiLU gate shared by the normal layers, the RM layers and (in tensor form) the
//! candle model.
//!
//! Gradients follow the convention of the backward passes: for a real loss L, the gradient of a complex value
//! z = u + iv is dL/du + i dL/dv. An upstream gradient G of an output Y means L = sum(Re(G) Re(Y) + Im(G) Im(Y)).
//!
//! Layer norm: y_f = (x_f - mu) / sigma * gamma_f + beta_f with the real variance var = mean(|x - mu|^2) and
//! sigma = sqrt(var + eps). The variance is the squared modulus, so it is never smaller than the spread of the values
//! (the complex square (x - mu)^2 lets phases cancel, and dividing by its near-zero root made the gradients explode).
//!
//! Split SiLU: silu(u) + i silu(v) with silu(t) = t / (1 + e^-t). Unlike the complex sigmoid it has no poles.

use crate::neural_networks::utils::dtype::{r, Real, C, ONE, ZERO};

/// Layer norm of one row: (output, mean, real variance).
pub fn layer_norm_row(x: &[C], gamma: &[C], beta: &[C], epsilon: f64) -> (Vec<C>, C, Real) {
    let n: Real = r(x.len() as f64);
    let mut mean = C::new(ZERO, ZERO);
    for v in x {
        mean += *v;
    }
    mean /= n;
    let var = x.iter().map(|v| (*v - mean).norm_sqr()).sum::<Real>() / n;
    let sigma = (var + r(epsilon)).sqrt();
    let out = x.iter().enumerate().map(|(f, v)| (*v - mean) / sigma * gamma[f] + beta[f]).collect();
    (out, mean, var)
}

/// Backward pass of `layer_norm_row` for the upstream gradient `upstream` of the output:
/// (input gradient, gamma gradient, beta gradient) of this row.
///
/// With d = x - mu, x_hat = d / sigma and u_f = G_f conj(gamma_f) (the gradient with respect to x_hat):
/// dL/dx_k = (u_k - mean(u)) / sigma - S d_k / (n sigma^3) with S = sum_f Re(conj(u_f) d_f),
/// dL/dgamma_f = G_f conj(x_hat_f), dL/dbeta_f = G_f.
pub fn layer_norm_row_backward(x: &[C], mean: C, var: Real, gamma: &[C], epsilon: f64, upstream: &[C]) -> (Vec<C>, Vec<C>, Vec<C>) {
    let n: Real = r(x.len() as f64);
    let sigma = (var + r(epsilon)).sqrt();
    let u: Vec<C> = upstream.iter().zip(gamma).map(|(g, gm)| *g * gm.conj()).collect();
    let mut u_mean = C::new(ZERO, ZERO);
    let mut s: Real = ZERO;
    for (k, v) in x.iter().enumerate() {
        let d = *v - mean;
        u_mean += u[k];
        s += (u[k].conj() * d).re;
    }
    u_mean /= n;
    let c = s / (n * sigma * sigma * sigma);
    let mut input_grad = Vec::with_capacity(x.len());
    let mut gamma_grad = Vec::with_capacity(x.len());
    for (k, v) in x.iter().enumerate() {
        let d = *v - mean;
        input_grad.push((u[k] - u_mean) / sigma - d * c);
        gamma_grad.push(upstream[k] * (d / sigma).conj());
    }
    (input_grad, gamma_grad, upstream.to_vec())
}

fn sigmoid_real(t: Real) -> Real {
    if t >= ZERO {
        ONE / (ONE + (-t).exp())
    } else {
        let e = t.exp();
        e / (ONE + e)
    }
}

fn silu_real(t: Real) -> Real {
    t * sigmoid_real(t)
}

fn silu_real_derivative(t: Real) -> Real {
    let s = sigmoid_real(t);
    s * (ONE + t * (ONE - s))
}

/// silu(Re z) + i silu(Im z).
pub fn split_silu(z: &C) -> C {
    C::new(silu_real(z.re), silu_real(z.im))
}

/// Gradient with respect to z of L = Re(conj(u) split_silu(z)), u being the gradient with respect to the output.
pub fn split_silu_backward(z: &C, u: &C) -> C {
    C::new(u.re * silu_real_derivative(z.re), u.im * silu_real_derivative(z.im))
}

/// Gated unit of the feed-forward layers: a * split_silu(b).
pub fn gated(a: &C, b: &C) -> C {
    *a * split_silu(b)
}

/// Gradients (dL/da, dL/db) of `gated` for the upstream gradient g of its output.
pub fn gated_backward(a: &C, b: &C, g: &C) -> (C, C) {
    (*g * split_silu(b).conj(), split_silu_backward(b, &(*g * a.conj())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L = sum Re(conj(G) y) and its central difference along the real and the imaginary part of every input.
    fn numerical<F: Fn(&[C]) -> Vec<C>>(f: F, x: &[C], g: &[C]) -> Vec<C> {
        let loss = |x: &[C]| f(x).iter().zip(g).map(|(y, g)| (g.conj() * y).re as f64).sum::<f64>();
        // f64 build: a tiny step; f32 build: a step large enough that rounding does not swamp the difference.
        let h = if std::mem::size_of::<Real>() == 8 { 1e-6 } else { 1e-2 };
        (0..x.len())
            .map(|k| {
                let mut grad = [0.0f64; 2];
                for (part, unit) in [C::new(r(h), ZERO), C::new(ZERO, r(h))].into_iter().enumerate() {
                    let (mut plus, mut minus) = (x.to_vec(), x.to_vec());
                    plus[k] += unit;
                    minus[k] -= unit;
                    grad[part] = (loss(&plus) - loss(&minus)) / (2.0 * h);
                }
                C::new(r(grad[0]), r(grad[1]))
            })
            .collect()
    }

    fn values(n: usize, seed: u64) -> Vec<C> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let mut next = || {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
                };
                C::new(r(next()), r(next()))
            })
            .collect()
    }

    fn assert_close(name: &str, expected: &[C], actual: &[C]) {
        // f64 build: central differences agree to ~1e-9; f32 build: to the precision of the f32 difference quotient.
        let tolerance = if std::mem::size_of::<Real>() == 8 { 1e-6 } else { 1e-2 };
        for (k, (e, a)) in expected.iter().zip(actual).enumerate() {
            let diff = (*e - *a).norm() as f64 / (e.norm() as f64).max(1.0);
            assert!(diff <= tolerance, "{name}[{k}]: numerical {e} vs analytical {a} (relative difference {diff:.2e})");
        }
    }

    #[test]
    fn layer_norm_gradients_match_finite_differences() {
        let n = 16;
        let (x, gamma, beta, g) = (values(n, 1), values(n, 2), values(n, 3), values(n, 4));
        let eps = 1e-8;
        let (_, mean, var) = layer_norm_row(&x, &gamma, &beta, eps);
        let (input_grad, gamma_grad, beta_grad) = layer_norm_row_backward(&x, mean, var, &gamma, eps, &g);
        assert_close("input", &numerical(|x| layer_norm_row(x, &gamma, &beta, eps).0, &x, &g), &input_grad);
        assert_close("gamma", &numerical(|gm| layer_norm_row(&x, gm, &beta, eps).0, &gamma, &g), &gamma_grad);
        assert_close("beta", &numerical(|bt| layer_norm_row(&x, &gamma, bt, eps).0, &beta, &g), &beta_grad);
    }

    #[test]
    fn layer_norm_output_has_zero_mean_and_unit_modulus_variance() {
        let x = values(32, 5);
        let ones = vec![C::new(ONE, ZERO); 32];
        let zeros = vec![C::new(ZERO, ZERO); 32];
        let (y, _, _) = layer_norm_row(&x, &ones, &zeros, 0.0);
        let mean: C = y.iter().sum::<C>() / r(32.0);
        let var = y.iter().map(|v| (*v - mean).norm_sqr()).sum::<Real>() / r(32.0);
        assert!(mean.norm() < r(1e-4) && (var - ONE).abs() < r(1e-4), "mean {mean}, variance {var}");
    }

    #[test]
    fn gated_gradients_match_finite_differences() {
        let n = 12;
        let (a, b, g) = (values(n, 6), values(n, 7), values(n, 8));
        let analytical: Vec<(C, C)> = (0..n).map(|k| gated_backward(&a[k], &b[k], &g[k])).collect();
        let da: Vec<C> = analytical.iter().map(|p| p.0).collect();
        let db: Vec<C> = analytical.iter().map(|p| p.1).collect();
        assert_close("a", &numerical(|a| a.iter().zip(&b).map(|(a, b)| gated(a, b)).collect(), &a, &g), &da);
        assert_close("b", &numerical(|b| a.iter().zip(b).map(|(a, b)| gated(a, b)).collect(), &b, &g), &db);
    }

    #[test]
    fn split_silu_is_bounded_where_the_complex_sigmoid_has_poles() {
        // The complex sigmoid 1 / (1 + e^-z) has poles at z = i pi (2k + 1).
        for k in [-3.0f64, -1.0, 1.0, 3.0] {
            let z = C::new(ZERO, r(k * std::f64::consts::PI));
            let y = split_silu(&z);
            assert!(y.norm().is_finite() && (y.norm() as f64) < 10.0, "split_silu({z}) = {y}");
        }
    }
}
