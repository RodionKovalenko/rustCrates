use num_traits::Zero;
use rand::{Rng, RngExt};

use crate::neural_networks::utils::dtype::{r, C, Real, ZERO};

/// Normalize complex vector; if norm is tiny, reinitialize randomly (better than returning same vector).
pub fn normalize(v: &Vec<C>) -> Vec<C> {
    let norm: Real = v.iter().map(|x| x.norm_sqr()).sum::<Real>().sqrt();
    if norm < r(1e-15) {
        // return a random normalized vector of the same size
        let mut rng = rand::rng();
        let mut random_vec: Vec<C> = (0..v.len())
            .map(|_| C::new(r(rng.random_range(-1.0f64..1.0f64)), r(rng.random_range(-1.0f64..1.0f64))))
            .collect();
        // normalize before returning
        let norm_r: Real = random_vec
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<Real>()
            .sqrt()
            .max(r(1e-12));
        random_vec.iter_mut().for_each(|x| *x /= C::new(norm_r, ZERO));
        random_vec
    } else {
        v.iter().map(|x| *x / C::new(norm, ZERO)).collect()
    }
}

/// Matrix-vector multiply (m: n x p, v: p)
pub fn matvecmul(m: &Vec<Vec<C>>, v: &Vec<C>) -> Vec<C> {
    let n = m.len();
    let p = m[0].len();
    assert_eq!(p, v.len());

    let mut res = vec![C::zero(); n];
    for i in 0..n {
        let mut sum = C::zero();
        for j in 0..p {
            sum += m[i][j] * v[j];
        }
        res[i] = sum;
    }
    res
}

/// Transpose matrix
pub fn transpose(m: &Vec<Vec<C>>) -> Vec<Vec<C>> {
    let n = m.len();
    let p = m[0].len();
    let mut t = vec![vec![C::zero(); n]; p];
    for i in 0..n {
        for j in 0..p {
            t[j][i] = m[i][j];
        }
    }
    t
}

/// Compute y = M^H * x without forming M^H explicitly.
/// M is n x p, x is length n, result y has length p:
/// y_j = sum_i conj(M[i][j]) * x_i
pub fn mat_conj_transpose_vecmul(m: &Vec<Vec<C>>, x: &Vec<C>) -> Vec<C> {
    let n = m.len();
    let p = m[0].len();
    assert_eq!(n, x.len());
    let mut y = vec![C::zero(); p];
    for j in 0..p {
        let mut sum = C::zero();
        for i in 0..n {
            sum += m[i][j].conj() * x[i];
        }
        y[j] = sum;
    }
    y
}

/// Deflation: M <- M - sigma * u * v^H
/// Here sigma is real (f64) expected non-negative.
pub fn deflate(m: &mut Vec<Vec<C>>, u: &Vec<C>, v: &Vec<C>, sigma: Real) {
    let n = m.len();
    let p = m[0].len();
    let s = C::new(sigma, ZERO);
    for i in 0..n {
        for j in 0..p {
            m[i][j] -= s * u[i] * v[j].conj();
        }
    }
}

/// Power iteration to find dominant singular triplet (sigma, u, v)
/// Returns sigma (f64, non-negative), u (len n), v (len p)
pub fn power_iteration_svd(m: &Vec<Vec<C>>, max_iter: usize, tol: Real) -> (Real, Vec<C>, Vec<C>) {
    let n = m.len();
    let p = m[0].len();
    let mut rng = rand::rng();

    // random v (length p)
    let mut v: Vec<C> = (0..p)
        .map(|_| C::new(r(rng.random_range(-1.0f64..1.0f64)), r(rng.random_range(-1.0f64..1.0f64))))
        .collect();
    v = normalize(&v);

    let mut u = vec![C::zero(); n];
    let mut sigma: Real = ZERO;

    for _ in 0..max_iter {
        // u = normalize( M * v )
        let mv = matvecmul(m, &v);
        let u_new = normalize(&mv);

        // v = normalize( M^H * u_new )
        let mtu = mat_conj_transpose_vecmul(m, &u_new);
        let v_new = normalize(&mtu);

        // sigma estimate = real(u_new^H * (M v))
        let mut s_complex: C = C::zero();
        for i in 0..n {
            s_complex += u_new[i].conj() * mv[i];
        }
        // take real/safe magnitude
        let s_val: Real = s_complex.re.abs(); // ensure non-negative

        // convergence check on v
        let diff: Real = v_new
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (*a - *b).norm())
            .sum();

        v = v_new;
        u = u_new;
        sigma = s_val;

        if diff < tol {
            break;
        }
    }

    (sigma, u, v)
}

/// Low rank approx: returns U (n x r) and V (r x p) such that M ≈ U * V
/// Conventions: U[:,k] = sqrt(sigma_k) * u_k, V[k,:] = sqrt(sigma_k) * conj(v_k)
pub fn low_rank_approx(m: &Vec<Vec<C>>, rank: usize, max_iter: usize, tol: Real) -> (Vec<Vec<C>>, Vec<Vec<C>>) {
    let n = m.len();
    let p = m[0].len();

    let mut residual = m.clone();

    let mut u_mat = vec![vec![C::zero(); rank]; n]; // n x r
    let mut v_mat = vec![vec![C::zero(); p]; rank]; // r x p

    for k in 0..rank {
        let (sigma, u, v) = power_iteration_svd(&residual, max_iter, tol);
        if sigma <= ZERO {
            break;
        }
        let s_sqrt = sigma.sqrt();

        // U[:,k] = sqrt(sigma) * u
        for i in 0..n {
            u_mat[i][k] = u[i] * C::new(s_sqrt, ZERO);
        }
        // V[k,:] = sqrt(sigma) * conj(v)
        for j in 0..p {
            v_mat[k][j] = v[j].conj() * C::new(s_sqrt, ZERO);
        }

        // deflate by sigma * u * v^H (sigma is f64)
        deflate(&mut residual, &u, &v, sigma);
    }

    (u_mat, v_mat)
}

pub fn reconstruction_error(w: &Vec<Vec<C>>, u: &Vec<Vec<C>>, v: &Vec<Vec<C>>) -> Real {
    let n = w.len();
    let p = w[0].len();

    // Compute W_approx = U * V
    let mut w_approx = vec![vec![C::zero(); p]; n];
    for i in 0..n {
        for j in 0..p {
            let mut sum = C::zero();
            for k in 0..u[0].len() {
                sum += u[i][k] * v[k][j];
            }
            w_approx[i][j] = sum;
        }
    }

    // Compute Frobenius norm of error W - W_approx
    let mut error_sum: Real = ZERO;
    for i in 0..n {
        for j in 0..p {
            let diff = w[i][j] - w_approx[i][j];
            error_sum += diff.norm_sqr();
        }
    }
    error_sum.sqrt()
}
