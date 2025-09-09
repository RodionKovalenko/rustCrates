use num_complex::Complex;
use num_traits::Zero;
use rand::Rng;

/// Normalize complex vector; if norm is tiny, reinitialize randomly (better than returning same vector).
pub fn normalize(v: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    if norm < 1e-15 {
        // return a random normalized vector of the same size
        let mut rng = rand::rng();
        let mut random_vec: Vec<Complex<f64>> = (0..v.len()).map(|_| Complex::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))).collect();
        // normalize before returning
        let norm_r = random_vec.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt().max(1e-12);
        random_vec.iter_mut().for_each(|x| *x /= Complex::new(norm_r, 0.0));
        random_vec
    } else {
        v.iter().map(|x| *x / Complex::new(norm, 0.0)).collect()
    }
}

/// Matrix-vector multiply (m: n x p, v: p)
pub fn matvecmul(m: &Vec<Vec<Complex<f64>>>, v: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = m.len();
    let p = m[0].len();
    assert_eq!(p, v.len());

    let mut res = vec![Complex::zero(); n];
    for i in 0..n {
        let mut sum = Complex::zero();
        for j in 0..p {
            sum += m[i][j] * v[j];
        }
        res[i] = sum;
    }
    res
}

/// Transpose matrix
pub fn transpose(m: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    let n = m.len();
    let p = m[0].len();
    let mut t = vec![vec![Complex::zero(); n]; p];
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
pub fn mat_conj_transpose_vecmul(m: &Vec<Vec<Complex<f64>>>, x: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = m.len();
    let p = m[0].len();
    assert_eq!(n, x.len());
    let mut y = vec![Complex::zero(); p];
    for j in 0..p {
        let mut sum = Complex::zero();
        for i in 0..n {
            sum += m[i][j].conj() * x[i];
        }
        y[j] = sum;
    }
    y
}

/// Deflation: M <- M - sigma * u * v^H
/// Here sigma is real (f64) expected non-negative.
pub fn deflate(m: &mut Vec<Vec<Complex<f64>>>, u: &Vec<Complex<f64>>, v: &Vec<Complex<f64>>, sigma: f64) {
    let n = m.len();
    let p = m[0].len();
    let s = Complex::new(sigma, 0.0);
    for i in 0..n {
        for j in 0..p {
            m[i][j] -= s * u[i] * v[j].conj();
        }
    }
}

/// Power iteration to find dominant singular triplet (sigma, u, v)
/// Returns sigma (f64, non-negative), u (len n), v (len p)
pub fn power_iteration_svd(m: &Vec<Vec<Complex<f64>>>, max_iter: usize, tol: f64) -> (f64, Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let n = m.len();
    let p = m[0].len();
    let mut rng = rand::rng();

    // random v (length p)
    let mut v: Vec<Complex<f64>> = (0..p).map(|_| Complex::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))).collect();
    v = normalize(&v);

    let mut u = vec![Complex::zero(); n];
    let mut sigma = 0.0f64;

    for _ in 0..max_iter {
        // u = normalize( M * v )
        let mv = matvecmul(m, &v);
        let u_new = normalize(&mv);

        // v = normalize( M^H * u_new )
        let mtu = mat_conj_transpose_vecmul(m, &u_new);
        let v_new = normalize(&mtu);

        // sigma estimate = real(u_new^H * (M v))
        let mut s_complex: Complex<f64> = Complex::zero();
        for i in 0..n {
            s_complex += u_new[i].conj() * mv[i];
        }
        // take real/safe magnitude
        let s_val: f64 = s_complex.re.abs(); // ensure non-negative

        // convergence check on v
        let diff: f64 = v_new.iter().zip(v.iter()).map(|(a, b)| (*a - *b).norm()).sum();

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
pub fn low_rank_approx(m: &Vec<Vec<Complex<f64>>>, rank: usize, max_iter: usize, tol: f64) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>) {
    let n = m.len();
    let p = m[0].len();

    let mut residual = m.clone();

    let mut u_mat = vec![vec![Complex::zero(); rank]; n]; // n x r
    let mut v_mat = vec![vec![Complex::zero(); p]; rank]; // r x p

    for k in 0..rank {
        let (sigma, u, v) = power_iteration_svd(&residual, max_iter, tol);
        if sigma <= 0.0 {
            break;
        }
        let s_sqrt = sigma.sqrt();

        // U[:,k] = sqrt(sigma) * u
        for i in 0..n {
            u_mat[i][k] = u[i] * Complex::new(s_sqrt, 0.0);
        }
        // V[k,:] = sqrt(sigma) * conj(v)
        for j in 0..p {
            v_mat[k][j] = v[j].conj() * Complex::new(s_sqrt, 0.0);
        }

        // deflate by sigma * u * v^H (sigma is f64)
        deflate(&mut residual, &u, &v, sigma);
    }

    (u_mat, v_mat)
}

pub fn reconstruction_error(w: &Vec<Vec<Complex<f64>>>, u: &Vec<Vec<Complex<f64>>>, v: &Vec<Vec<Complex<f64>>>) -> f64 {
    let n = w.len();
    let p = w[0].len();

    // Compute W_approx = U * V
    let mut w_approx = vec![vec![Complex::zero(); p]; n];
    for i in 0..n {
        for j in 0..p {
            let mut sum = Complex::zero();
            for k in 0..u[0].len() {
                sum += u[i][k] * v[k][j];
            }
            w_approx[i][j] = sum;
        }
    }

    // Compute Frobenius norm of error W - W_approx
    let mut error_sum = 0.0;
    for i in 0..n {
        for j in 0..p {
            let diff = w[i][j] - w_approx[i][j];
            error_sum += diff.norm_sqr();
        }
    }
    error_sum.sqrt()
}
