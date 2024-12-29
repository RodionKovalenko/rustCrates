use crate::neural_networks::utils::activation::gauss;
use crate::utils::linalg::{get_determinant, pseudoinverse};
use std::f64::consts::PI;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

pub fn exp_max_1d<T, V, G>(
    data: &Vec<T>,
    m: &Vec<V>,
    s: &Vec<G>,
    n_iter: &i32,
) -> (Vec<f64>, Vec<f64>, Vec<f64>)
where
    T: Debug
        + Clone
        + Into<f64>
        + Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + Sub<Output = T>,
    V: Debug + Clone + Into<f64> + Mul<Output = V> + Add<Output = V> + Div<Output = V>,
    G: Debug + Clone + Into<f64> + Mul<Output = G> + Add<Output = G> + Div<Output = G>,
{
    // mean
    let mut m_k: Vec<f64> = vec![0.0; m.len()];
    for i in 0..m.len() {
        m_k[i] = m[i].clone().into();
    }

    // variance
    let mut s_k: Vec<f64> = vec![0.0; m.len()];
    for i in 0..m.len() {
        s_k[i] = s[i].clone().into();

        if s_k[i] == 0.0 {
            s_k[i] = 0.01;
        }
    }

    let mut gauss_v_k: Vec<Vec<f64>> = vec![vec![0.0; m.len()]; data.len()];
    let mut p_ijk: Vec<Vec<f64>> = vec![vec![0.0; m.len()]; data.len()];
    let mut p_k: Vec<f64> = vec![0.0; m.len()];
    let d = data.len() as f64;

    // general propability p pro cluster
    let mut p: Vec<f64> = vec![1.0 / m.len() as f64; m.len()];

    for _ in 0..*n_iter {
        let mut p_ij: Vec<f64> = vec![0.0; data.len()];
        let mut p_ijk_sum: Vec<f64> = vec![0.0; m.len()];

        // expectation step
        //----------------------------------------------------------------------
        for i in 0..data.len() {
            for k in 0..p.len() {
                gauss_v_k[i][k] = gauss(&data[i], &m_k[k], &s_k[k]);
                p_ij[i] += p[k] * gauss_v_k[i][k];
            }
            for k in 0..p.len() {
                p_ijk[i][k] = (p[k] * gauss_v_k[i][k]) / p_ij[i];
                p_ijk_sum[k] += p_ijk[i][k];
            }
        }

        let _ln_sum_check_before = check_ln_gauss_1d(&gauss_v_k, &p);
        //----------------------------------------------------------------------

        // maximization step
        //----------------------------------------------------------------------
        for k in 0..p.len() {
            p_k[k] = p_ijk_sum[k];

            // Update mean
            let mut sum_k: f64 = 0.0;
            for i in 0..data.len() {
                sum_k += data[i].clone().into() * p_ijk[i][k];
            }

            m_k[k] = sum_k / p_k[k];

            // Update variance
            sum_k = 0.0;
            for i in 0..data.len() {
                sum_k += (data[i].clone().into() - m_k[k]).powf(2.0) * p_ijk[i][k];
            }

            s_k[k] = sum_k / p_k[k];

            p[k] = p_k[k] / d;
        }
        //----------------------------------------------------------------------

        // check convergence
        let _ln_sum_check_after = check_ln_gauss_1d(&gauss_v_k, &p);
        let tolarence = (_ln_sum_check_after - _ln_sum_check_before).abs();

        if tolarence < 0.00000001 {
            println!("ln_sum_check_after: {:?}", tolarence);
            println!("Converged");
            break;
        }
    }

    (p, m_k, s_k)
}

pub fn check_ln_gauss_1d(gauss_v_k: &Vec<Vec<f64>>, p: &Vec<f64>) -> f64 {
    let mut sum: f64 = 0.0;

    for i in 0..gauss_v_k.len() {
        let mut sum_partial = 0.0;
        for k in 0..p.len() {
            sum_partial += p[k] * gauss_v_k[i][k];
        }
        sum += sum_partial.ln();
    }

    sum
}

pub fn check_ln_gauss(gauss_v_k: &Vec<Vec<f64>>, p: &Vec<f64>) -> f64 {
    let mut sum: f64 = 0.0;

    for i in 0..gauss_v_k.len() {
        let mut sum_partial = 0.0;
        for j in 0..gauss_v_k[i].len() {
            sum_partial += p[j] * gauss_v_k[i][j];
        }
        sum += sum_partial.ln();
    }

    sum
}

pub fn exp_max_2d<T, V, G>(
    data: &Vec<Vec<T>>,
    m: &Vec<Vec<V>>,
    s: &Vec<Vec<Vec<G>>>,
    n_iter: &i32,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>)
where
    T: Debug
        + Clone
        + Into<f64>
        + Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + Sub<Output = T>,
    V: Debug + Clone + Into<f64> + Mul<Output = V> + Add<Output = V> + Div<Output = V>,
    G: Debug + Clone + Into<f64> + Mul<Output = G> + Add<Output = G> + Div<Output = G>,
{
    let n = data.len();
    let d = data[0].len();
    let k = m.len(); // Number of clusters (number of means)

    // Initialize the mixing coefficients, means, and covariances
    let mut pi: Vec<f64> = vec![1.0 / k as f64; k];

    let data_f64: Vec<Vec<f64>> = data
        .iter() // Iterate over the outer Vec<Vec<T>>
        .map(|inner| {
            inner
                .iter() // Iterate over each inner Vec<T>
                .map(|x| Into::<f64>::into(x.clone())) // Convert each element to f64
                .collect::<Vec<f64>>() // Collect them into a Vec<f64>
        })
        .collect();

    let mut mu: Vec<Vec<f64>> = m
        .iter()
        .map(|mean| mean.iter().map(|x| Into::<f64>::into(x.clone())).collect())
        .collect();

    // Convert s (covariances) into Vec<Vec<Vec<f64>>>
    let mut sigma: Vec<Vec<Vec<f64>>> = s
        .iter()
        .map(|cov| {
            cov.iter()
                .map(|x| x.iter().map(|y| Into::<f64>::into(y.clone())).collect())
                .collect()
        })
        .collect();

    let mut log_likelihoods = Vec::new();

    for iteration in 0..*n_iter {
        // E-step: Compute responsibilities (resp)
        let mut resp: Vec<Vec<f64>> = vec![vec![0.0; k]; n];

        // Fill the responsibilities matrix
        for i in 0..k {
            let mu_i = &mu[i]; // mean for the i-th cluster
            let sigma_i = &sigma[i]; // covariance for the i-th cluster

            for j in 0..n {
                let likelihood = compute_likelihood(&data[j], mu_i, sigma_i); 
                resp[j][i] = pi[i] * likelihood;
            }
        }

        // Normalize responsibilities
        for i in 0..n {
            let sum: f64 = resp[i].iter().sum();
            for j in 0..k {
                resp[i][j] /= sum;
            }
        }

        // M-step: Update the parameters (means, covariances, and pi)
        let mut n_k: Vec<f64> = vec![0.0; k];
        for i in 0..k {
            // Update means
            let mut mu_new: Vec<f64> = vec![0.0; d];
            for j in 0..n {
                let weight = resp[j][i];
                for m in 0..d {
                    mu_new[m] += weight * data_f64[j][m];
                }
            }

            n_k[i] = resp.iter().map(|r| r[i]).sum();
            for m in 0..d {
                mu_new[m] /= n_k[i];
            }
            mu[i] = mu_new;

            // Update covariances (s)
            let mut sigma_new: Vec<Vec<f64>> = vec![vec![0.0; d]; d];
            for j in 0..n {
                let weight = resp[j][i];

                // Compute (data[j] - mu[i])
                let diff: Vec<f64> = data_f64[j]
                    .iter()
                    .zip(mu[i].iter())
                    .map(|(x, m)| x - m)
                    .collect();

                // Weighted outer product of (diff * diff^T)
                for row in 0..d {
                    for col in 0..d {
                        sigma_new[row][col] += weight * diff[row] * diff[col];
                    }
                }
            }

            // Normalize covariance by Nk[i]
            for row in 0..d {
                for col in 0..d {
                    sigma_new[row][col] /= n_k[i];
                }
            }
            sigma[i] = sigma_new;
        }

        // Update mixing coefficients
        for i in 0..k {
            pi[i] = n_k[i] / n as f64;
        }

        // Compute log-likelihood
        let log_likelihood: f64 = (0..n)
            .map(|i| {
                (0..k)
                    .map(|j| pi[j] * compute_likelihood(&data[i], &mu[j], &sigma[j]))
                    .sum::<f64>()
                    .ln()
            })
            .sum();

        log_likelihoods.push(log_likelihood);

        let iter_u = iteration as usize;

        // Check for convergence (could break here based on tolerance)
        if iteration > 0 {
            let log_likeli_diff: f64 =
                (log_likelihoods[iter_u] - log_likelihoods[iter_u - 1]).abs();

            println!("log_likelihoods diff: {}", &log_likeli_diff);
            if log_likeli_diff < 1e-6 {
                println!("Converged at iteration: {}", iteration);
                break;
            }
        }
    }

    // Return the final parameters
    (pi, mu, sigma)
}

fn compute_likelihood<T, V, G>(data: &Vec<T>, mu: &Vec<V>, sigma: &Vec<Vec<G>>) -> f64
where
    T: Into<f64> + Clone,
    V: Into<f64> + Clone,
    G: Into<f64> + Clone,
{
    let d = data.len(); // Dimensionality of the data point (number of features)

    // Convert data point and mean vector to f64 for calculation
    let data_point: Vec<f64> = data.iter().map(|x| x.clone().into()).collect();
    let mu_vec: Vec<f64> = mu.iter().map(|x| x.clone().into()).collect();

    // Convert covariance matrix elements to f64
    let sigma_m: Vec<Vec<f64>> = sigma
        .iter()
        .map(|x| x.iter().map(|x| x.clone().into()).collect())
        .collect();

    // Compute the difference vector (x - mu)
    let diff: Vec<f64> = data_point
        .iter()
        .zip(mu_vec.iter())
        .map(|(data_val, mu_val)| data_val - mu_val)
        .collect();

    // Calculate the covariance matrix inverse and determinantz
    let sigma_inv = pseudoinverse(&sigma_m).unwrap(); // Ensure robust error handling
    let mut sigma_det = get_determinant(&sigma_m); // Ensure this handles edge cases (singular matrices)

    if sigma_det == 0.0 {
        sigma_det = 0.001;
    }

    // Compute the quadratic form: (x - mu)^T * Sigma_inv * (x - mu)
    // Step 1: (x - mu) * Sigma_inv, result is a vector
    let diff_times_sigma_inv: Vec<f64> = (0..d)
        .map(|i| (0..d).map(|j| diff[j] * sigma_inv[j][i]).sum()) // Matrix-vector multiplication
        .collect();

    // Step 2: (x - mu)^T * result, this will be a scalar (dot product)
    let quadratic_form = diff
        .iter()
        .zip(diff_times_sigma_inv.iter())
        .map(|(diff_val, sigma_inv_val)| diff_val * sigma_inv_val)
        .sum::<f64>();

    // Compute the multivariate normal PDF (likelihood)
    let constant = 1.0 / ((2.0 * PI).powf(d as f64 / 2.0) * sigma_det.sqrt());

    // Now quadratic_form is a scalar, we can safely multiply it with -0.5
    let exponent = (-0.5) * quadratic_form;

    // Compute the final likelihood: constant * exp(exponent)
    constant * exponent.exp() // This is the likelihood of the data point
}
