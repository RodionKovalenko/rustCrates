use crate::neural_networks::utils::activation::gauss;
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

pub fn check_ln_gauss(gauss_v_k: &Vec<Vec<Vec<f64>>>, p: &Vec<f64>) -> f64 {
    let mut sum: f64 = 0.0;

    for i in 0..gauss_v_k.len() {
        for j in 0..gauss_v_k[i].len() {
            let mut sum_partial = 0.0;
            for k in 0..p.len() {
                sum_partial += p[k] * gauss_v_k[i][j][k];
            }
            sum += sum_partial.ln();
        }
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
    // mean (2D array for k clusters, n features)
    let mut m_k: Vec<Vec<f64>> = vec![vec![0.0; data[0].len()]; m.len()];
    for k in 0..m.len() {
        for d in 0..m[k].len() {
            m_k[k][d] = m[k][d].clone().into(); // Copy each mean per feature
        }
    }

    // variance (3D array for k clusters, n features, n features)
    let mut s_k: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; data[0].len()]; data[0].len()]; m.len()];
    for k in 0..m.len() {
        for i in 0..data[0].len() {
            for j in 0..data[0].len() {
                s_k[k][i][j] = s[k][i][j].clone().into(); // Copy each variance/covariance matrix
                if s_k[k][i][j] == 0.0 {
                    s_k[k][i][j] = 0.01; // Avoid division by zero
                }
            }
        }
    }

    let mut gauss_v_k: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![0.0; m.len()]; data[0].len()]; data.len()];
    let mut p_ijk: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; m.len()]; data[0].len()]; data.len()];
    let mut p_k: Vec<f64> = vec![0.0; m.len()];
    let d = (data.len() * data[0].len()) as f64;

    // General probability p for each cluster
    let mut p: Vec<f64> = vec![1.0 / m.len() as f64; m.len()];

    for _ in 0..*n_iter {
        let mut p_ij: Vec<Vec<f64>> = vec![vec![0.0; data[0].len()]; data.len()];
        let mut p_ijk_sum: Vec<f64> = vec![0.0; m.len()];

        // Expectation step
        //----------------------------------------------------------------------
        // for i in 0..data.len() {
        //     for j in 0..data[i].len() {
        //         for k in 0..p.len() {
        //             gauss_v_k[i][j][k] = gauss(&data[i][j], &m_k[k], &s_k[k]);
        //             p_ij[i][j] += p[k] * gauss_v_k[i][j][k];
        //         }
        //         for k in 0..p.len() {
        //             p_ijk[i][j][k] = (p[k] * gauss_v_k[i][j][k]) / p_ij[i][j];
        //             p_ijk_sum[k] += p_ijk[i][j][k];
        //         }
        //     }
        // }

        let _ln_sum_check_before = check_ln_gauss(&gauss_v_k, &p);
        //----------------------------------------------------------------------

        // Maximization step
        //----------------------------------------------------------------------
        for k in 0..p.len() {
            p_k[k] = p_ijk_sum[k];

            // Update mean (mean for each cluster, each feature)
            let mut sum_k: Vec<f64> = vec![0.0; data[0].len()]; // Sum for each feature
            for i in 0..data.len() {
                for j in 0..data[i].len() {
                    for f in 0..data[i].len() {
                        sum_k[f] += data[i][j].clone().into() * p_ijk[i][j][k];
                    }
                }
            }

            for f in 0..data[0].len() {
                m_k[k][f] = sum_k[f] / p_k[k];
            }

            // Update variance (covariance matrix for each cluster)
            let mut sum_k: Vec<Vec<f64>> = vec![vec![0.0; data[0].len()]; data[0].len()];
            for i in 0..data.len() {
                for j in 0..data[i].len() {
                    for f1 in 0..data[0].len() {
                        for f2 in 0..data[0].len() {
                            sum_k[f1][f2] += (data[i][j].clone().into() - m_k[k][f1])
                                * (data[i][j].clone().into() - m_k[k][f2])
                                * p_ijk[i][j][k];
                        }
                    }
                }
            }

            for f1 in 0..data[0].len() {
                for f2 in 0..data[0].len() {
                    s_k[k][f1][f2] = sum_k[f1][f2] / p_k[k];
                }
            }

            p[k] = p_k[k] / d;
        }
        //----------------------------------------------------------------------

        // Check convergence
        let _ln_sum_check_after = check_ln_gauss(&gauss_v_k, &p);
        let tolerance = (_ln_sum_check_after - _ln_sum_check_before).abs();

        if tolerance < 0.000001 {
            println!("ln_sum_check_after: {:?}", tolerance);
            println!("Converged");
            break;
        }
    }

    (p, m_k, s_k)
}
