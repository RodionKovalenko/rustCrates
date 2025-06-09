use rand::rng;
use rand_distr::{Distribution, Normal};

pub struct Performer {
    projection_matrix: Vec<Vec<f64>>, // shape: (d, m)
    scale: f64,
    temperature: f64,
}

impl Performer {
    pub fn new(d: usize, m: usize, temperature: f64) -> Self {
        let mut rng = rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let scale = 1.0 / (m as f64).sqrt();

        // Create random Gaussian projection matrix: shape (d, m)
        let mut w = vec![vec![0.0; m]; d];
        for i in 0..d {
            for j in 0..m {
                w[i][j] = normal.sample(&mut rng);
            }
        }

        Performer { projection_matrix: w, scale, temperature }
    }

    /// φ(x) = exp((wᵀx - ||x||²/2) / temperature) for Performer
    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        let d = x[0].len();
        let m = self.projection_matrix[0].len();
        let mut features = vec![vec![0.0; m]; n];
        let epsilon = 1e-6;

        for i in 0..n {
            // Compute ||x||² / 2
            let x_norm_sq = x[i].iter().map(|v| v * v).sum::<f64>() / 2.0;

            for j in 0..m {
                // Compute dot product w_j^T x
                let mut dot = 0.0;
                for k in 0..d {
                    dot += x[i][k] * self.projection_matrix[k][j];
                }
                // Scale and temperature inside exp
                let scaled = (dot - x_norm_sq) / self.temperature;
                features[i][j] = (scaled).exp() * self.scale + epsilon;
            }
        }

        features
    }

    pub fn approximate_attention(&self, q: &[Vec<f64>], k: &[Vec<f64>], v: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let q_phi = self.transform(q); // (n_q, m)
        let k_phi = self.transform(k); // (n_k, m)

        let m = k_phi[0].len();
        let d_v = v[0].len();
        let n_q = q.len();
        let n_k = k.len();

        // Compute φ(K)^T V: (m × d_v)
        let mut k_phi_t_v = vec![vec![0.0; d_v]; m];
        for i in 0..n_k {
            for j in 0..m {
                for d in 0..d_v {
                    k_phi_t_v[j][d] += k_phi[i][j] * v[i][d];
                }
            }
        }

        // Compute φ(K)^T 1: (m)
        let mut k_phi_sum = vec![0.0; m];
        for i in 0..n_k {
            for j in 0..m {
                k_phi_sum[j] += k_phi[i][j];
            }
        }

        // Final attention output: φ(Q)(φ(K)^T V) / φ(Q)(φ(K)^T 1)
        let mut output = vec![vec![0.0; d_v]; n_q];
        for i in 0..n_q {
            // Compute denominator: scalar
            let denom: f64 = q_phi[i].iter().zip(&k_phi_sum).map(|(qf, kf)| qf * kf).sum::<f64>().max(1e-6); // avoid divide-by-zero

            for j in 0..d_v {
                let mut num = 0.0;
                for k in 0..m {
                    num += q_phi[i][k] * k_phi_t_v[k][j];
                }
                output[i][j] = num / denom;
            }
        }

        output
    }
}
