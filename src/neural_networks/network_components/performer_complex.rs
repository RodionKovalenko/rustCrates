use num_complex::Complex;
use rand::rng;
use rand_distr::{Distribution, Normal};

pub struct PerformerComplex {
    projection_matrix: Vec<Vec<Complex<f64>>>, // shape: (d, m)
    scale: f64,
    temperature: f64,
}

impl PerformerComplex {
    pub fn new(d: usize, m: usize, temperature: f64) -> Self {
        let mut rng = rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let scale = 1.0 / (m as f64).sqrt();

        let mut w = vec![vec![Complex::new(0.0, 0.0); m]; d];
        for i in 0..d {
            for j in 0..m {
                w[i][j] = Complex::new(normal.sample(&mut rng), normal.sample(&mut rng));
            }
        }

        PerformerComplex { projection_matrix: w, scale, temperature }
    }

    /// φ(x) = exp((wᵀx - ||x||²/2) / temperature) * scale
    pub fn transform(&self, x: &[Vec<Complex<f64>>]) -> Vec<Vec<Complex<f64>>> {
        let n = x.len();
        let d = x[0].len();
        let m = self.projection_matrix[0].len();
        let mut features = vec![vec![Complex::new(0.0, 0.0); m]; n];
        let epsilon = Complex::new(1e-6, 0.0);

        for i in 0..n {
            let x_norm_sq: f64 = x[i].iter().map(|v| v.norm_sqr()).sum::<f64>() / 2.0;

            for j in 0..m {
                let mut dot = Complex::new(0.0, 0.0);
                for k in 0..d {
                    dot += x[i][k] * self.projection_matrix[k][j];
                }
                let scaled = (dot - Complex::new(x_norm_sq, 0.0)) / self.temperature;
                features[i][j] = scaled.exp() * self.scale + epsilon;
            }
        }

        features
    }

    pub fn approximate_attention(&self, q: &[Vec<Complex<f64>>], k: &[Vec<Complex<f64>>], v: &[Vec<Complex<f64>>]) -> Vec<Vec<Complex<f64>>> {
        let q_phi = self.transform(q); // (n_q, m)
        let k_phi = self.transform(k); // (n_k, m)

        let m = k_phi[0].len();
        let d_v = v[0].len();
        let n_q = q.len();
        let n_k = k.len();

        // φ(K)^T V: (m × d_v)
        let mut k_phi_t_v = vec![vec![Complex::new(0.0, 0.0); d_v]; m];
        for i in 0..n_k {
            for j in 0..m {
                for d in 0..d_v {
                    k_phi_t_v[j][d] += k_phi[i][j] * v[i][d];
                }
            }
        }

        // φ(K)^T 1: (m)
        let mut k_phi_sum = vec![Complex::new(0.0, 0.0); m];
        for i in 0..n_k {
            for j in 0..m {
                k_phi_sum[j] += k_phi[i][j];
            }
        }

        // Final attention output: φ(Q)(φ(K)^T V) / φ(Q)(φ(K)^T 1)
        let mut output = vec![vec![Complex::new(0.0, 0.0); d_v]; n_q];
        for i in 0..n_q {
            let denom: Complex<f64> = q_phi[i].iter().zip(&k_phi_sum).map(|(qf, kf)| qf * kf).sum::<Complex<f64>>();
            let denom = if denom.norm_sqr() < 1e-12 { Complex::new(1e-6, 0.0) } else { denom };

            for j in 0..d_v {
                let mut num = Complex::new(0.0, 0.0);
                for k in 0..m {
                    num += q_phi[i][k] * k_phi_t_v[k][j];
                }
                output[i][j] = num / denom;
            }
        }

        output
    }
}
