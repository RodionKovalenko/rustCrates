use num::Complex;

pub static B_1: f64 = 0.9;
pub static B_2: f64 = 0.999;
pub static EPSILON: f64 = 1e-4;
pub static WEIGHT_DECAY: f64 = 0.0001;
pub static MAX_NORM: f64 = 3.0;

// AdamW optimizer for complex weights (matrix)
pub fn calculate_adam_w(
    weights: &mut Vec<Vec<Complex<f64>>>,
    weight_gradients: &Vec<Vec<Complex<f64>>>,
    prev_m: &mut Vec<Vec<Complex<f64>>>,
    prev_v: &mut Vec<Vec<Complex<f64>>>, // real part used, imag=0
    learning_rate: f64,
    t: usize,
) -> Vec<Vec<Complex<f64>>> {
    let t = t.max(1) as i32;

    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            // 1️⃣ Gradient clipping (complex norm)
            let g_t = weight_gradients[i][j];

            // 2️⃣ First moment update
            prev_m[i][j] = prev_m[i][j] * B_1 + g_t * (1.0 - B_1);

            // 3️⃣ Second moment update: only .re is used, imag=0
            let g_norm_sqr = g_t.norm_sqr();
            let new_v_re = prev_v[i][j].re * B_2 + (1.0 - B_2) * g_norm_sqr;
            prev_v[i][j] = Complex::new(new_v_re, 0.0);

            // 4️⃣ Bias corrections
            let m_hat: Complex<f64> = prev_m[i][j] / (1.0 - B_1.powi(t));
            let v_hat: f64 = new_v_re / (1.0 - B_2.powi(t));

            // 5️⃣ Adaptive learning rate (real scalar)
            let denom: f64 = v_hat.sqrt() + EPSILON;
            let adaptive_lr: f64 = learning_rate / denom;

            // 6️⃣ AdamW update (with decoupled weight decay)
            weights[i][j] = weights[i][j] * (1.0 - learning_rate * WEIGHT_DECAY) - m_hat * adaptive_lr;
        }
    }

    weights.clone()
}

// AdamW optimizer for complex biases (vector)
pub fn calculate_adam_w_bias(
    bias: &[Complex<f64>],
    gradient: &[Complex<f64>],
    prev_m: &mut Vec<Complex<f64>>,
    prev_v: &mut Vec<Complex<f64>>, // real part used, imag=0
    learning_rate: f64,
    time_step: usize,
) -> Vec<Complex<f64>> {
    let mut bias = bias.to_vec();
    let t_i = time_step.max(1) as i32;

    for (i, b) in bias.iter_mut().enumerate() {
        let g_t = gradient[i];

        // 1️⃣ First moment update
        prev_m[i] = prev_m[i] * B_1 + g_t * (1.0 - B_1);

        // 2️⃣ Second moment update (real part only)
        let new_v_re = prev_v[i].re * B_2 + (1.0 - B_2) * g_t.norm_sqr();
        prev_v[i] = Complex::new(new_v_re, 0.0);

        // 3️⃣ Bias corrections
        let m_hat: Complex<f64> = prev_m[i] / (1.0 - B_1.powi(t_i));
        let v_hat: f64 = new_v_re / (1.0 - B_2.powi(t_i));

        // 4️⃣ Adaptive learning rate (real scalar)
        let denom: f64 = v_hat.sqrt() + EPSILON;
        let adaptive_lr: f64 = learning_rate / denom;

        // 5️⃣ AdamW update (decoupled weight decay)
        *b = *b * (1.0 - learning_rate * WEIGHT_DECAY) - m_hat * adaptive_lr;
    }

    bias
}

pub fn sgd_weights(weights: &mut Vec<Vec<Complex<f64>>>, weight_gradients: &Vec<Vec<Complex<f64>>>, learning_rate: f64) -> Vec<Vec<Complex<f64>>> {
    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            let mut g_t = weight_gradients[i][j];

            // optional gradient clipping
            let g_norm = g_t.norm();
            if g_norm > MAX_NORM {
                g_t *= MAX_NORM / g_norm;
            }

            // optional weight decay (L2 regularization)
            weights[i][j] = weights[i][j] * (1.0 - learning_rate * WEIGHT_DECAY) - g_t * Complex::new(learning_rate, 0.0);
        }
    }

    weights.clone()
}

pub fn sgd_bias(bias: &[Complex<f64>], gradient: &[Complex<f64>], learning_rate: f64) -> Vec<Complex<f64>> {
    let mut bias = bias.to_vec();

    for (b, &g_t) in bias.iter_mut().zip(gradient.iter()) {
        // optional gradient clipping
        let g_norm = g_t.norm();
        let mut g_t = g_t;
        if g_norm > MAX_NORM {
            g_t *= MAX_NORM / g_norm;
        }

        // optional weight decay (L2 regularization)
        *b = *b * (1.0 - learning_rate * WEIGHT_DECAY) - g_t * Complex::new(learning_rate, 0.0);
    }

    bias.to_vec()
}

pub fn average_gradient_polar(sums: &[Vec<Complex<f64>>], batch_size: f64) -> Vec<Vec<Complex<f64>>> {
    let rows = sums.len();
    let cols = sums[0].len();

    let mut out = vec![vec![Complex::new(0.0, 0.0); cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            let s = sums[i][j];
            // 1) Cartesian mean:
            let mean_re = s.re / batch_size;
            let mean_im = s.im / batch_size;
            // 2) Polar conversion:
            let mag = (mean_re.powi(2) + mean_im.powi(2)).sqrt();
            let ang = mean_im.atan2(mean_re);
            out[i][j] = Complex::from_polar(mag, ang);
        }
    }

    out
}

/// 1D version: same thing but for a flat Vec of summed gradients.
pub fn average_gradient_polar_1d(sums: &[Complex<f64>], batch_size: f64) -> Vec<Complex<f64>> {
    sums.iter()
        .map(|&s| {
            let mean_re = s.re / batch_size;
            let mean_im = s.im / batch_size;
            let mag = (mean_re.powi(2) + mean_im.powi(2)).sqrt();
            let ang = mean_im.atan2(mean_re);
            Complex::from_polar(mag, ang)
        })
        .collect()
}
