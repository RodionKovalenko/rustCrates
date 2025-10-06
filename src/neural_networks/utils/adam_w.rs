use num::Complex;

pub static B_1: f64 = 0.9;
pub static B_2: f64 = 0.999;
pub static EPSILON: f64 = 1e-6;
pub static WEIGHT_DECAY: f64 = 0.01;
pub static MAX_NORM: f64 = 10.0;
pub static WARMUP_STEPS: usize = 50;

// Assume this helper function exists or add it
pub fn is_nan_or_inf(c: &Complex<f64>) -> bool {
    c.re.is_nan() || c.re.is_infinite() || c.im.is_nan() || c.im.is_infinite() || c.norm_sqr().is_nan() || c.norm_sqr().is_infinite() || c.norm_sqr() > 1e10
    // Add threshold for very large values
}

// AdamW optimizer for complex weights (matrix)
pub fn calculate_adam_w(
    weights: &mut Vec<Vec<Complex<f64>>>,
    weight_gradients: &Vec<Vec<Complex<f64>>>,
    prev_m: &mut Vec<Vec<Complex<f64>>>,
    prev_v: &mut Vec<Vec<Complex<f64>>>, // real part used, imag=0
    learning_rate: f64,
    t: usize,
) {
    let t = t.max(1) as i32;
    let current_lr = get_current_learning_rate(learning_rate, t as usize);

    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            // 1️⃣ Gradient clipping (complex norm)
            let g_t = weight_gradients[i][j];

            if is_nan_or_inf(&g_t) {
                continue;
            }

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
            let adaptive_lr: f64 = current_lr / denom;

            // 6️⃣ AdamW update (with decoupled weight decay)
            weights[i][j] = weights[i][j] * (1.0 - current_lr * WEIGHT_DECAY) - m_hat * adaptive_lr;

            // // Second moment update - treat real and imag separately
            // let g_re_sq = g_t.re * g_t.re;
            // let g_im_sq = g_t.im * g_t.im;
            // let new_v_re = prev_v[i][j].re * B_2 + (1.0 - B_2) * g_re_sq;
            // let new_v_im = prev_v[i][j].im * B_2 + (1.0 - B_2) * g_im_sq;
            // prev_v[i][j] = Complex::new(new_v_re, new_v_im);

            // // Bias corrections
            // let m_hat_re = prev_m[i][j].re / (1.0 - B_1.powi(t));
            // let m_hat_im = prev_m[i][j].im / (1.0 - B_1.powi(t));
            // let v_hat_re = new_v_re / (1.0 - B_2.powi(t));
            // let v_hat_im = new_v_im / (1.0 - B_2.powi(t));

            // // Adaptive learning rate (separate for real and imag)
            // let denom_re = v_hat_re.sqrt() + EPSILON;
            // let denom_im = v_hat_im.sqrt() + EPSILON;

            // // AdamW update (with decoupled weight decay)
            // let weight_decay_factor = 1.0 - current_lr * WEIGHT_DECAY;
            // weights[i][j].re = weights[i][j].re * weight_decay_factor - m_hat_re * (current_lr / denom_re);
            // weights[i][j].im = weights[i][j].im * weight_decay_factor - m_hat_im * (current_lr / denom_im);
        }
    }
}

// AdamW optimizer for complex biases (vector)
pub fn calculate_adam_w_bias(
    bias: &[Complex<f64>],
    gradient: &[Complex<f64>],
    prev_m: &mut Vec<Complex<f64>>,
    prev_v: &mut Vec<Complex<f64>>, // real part used, imag=0
    learning_rate: f64,
    time_step: usize,
) {
    let mut bias = bias.to_vec();
    let t_i = time_step.max(1) as i32;

    for (i, b) in bias.iter_mut().enumerate() {
        let g_t = gradient[i];
        let current_lr = get_current_learning_rate(learning_rate, t_i as usize);

        if is_nan_or_inf(&g_t) {
            continue;
        }

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
        let adaptive_lr: f64 = current_lr / denom;

        // 5️⃣ AdamW update (decoupled weight decay)
        *b = *b * (1.0 - current_lr * WEIGHT_DECAY) - m_hat * adaptive_lr;

        // // Second moment update - treat real and imag separately
        // let g_re_sq = g_t.re * g_t.re;
        // let g_im_sq = g_t.im * g_t.im;
        // let new_v_re = prev_v[i].re * B_2 + (1.0 - B_2) * g_re_sq;
        // let new_v_im = prev_v[i].im * B_2 + (1.0 - B_2) * g_im_sq;
        // prev_v[i] = Complex::new(new_v_re, new_v_im);

        // // Bias corrections
        // let m_hat_re = prev_m[i].re / (1.0 - B_1.powi(t_i));
        // let m_hat_im = prev_m[i].im / (1.0 - B_1.powi(t_i));
        // let v_hat_re = new_v_re / (1.0 - B_2.powi(t_i));
        // let v_hat_im = new_v_im / (1.0 - B_2.powi(t_i));

        // // Adaptive learning rate (separate for real and imag)
        // let denom_re = v_hat_re.sqrt() + EPSILON;
        // let denom_im = v_hat_im.sqrt() + EPSILON;

        // // AdamW update (with decoupled weight decay)
        // let weight_decay_factor = 1.0 - current_lr * WEIGHT_DECAY;
        // bias[i].re = bias[i].re * weight_decay_factor - m_hat_re * (current_lr / denom_re);
        // bias[i].im = bias[i].im * weight_decay_factor - m_hat_im * (current_lr / denom_im);
    }
}

pub fn get_current_learning_rate(base_lr: f64, step: usize) -> f64 {
    if step < WARMUP_STEPS {
        // Linear warmup
        base_lr * (step as f64) / (WARMUP_STEPS as f64)
    } else {
        let total_steps = 2000;
        let progress = (step - WARMUP_STEPS) as f64 / (total_steps - WARMUP_STEPS) as f64;
        base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
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
