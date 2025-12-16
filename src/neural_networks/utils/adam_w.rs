use num::Complex;

pub static B_1: f64 = 0.9;
pub static B_2: f64 = 0.999;
pub static EPSILON: f64 = 1e-8;
pub static WEIGHT_DECAY: f64 = 0.001;
pub static MAX_NORM: f64 = 2.0;
pub const MAX_ELEMENT: f64 = 20.0;
pub static WARMUP_STEPS: usize = 2000;

// Assume this helper function exists or add it
pub fn is_nan_or_inf(c: &Complex<f64>) -> bool {
    c.re.is_nan() || c.re.is_infinite() || c.im.is_nan() || c.im.is_infinite() || c.norm_sqr().is_nan() || c.norm_sqr().is_infinite() || c.norm_sqr() > 1e10
    // Add threshold for very large values
}

pub fn calculate_adam_w(
    weights: &mut Vec<Vec<Complex<f64>>>,
    weight_gradients: &Vec<Vec<Complex<f64>>>,
    prev_m: &mut Vec<Vec<Complex<f64>>>,
    prev_v: &mut Vec<Vec<Complex<f64>>>,
    prev_v_hat: &mut Vec<Vec<Complex<f64>>>,
    learning_rate: f64,
    t: usize,
) {
    let t = t.max(1) as i32;
    let current_lr = get_current_learning_rate(learning_rate, t as usize);

    // hyperparameters for explosion control
    let max_grad = 15.0; // max norm for gradient
    let max_m = 100.0; // optional cap for prev_m
    let max_vhat = 1e6; // optional cap for prev_v_hat

    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            let mut g_t = weight_gradients[i][j];

            // ---------- 0️⃣ Clip gradient ----------
            if g_t.norm() > max_grad {
                g_t = g_t / g_t.norm() * max_grad;
            }

            if is_nan_or_inf(&g_t) {
                println!("Gradient contains NaN or Inf in adam w weights: {:?}", g_t);
                continue;
            }

            // ---------- 1️⃣ First moment update ----------
            prev_m[i][j] = prev_m[i][j] * B_1 + (1.0 - B_1) * g_t;

            // optional cap on prev_m
            if prev_m[i][j].norm() > max_m {
                prev_m[i][j] = prev_m[i][j] / prev_m[i][j].norm() * max_m;
            }

            // ---------- 2️⃣ Second moment update (real part only) ----------
            let g_norm2 = g_t.norm_sqr();
            prev_v[i][j].re = prev_v[i][j].re * B_2 + (1.0 - B_2) * g_norm2;

            // ---------- 3️⃣ AMSGrad ----------
            prev_v_hat[i][j].re = prev_v_hat[i][j].re.max(prev_v[i][j].re);
            prev_v_hat[i][j].re = prev_v_hat[i][j].re.min(max_vhat);

            // ---------- 4️⃣ Bias-corrected moments (FIXED: cap bias correction) ----------
            let t_safe = t.min(10000) as i32;
            let bias_correct_m = (1.0 - B_1.powi(t_safe)).recip().min(1e6);
            let bias_correct_v = (1.0 - B_2.powi(t_safe)).recip().min(1e6);

            let m_t_hat = prev_m[i][j] * bias_correct_m;
            let v_t_hat = prev_v_hat[i][j].re * bias_correct_v;

            // ---------- 5️⃣ Adaptive learning rate (FIXED: clip by norm) ----------
            let adaptive_lr_raw = m_t_hat / (v_t_hat.sqrt().max(EPSILON));
            let adaptive_lr = if adaptive_lr_raw.norm() > 1.0 {
                adaptive_lr_raw / adaptive_lr_raw.norm() * 1.0
            } else {
                adaptive_lr_raw
            };

            // ---------- 6️⃣ AdamW update ----------
            weights[i][j] = weights[i][j] - (current_lr * WEIGHT_DECAY * weights[i][j]) - (current_lr * adaptive_lr);
        }
    }
}

pub fn calculate_adam_w_bias(
    bias: &mut Vec<Complex<f64>>,
    gradient: &[Complex<f64>],
    prev_m: &mut Vec<Complex<f64>>,
    prev_v: &mut Vec<Complex<f64>>,
    prev_v_hat: &mut Vec<Complex<f64>>,
    learning_rate: f64,
    time_step: usize,
) {
    let t_i = time_step.max(1) as i32;
    let current_lr = get_current_learning_rate(learning_rate, t_i as usize);

    // hyperparameters for explosion control
    let max_grad = 15.0;
    let max_m = 100.0;
    let max_vhat = 1e6;

    for (i, b) in bias.iter_mut().enumerate() {
        let mut g_t = gradient[i];

        // ---------- 0️⃣ Clip gradient ----------
        if g_t.norm() > max_grad {
            g_t = g_t / g_t.norm() * max_grad;
        }

        if is_nan_or_inf(&g_t) {
            println!("Gradient contains NaN or Inf in adam w bias: {:?}", g_t);
            continue;
        }

        // ---------- 1️⃣ First moment update ----------
        prev_m[i] = prev_m[i] * B_1 + (1.0 - B_1) * g_t;

        // optional cap on prev_m
        if prev_m[i].norm() > max_m {
            prev_m[i] = prev_m[i] / prev_m[i].norm() * max_m;
        }

        // ---------- 2️⃣ Second moment update (real part only) ----------
        let g_norm2 = g_t.norm_sqr();
        prev_v[i].re = prev_v[i].re * B_2 + (1.0 - B_2) * g_norm2;

        // ---------- 3️⃣ AMSGrad ----------
        prev_v_hat[i].re = prev_v_hat[i].re.max(prev_v[i].re);
        prev_v_hat[i].re = prev_v_hat[i].re.min(max_vhat);

        // ---------- 4️⃣ Bias-corrected moments (FIXED: cap bias correction) ----------
        let t_safe = t_i.min(10000) as i32;
        let bias_correct_m = (1.0 - B_1.powi(t_safe)).recip().min(1e6);
        let bias_correct_v = (1.0 - B_2.powi(t_safe)).recip().min(1e6);

        let m_hat = prev_m[i] * bias_correct_m;
        let v_hat = prev_v_hat[i].re * bias_correct_v;

        // ---------- 5️⃣ Adaptive learning rate (FIXED: clip by norm) ----------
        let adaptive_lr_raw = m_hat / (v_hat.sqrt().max(EPSILON));
        let adaptive_lr = if adaptive_lr_raw.norm() > 1.0 {
            adaptive_lr_raw / adaptive_lr_raw.norm() * 1.0
        } else {
            adaptive_lr_raw
        };

        // ---------- 6️⃣ AdamW update ----------
        *b = *b - (current_lr * WEIGHT_DECAY * *b) - (current_lr * adaptive_lr);
    }
}

// pub fn calculate_adam_w(weights: &mut Vec<Vec<Complex<f64>>>, weight_gradients: &Vec<Vec<Complex<f64>>>, prev_m: &mut Vec<Vec<Complex<f64>>>, prev_v: &mut Vec<Vec<Complex<f64>>>, prev_v_hat: &mut Vec<Vec<Complex<f64>>>, learning_rate: f64, t: usize) {
//     let t = t.max(1) as i32;
//     let current_lr = get_current_learning_rate(learning_rate, t as usize);

//     //normalize_gradients(weights);
//     // normalize_gradients(prev_m);
//     // normalize_gradients(prev_v);

//     for i in 0..weights.len() {
//         for j in 0..weights[i].len() {
//             // 1️⃣ Gradient clipping (complex norm)
//             let orig_gt = weight_gradients[i][j];
//             let g_t = weight_gradients[i][j];

//             // if g_t.norm_sqr() > MAX_NORM {
//             //     g_t = g_t / g_t.norm_sqr();
//             // }

//             if is_nan_or_inf(&g_t) {
//                 println!("Gradient contains NaN or Inf in adam w weights: {:?}, original g_t: {:?}", g_t, orig_gt);
//                 continue;
//             }

//             // 2️⃣ First moment update (A)
//             prev_m[i][j] = prev_m[i][j] * B_1 + (1.0 - B_1) * g_t;

//             // 3️⃣Second moment update (F)
//             prev_v[i][j] = prev_v[i][j] * B_2 + (1.0 - B_2) * g_t * g_t;

//             prev_v_hat[i][j] = if prev_v_hat[i][j].norm() > prev_v[i][j].norm() { prev_v_hat[i][j] } else { prev_v[i][j] };

//             let m_t_hat = prev_m[i][j] / (1.0 - B_1.powi(t));
//             let v_t_hat = prev_v_hat[i][j] / (1.0 - B_2.powi(t));

//             // 4️⃣ Adaptive learning rate
//             let adaptive_lr = (current_lr * m_t_hat) / (v_t_hat.sqrt() + EPSILON);

//             // 5️⃣ AdamW update (with decoupled weight decay)
//             weights[i][j] = weights[i][j] - (current_lr * WEIGHT_DECAY * weights[i][j]) - adaptive_lr;
//         }
//     }
// }

// // AdamW optimizer for complex biases (vector)
// pub fn calculate_adam_w_bias(
//     bias: &mut Vec<Complex<f64>>,
//     gradient: &[Complex<f64>],
//     prev_m: &mut Vec<Complex<f64>>,
//     prev_v: &mut Vec<Complex<f64>>,
//     prev_v_hat: &mut Vec<Complex<f64>>,
//     learning_rate: f64,
//     time_step: usize,
// ) {
//     let t_i = time_step.max(1) as i32;

//     // normalize_bias(bias);
//     // normalize_bias(prev_m);
//     // normalize_bias(prev_v);

//     for (i, b) in bias.iter_mut().enumerate() {
//         let g_t = gradient[i];
//         let current_lr = get_current_learning_rate(learning_rate, t_i as usize);

//         // if g_t.norm_sqr() > MAX_NORM {
//         //     g_t = g_t / g_t.norm_sqr();
//         // }

//         if is_nan_or_inf(&g_t) {
//             println!("Gradient contains NaN or Inf in adam w bias: {:?}", g_t);
//             continue;
//         }

//         // 1️⃣ First moment update
//         prev_m[i] = prev_m[i] * B_1 + (1.0 - B_1) * g_t;

//         // 2️⃣ Second moment update
//         prev_v[i] = prev_v[i] * B_2 + (1.0 - B_2) * g_t.norm_sqr();
//         prev_v_hat[i] = if prev_v_hat[i].norm() > prev_v[i].norm() { prev_v_hat[i] } else { prev_v[i] };

//         // 3️⃣ Bias corrections
//         let m_hat: Complex<f64> = prev_m[i] / (1.0 - B_1.powi(t_i));
//         let v_hat: Complex<f64> = prev_v_hat[i] / (1.0 - B_2.powi(t_i));

//         // 4️⃣ Adaptive learning rate
//         let adaptive_lr = (current_lr * m_hat) / (v_hat.sqrt() + EPSILON);

//         // 5️⃣ AdamW update (with decoupled weight decay)
//         *b = *b - (current_lr * WEIGHT_DECAY * *b) - adaptive_lr;
//     }
// }

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
            if g_t.norm_sqr() > MAX_NORM {
                let scale = MAX_NORM / g_t.norm_sqr();
                g_t = g_t * scale; // g_t is now clipped to have magnitude <= MAX_NORM
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
