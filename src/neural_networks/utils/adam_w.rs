use num::Complex;

pub static B_1: f64 = 0.9;
pub static B_2: f64 = 0.999;
pub static EPSILON: f64 = 1e-8;
pub static WEIGHT_DECAY: f64 = 0.001;
pub static MAX_NORM: f64 = 1.0;
pub const MAX_ELEMENT: f64 = 1.0;
pub static WARMUP_STEPS: usize = 200;

// Assume this helper function exists or add it
pub fn is_nan_or_inf(c: &Complex<f64>) -> bool {
    c.re.is_nan() || c.re.is_infinite() || c.im.is_nan() || c.im.is_infinite() || c.norm_sqr().is_nan() || c.norm_sqr().is_infinite() || c.norm_sqr() > 1e10
    // Add threshold for very large values
}

pub fn calculate_adam_w(
    weights: &mut Vec<Vec<Complex<f64>>>,
    weight_gradients: &Vec<Vec<Complex<f64>>>,
    prev_m: &mut Vec<Vec<Complex<f64>>>,
    prev_v: &mut Vec<Vec<Complex<f64>>>,     // stays Complex<f64>
    prev_v_hat: &mut Vec<Vec<Complex<f64>>>, // stays Complex<f64>
    learning_rate: f64,
    t: usize,
) {
    let t = t.max(1) as i32;
    let current_lr = get_current_learning_rate(learning_rate, t as usize);

    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            let g = weight_gradients[i][j];

            if is_nan_or_inf(&g) {
                panic!("Gradient contains NaN or Inf in adam w weights: {:?}", g);
            }

            // 1️⃣ First moment (complex)
            prev_m[i][j] = prev_m[i][j] * B_1 + (1.0 - B_1) * g;

            // 2️⃣ Second moment (REAL stored in Complex.re)
            let g2 = g.norm_sqr(); // |g|² ≥ 0
            prev_v[i][j] = Complex::new(prev_v[i][j].re * B_2 + (1.0 - B_2) * g2, 0.0);

            // 3️⃣ AMSGrad (compare REAL parts only)
            if prev_v_hat[i][j].re < prev_v[i][j].re {
                prev_v_hat[i][j] = prev_v[i][j];
            }

            // 4️⃣ Bias correction
            let m_hat = prev_m[i][j] / (1.0 - B_1.powi(t));
            let v_hat_re = prev_v_hat[i][j].re / (1.0 - B_2.powi(t));

            // 5️⃣ Adaptive step (complex / real)
            let denom = v_hat_re.sqrt() + EPSILON;
            let adaptive_step = (current_lr * m_hat) / denom;

            // 6️⃣ AdamW update (decoupled weight decay)
            weights[i][j] = weights[i][j] - current_lr * WEIGHT_DECAY * weights[i][j] - adaptive_step;
        }
    }
}
// AdamW optimizer for complex biases (vector)
pub fn calculate_adam_w_bias(
    bias: &mut Vec<Complex<f64>>,
    gradient: &Vec<Complex<f64>>,
    prev_m: &mut Vec<Complex<f64>>,
    prev_v: &mut Vec<Complex<f64>>,     // stays Complex<f64>
    prev_v_hat: &mut Vec<Complex<f64>>, // stays Complex<f64>
    learning_rate: f64,
    time_step: usize,
) {
    let t = time_step.max(1) as i32;
    let current_lr = get_current_learning_rate(learning_rate, t as usize);

    for i in 0..bias.len() {
        let g = gradient[i];

        if is_nan_or_inf(&g) {
            panic!("Gradient contains NaN or Inf in adam w bias: {:?}", g);
        }

        // 1️⃣ First moment (complex)
        prev_m[i] = prev_m[i] * B_1 + (1.0 - B_1) * g;

        // 2️⃣ Second moment (REAL stored in Complex.re)
        let g2 = g.norm_sqr();
        prev_v[i] = Complex::new(prev_v[i].re * B_2 + (1.0 - B_2) * g2, 0.0);

        // 3️⃣ AMSGrad
        if prev_v_hat[i].re < prev_v[i].re {
            prev_v_hat[i] = prev_v[i];
        }

        // 4️⃣ Bias correction
        let m_hat = prev_m[i] / (1.0 - B_1.powi(t));
        let v_hat_re = prev_v_hat[i].re / (1.0 - B_2.powi(t));

        // 5️⃣ Adaptive step
        let denom = v_hat_re.sqrt() + EPSILON;
        let adaptive_step = (current_lr * m_hat) / denom;

        // 6️⃣ AdamW update
        bias[i] = bias[i] - current_lr * WEIGHT_DECAY * bias[i] - adaptive_step;
    }
}

// pub fn calculate_adam_w(
//     weights: &mut Vec<Vec<Complex<f64>>>,
//     weight_gradients: &Vec<Vec<Complex<f64>>>,
//     prev_m: &mut Vec<Vec<Complex<f64>>>,
//     prev_v: &mut Vec<Vec<Complex<f64>>>,
//     prev_v_hat: &mut Vec<Vec<Complex<f64>>>,
//     learning_rate: f64,
//     t: usize,
// ) {
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
//                 panic!("Gradient contains NaN or Inf in adam w weights: {:?}, original g_t: {:?}", g_t, orig_gt);
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
//             panic!("Gradient contains NaN or Inf in adam w bias: {:?}", g_t);
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

pub fn calculate_adam_w_bias_f32(
    bias: &mut Vec<f32>,
    gradient: &Vec<Complex<f64>>,
    prev_m: &mut Vec<Complex<f64>>,
    prev_v: &mut Vec<Complex<f64>>,     // stays Complex<f64>
    prev_v_hat: &mut Vec<Complex<f64>>, // stays Complex<f64>
    learning_rate: f64,
    time_step: usize,
) {
    let t = time_step.max(1) as i32;
    let current_lr = get_current_learning_rate(learning_rate, t as usize);

    for i in 0..bias.len() {
        let g = gradient[i];

        if is_nan_or_inf(&g) {
            panic!("Gradient contains NaN or Inf in adam w bias: {:?}", g);
        }

        // 1️⃣ First moment (complex)
        prev_m[i] = prev_m[i] * B_1 + (1.0 - B_1) * g;

        // 2️⃣ Second moment (REAL stored in Complex.re)
        let g2 = g.norm_sqr();
        prev_v[i] = Complex::new(prev_v[i].re * B_2 + (1.0 - B_2) * g2, 0.0);

        // 3️⃣ AMSGrad
        if prev_v_hat[i].re < prev_v[i].re {
            prev_v_hat[i] = prev_v[i];
        }

        // 4️⃣ Bias correction
        let m_hat = prev_m[i] / (1.0 - B_1.powi(t));
        let v_hat_re = prev_v_hat[i].re / (1.0 - B_2.powi(t));

        // 5️⃣ Adaptive step
        let denom = v_hat_re.sqrt() + EPSILON;
        let adaptive_step = (current_lr * m_hat) / denom;

        // 6️⃣ AdamW update - convert to f32
        let bias_complex = Complex::new(bias[i] as f64, 0.0);
        let updated_bias = bias_complex - current_lr * WEIGHT_DECAY * bias_complex - adaptive_step;
        bias[i] = updated_bias.re as f32;
    }
}

// f32 version for weights stored as Vec<Vec<f32>>
pub fn calculate_adam_w_f32(
    weights: &mut Vec<Vec<f32>>,
    weight_gradients: &Vec<Vec<Complex<f64>>>,
    prev_m: &mut Vec<Vec<Complex<f64>>>,
    prev_v: &mut Vec<Vec<Complex<f64>>>,     // stays Complex<f64>
    prev_v_hat: &mut Vec<Vec<Complex<f64>>>, // stays Complex<f64>
    learning_rate: f64,
    t: usize,
) {
    let t = t.max(1) as i32;
    let current_lr = get_current_learning_rate(learning_rate, t as usize);

    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            let g = weight_gradients[i][j];

            if is_nan_or_inf(&g) {
                panic!("Gradient contains NaN or Inf in adam w weights: {:?}", g);
            }

            // 1️⃣ First moment (complex)
            prev_m[i][j] = prev_m[i][j] * B_1 + (1.0 - B_1) * g;

            // 2️⃣ Second moment (REAL stored in Complex.re)
            let g2 = g.norm_sqr(); // |g|² ≥ 0
            prev_v[i][j] = Complex::new(prev_v[i][j].re * B_2 + (1.0 - B_2) * g2, 0.0);

            // 3️⃣ AMSGrad (compare REAL parts only)
            if prev_v_hat[i][j].re < prev_v[i][j].re {
                prev_v_hat[i][j] = prev_v[i][j];
            }

            // 4️⃣ Bias correction
            let m_hat = prev_m[i][j] / (1.0 - B_1.powi(t));
            let v_hat_re = prev_v_hat[i][j].re / (1.0 - B_2.powi(t));

            // 5️⃣ Adaptive step (complex / real)
            let denom = v_hat_re.sqrt() + EPSILON;
            let adaptive_step = (current_lr * m_hat) / denom;

            // 6️⃣ AdamW update (decoupled weight decay) - convert to f32
            let weight_complex = Complex::new(weights[i][j] as f64, 0.0);
            let updated_weight = weight_complex - current_lr * WEIGHT_DECAY * weight_complex - adaptive_step;
            weights[i][j] = updated_weight.re as f32;
        }
    }
}
