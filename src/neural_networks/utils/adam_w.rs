use num::Complex;

pub static B_1: f64 = 0.9;
pub static B_2: f64 = 0.999;
pub static EPSILON: f64 = 1e-8;
pub static WEIGHT_DECAY: f64 = 0.01;
pub static MAX_NORM: f64 = 1.0;

pub fn calculate_adam_w(weights: &[Vec<Complex<f64>>], gradient: &[Vec<Complex<f64>>], prev_m: &mut Vec<Vec<Complex<f64>>>, prev_v: &mut Vec<Vec<Complex<f64>>>, learning_rate: f64, time_step: usize) -> Vec<Vec<Complex<f64>>> {
    let time_step = time_step.max(1);

    let updated_weights: Vec<Vec<Complex<f64>>> = weights
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, &w)| {
                    let mut g_t = gradient[i][j];

                    // Clip gradients to prevent explosion
                    let g_norm = g_t.norm();
                    if g_norm > MAX_NORM {
                        g_t = g_t * (MAX_NORM / g_norm);
                    }

                    // Update moments
                    prev_m[i][j] = B_1 * prev_m[i][j] + (1.0 - B_1) * g_t;
                    prev_v[i][j] = B_2 * prev_v[i][j] + (1.0 - B_2) * g_t * g_t.conj();

                    // Bias correction
                    let m_hat = prev_m[i][j] / (1.0 - B_1.powi(time_step as i32));
                    let v_hat = prev_v[i][j] / (1.0 - B_2.powi(time_step as i32));

                    // Adaptive learning rate
                    // let v_hat_norm = v_hat.norm().max(EPSILON);
                    let v_hat_norm = v_hat.norm().max(EPSILON);
                    let adaptive_lr = learning_rate / (v_hat_norm.sqrt() + EPSILON);

                    // AdamW update (weight decay + Adam)
                    let decayed_weight = w * (1.0 - learning_rate * WEIGHT_DECAY);
                    let weight_update = decayed_weight - Complex::new(adaptive_lr, 0.0) * m_hat;

                    weight_update
                })
                .collect()
        })
        .collect();

    // let max = updated_weights.iter().flat_map(|w| w.iter()).max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Less));
    // println!("max in adamW updated weights: {:?}", max);

    updated_weights
}

pub fn calculate_adam_w_bias(bias: &[Complex<f64>], gradient: &[Complex<f64>], prev_m: &mut Vec<Complex<f64>>, prev_v: &mut Vec<Complex<f64>>, learning_rate: f64, time_step: usize) -> Vec<Complex<f64>> {
    let time_step = time_step.max(1);

    let updated_bias: Vec<Complex<f64>> = bias
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let mut g_t = gradient[i];

            // Clip gradients to prevent explosion
            let g_norm = g_t.norm();
            if g_norm > MAX_NORM {
                g_t = g_t * (MAX_NORM / g_norm);
            }

            // Update moments
            prev_m[i] = B_1 * prev_m[i] + (1.0 - B_1) * g_t;
            prev_v[i] = B_2 * prev_v[i] + (1.0 - B_2) * g_t * g_t.conj();

            // Bias correction with numerical safety
            let m_hat = prev_m[i] / (1.0 - B_1.powi(time_step as i32));
            let v_hat = prev_v[i] / (1.0 - B_2.powi(time_step as i32));

            // let v_hat_norm = v_hat.norm().max(EPSILON);
            let v_hat_norm = v_hat.norm();
            let adaptive_lr = learning_rate / (v_hat_norm.sqrt() + EPSILON);

            // AdamW update (weight decay + Adam)
            let decayed_bias = b * (1.0 - learning_rate * WEIGHT_DECAY);
            let bias_update = decayed_bias - Complex::new(adaptive_lr, 0.0) * m_hat;

            bias_update
        })
        .collect();

    // let max = updated_bias.iter().max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Less));
    // println!("max in adamW updated bias: {:?}", max);

    updated_bias
}
