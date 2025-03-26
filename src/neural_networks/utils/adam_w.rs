use num::Complex;
use rand::Rng;

pub static B_1: f64 = 0.9; // First moment decay rate
pub static B_2: f64 = 0.999; // Second moment decay rate
pub static EPSILON: f64 = 1e-8; // Small constant to prevent division by zero
pub static WEIGHT_DECAY: f64 = 0.01; // Weight decay coefficient

pub fn calculate_adam_w(weights: &Vec<Vec<Complex<f64>>>, gradient: &Vec<Vec<Complex<f64>>>, prev_m: &mut Vec<Vec<Complex<f64>>>, prev_v: &mut Vec<Vec<Complex<f64>>>, learning_rate: f64, time_step: usize) -> Vec<Vec<Complex<f64>>> {
    let mut updated_weights = weights.clone();
    let time_step = time_step.max(1); // Ensure time_step is at least 1

    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            let g_t = gradient[i][j];

            // Update moments - using only the real part for stability
            prev_m[i][j] = B_1 * prev_m[i][j] + (1.0 - B_1) * g_t;
            prev_v[i][j] = B_2 * prev_v[i][j] + (1.0 - B_2) * g_t * g_t.conj(); // Using conjugate for proper complex magnitude

            // Bias correction with numerical safety
            let b1_correction = 1.0 - B_1.powi(time_step as i32);
            let b2_correction = 1.0 - B_2.powi(time_step as i32);

            let m_hat = prev_m[i][j] / Complex::new(b1_correction.max(EPSILON), 0.0);
            let v_hat = prev_v[i][j] / Complex::new(b2_correction.max(EPSILON), 0.0);

            // Safe adaptive learning rate calculation
            let v_hat_real = v_hat.re.max(0.0); // Ensure non-negative for sqrt
            let adaptive_lr = learning_rate / (v_hat_real.sqrt() + EPSILON);

            // AdamW weight update with numerical safety
            let weight_update = updated_weights[i][j] - Complex::new(adaptive_lr, 0.0) * m_hat - Complex::new(learning_rate * WEIGHT_DECAY, 0.0) * weights[i][j];

            // Clamping the updated weights between -1 and 1
            updated_weights[i][j] = Complex::new(weight_update.re.clamp(-1.0, 1.0), weight_update.im.clamp(-1.0, 1.0));

            if updated_weights[i][j].norm() > 2.0 {
                updated_weights[i][j] = Complex::new(rand::rng().random_range(-0.5..0.5), rand::rng().random_range(-0.5..0.5));
            }
        }
    }
    updated_weights
}

pub fn calculate_adam_w_bias(bias: &Vec<Complex<f64>>, gradient: &Vec<Complex<f64>>, prev_m: &mut Vec<Complex<f64>>, prev_v: &mut Vec<Complex<f64>>, learning_rate: f64, time_step: usize) -> Vec<Complex<f64>> {
    let mut updated_bias = bias.clone();
    let time_step = time_step.max(1); // Ensure time_step is at least 1

    for i in 0..bias.len() {
        let g_t = gradient[i];

        // Update moments
        prev_m[i] = B_1 * prev_m[i] + (1.0 - B_1) * g_t;
        prev_v[i] = B_2 * prev_v[i] + (1.0 - B_2) * g_t * g_t.conj();

        // Bias correction with numerical safety
        let b1_correction = 1.0 - B_1.powi(time_step as i32);
        let b2_correction = 1.0 - B_2.powi(time_step as i32);

        let m_hat = prev_m[i] / Complex::new(b1_correction.max(EPSILON), 0.0);
        let v_hat = prev_v[i] / Complex::new(b2_correction.max(EPSILON), 0.0);

        // Safe adaptive learning rate calculation
        let v_hat_real = v_hat.re.max(0.0); // Ensure non-negative for sqrt
        let adaptive_lr = learning_rate / (v_hat_real.sqrt() + EPSILON);

        // AdamW bias update
        let bias_update = updated_bias[i] - Complex::new(adaptive_lr, 0.0) * m_hat;

        // Clamping the updated bias between -1 and 1
        updated_bias[i] = Complex::new(bias_update.re.clamp(-1.0, 1.0), bias_update.im.clamp(-1.0, 1.0));

        if updated_bias[i].norm() > 2.0 {
            updated_bias[i] = Complex::new(rand::rng().random_range(-0.5..0.5), rand::rng().random_range(-0.5..0.5));
        }
    }

    updated_bias
}
