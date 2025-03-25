use num::Complex;

pub static B_1: f64 = 0.9; // First moment decay rate
pub static B_2: f64 = 0.999; // Second moment decay rate
pub static EPSILON: f64 = 1e-8; // Small constant to prevent division by zero
pub static WEIGHT_DECAY: f64 = 0.01; // Weight decay coefficient

pub fn calculate_adam_w(weight: &Vec<Vec<Complex<f64>>>, gradient: &Vec<Vec<Complex<f64>>>, prev_m: &mut Vec<Vec<Complex<f64>>>, prev_v: &mut Vec<Vec<Complex<f64>>>, learning_rate: f64, time_step: usize) -> Vec<Vec<Complex<f64>>> {
    let mut updated_weights = weight.clone();

    for i in 0..weight.len() {
        for j in 0..weight[i].len() {
            let g_t = gradient[i][j];

            // Update biased first moment estimate (momentum)
            prev_m[i][j] = B_1 * prev_m[i][j] + (1.0 - B_1) * g_t;

            // Update biased second raw moment estimate (RMS scaling)
            prev_v[i][j] = B_2 * prev_v[i][j] + (1.0 - B_2) * (g_t * g_t);

            // Bias correction for first moment
            let m_hat = prev_m[i][j] / (1.0 - B_1.powi(time_step as i32));

            // Bias correction for second moment
            let v_hat = prev_v[i][j] / (1.0 - B_2.powi(time_step as i32));

            // Compute adaptive learning rate
            let adaptive_lr = learning_rate / (v_hat.sqrt() + Complex::new(EPSILON, 0.0));

            // AdamW weight update
            updated_weights[i][j] = updated_weights[i][j] - adaptive_lr * m_hat - learning_rate * WEIGHT_DECAY * weight[i][j];
        }
    }
    updated_weights
}

pub fn calculate_adam_w_bias(bias: &Vec<Complex<f64>>, gradient: &Vec<Complex<f64>>, prev_m: &mut Vec<Complex<f64>>, prev_v: &mut Vec<Complex<f64>>, learning_rate: f64, time_step: usize) -> Vec<Complex<f64>> {
    let mut updated_bias = bias.clone();

    for i in 0..bias.len() {
        let g_t = gradient[i];

        // First moment estimate (momentum)
        prev_m[i] = B_1 * prev_m[i] + (1.0 - B_1) * g_t;

        // Second moment estimate (variance)
        prev_v[i] = B_2 * prev_v[i] + (1.0 - B_2) * (g_t * g_t);

        // Bias correction
        let m_hat = prev_m[i] / (1.0 - B_1.powi(time_step as i32));
        let v_hat = prev_v[i] / (1.0 - B_2.powi(time_step as i32));

        // Adaptive learning rate
        let adaptive_lr = learning_rate / (v_hat.sqrt() + Complex::new(EPSILON, 0.0));

        // AdamW bias update (NO weight decay)
        updated_bias[i] = updated_bias[i] - adaptive_lr * m_hat;
    }

    updated_bias
}
