use crate::wavelet_transform::dwt_coeffients::*;
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;

// Return a Low Pass Filter, a filter of moving averages for a specific discrete wavelet type
pub fn get_low_pass_filter(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    match dw_type {
        DiscreteWaletetType::DB_1 => DB_1.to_vec(),
        DiscreteWaletetType::DB_2 => DB_2.to_vec(),
        DiscreteWaletetType::DB_3 => DB_3.to_vec(),
        DiscreteWaletetType::DB_4 => DB_4.to_vec(),
        DiscreteWaletetType::DB_5 => DB_5.to_vec(),
        DiscreteWaletetType::DB_6 => DB_6.to_vec(),
        DiscreteWaletetType::DB_7 => DB_7.to_vec(),
        DiscreteWaletetType::DB_8 => DB_8.to_vec(),
        DiscreteWaletetType::DB_9 => DB_9.to_vec(),
        DiscreteWaletetType::DB_10 => DB_10.to_vec(),
        DiscreteWaletetType::DB_11 => DB_11.to_vec(),
        DiscreteWaletetType::DB_12 => DB_12.to_vec(),
        DiscreteWaletetType::DB_13 => DB_13.to_vec(),
        DiscreteWaletetType::DB_14 => DB_14.to_vec(),
        DiscreteWaletetType::DB_15 => DB_15.to_vec(),
        DiscreteWaletetType::DB_16 => DB_16.to_vec(),
        DiscreteWaletetType::DB_17 => DB_17.to_vec(),
        DiscreteWaletetType::DB_18 => DB_18.to_vec(),
        DiscreteWaletetType::DB_19 => DB_19.to_vec(),
        DiscreteWaletetType::DB_20 => DB_20.to_vec(),
        DiscreteWaletetType::DB_21 => DB_21.to_vec(),
        DiscreteWaletetType::DB_22 => DB_22.to_vec(),
        DiscreteWaletetType::DB_23 => DB_23.to_vec(),
        DiscreteWaletetType::DB_24 => DB_24.to_vec(),
        DiscreteWaletetType::DB_25 => DB_25.to_vec(),
        DiscreteWaletetType::DB_26 => DB_26.to_vec(),
        DiscreteWaletetType::DB_27 => DB_27.to_vec(),
        DiscreteWaletetType::DB_28 => DB_28.to_vec(),
        DiscreteWaletetType::DB_29 => DB_29.to_vec(),
        DiscreteWaletetType::DB_30 => DB_30.to_vec(),
        DiscreteWaletetType::DB_31 => DB_31.to_vec(),
        DiscreteWaletetType::DB_32 => DB_32.to_vec(),
        DiscreteWaletetType::DB_33 => DB_33.to_vec(),
        DiscreteWaletetType::DB_34 => DB_34.to_vec(),
        DiscreteWaletetType::DB_35 => DB_35.to_vec(),
        DiscreteWaletetType::DB_36 => DB_36.to_vec(),
        DiscreteWaletetType::DB_37 => DB_37.to_vec(),
        DiscreteWaletetType::DB_38 => DB_38.to_vec(),
        DiscreteWaletetType::SYM_2 => SYM_2.to_vec(),
        DiscreteWaletetType::SYM_3 => SYM_3.to_vec(),
        DiscreteWaletetType::SYM_4 => SYM_4.to_vec(),
        DiscreteWaletetType::SYM_5 => SYM_5.to_vec(),
        DiscreteWaletetType::SYM_6 => SYM_6.to_vec(),
        DiscreteWaletetType::SYM_7 => SYM_7.to_vec(),
        DiscreteWaletetType::SYM_8 => SYM_8.to_vec(),
        DiscreteWaletetType::SYM_9 => SYM_9.to_vec(),
        DiscreteWaletetType::SYM_10 => SYM_10.to_vec(),
        DiscreteWaletetType::SYM_11 => SYM_11.to_vec(),
        DiscreteWaletetType::SYM_12 => SYM_12.to_vec(),
        DiscreteWaletetType::SYM_13 => SYM_13.to_vec(),
        DiscreteWaletetType::SYM_14 => SYM_14.to_vec(),
        DiscreteWaletetType::SYM_15 => SYM_15.to_vec(),
        DiscreteWaletetType::SYM_16 => SYM_16.to_vec(),
        DiscreteWaletetType::SYM_17 => SYM_17.to_vec(),
        DiscreteWaletetType::SYM_18 => SYM_18.to_vec(),
        DiscreteWaletetType::SYM_19 => SYM_19.to_vec(),
        DiscreteWaletetType::SYM_20 => SYM_20.to_vec(),
        DiscreteWaletetType::COIF_1 => COIF_1.to_vec(),
        DiscreteWaletetType::COIF_2 => COIF_2.to_vec(),
        DiscreteWaletetType::COIF_3 => COIF_3.to_vec(),
        DiscreteWaletetType::COIF_4 => COIF_4.to_vec(),
        DiscreteWaletetType::COIF_6 => COIF_6.to_vec(),
        DiscreteWaletetType::COIF_7 => COIF_7.to_vec(),
        DiscreteWaletetType::COIF_8 => COIF_8.to_vec(),
        DiscreteWaletetType::COIF_9 => COIF_9.to_vec(),
        DiscreteWaletetType::COIF_10 => COIF_10.to_vec(),
        DiscreteWaletetType::COIF_11 => COIF_11.to_vec(),
        DiscreteWaletetType::COIF_12 => COIF_12.to_vec(),
        DiscreteWaletetType::COIF_13 => COIF_13.to_vec(),
        DiscreteWaletetType::COIF_14 => COIF_14.to_vec(),
        DiscreteWaletetType::COIF_15 => COIF_15.to_vec(),
        DiscreteWaletetType::COIF_16 => COIF_16.to_vec(),
        DiscreteWaletetType::COIF_17 => COIF_17.to_vec(),
        DiscreteWaletetType::BIOR_1_0 => BIOR_1_0.to_vec(),
        DiscreteWaletetType::BIOR_1_1 => BIOR_1_1.to_vec(),
        DiscreteWaletetType::BIOR_1_3 => BIOR_1_3.to_vec(),
        DiscreteWaletetType::BIOR_1_5 => BIOR_1_5.to_vec(),
        DiscreteWaletetType::BIOR_2_0 => BIOR_2_0.to_vec(),
        DiscreteWaletetType::BIOR_2_2 => BIOR_2_2.to_vec(),
        DiscreteWaletetType::BIOR_2_4 => BIOR_2_4.to_vec(),
        DiscreteWaletetType::BIOR_2_6 => BIOR_2_6.to_vec(),
        DiscreteWaletetType::BIOR_2_8 => BIOR_2_8.to_vec(),
        DiscreteWaletetType::BIOR_3_0 => BIOR_3_0.to_vec(),
        DiscreteWaletetType::BIOR_3_1 => BIOR_3_1.to_vec(),
        DiscreteWaletetType::BIOR_3_3 => BIOR_3_3.to_vec(),
        DiscreteWaletetType::BIOR_3_5 => BIOR_3_5.to_vec(),
        DiscreteWaletetType::BIOR_3_7 => BIOR_3_7.to_vec(),
        DiscreteWaletetType::BIOR_3_9 => BIOR_3_9.to_vec(),
        DiscreteWaletetType::BIOR_4_0 => BIOR_4_0.to_vec(),
        DiscreteWaletetType::BIOR_4_4 => BIOR_4_4.to_vec(),
        DiscreteWaletetType::BIOR_5_0 => BIOR_5_0.to_vec(),
        DiscreteWaletetType::BIOR_5_5 => BIOR_5_5.to_vec(),
        DiscreteWaletetType::BIOR_6_0 => BIOR_6_0.to_vec(),
        DiscreteWaletetType::BIOR_6_8 => BIOR_6_8.to_vec(),
        DiscreteWaletetType::DMEY => DMEY.to_vec(),
        _ => Vec::new()
    }
}

// Return a High Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_high_pass_filter(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    let low_pass_filter = get_low_pass_filter(&dw_type);
    let mut high_pass_filter = Vec::new();

    let mut v: f64;
    for (i, el) in low_pass_filter.iter().rev().enumerate() {
        v = el.clone();

        if (i % 2 != 0) {
            v *= -1.0;
        }

        high_pass_filter.push(v);
    }

    high_pass_filter
}

// Return a Inverse Low Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_inverse_low_pass_filter(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    let low_pass_filter = get_low_pass_filter(&dw_type);
    let mut high_pass_filter = Vec::new();

    let mut v: f64;
    for (i, el) in low_pass_filter.iter().rev().enumerate() {
        v = el.clone();

        // if (i % 1 != 0) {
        //     v *= -1.0;
        // }

        high_pass_filter.push(v);
    }

    high_pass_filter
}

// Return a Inverse High Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_inverse_high_pass_filter(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    let low_pass_filter = get_high_pass_filter(&dw_type);
    let mut high_pass_filter = Vec::new();

    let mut v: f64;
    for (i, el) in low_pass_filter.iter().rev().enumerate() {
        v = el.clone();

        // if (i % 2 == 1 || i.clone() == 0) {
        //     v *= -1.0;
        // }

        high_pass_filter.push(v);
    }

    high_pass_filter
}