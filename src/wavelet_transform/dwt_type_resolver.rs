use crate::wavelet_transform::dwt_coeffients::*;
use crate::wavelet_transform::dwt_inverse_coeffients::*;
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;

// Return a Low Pass Filter, a filter of moving averages for a specific discrete wavelet type
pub fn get_low_pass_filter(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    match dw_type {
        DiscreteWaletetType::DB1 => DB_1.to_vec(),
        DiscreteWaletetType::DB2 => DB_2.to_vec(),
        DiscreteWaletetType::DB3 => DB_3.to_vec(),
        DiscreteWaletetType::DB4 => DB_4.to_vec(),
        DiscreteWaletetType::DB5 => DB_5.to_vec(),
        DiscreteWaletetType::DB6 => DB_6.to_vec(),
        DiscreteWaletetType::DB7 => DB_7.to_vec(),
        DiscreteWaletetType::DB8 => DB_8.to_vec(),
        DiscreteWaletetType::DB9 => DB_9.to_vec(),
        DiscreteWaletetType::DB10 => DB_10.to_vec(),
        DiscreteWaletetType::DB11 => DB_11.to_vec(),
        DiscreteWaletetType::DB12 => DB_12.to_vec(),
        DiscreteWaletetType::DB13 => DB_13.to_vec(),
        DiscreteWaletetType::DB14 => DB_14.to_vec(),
        DiscreteWaletetType::DB15 => DB_15.to_vec(),
        DiscreteWaletetType::DB16 => DB_16.to_vec(),
        DiscreteWaletetType::DB17 => DB_17.to_vec(),
        DiscreteWaletetType::DB18 => DB_18.to_vec(),
        DiscreteWaletetType::DB19 => DB_19.to_vec(),
        DiscreteWaletetType::DB20 => DB_20.to_vec(),
        DiscreteWaletetType::DB21 => DB_21.to_vec(),
        DiscreteWaletetType::DB22 => DB_22.to_vec(),
        DiscreteWaletetType::DB23 => DB_23.to_vec(),
        DiscreteWaletetType::DB24 => DB_24.to_vec(),
        DiscreteWaletetType::DB25 => DB_25.to_vec(),
        DiscreteWaletetType::DB26 => DB_26.to_vec(),
        DiscreteWaletetType::DB27 => DB_27.to_vec(),
        DiscreteWaletetType::DB28 => DB_28.to_vec(),
        DiscreteWaletetType::DB29 => DB_29.to_vec(),
        DiscreteWaletetType::DB30 => DB_30.to_vec(),
        DiscreteWaletetType::DB31 => DB_31.to_vec(),
        DiscreteWaletetType::DB32 => DB_32.to_vec(),
        DiscreteWaletetType::DB33 => DB_33.to_vec(),
        DiscreteWaletetType::DB34 => DB_34.to_vec(),
        DiscreteWaletetType::DB35 => DB_35.to_vec(),
        DiscreteWaletetType::DB36 => DB_36.to_vec(),
        DiscreteWaletetType::DB37 => DB_37.to_vec(),
        DiscreteWaletetType::DB38 => DB_38.to_vec(),
        DiscreteWaletetType::SYM2 => SYM_2.to_vec(),
        DiscreteWaletetType::SYM3 => SYM_3.to_vec(),
        DiscreteWaletetType::SYM4 => SYM_4.to_vec(),
        DiscreteWaletetType::SYM5 => SYM_5.to_vec(),
        DiscreteWaletetType::SYM6 => SYM_6.to_vec(),
        DiscreteWaletetType::SYM7 => SYM_7.to_vec(),
        DiscreteWaletetType::SYM8 => SYM_8.to_vec(),
        DiscreteWaletetType::SYM9 => SYM_9.to_vec(),
        DiscreteWaletetType::SYM10 => SYM_10.to_vec(),
        DiscreteWaletetType::SYM11 => SYM_11.to_vec(),
        DiscreteWaletetType::SYM12 => SYM_12.to_vec(),
        DiscreteWaletetType::SYM13 => SYM_13.to_vec(),
        DiscreteWaletetType::SYM14 => SYM_14.to_vec(),
        DiscreteWaletetType::SYM15 => SYM_15.to_vec(),
        DiscreteWaletetType::SYM16 => SYM_16.to_vec(),
        DiscreteWaletetType::SYM17 => SYM_17.to_vec(),
        DiscreteWaletetType::SYM18 => SYM_18.to_vec(),
        DiscreteWaletetType::SYM19 => SYM_19.to_vec(),
        DiscreteWaletetType::SYM20 => SYM_20.to_vec(),
        DiscreteWaletetType::COIF1 => COIF_1.to_vec(),
        DiscreteWaletetType::COIF2 => COIF_2.to_vec(),
        DiscreteWaletetType::COIF3 => COIF_3.to_vec(),
        DiscreteWaletetType::COIF4 => COIF_4.to_vec(),
        DiscreteWaletetType::COIF6 => COIF_6.to_vec(),
        DiscreteWaletetType::COIF7 => COIF_7.to_vec(),
        DiscreteWaletetType::COIF8 => COIF_8.to_vec(),
        DiscreteWaletetType::COIF9 => COIF_9.to_vec(),
        DiscreteWaletetType::COIF10 => COIF_10.to_vec(),
        DiscreteWaletetType::COIF11 => COIF_11.to_vec(),
        DiscreteWaletetType::COIF12 => COIF_12.to_vec(),
        DiscreteWaletetType::COIF13 => COIF_13.to_vec(),
        DiscreteWaletetType::COIF14 => COIF_14.to_vec(),
        DiscreteWaletetType::COIF15 => COIF_15.to_vec(),
        DiscreteWaletetType::COIF16 => COIF_16.to_vec(),
        DiscreteWaletetType::COIF17 => COIF_17.to_vec(),
        DiscreteWaletetType::BIOR10 => BIOR_1_0.to_vec(),
        DiscreteWaletetType::BIOR11 => BIOR_1_1.to_vec(),
        DiscreteWaletetType::BIOR13 => BIOR_1_3.to_vec(),
        DiscreteWaletetType::BIOR15 => BIOR_1_5.to_vec(),
        DiscreteWaletetType::BIOR20 => BIOR_2_0.to_vec(),
        DiscreteWaletetType::BIOR22 => BIOR_2_2.to_vec(),
        DiscreteWaletetType::BIOR24 => BIOR_2_4.to_vec(),
        DiscreteWaletetType::BIOR26 => BIOR_2_6.to_vec(),
        DiscreteWaletetType::BIOR28 => BIOR_2_8.to_vec(),
        DiscreteWaletetType::BIOR30 => BIOR_3_0.to_vec(),
        DiscreteWaletetType::BIOR31 => BIOR_3_1.to_vec(),
        DiscreteWaletetType::BIOR33 => BIOR_3_3.to_vec(),
        DiscreteWaletetType::BIOR35 => BIOR_3_5.to_vec(),
        DiscreteWaletetType::BIOR37 => BIOR_3_7.to_vec(),
        DiscreteWaletetType::BIOR39 => BIOR_3_9.to_vec(),
        DiscreteWaletetType::BIOR40 => BIOR_4_0.to_vec(),
        DiscreteWaletetType::BIOR44 => BIOR_4_4.to_vec(),
        DiscreteWaletetType::BIOR50 => BIOR_5_0.to_vec(),
        DiscreteWaletetType::BIOR55 => BIOR_5_5.to_vec(),
        DiscreteWaletetType::BIOR60 => BIOR_6_0.to_vec(),
        DiscreteWaletetType::BIOR68 => BIOR_6_8.to_vec(),
        DiscreteWaletetType::DMEY => DMEY.to_vec()
    }
}

// Return a High Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_high_pass_filter(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    let low_pass_filter = get_high_pass_filter_non_symmetric(&dw_type);
    let mut high_pass_filter: Vec<f64> = Vec::new();

    let mut v: f64;
    for (i, el) in low_pass_filter.iter().rev().enumerate() {
        v = el.clone();

        if i % 2 == 1 {
            v *= -1.0;
        }

        high_pass_filter.push(v);
    }

    high_pass_filter
}

// Return a Inverse Low Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_inverse_low_pass_filter(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    let low_pass_filter = get_high_pass_filter_non_symmetric(&dw_type);
    let mut high_pass_filter = Vec::new();

    let mut v: f64;
    for (_i, el) in low_pass_filter.iter().rev().enumerate() {
        v = el.clone();

        high_pass_filter.push(v);
    }

    high_pass_filter
}

// Return a Inverse High Pass Filter, a filter of moving difference for a specific discrete wavelet type
pub fn get_inverse_high_pass_filter(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    let high_pass_filter_coef;
    let mut high_pass_filter: Vec<f64> = Vec::new();
    let mut is_default: bool = false;

    high_pass_filter_coef = match dw_type {
        DiscreteWaletetType::BIOR11 | DiscreteWaletetType::BIOR13 | DiscreteWaletetType::BIOR15
        | DiscreteWaletetType::BIOR22 | DiscreteWaletetType::BIOR24 | DiscreteWaletetType::BIOR26
        | DiscreteWaletetType::BIOR28 | DiscreteWaletetType::BIOR31 | DiscreteWaletetType::BIOR33
        | DiscreteWaletetType::BIOR35 | DiscreteWaletetType::BIOR37 | DiscreteWaletetType::BIOR39
        | DiscreteWaletetType::BIOR44 | DiscreteWaletetType::BIOR55 | DiscreteWaletetType::BIOR68
        => get_low_pass_filter(dw_type),
        _ => {
            is_default = true;
            get_high_pass_filter(&dw_type)
        }
    };

    if is_default {
        let mut v: f64;
        for (_i, el) in high_pass_filter_coef.iter().rev().enumerate() {
            v = el.clone();

            high_pass_filter.push(v);
        }
    } else {
        fill_array_and_negate_odd(&high_pass_filter_coef, &mut high_pass_filter);
    }

    high_pass_filter
}

pub fn get_high_pass_filter_non_symmetric(dw_type: &DiscreteWaletetType) -> Vec<f64> {
    let high_pass_filter: Vec<f64> = match dw_type {
        DiscreteWaletetType::BIOR11 => INVERSE_BIOR_1_1.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR13 => INVERSE_BIOR_1_3.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR15 => INVERSE_BIOR_1_5.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR22 => INVERSE_BIOR_2_2.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR24 => INVERSE_BIOR_2_4.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR26 => INVERSE_BIOR_2_6.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR28 => INVERSE_BIOR_2_8.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR31 => INVERSE_BIOR_3_1.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR33 => INVERSE_BIOR_3_3.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR35 => INVERSE_BIOR_3_5.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR37 => INVERSE_BIOR_3_7.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR39 => INVERSE_BIOR_3_9.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR44 => INVERSE_BIOR_4_4.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR55 => INVERSE_BIOR_5_5.to_vec().into_iter().rev().collect(),
        DiscreteWaletetType::BIOR68 => INVERSE_BIOR_6_8.to_vec().into_iter().rev().collect(),
        _ => {
            get_low_pass_filter(&dw_type)
        }
    };

    high_pass_filter
}

pub fn fill_array_and_negate_odd(filter_coef: &Vec<f64>, data: &mut Vec<f64>) {
    let mut v: f64;

    for (i, el) in filter_coef.iter().enumerate() {
        v = el.clone();

        if i % 2 != 1 {
            v *= -1.0;
        }

        data.push(v);
    }
}