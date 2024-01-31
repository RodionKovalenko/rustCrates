use std::fmt::Debug;
use std::ops::{AddAssign, Mul};
use num_traits::{FromPrimitive, ToPrimitive};
use crate::wavelet_transform::dwt_type_resolver::{get_high_pass_filter, get_inverse_high_pass_filter, get_inverse_low_pass_filter, get_low_pass_filter};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;

pub fn transform_2_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive>(data: Vec<Vec<T>>, dw_type: &DiscreteWaletetType) {
    let mut data_trans: Vec<Vec<f64>> = Vec::new();

    for r in data.iter() {
        data_trans.push(transform_1_d(&r, &dw_type));
    }
}

pub fn transform_1_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive>(data: &Vec<T>, dw_type: &DiscreteWaletetType) -> Vec<f64> {
    // moving averages filter
    let low_pass_filter: Vec<f64> = get_low_pass_filter(&dw_type);
    // moving differences filter
    let high_pass_filter: Vec<f64> = get_high_pass_filter(&dw_type);

    // println!("low pass filter : {:?}", &low_pass_filter);
    // println!("high pass filter : {:?}", &high_pass_filter);

    let mut middle_index = data.len() >> 1;
    let mut data_clone = data.clone();

    let mut value_high: f64;
    let mut value_low: f64;

    let mut index_high;
    let mut index_low;
    let mut ind_transform = 0;

    let mut n: i32 = data.len() as i32;
    let mut extra_padding = 0;

    if data.len() % 2 != 0 {
        extra_padding = 1;
    }

    middle_index = (n.clone() >> 1) as usize;

    // Padding before padding before data
    for _i in 0..(high_pass_filter.len() - 2) {
        data_clone.insert(0, T::from_f64(0.0).unwrap());
    }

    n = data_clone.len() as i32;
    middle_index = (n.clone() >> 1) as usize;

    // Padding after
    if (data_clone.len() % high_pass_filter.len() != 0) {
        let mut index = high_pass_filter.len() - (data_clone.len() % high_pass_filter.len());
        for _i in 0..index {
            data_clone.push(T::from_f64(0.0).unwrap());
        }
    }

    // for even number of elements in array add one and calculate middle index again
    if data.len() % 2 != 0 {
        n += 1;
        middle_index = (n.clone() >> 1) as usize;
    }

    let mut data_trans: Vec<f64> = vec![0.0; n.clone() as usize];

    for i in (0..n).step_by(2) {
        value_low = 0.0;
        value_high = 0.0;

        for (l_p_ind, low_p_f_value) in low_pass_filter.iter().enumerate() {
            index_low = ((i.clone() + l_p_ind as i32) % n.clone()) as usize;
            // High-Pass
            value_low += data_clone[index_low].to_f64().unwrap() * low_p_f_value.clone();
            // Low-Pass
            value_high += data_clone[index_low.clone()].to_f64().unwrap() * high_pass_filter[l_p_ind.clone()].clone();
        }

        index_high = (ind_transform.clone() + middle_index.clone() as i32) as usize;

        set_value(&mut data_trans, value_low, &(ind_transform.clone() as usize));
        set_value(&mut data_trans, value_high, &index_high);

        ind_transform += 1;
    }

    data_trans
}

pub fn inverse_transform_1_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive>
(data: &Vec<T>, dw_type: &DiscreteWaletetType) -> Vec<f64> {
    // inverse low pass filter (moving averages filter)
    let inverse_low_pass_filter: Vec<f64> = get_inverse_low_pass_filter(&dw_type);
    // inverse high pass filter (moving differences filter)
    let inverse_high_pass_filter: Vec<f64> = get_inverse_high_pass_filter(&dw_type);

    // println!("inverse low pass filter : {:?}", &inverse_low_pass_filter);
    // println!("inverse high pass filter : {:?}", &inverse_high_pass_filter);

    let mut data_clone = vec![0.0; data.len()];

    let mut value_high: f64;
    let mut value_low: f64;

    let mut index_high;
    let mut index_low = 0;
    let mut ind_transform = 0;
    let mut ind = 0;
    let mut ind_next = 0;

    // total number of elements in data wavelets
    let mut n: i32 = data.len() as i32;
    // original data length l
    let mut l = (data.len() - (inverse_high_pass_filter.len() - 2)) as i32;

    let mut middle_index = data.len() >> 1;
    let mut data_trans: Vec<f64> = vec![0.0; l.clone() as usize];

    for i in 0..data_clone.len() >> 1 {
        index_high = (i.clone() + middle_index.clone()) % n.clone() as usize;
        data_clone[2 * i] = data[i.clone()].to_f64().unwrap();
        data_clone[2 * i.clone() + 1] = data[index_high.clone()].to_f64().unwrap();
    }

    println!("data clone: {:?} : ", &data_clone);

    for i in (0..l).step_by(2) {
        value_low = 0.0;
        value_high = 0.0;

        for l_p_ind in (0..inverse_low_pass_filter.len()).step_by(2) {
            index_low = ((l_p_ind.clone() as i32 + 1) % inverse_low_pass_filter.len() as i32) as usize;
            ind = ((l_p_ind.clone() as i32 + i.clone()) % data_clone.len() as i32) as usize;
            ind_next = ((l_p_ind.clone() as i32 + i.clone() + 1) % data_clone.len() as i32) as usize;

            // trend value
            value_low += data_clone[ind.clone()].to_f64().unwrap() * &inverse_low_pass_filter[index_low.clone()];
            value_low += data_clone[ind_next.clone()].to_f64().unwrap() * &inverse_high_pass_filter[index_low.clone()];

            // fluctuation value
            value_high += data_clone[ind.clone()].to_f64().unwrap() * &inverse_low_pass_filter[l_p_ind.clone()];
            value_high += data_clone[ind_next.clone()].to_f64().unwrap() * &inverse_high_pass_filter[l_p_ind.clone()];
        }

        set_value(&mut data_trans, value_low.clone(), &(ind_transform.clone() as usize));
        ind_transform += 1;
        set_value(&mut data_trans, value_high.clone(), &(ind_transform.clone() as usize));
        ind_transform += 1;
    }

    data_trans
}

pub fn set_value(data_trans: &mut Vec<f64>, value: f64, i: &usize) {
    if i >= &data_trans.len() {
        data_trans.push(value);
    } else {
        data_trans[i.clone()] = value;
    }
}