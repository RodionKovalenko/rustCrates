use std::fmt::Debug;
use std::ops::{AddAssign, Mul, Neg};
use num_traits::{FromPrimitive, ToPrimitive};
use crate::wavelet_transform::dwt_type_resolver::{get_high_pass_filter, get_inverse_high_pass_filter, get_inverse_low_pass_filter, get_low_pass_filter};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
use crate::wavelet_transform::modes::WaveletMode;

pub fn transform_2_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<Vec<f64>> {
    let mut data_trans: Vec<Vec<f64>> = Vec::new();

    for r in data.iter() {
        data_trans.push(transform_1_d(&r, &dw_type, mode));
    }

    data_trans
}

pub fn transform_1_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data: &Vec<T>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<f64> {
    // moving averages filter
    let low_pass_filter: Vec<f64> = get_low_pass_filter(&dw_type);
    // moving differences filter
    let high_pass_filter: Vec<f64> = get_high_pass_filter(&dw_type);

    println!("low pass filter : {:?}", &low_pass_filter);
    println!("high pass filter : {:?}", &high_pass_filter);

    let mut middle_index = data.len() >> 1;
    let mut data_clone = data.clone();

    let mut value_high: f64;
    let mut value_low: f64;

    let mut index_high;
    let mut index_low;
    let mut ind_transform = 0;

    let mut n: i32 = data.len() as i32;
    let padding_size_before = high_pass_filter.len() - 2;

    middle_index = (n.clone() >> 1) as usize;

    // Padding before padding before data
    insert_padding_before(&mut data_clone, mode, padding_size_before);

    // update n and middle index
    n = data_clone.len() as i32;
    middle_index = (n.clone() >> 1) as usize;

    // Padding after
    if (data_clone.len() % high_pass_filter.len() != 0) {
        let index = (high_pass_filter.len() - (data_clone.len() % high_pass_filter.len()));
        insert_padding_after(&mut data_clone, mode, index, padding_size_before.clone());
    }

    // println!("data clone after padding: {:?}", &data_clone);
    // println!("data clone after padding length: {:?}", &data_clone.len());

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

pub fn inverse_transform_1_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data: &Vec<T>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<f64> {
    // inverse low pass filter (moving averages filter)
    let inverse_low_pass_filter: Vec<f64> = get_inverse_low_pass_filter(&dw_type);
    // inverse high pass filter (moving differences filter)
    let inverse_high_pass_filter: Vec<f64> = get_inverse_high_pass_filter(&dw_type);

    println!("inverse low pass filter : {:?}", &inverse_low_pass_filter);
    println!("inverse high pass filter : {:?}", &inverse_high_pass_filter);

    let mut value_high: f64;
    let mut value_low: f64;

    let mut index_low = 0;
    let mut ind_transform = 0;
    let mut ind_trend = 0;
    let mut index_high = 0;

    // original length of data l
    let mut l = (data.len() - (inverse_high_pass_filter.len() - 2)) as i32;

    let middle_index = data.len() >> 1;
    let mut data_trans: Vec<f64> = vec![0.0; l.clone() as usize];

    for i in (0..l >> 1).step_by(1) {
        value_low = 0.0;
        value_high = 0.0;

        for l_p_ind in (0..inverse_low_pass_filter.len()).step_by(2) {
            index_low = ((l_p_ind.clone() as i32 + 1) % inverse_low_pass_filter.len() as i32) as usize;
            ind_trend = ((l_p_ind.clone() as i32 / 2 + i.clone()) % data.len() as i32) as usize;
            index_high = ((ind_trend.clone() as i32 + middle_index.clone() as i32) % data.len() as i32) as usize;

            // value low = trend value + fluctuation value
            value_low += data[ind_trend.clone()].to_f64().unwrap() * &inverse_low_pass_filter[index_low.clone()];
            value_low += data[index_high.clone()].to_f64().unwrap() * &inverse_high_pass_filter[index_low.clone()];

            // high value = trend value + fluctuation value
            value_high += data[ind_trend.clone()].to_f64().unwrap() * &inverse_low_pass_filter[l_p_ind.clone()];
            value_high += data[index_high.clone()].to_f64().unwrap() * &inverse_high_pass_filter[l_p_ind.clone()];
        }

        set_value(&mut data_trans, value_low.clone(), &(ind_transform.clone()));
        ind_transform += 1;
        set_value(&mut data_trans, value_high.clone(), &(ind_transform.clone()));
        ind_transform += 1;
    }

    data_trans
}

pub fn inverse_transform_2_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<Vec<f64>> {
    let mut data_trans: Vec<Vec<f64>> = Vec::new();

    for r in data.iter() {
        data_trans.push(inverse_transform_1_d(&r, &dw_type, mode));
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

pub fn insert_padding_before<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data_trans: &mut Vec<T>, mode: &WaveletMode, size: usize)
{
    let mut tmp_ind = 0;
    let mut val: f64;
    let orig_len = data_trans.len();
    let origin_data = data_trans.clone();

    for _i in 0..size {
        match mode {
            WaveletMode::ZERO => data_trans.insert(0, T::from_f64(0.0).unwrap()),
            WaveletMode::CONSTANT => data_trans.insert(0, data_trans[_i].clone()),
            WaveletMode::SYMMETRIC => {
                tmp_ind = _i % orig_len.clone();
                data_trans.insert(0, origin_data[tmp_ind].clone());
            }
            WaveletMode::ANTISYMMETRIC => {
                tmp_ind = _i % orig_len.clone();
                data_trans.insert(0, -origin_data[tmp_ind].clone());
            }
            WaveletMode::REFLECT => {
                tmp_ind = (_i + 1) % orig_len.clone();
                data_trans.insert(0, origin_data[tmp_ind].clone());
            }
            WaveletMode::ANTIREFLECT => {
                tmp_ind = ((_i + 1) % orig_len.clone());
                val = 2.0 * T::into(origin_data[0].clone()) - T::into(origin_data[tmp_ind].clone());
                data_trans.insert(0, T::from_f64(val).unwrap());
            }
            WaveletMode::PERIODIC => {
                tmp_ind = origin_data.len() - 1 - ((_i) % origin_data.len());
                data_trans.insert(0, origin_data[tmp_ind].clone());
            }
            // WaveletMode::SMOOTH => 0,
            // WaveletMode::PERIODIZATION => 0,
            _ => {}
        }
    }
}

pub fn insert_padding_after<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data_trans: &mut Vec<T>, mode: &WaveletMode, size: usize, padding_len_before: usize)
{
    let orig_len = data_trans.len() - padding_len_before.clone();
    let origin_data = data_trans.clone();
    let orig_half_len = (data_trans.len() - padding_len_before) >> 1;
    let mut tmp_ind = 0;
    let mut val: f64;

    for _i in 0..size {
        match mode {
            WaveletMode::ZERO => data_trans.push(T::from_f64(0.0).unwrap()),
            WaveletMode::CONSTANT => data_trans.push(data_trans[data_trans.len() - 1].clone()),
            WaveletMode::SYMMETRIC => {
                tmp_ind = (orig_len.clone() - (_i % orig_half_len.clone())) - 1;
                data_trans.push(origin_data[tmp_ind]);
            }
            WaveletMode::ANTISYMMETRIC => {
                tmp_ind = (orig_len.clone() - (_i % orig_half_len.clone())) - 1;
                data_trans.push(-origin_data[tmp_ind]);
            }
            WaveletMode::REFLECT => {
                tmp_ind = (origin_data.len() - ((_i + 2) % origin_data.len()));
                data_trans.push(origin_data[tmp_ind]);
            }
            WaveletMode::ANTIREFLECT => {
                tmp_ind = (origin_data.len() - ((_i + 2) % origin_data.len()));
                val = 2.0 * T::into(origin_data[(orig_len.clone() - 1)].clone()) - T::into(origin_data[tmp_ind].clone());
                data_trans.push(T::from_f64(val).unwrap());
            }
            WaveletMode::PERIODIC => {
                tmp_ind = _i % orig_len.clone() + padding_len_before.clone();
                data_trans.push(origin_data[tmp_ind]);
            }
            // WaveletMode::SMOOTH => 0,
            // WaveletMode::PERIODIZATION => 0,
            _ => {}
        }
    }
}