use std::fmt::Debug;
use std::ops::{AddAssign, Mul, Neg};
use num_traits::{FromPrimitive, Num, ToPrimitive};
use crate::utils::data_converter::{convert_to_f64_1d, convert_to_f64_2d};
use crate::utils::num_trait::{ArrayType, NumTrait};
use crate::wavelet_transform::dwt_type_resolver::{get_high_pass_filter, get_inverse_high_pass_filter, get_inverse_low_pass_filter, get_low_pass_filter};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
use crate::wavelet_transform::modes::WaveletMode;

pub fn transform_2_d_partial<T: NumTrait>(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<Vec<f64>> {
    let mut data_trans: Vec<Vec<f64>> = Vec::new();

    for r in data.iter() {
        data_trans.push(transform_1_d(&r, &dw_type, mode));
    }

    data_trans
}

pub fn transform_2_d_partial_f64(data: &Vec<Vec<f64>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<Vec<f64>> {
    let mut data_trans: Vec<Vec<f64>> = Vec::new();

    for r in data.iter() {
        data_trans.push(transform_1_df64(&r, &dw_type, mode));
    }

    data_trans
}

pub fn transform_2_d<T: ArrayType>(data: &T, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<Vec<f64>> {
    let data_f64: Vec<Vec<f64>> = convert_to_f64_2d(data);
    let mut data_trans: Vec<Vec<f64>> = transform_2_d_partial(&data_f64, &dw_type, &mode);

    // println!("before transposed array: {:?}", &data_trans);
    data_trans = transpose(data_trans);
    // println!("transposed array: {:?}", &data_trans);

    data_trans = transform_2_d_partial(&data_trans, &dw_type, &mode);

    transpose(data_trans)
}

fn transpose<T>(original: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!original.is_empty());
    let mut transposed = (0..original[0].len()).map(|_| vec![]).collect::<Vec<_>>();

    for original_row in original {
        for (item, transposed_row) in original_row.into_iter().zip(&mut transposed) {
            transposed_row.push(item);
        }
    }

    transposed
}

pub fn transform_2_df64(data: &Vec<Vec<f64>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<Vec<f64>> {
    let mut data_trans: Vec<Vec<f64>> = transform_2_d_partial_f64(&data, &dw_type, &mode);

    // println!("before transposed array: {:?}", &data_trans);
    data_trans = transpose(data_trans);
    // println!("transposed array: {:?}", &data_trans);

    data_trans = transform_2_d_partial_f64(&data_trans, &dw_type, &mode);

    transpose(data_trans)
}

pub fn transform_1_d<T: NumTrait>(data: &Vec<T>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<f64> {
    let data_clone: Vec<f64> = convert_to_f64_1d(data);

    transform_1_df64(&data_clone, &dw_type, &mode)
}
pub fn transform_1_df64(data: &Vec<f64>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<f64> {
    // moving averages filter
    let low_pass_filter: Vec<f64> = get_low_pass_filter(&dw_type);
    // moving differences filter
    let high_pass_filter: Vec<f64> = get_high_pass_filter(&dw_type);

    // println!("low pass filter : {:?}", &low_pass_filter);
    // println!("high pass filter : {:?}", &high_pass_filter);

    let mut middle_index;
    let mut data_clone: Vec<f64> = convert_to_f64_1d(data);

    let mut value_high: f64;
    let mut value_low: f64;

    let mut index_high;
    let mut index_low;
    let mut ind_transform = 0;
    let mut n: i32;
    let padding_size_before = high_pass_filter.len() - 2;
    // Padding before padding before data
    insert_padding_before(&mut data_clone, mode, padding_size_before);

    // update n and middle index
    n = data_clone.len() as i32;
    middle_index = (n.clone() >> 1) as usize;

    // Padding after
    let mut padding_size_after = high_pass_filter.len() - (data_clone.len() % high_pass_filter.len()) + 1;

    if data_clone.len() % high_pass_filter.len() != 0 {
        padding_size_after = high_pass_filter.len() - (data_clone.len() % high_pass_filter.len()) + 2;
    }

    insert_padding_after(&mut data_clone, mode, padding_size_after, padding_size_before.clone());

    // println!("data clone after padding: {:?}", &data_clone);
    // println!("data clone after padding length: {:?}", &data_clone.len());

    // for even number of elements in array add one and calculate middle index again
    if data.len() % 2 != 0 {
        n += 1;
        middle_index = (n >> 1) as usize;
    }

    let mut data_trans: Vec<f64> = vec![0.0; n.clone() as usize];

    for i in (0..n).step_by(2) {
        value_low = 0.0;
        value_high = 0.0;

        for (l_p_ind, low_p_f_value) in low_pass_filter.iter().enumerate() {
            index_low = ((i + l_p_ind as i32) % (data_clone.len() as i32)) as usize;

            // Low-Pass
            value_low += data_clone[index_low] * low_p_f_value;

            // High-Pass
            value_high += data_clone[index_low] * high_pass_filter[l_p_ind];
        }

        index_high = (ind_transform + middle_index as i32) as usize;

        set_value(&mut data_trans, value_low, &(ind_transform as usize));
        set_value(&mut data_trans, value_high, &index_high);

        ind_transform += 1;
    }


    data_trans
}

pub fn inverse_transform_1_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data: &Vec<T>, dw_type: &DiscreteWaletetType, _mode: &WaveletMode, _level: u32) -> Vec<f64> {
    // inverse low pass filter (moving averages filter)
    let inverse_low_pass_filter: Vec<f64> = get_inverse_low_pass_filter(&dw_type);
    // inverse high pass filter (moving differences filter)
    let inverse_high_pass_filter: Vec<f64> = get_inverse_high_pass_filter(&dw_type);

    let mut value_high: f64;
    let mut value_low: f64;

    let mut index_low;
    let mut ind_transform = 0;
    let mut ind_trend;
    let mut index_high;

    // original length of data l
    let l = (data.len() - (inverse_high_pass_filter.len() - 2)) as i32;

    let middle_index = data.len() >> 1;
    let mut data_trans: Vec<f64> = vec![0.0; l.clone() as usize];

    for i in (0..l >> 1).step_by(1) {
        value_low = 0.0;
        value_high = 0.0;

        for l_p_ind in (0..inverse_low_pass_filter.len()).step_by(2) {
            index_low = ((l_p_ind as i32 + 1) % inverse_low_pass_filter.len() as i32) as usize;
            ind_trend = ((l_p_ind as i32 / 2 + i) % data.len() as i32) as usize;
            index_high = ((ind_trend as i32 + middle_index as i32) % data.len() as i32) as usize;

            // value low = trend value + fluctuation value
            value_low += data[ind_trend].to_f64().unwrap() * &inverse_low_pass_filter[index_low];
            value_low += data[index_high].to_f64().unwrap() * &inverse_high_pass_filter[index_low];

            // high value = trend value + fluctuation value
            value_high += data[ind_trend].to_f64().unwrap() * &inverse_low_pass_filter[l_p_ind];
            value_high += data[index_high].to_f64().unwrap() * &inverse_high_pass_filter[l_p_ind];
        }

        set_value(&mut data_trans, value_low, &(ind_transform));
        ind_transform += 1;
        set_value(&mut data_trans, value_high, &(ind_transform));
        ind_transform += 1;
    }

    data_trans
}

pub fn inverse_transform_2_d_partial<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode, level: u32) -> Vec<Vec<f64>> {
    let mut data_trans: Vec<Vec<f64>> = Vec::new();

    for r in data.iter() {
        data_trans.push(inverse_transform_1_d(&r, &dw_type, mode, level.clone()));
    }

    data_trans
}

pub fn inverse_transform_2_d<T: Debug + Copy + FromPrimitive + Mul<T, Output=T> + Into<f64> + AddAssign + ToPrimitive + Neg<Output=T>>
(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode, level: u32) -> Vec<Vec<f64>> {
    let mut data_trans: Vec<Vec<f64>> = inverse_transform_2_d_partial(&data, &dw_type, &mode, level);

    data_trans = transpose(data_trans);

    transpose(inverse_transform_2_d_partial(&data_trans, &dw_type, &mode, level.clone()))
}

pub fn set_value<T: Num + Clone>(data_trans: &mut Vec<T>, value: T, i: &usize) {
    if i >= &data_trans.len() {
        data_trans.push(value);
    } else {
        data_trans[i.clone()] = value;
    }
}

pub fn set_value_2d<T: Num + Clone>(data_trans: &mut Vec<Vec<T>>, value: Vec<T>, i: &usize) {
    if i >= &data_trans.len() {
        data_trans.push(value);
    } else {
        data_trans[i.clone()] = value;
    }
}

pub fn insert_padding_before(data_trans: &mut Vec<f64>, mode: &WaveletMode, size: usize) {
    let mut tmp_ind: usize = 0;
    let mut val: f64;
    let orig_len = data_trans.len();
    let mut origin_data = data_trans.clone();

    for _i in 0..size {
        match mode {
            WaveletMode::ZERO => data_trans.insert(0, 0.0),
            WaveletMode::CONSTANT => data_trans.insert(0, data_trans[_i].clone()),
            WaveletMode::SYMMETRIC => {
                tmp_ind = data_trans.len() % orig_len.clone() + _i % orig_len.clone();
                data_trans.insert(0, data_trans[tmp_ind].clone());
            }
            WaveletMode::ANTISYMMETRIC => {
                tmp_ind = data_trans.len() % orig_len.clone() + _i % orig_len.clone();
                data_trans.insert(0, -data_trans[tmp_ind].clone());
            }
            WaveletMode::REFLECT => {
                if _i > 0 {
                    tmp_ind = (tmp_ind + 2) % data_trans.len();
                }
                if tmp_ind == 0 {
                    tmp_ind = 1;
                }

                data_trans.insert(0, data_trans[tmp_ind].clone());
            }
            WaveletMode::ANTIREFLECT => {
                if tmp_ind % (origin_data.len() - 1) == 0 {
                    tmp_ind = 0;
                }

                val = 2.0 * data_trans[_i % (origin_data.len() - 1)].clone() - data_trans[(_i % (origin_data.len() - 1) + 1 + tmp_ind) % data_trans.len()].clone();
                data_trans.insert(0, val);

                tmp_ind += 1;
            }
            WaveletMode::PERIODIC => {
                tmp_ind = origin_data.len() - 1 - ((_i) % origin_data.len());
                data_trans.insert(0, origin_data[tmp_ind].clone());
            }
            WaveletMode::PERIODIZATION => {
                if _i == 0 && origin_data.len() % 2 == 1 {
                    data_trans.push(origin_data[origin_data.len() - 1].clone());
                    origin_data = data_trans.clone();
                }
                tmp_ind = origin_data.len() - 1 - ((_i) % origin_data.len());
                data_trans.insert(0, origin_data[tmp_ind].clone());
            }
        }
    }
}

pub fn insert_padding_after(data_trans: &mut Vec<f64>, mode: &WaveletMode, size: usize, padding_len_before: usize)
{
    let origin_data = data_trans.clone();
    let mut tmp_ind = 0;
    let mut val: f64;

    for _i in 0..size {
        match mode {
            WaveletMode::ZERO => data_trans.push(0.0),
            WaveletMode::CONSTANT => data_trans.push(data_trans[data_trans.len() - 1].clone()),
            WaveletMode::SYMMETRIC => {
                tmp_ind = data_trans.len() - (data_trans.len() % origin_data.len() + _i % origin_data.len()) - 1;
                data_trans.push(data_trans[tmp_ind].clone());
            }
            WaveletMode::ANTISYMMETRIC => {
                tmp_ind = data_trans.len() - (data_trans.len() % origin_data.len() + _i % origin_data.len()) - 1;
                data_trans.push(-data_trans[tmp_ind].clone());
            }
            WaveletMode::REFLECT => {
                tmp_ind = origin_data.len() - ((_i + 1) % origin_data.len()) - 1;
                data_trans.push(origin_data[tmp_ind].clone());
            }
            WaveletMode::ANTIREFLECT => {
                if tmp_ind % (origin_data.len() - 1) == 0 {
                    tmp_ind = 0;
                }

                let ind = _i % (origin_data.len() - 1);
                val = 2.0 * data_trans[data_trans.len() - ind - 1].clone() - data_trans[data_trans.len() - (ind + 2 + tmp_ind)].clone();

                data_trans.push(val);
                tmp_ind += 1;
            }
            WaveletMode::PERIODIC => {
                tmp_ind = _i % (origin_data.len() - padding_len_before.clone()) + padding_len_before.clone();
                data_trans.push(origin_data[tmp_ind].clone());
            }
            WaveletMode::PERIODIZATION => {
                tmp_ind = _i % (origin_data.len() - padding_len_before.clone()) + padding_len_before.clone();
                data_trans.push(origin_data[tmp_ind].clone());
            }
        }
    }
}

pub fn get_ll_hl_lh_hh<T>(data: &Vec<Vec<T>>) -> Vec<Vec<Vec<T>>>
    where T: Num + Clone + PartialOrd + ToPrimitive + Debug + Copy {
    // top left: average approximation
    let mut ll: Vec<Vec<T>> = Vec::new();
    let mut lh: Vec<Vec<T>> = Vec::new();
    let mut hl: Vec<Vec<T>> = Vec::new();
    let mut hh: Vec<Vec<T>> = Vec::new();

    let half_row_ind = data.len() >> 1;
    let half_col_ind = data[0].len() >> 1;

    for (i, row) in data.iter().enumerate() {
        if i < half_row_ind {
            ll.push(row[0..half_col_ind].to_vec());
            lh.push(row[half_col_ind..].to_vec());
        } else {
            hl.push(row[0..half_col_ind].to_vec());
            hh.push(row[half_col_ind..].to_vec());
        }
    }

    vec![ll, hl, lh, hh]
}

pub fn combine_ll_lh_hl_hh(ll_lh_hl_hh: &Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
    let len_r = ll_lh_hl_hh[0].len() << 1;
    let len_c = ll_lh_hl_hh[0][0].len() << 1;

    let mut combined_vec: Vec<Vec<f64>> = ll_lh_hl_hh[0].clone();
    let mut ind_r: usize;
    let mut ind_c: usize;

    for i in 0..len_r {
        for j in 0..len_c {
            if i.clone() >= combined_vec.len() {
                combined_vec.push(Vec::new());
            }

            ind_r = &i % ll_lh_hl_hh[0].len();
            ind_c = &j % ll_lh_hl_hh[0][0].len();

            if j.clone() >= ll_lh_hl_hh[0][0].len() {
                combined_vec[i.clone()].push(ll_lh_hl_hh[1][ind_r.clone()][ind_c.clone()].clone());
            }
            if i.clone() >= ll_lh_hl_hh[0].len() {
                if j.clone() < ll_lh_hl_hh[0][0].len() {
                    combined_vec[i.clone()].push(ll_lh_hl_hh[2][ind_r.clone()][ind_c.clone()].clone());
                } else {
                    combined_vec[i.clone()].push(ll_lh_hl_hh[3][ind_r.clone()][ind_c.clone()].clone());
                }
            }
        }
    }

    combined_vec
}