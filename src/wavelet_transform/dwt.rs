use crate::wavelet_transform::dwt_type_resolver::{get_high_pass_filter, get_inverse_high_pass_filter, get_inverse_low_pass_filter, get_low_pass_filter};
use crate::wavelet_transform::dwt_types::DiscreteWaletetType;
use crate::wavelet_transform::modes::WaveletMode;
use num_traits::Num;
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

pub fn dwt_2d_partial<T>(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<Vec<T>>
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    let mut data_trans: Vec<Vec<T>> = Vec::new();

    for r in data.iter() {
        data_trans.push(dwt_1d(&r, &dw_type, mode));
    }

    data_trans
}

pub fn dwt_2d_full<T>(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<Vec<T>>
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    let mut data_trans: Vec<Vec<T>> = dwt_2d_partial(data, &dw_type, &mode);

    // println!("before transposed array: {:?}", &data_trans);
    data_trans = transpose(data_trans);
    // println!("transposed array: {:?}", &data_trans);

    data_trans = dwt_2d_partial(&data_trans, &dw_type, &mode);

    transpose(data_trans)
}

pub fn dwt_1d<T>(data: &Vec<T>, dw_type: &DiscreteWaletetType, mode: &WaveletMode) -> Vec<T>
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    let low_pass_filter: Vec<f64> = get_low_pass_filter(dw_type);
    let high_pass_filter: Vec<f64> = get_high_pass_filter(dw_type);

    let mut data_clone = data.clone();

    let mut ind_transform: usize = 0;
    let padding_size_before: usize = high_pass_filter.len() - 2;

    insert_padding_before(&mut data_clone, mode, padding_size_before);

    let mut n = data_clone.len();
    let mut middle_index = n >> 1;

    let mut padding_size_after = high_pass_filter.len() - (data_clone.len() % high_pass_filter.len()) + 1;

    if data_clone.len() % high_pass_filter.len() != 0 {
        padding_size_after = high_pass_filter.len() - (data_clone.len() % high_pass_filter.len()) + 2;
    }

    insert_padding_after(&mut data_clone, mode, padding_size_after, padding_size_before);

    if data.len() % 2 != 0 {
        n += 1;
        middle_index = n >> 1;
    }

    let mut data_trans: Vec<T> = vec![T::zero(); n];

    for i in (0..n).step_by(2) {
        let mut value_low = T::zero();
        let mut value_high = T::zero();

        for (l_p_ind, &low_p_f_value) in low_pass_filter.iter().enumerate() {
            let index = (i + l_p_ind) % data_clone.len();

            value_low = value_low + data_clone[index] * low_p_f_value;
            value_high = value_high + data_clone[index] * high_pass_filter[l_p_ind];
        }

        let index_high = ind_transform + middle_index;

        set_value(&mut data_trans, value_low, &ind_transform);
        set_value(&mut data_trans, value_high, &index_high);

        ind_transform += 1;
    }

    data_trans
}

pub fn inverse_dwt_1d<T>(data: &Vec<T>, dw_type: &DiscreteWaletetType, _mode: &WaveletMode, _level: u32) -> Vec<T>
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    let inverse_low_pass_filter: Vec<f64> = get_inverse_low_pass_filter(dw_type);
    let inverse_high_pass_filter: Vec<f64> = get_inverse_high_pass_filter(dw_type);

    let middle_index = data.len() >> 1;
    let l = data.len() - (inverse_high_pass_filter.len() - 2);

    let mut data_trans: Vec<T> = vec![T::zero(); l];
    let mut ind_transform = 0;

    for i in 0..(l >> 1) {
        let mut value_low = T::zero();
        let mut value_high = T::zero();

        for l_p_ind in (0..inverse_low_pass_filter.len()).step_by(2) {
            let index_low = (l_p_ind + 1) % inverse_low_pass_filter.len();
            let ind_trend = (l_p_ind / 2 + i) % data.len();
            let index_high = (ind_trend + middle_index) % data.len();

            value_low = value_low + data[ind_trend] * inverse_low_pass_filter[index_low] + data[index_high] * inverse_high_pass_filter[index_low];

            value_high = value_high + data[ind_trend] * inverse_low_pass_filter[l_p_ind] + data[index_high] * inverse_high_pass_filter[l_p_ind];
        }

        data_trans[ind_transform] = value_low;
        ind_transform += 1;
        data_trans[ind_transform] = value_high;
        ind_transform += 1;
    }

    data_trans
}

pub fn inverse_dwt_2d_partial<T>(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode, level: u32) -> Vec<Vec<T>>
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    let mut data_trans: Vec<Vec<T>> = Vec::new();

    for r in data.iter() {
        data_trans.push(inverse_dwt_1d(&r, &dw_type, mode, level.clone()));
    }

    data_trans
}

pub fn inverse_dwt_2d_full<T>(data: &Vec<Vec<T>>, dw_type: &DiscreteWaletetType, mode: &WaveletMode, level: u32) -> Vec<Vec<T>>
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    let mut data_trans: Vec<Vec<T>> = inverse_dwt_2d_partial(&data, &dw_type, &mode, level);

    data_trans = transpose(data_trans);

    transpose(inverse_dwt_2d_partial(&data_trans, &dw_type, &mode, level.clone()))
}

pub fn set_value<T>(data_trans: &mut Vec<T>, value: T, i: &usize)
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    if i >= &data_trans.len() {
        data_trans.push(value);
    } else {
        data_trans[i.clone()] = value;
    }
}

pub fn set_value_2d<T>(data_trans: &mut Vec<Vec<T>>, value: Vec<T>, i: &usize)
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    if i >= &data_trans.len() {
        data_trans.push(value);
    } else {
        data_trans[i.clone()] = value;
    }
}

pub fn insert_padding_before<T>(data_trans: &mut Vec<T>, mode: &WaveletMode, size: usize)
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    let mut tmp_ind: usize = 0;
    let orig_len = data_trans.len();
    let mut origin_data = data_trans.clone();

    for i in 0..size {
        match mode {
            WaveletMode::ZERO => data_trans.insert(0, T::zero()),

            WaveletMode::CONSTANT => data_trans.insert(0, data_trans[i].clone()),

            WaveletMode::SYMMETRIC => {
                tmp_ind = data_trans.len() % orig_len + i % orig_len;
                data_trans.insert(0, data_trans[tmp_ind].clone());
            }

            WaveletMode::ANTISYMMETRIC => {
                tmp_ind = data_trans.len() % orig_len + i % orig_len;
                data_trans.insert(0, -data_trans[tmp_ind].clone());
            }

            WaveletMode::REFLECT => {
                if i > 0 {
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

                let two = T::one() + T::one();
                let val = two * data_trans[i % (origin_data.len() - 1)].clone() - data_trans[(i % (origin_data.len() - 1) + 1 + tmp_ind) % data_trans.len()].clone();
                data_trans.insert(0, val);

                tmp_ind += 1;
            }

            WaveletMode::PERIODIC => {
                tmp_ind = origin_data.len() - 1 - (i % origin_data.len());
                data_trans.insert(0, origin_data[tmp_ind].clone());
            }

            WaveletMode::PERIODIZATION => {
                if i == 0 && origin_data.len() % 2 == 1 {
                    data_trans.push(origin_data[origin_data.len() - 1].clone());
                    origin_data = data_trans.clone();
                }
                tmp_ind = origin_data.len() - 1 - (i % origin_data.len());
                data_trans.insert(0, origin_data[tmp_ind].clone());
            }
        }
    }
}

pub fn insert_padding_after<T>(data_trans: &mut Vec<T>, mode: &WaveletMode, size: usize, padding_len_before: usize)
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    let origin_data = data_trans.clone();
    let mut tmp_ind = 0;
    let mut val: T;

    for i in 0..size {
        match mode {
            WaveletMode::ZERO => data_trans.push(T::zero()),

            WaveletMode::CONSTANT => {
                data_trans.push(data_trans[data_trans.len() - 1].clone());
            }

            WaveletMode::SYMMETRIC => {
                tmp_ind = data_trans.len() - (data_trans.len() % origin_data.len() + i % origin_data.len()) - 1;
                data_trans.push(data_trans[tmp_ind].clone());
            }

            WaveletMode::ANTISYMMETRIC => {
                tmp_ind = data_trans.len() - (data_trans.len() % origin_data.len() + i % origin_data.len()) - 1;
                data_trans.push(-data_trans[tmp_ind].clone());
            }

            WaveletMode::REFLECT => {
                tmp_ind = origin_data.len() - ((i + 1) % origin_data.len()) - 1;
                data_trans.push(origin_data[tmp_ind].clone());
            }

            WaveletMode::ANTIREFLECT => {
                if tmp_ind % (origin_data.len() - 1) == 0 {
                    tmp_ind = 0;
                }

                let two = T::one() + T::one();
                let ind = i % (origin_data.len() - 1);

                val = two * data_trans[data_trans.len() - ind - 1].clone() - data_trans[data_trans.len() - (ind + 2 + tmp_ind)].clone();

                data_trans.push(val);
                tmp_ind += 1;
            }

            WaveletMode::PERIODIC => {
                tmp_ind = i % (origin_data.len() - padding_len_before) + padding_len_before;
                data_trans.push(origin_data[tmp_ind].clone());
            }

            WaveletMode::PERIODIZATION => {
                tmp_ind = i % (origin_data.len() - padding_len_before) + padding_len_before;
                data_trans.push(origin_data[tmp_ind].clone());
            }
        }
    }
}

pub fn get_ll_hh<T>(data: &Vec<Vec<T>>) -> Vec<Vec<Vec<T>>>
where
    T: Num + Clone + Debug + Copy,
{
    // top left: average approximation
    let mut ll: Vec<Vec<T>> = Vec::new();
    let mut hh: Vec<Vec<T>> = Vec::new();

    let half_col_ind = data[0].len() >> 1;

    for row in data.iter() {
        ll.push(row[0..half_col_ind].to_vec());
        hh.push(row[half_col_ind..].to_vec());
    }

    vec![ll, hh]
}

pub fn combine_ll_hh<T>(ll_hh: &Vec<Vec<Vec<T>>>) -> Vec<Vec<T>>
where
    T: Num + Clone + Debug + Copy,
{
    let len_r = ll_hh[0].len();

    let mut combined_vec: Vec<Vec<T>> = ll_hh[0].clone();
    let mut ind_r: usize;

    for i in 0..len_r {
        ind_r = &i % ll_hh[0].len();
        combined_vec[i].extend_from_slice(&ll_hh[1][ind_r]);
    }

    combined_vec
}

pub fn get_ll_hl_lh_hh<T>(data: &Vec<Vec<T>>) -> Vec<Vec<Vec<T>>>
where
    T: Num + Clone + Debug + Copy,
{
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

pub fn combine_ll_lh_hl_hh<T>(ll_lh_hl_hh: &Vec<Vec<Vec<T>>>) -> Vec<Vec<T>>
where
    T: Num + Clone + Debug + Copy,
{
    let len_r = ll_lh_hl_hh[0].len() << 1;
    let len_c = ll_lh_hl_hh[0][0].len() << 1;

    let mut combined_vec: Vec<Vec<T>> = ll_lh_hl_hh[0].clone();
    let mut ind_r: usize;
    let mut ind_c: usize;

    for i in 0..len_r {
        for j in 0..len_c {
            if i >= combined_vec.len() {
                combined_vec.push(Vec::new());
            }

            ind_r = &i % ll_lh_hl_hh[0].len();
            ind_c = &j % ll_lh_hl_hh[0][0].len();

            if j >= ll_lh_hl_hh[0][0].len() {
                combined_vec[i].push(ll_lh_hl_hh[2][ind_r][ind_c].clone());
            }
            if i >= ll_lh_hl_hh[0].len() {
                combined_vec[i] = ll_lh_hl_hh[1][ind_r].clone();

                combined_vec[i].extend_from_slice(&ll_lh_hl_hh[3][ind_r]);
            }
        }
    }

    combined_vec
}

fn transpose<T>(original: Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Num + Clone + Debug + Copy + Neg<Output = T> + Sub<Output = T> + Add<Output = T> + Mul<f64, Output = T>,
{
    assert!(!original.is_empty());
    let mut transposed = (0..original[0].len()).map(|_| vec![]).collect::<Vec<_>>();

    for original_row in original {
        for (item, transposed_row) in original_row.into_iter().zip(&mut transposed) {
            transposed_row.push(item);
        }
    }

    transposed
}
