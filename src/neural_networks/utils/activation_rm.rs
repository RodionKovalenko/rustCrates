use crate::neural_networks::{
    network_layers::layer::ActivationType,
    utils::{
        activation::{gelu_complex, sigmoid_complex, tanh_complex},
        dtype::{C, ZERO},
        matrix::RowMajorMatrix,
    },
};

pub fn swish_rm(b: &RowMajorMatrix<C>) -> RowMajorMatrix<C> {
    let mut out = RowMajorMatrix::from_data(b.rows, b.cols, vec![C::new(ZERO, ZERO); b.rows * b.cols]);
    for i in 0..out.data.len() {
        let z = b.data[i];
        out.data[i] = z * sigmoid_complex(&z);
    }
    out
}

pub fn activate_output_complex_rm(mut data: &RowMajorMatrix<C>, activation: &ActivationType) -> RowMajorMatrix<C> {
    let mut input_data = data.clone();
    match activation {
        ActivationType::LINEAR => input_data,
        ActivationType::TANH => {
            for v in input_data.data.iter_mut() {
                *v = tanh_complex(*v);
            }
            input_data
        }
        ActivationType::RELU => {
            for v in input_data.data.iter_mut() {
                *v = if v.re > ZERO { *v } else { C::new(ZERO, ZERO) };
            }
            input_data
        }
        ActivationType::LEAKYRELU => {
            for v in input_data.data.iter_mut() {
                *v = if v.re > ZERO { *v } else { *v * 0.01 };
            }
            input_data
        }
        ActivationType::SIGMOID => {
            for v in input_data.data.iter_mut() {
                *v = sigmoid_complex(v);
            }
            input_data
        }
        ActivationType::GELU => {
            for v in input_data.data.iter_mut() {
                *v = gelu_complex(v);
            }
            input_data
        }
        ActivationType::SWiGLU => {
            let cols = data.cols;
            assert!(cols % 2 == 0, "SWiGLU expects an even column count");
            let half = cols / 2;
            let mut out = RowMajorMatrix::from_data(data.rows, half, vec![C::new(ZERO, ZERO); data.rows * half]);
            for r_i in 0..data.rows {
                let row = data.row_range(r_i);
                let out_row = out.row_range(r_i);
                for c in 0..half {
                    let a = data.data[row.start + c];
                    let b = data.data[row.start + half + c];
                    out.data[out_row.start + c] = a * b * sigmoid_complex(&b);
                }
            }
            out
        }
        _ => input_data,
    }
}
