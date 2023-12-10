use std::fmt::Debug;
use num::Float;
use rand::{Rng};

pub fn initialize_weights<T: Clone>(
    num_layer_inputs_dim2: i32,
    number_hidden_neurons: i32,
    type_value: &T
) -> Vec<Vec<T>> {
    let mut weight_matrix: Vec<Vec<T>> = Vec::new();
    let mut rng = rand::thread_rng();

    for _i in 0..num_layer_inputs_dim2 {
        weight_matrix.push(Vec::new());
    }

    for i in 0..num_layer_inputs_dim2 as usize {
        for j in 0..number_hidden_neurons as usize {
            // weight_matrix[i][j] = rng.gen_range(-0.06, 0.6);
            let rand_value: f64 = rng.gen_range(-0.6,0.6);
            set_weights(&mut weight_matrix, &i, &j, type_value.clone());
            //  print!(",weight: {:?}", weight_matrix[i][j]);
        }

        // print!(",weights: {:?}", weight_matrix);
        // println!("");
    }

    //println!("size of weight matrix: number of rows: {}, columns: {}", weight_matrix.len(), weight_matrix[0].len());

    weight_matrix
}

pub fn set_weights<T>(weight_matrix: &mut Vec<Vec<T>>, i: &usize, j: &usize, value: T) {

    if j >= &weight_matrix[*i].len() {
        weight_matrix[*i].push(value);
    } else {
        weight_matrix[*i][*j] = value;
    }
}