use std::fmt::Debug;
use num_traits::{Float, FromPrimitive};
use rand::Rng;

pub fn initialize_weights(
    num_layer_inputs_dim2: usize,
    number_hidden_neurons: usize,
    weight_matrix: &mut Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();

    for _i in 0..num_layer_inputs_dim2 {
        weight_matrix.push(Vec::new());
    }

    for i in 0..num_layer_inputs_dim2 as usize {
        for j in 0..number_hidden_neurons.clone() as usize {
            let random_value = rng.gen_range(-0.6..0.6);

            set_weights(weight_matrix, i.clone(), j.clone(), random_value);
           // print!(",weight: {:?}", weight_matrix[i][j]);
        }
        // println!("");
    }

    //println!("size of weight matrix: number of rows: {}, columns: {}", weight_matrix.len(), weight_matrix[0].len());
}


pub fn set_weights<T: Debug + Clone + Float + FromPrimitive>(weight_matrix: &mut Vec<Vec<T>>, i: usize, j: usize, value: T) {
    if j >= weight_matrix[i].len() {
        weight_matrix[i.clone()].push(value);
    } else {
        weight_matrix[i.clone()][j.clone()] = value;
    }
}