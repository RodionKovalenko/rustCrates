use crate::network_components::layer::{Layer};
use std::fmt::Debug;
use rand::Rng;

pub fn initialize_weights(
    num_layer_inputs_dim1: usize,
    num_layer_inputs_dim2: usize,
    number_hidden_neurons: usize,
) -> Vec<Vec<f64>> {
    let mut weight_matrix = vec![
        vec![0.0f64; number_hidden_neurons as usize];
        num_layer_inputs_dim2 as usize
    ];
    let mut rng = rand::thread_rng();

    for i in 0..num_layer_inputs_dim2 as usize {
        for j in 0..number_hidden_neurons as usize {
           weight_matrix[i][j] = rng.gen_range(-0.06, 0.6);
          //  print!(",weight: {:?}", weight_matrix[i][j]);
        }
     //   println!("");
    }

    println!("size of weight matrix: number of rows: {}, columns: {}", weight_matrix.len(), weight_matrix[0].len());

    weight_matrix
}