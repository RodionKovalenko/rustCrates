use std::fmt::Debug;
use rand::Rng;

pub fn initialize_weights<T: From<f64> + Clone + Debug>(
    num_layer_inputs_dim2: usize,
    number_hidden_neurons: usize,
) -> Vec<Vec<T>> {
    let mut weight_matrix: Vec<Vec<T>> = Vec::new();
    let mut rng = rand::thread_rng();

    for _i in 0..num_layer_inputs_dim2 {
        weight_matrix.push(Vec::new());
    }

    for i in 0..num_layer_inputs_dim2 as usize {
        for j in 0..number_hidden_neurons as usize {
            // weight_matrix[i][j] = rng.gen_range(-0.06, 0.6);
            let value = rng.gen_range(-0.6, 0.6);
            if j >= weight_matrix[i].len() {
                weight_matrix[i].push(T::from(value));
            } else {
                weight_matrix[i][j] = T::from(value);
            }
            //  print!(",weight: {:?}", weight_matrix[i][j]);
        }
        //   println!("");
    }

    println!("size of weight matrix: number of rows: {}, columns: {}", weight_matrix.len(), weight_matrix[0].len());

    weight_matrix
}