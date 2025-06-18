use core::fmt::Debug;
use num::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::neural_networks::network_components::linear_layer::LinearLayer;

use super::{gradient_struct::Gradient, layer_input_struct::LayerInput, layer_output_struct::LayerOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLinearLayer {
    pub layers: Vec<LinearLayer>,
    pub learning_rate: f64,
    #[serde(skip)]
    pub input_batch: Option<Vec<Vec<Vec<Complex<f64>>>>>,
    #[serde(skip)]
    pub gradient: Option<Gradient>,
    #[serde(skip)]
    pub previous_gradient: Option<Gradient>,
    pub time_step: usize,
}

impl MultiLinearLayer {
    pub fn new(learning_rate: f64, rows: usize, cols: usize, num_layers: usize) -> Self {
        let mut layers: Vec<LinearLayer> = vec![];
        let num_cols_compressed = cols / num_layers;

        for _i in 0..num_layers {
            let linear_layer = LinearLayer::new(learning_rate, rows, num_cols_compressed);
            layers.push(linear_layer);
        }

        Self {
            layers: layers,
            learning_rate: learning_rate,
            input_batch: None,
            gradient: None,
            previous_gradient: None,
            time_step: 0,
        }
    }
    pub fn forward(&mut self, input: &LayerInput) -> LayerOutput {
        let input_batch: Vec<Vec<Vec<Complex<f64>>>> = input.get_input_batch();
        self.input_batch = Some(input_batch.clone());
        self.time_step = input.get_time_step();

        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        let lin_layer_output_chunks = self
            .layers
            .par_iter_mut()
            .map(|lin_layer| {
                let lin_output = lin_layer.forward(&input);
                lin_output.get_output_batch()
            })
            .collect::<Vec<_>>();

        let mut output_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); 0]; seq_len]; batch_size];

        // Parallelize over (b, i) pairs
        output_batch.par_iter_mut().enumerate().for_each(|(b, batch_row)| {
            for i in 0..seq_len {
                let mut combined_output = Vec::new();
                for lin_layer_output_chunk in &lin_layer_output_chunks {
                    combined_output.extend_from_slice(&lin_layer_output_chunk[b][i]);
                }
                batch_row[i] = combined_output;
            }
        });

        let mut layer_output = LayerOutput::new_default();
        layer_output.set_output_batch(output_batch);

        layer_output
    }

    pub fn backward(&mut self, previous_gradient: &Gradient) -> Gradient {
        let input_batch = self.input_batch.as_ref().expect("Input batch is missing in linear layer");
        let mut gradient = Gradient::new_default();
        let batch_size = input_batch.len();
        let seq_len = input_batch[0].len();

        let gradient_input_batch_chunks = self
            .layers
            .par_iter_mut()
            .map(|lin_layer| {
                let gradient = lin_layer.backward(&previous_gradient);
                gradient.get_gradient_input_batch()
            })
            .collect::<Vec<_>>();

        let mut gradient_input_batch: Vec<Vec<Vec<Complex<f64>>>> = vec![vec![vec![Complex::new(0.0, 0.0); 0]; seq_len]; batch_size];

        // Parallelize over (b, i) pairs
        gradient_input_batch.par_iter_mut().enumerate().for_each(|(b, batch_row)| {
            for i in 0..seq_len {
                let mut combined_output = Vec::new();
                for lin_layer_output_chunk in &gradient_input_batch_chunks {
                    combined_output.extend_from_slice(&lin_layer_output_chunk[b][i]);
                }
                batch_row[i] = combined_output;
            }
        });

        gradient.set_gradient_input_batch(gradient_input_batch);

        gradient
    }

    pub fn update_parameters(&mut self) {
        self.layers.par_iter_mut().for_each(|lin_layer| lin_layer.update_parameters());
    }
}
