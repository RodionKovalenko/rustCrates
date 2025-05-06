use std::fmt::Debug;

use crate::neural_networks::training::xquad_structs::{load_data_xquad_de, XQuADDataset};

// Define a generic trait for the Data structure
pub trait DataTrait<T: Debug + Clone, O: Debug + Clone> {
    fn new(input: Vec<T>, target: Vec<O>) -> Self;
    fn get_input(&self) -> &Vec<T>;
    fn get_target(&self) -> &Vec<O>;
    fn set_input(&mut self, val: Vec<T>);
    fn set_target(&mut self, val: Vec<O>);
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> Option<(Vec<T>, Vec<O>)>;
}

// Define the Data struct
#[derive(Debug, Clone)] // Automatically derive Debug + Clone
pub struct Dataset<T: Debug + Clone, O: Debug + Clone> {
    pub input: Vec<T>,  // List of tokens for input, flattened
    pub target: Vec<O>, // List of target labels, flattened
}

impl<T: Debug + Clone, O: Debug + Clone> Dataset<T, O> {
    pub fn iter(&self) -> impl Iterator<Item = (&T, &O)> {
        self.input.iter().zip(self.target.iter())
    }

    // Fetch a specific batch (index is the batch number)
    pub fn get_batch(&self, batch_idx: usize, batch_size: usize) -> Option<(Vec<T>, Vec<O>)> {
        let start_idx = batch_idx * batch_size;
        let end_idx = ((batch_idx + 1) * batch_size).min(self.input.len());

        if start_idx < self.input.len() {
            let batch_input = &self.input[start_idx..end_idx];
            let batch_target = &self.target[start_idx..end_idx];
            Some((batch_input.to_vec(), batch_target.to_vec()))
        } else {
            None
        }
    }

    // Split the dataset into batches
    pub fn split_into_batches(&self, batch_size: usize) -> Vec<Dataset<T, O>> {
        let mut batches = Vec::new();
        let num_batches = (self.input.len() + batch_size - 1) / batch_size; // Round up division

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = ((batch_idx + 1) * batch_size).min(self.input.len());

            let batch_input: Vec<T> = self.input[start_idx..end_idx].to_vec();
            let batch_target: Vec<O> = self.target[start_idx..end_idx].to_vec();

            let batch = Dataset::new(batch_input, batch_target);

            //println!("batch dataset: {:?}", &batch);
            batches.push(batch);
        }

        batches
    }
    pub fn extend_input_with_target(&self, input_batch: &Vec<String>, target_batch: &Vec<String>) -> Vec<String> {
        input_batch.clone().iter().zip(target_batch.iter()).map(|(input, target)| format!("<bos> {} <sep> {} <eos>", input, target)).collect()
    }

    pub fn extend_target(&self, target_batch: &Vec<String>) -> Vec<String> {
        target_batch.clone().iter().map(|target: &String| format!("{} <eos>", target)).collect()
    }
}

pub fn extend_input_with_bos(input_batch: &Vec<String>) -> Vec<String> {
    input_batch.clone().iter().map(|input: &String| format!("<bos> {} <sep> ", input)).collect()
}

// Implement the DataTrait for the Dataset struct
impl<T: Debug + Clone, O: Debug + Clone> DataTrait<T, O> for Dataset<T, O> {
    // Create a new instance of Dataset
    fn new(input: Vec<T>, target: Vec<O>) -> Self {
        Dataset { input, target }
    }

    // Get a reference to the input data
    fn get_input(&self) -> &Vec<T> {
        &self.input
    }

    // Get a reference to the target data
    fn get_target(&self) -> &Vec<O> {
        &self.target
    }

    // Set the input data
    fn set_input(&mut self, val: Vec<T>) {
        self.input = val;
    }

    // Set the target data
    fn set_target(&mut self, val: Vec<O>) {
        self.target = val;
    }

    // Fetch a batch
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> Option<(Vec<T>, Vec<O>)> {
        self.get_batch(batch_idx, batch_size)
    }
}

pub fn load_data_xquad_de_as_dataset() -> Result<Dataset<String, String>, Box<dyn std::error::Error>> {
    println!("Dataset is being loaded...");
    let dataset: XQuADDataset = load_data_xquad_de()?; // Returns nested structure

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    // Traverse nested structure to extract context, question, and answers
    for article in dataset.data {
        for paragraph in article.paragraphs {
            let context = paragraph.context;
            for qa in paragraph.qas {
                let input = format!("Context: {} \n <sep> Question: {}", context, qa.question);
                let target = qa.answers.get(0).map(|a| a.text.clone()).unwrap_or_else(|| "N/A".to_string());

                inputs.push(input);
                targets.push(target);
            }
        }
    }

    println!("Dataset is loaded: {} instances", inputs.len());

    Ok(Dataset::new(inputs, targets))
}
