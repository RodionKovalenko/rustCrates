use std::fmt::Debug;

use crate::neural_networks::training::xquad_structs::{load_data_xquad_de, XQuADDataset};
use rand::rng;
use rand::seq::SliceRandom;

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
    pub total_training_records_size: usize,
    pub total_validation_records_size: usize,
}

impl<T: Debug + Clone, O: Debug + Clone> Dataset<T, O> {
    pub fn iter(&self) -> impl Iterator<Item = (&T, &O)> {
        self.input.iter().zip(self.target.iter())
    }

    // Setup train/validation/test split with validation set fixed at 10%
    // test_ratio: Optional fraction for test set (e.g., 0.1 for 10%)
    // If test_ratio is None: 90% train, 10% validation
    // If test_ratio is Some(0.1): 80% train, 10% validation, 10% test
    pub fn setup_splits(&mut self, test_ratio: Option<f64>) {
        let total_size = self.input.len();
        let val_ratio = 0.1; // Fixed 10% for validation
        
        if let Some(test_r) = test_ratio {
            // Three-way split: train / val (10%) / test
            let test_size = (total_size as f64 * test_r).round() as usize;
            let val_size = (total_size as f64 * val_ratio).round() as usize;
            let train_size = total_size.saturating_sub(val_size).saturating_sub(test_size);
            
            self.total_training_records_size = train_size;
            self.total_validation_records_size = val_size;
        } else {
            // Two-way split: train / val (10%)
            let val_size = (total_size as f64 * val_ratio).round() as usize;
            let train_size = total_size.saturating_sub(val_size);
            
            self.total_training_records_size = train_size;
            self.total_validation_records_size = val_size;
        }
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

    // Split the TRAINING data into batches (shuffled)
    // Only processes data up to total_training_records_size to preserve validation/test sets
    pub fn split_into_batches(&self, batch_size: usize) -> Vec<Dataset<T, O>> {
        let train_size = if self.total_training_records_size > 0 {
            self.total_training_records_size
        } else {
            self.input.len() // If not set, use all data
        };

        let mut indices: Vec<usize> = (0..train_size).collect();
        let mut rng = rng();
        indices.shuffle(&mut rng);

        let mut shuffled_input: Vec<T> = Vec::with_capacity(train_size);
        let mut shuffled_target: Vec<O> = Vec::with_capacity(train_size);
        for &i in &indices {
            shuffled_input.push(self.input[i].clone());
            shuffled_target.push(self.target[i].clone());
        }

        let mut batches = Vec::new();
        let num_batches = (shuffled_input.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = ((batch_idx + 1) * batch_size).min(shuffled_input.len());

            let batch_input: Vec<T> = shuffled_input[start_idx..end_idx].to_vec();
            let batch_target: Vec<O> = shuffled_target[start_idx..end_idx].to_vec();

            let batch = Dataset::new(batch_input, batch_target);
            batches.push(batch);
        }

        batches
    }

    // Get validation batches (NOT shuffled) - preserves order for consistent evaluation
    pub fn get_validation_batches(&self, batch_size: usize) -> Vec<Dataset<T, O>> {
        let train_size = self.total_training_records_size;
        let val_size = self.total_validation_records_size;
        
        if val_size == 0 || train_size >= self.input.len() {
            return Vec::new();
        }

        let val_start = train_size;
        let val_end = (train_size + val_size).min(self.input.len());

        let mut batches = Vec::new();
        let num_batches = (val_size + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start_idx = val_start + batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(val_end);

            let batch_input: Vec<T> = self.input[start_idx..end_idx].to_vec();
            let batch_target: Vec<O> = self.target[start_idx..end_idx].to_vec();

            let batch = Dataset::new(batch_input, batch_target);
            batches.push(batch);
        }

        batches
    }

    // Get test batches (NOT shuffled) - preserves order, starts after train+val
    pub fn get_test_batches(&self, batch_size: usize) -> Vec<Dataset<T, O>> {
        let test_start = self.total_training_records_size + self.total_validation_records_size;
        
        if test_start >= self.input.len() {
            return Vec::new();
        }

        let mut batches = Vec::new();
        let test_size = self.input.len() - test_start;
        let num_batches = (test_size + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start_idx = test_start + batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(self.input.len());

            let batch_input: Vec<T> = self.input[start_idx..end_idx].to_vec();
            let batch_target: Vec<O> = self.target[start_idx..end_idx].to_vec();

            let batch = Dataset::new(batch_input, batch_target);
            batches.push(batch);
        }

        batches
    }
    pub fn extend_input_with_target(&self, input_batch: &Vec<String>, target_batch: &Vec<String>) -> Vec<String> {
        input_batch.clone().iter().zip(target_batch.iter()).map(|(input, target)| format!("<bos> {} <sep> {} <eos>", input, target)).collect()
    }

    pub fn extend_target(&self, target_batch: &Vec<String>) -> Vec<String> {
        target_batch.clone().iter().map(|target: &String| format!(" <sep> {} <eos>", target)).collect()
    }
}

pub fn extend_input_with_bos(input_batch: &Vec<String>) -> Vec<String> {
    input_batch.clone().iter().map(|input: &String| format!("<bos> {}", input)).collect()
}

pub fn concat_batches(a: &Vec<Vec<u32>>, b: &Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    a.iter()
        .zip(b.iter())
        .map(|(row_a, row_b)| {
            let mut combined = row_a.clone();
            combined.extend(row_b);
            combined
        })
        .collect()
}

// Implement the DataTrait for the Dataset struct
impl<T: Debug + Clone, O: Debug + Clone> DataTrait<T, O> for Dataset<T, O> {
    // Create a new instance of Dataset
    fn new(input: Vec<T>, target: Vec<O>) -> Self {
       let mut dataset =  Dataset { input, target, total_training_records_size: 0, total_validation_records_size: 0 };
       dataset.setup_splits(Some(0.1));

       dataset
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
