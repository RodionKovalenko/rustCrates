use std::fmt::Debug;

// Define a generic trait for the Data structure
pub trait DataTrait<T: Debug, O: Debug> {
    fn new(input: Vec<Vec<T>>, target: Vec<Vec<O>>) -> Self;
    fn get_input(&self) -> &Vec<Vec<T>>;
    fn get_target(&self) -> &Vec<Vec<O>>;
    fn set_input(&mut self, val: Vec<Vec<T>>);
    fn set_target(&mut self, val: Vec<Vec<O>>);
}

// Define the Data struct
#[derive(Debug, Clone)] // Automatically derive Debug and Clone
pub struct Data<T: Debug, O: Debug> {
    pub input: Vec<Vec<T>>,
    pub target: Vec<Vec<O>>,
}

// Implement the DataTrait for the Data struct
impl<T: Debug, O: Debug> DataTrait<T, O> for Data<T, O> {
    // Create a new instance of Data
    fn new(input: Vec<Vec<T>>, target: Vec<Vec<O>>) -> Self {
        Data { input, target }
    }

    // Get a reference to the input data
    fn get_input(&self) -> &Vec<Vec<T>> {
        &self.input
    }

    // Get a reference to the target data
    fn get_target(&self) -> &Vec<Vec<O>> {
        &self.target
    }

    // Set the input data
    fn set_input(&mut self, val: Vec<Vec<T>>) {
        self.input = val;
    }

    // Set the target data
    fn set_target(&mut self, val: Vec<Vec<O>>) {
        self.target = val;
    }
}