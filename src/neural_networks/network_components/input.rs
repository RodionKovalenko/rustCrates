use std::fmt::Debug;

// Define a generic trait for the Data structure
pub trait DataTrait<T: Debug, O: Debug> {
    fn new(input: Vec<T>, target: Vec<O>) -> Self;
    fn get_input(&self) -> &Vec<T>;
    fn get_target(&self) -> &Vec<O>;
    fn set_input(&mut self, val: Vec<T>);
    fn set_target(&mut self, val: Vec<O>);
}

// Define the Data struct
#[derive(Debug, Clone)] // Automatically derive Debug and Clone
pub struct Dataset<T: Debug, O: Debug> {
    pub input: Vec<T>,
    pub target: Vec<O>,
}

// Implement the `iter` method for Dataset when T and O are Sized
impl<T: Debug, O: Debug> Dataset<T, O> {
    pub fn iter(&self) -> impl Iterator<Item = (&T, &O)> {
        self.input.iter().zip(self.target.iter())
    }
}

// Implement the DataTrait for the Data struct
impl<T: Debug, O: Debug> DataTrait<T, O> for Dataset<T, O> {
    // Create a new instance of Data
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
}