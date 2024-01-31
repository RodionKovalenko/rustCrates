use std::fmt::Debug;

pub trait DataTrait<T: Debug, O: Debug> {
    fn new(input: Vec<Vec<T>>, target: Vec<Vec<O>>) -> Self;
    fn get_input(&self) -> Vec<Vec<T>>;
    fn get_target(&self) -> Vec<Vec<O>>;
    fn set_input(&mut self, val: Vec<Vec<T>>);
    fn set_target(&mut self, val: Vec<Vec<O>>);
}

pub struct Data<T: Debug, O: Debug> {
    pub input: Vec<Vec<T>>,
    pub target: Vec<Vec<O>>,
}

impl<T: Debug + Clone + From<f64>, O: Debug + Clone + From<f64>> Data<T, O> {
    pub fn get_input(&self) -> Vec<Vec<T>> {
        self.input.clone()
    }
    pub fn get_target(&self) -> Vec<Vec<O>> {
        self.target.clone()
    }
}

impl<T: Debug + Clone + From<f64>, O: Debug + Clone + From<f64>> Clone for Data<T, O> {
    fn clone(&self) -> Self {
        Data {
            input: self.get_input(),
            target: self.get_target(),
        }
    }
}

impl <T: Debug + Clone + From<f64>, O: Debug + Clone + From<f64>> DataTrait<T, O> for Data<T, O> {
    fn new(input: Vec<Vec<T>>, target: Vec<Vec<O>>) -> Data<T, O> {
        Data { input, target }
    }
    fn get_input(&self) -> Vec<Vec<T>> {
        self.input.clone()
    }

    fn get_target(&self) -> Vec<Vec<O>> {
        self.target.clone()
    }

    fn set_input(&mut self, val: Vec<Vec<T>>) {
        self.input = val;
    }

    fn set_target(&mut self, val: Vec<Vec<O>>) {
        self.target = val;
    }
}