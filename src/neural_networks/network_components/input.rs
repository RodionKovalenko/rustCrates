use std::fmt::Debug;

pub trait DataTrait<T> {
    fn new(input: Vec<Vec<T>>, target: Vec<Vec<T>>) -> Self;
    fn get_input(&self) -> Vec<Vec<T>>;
    fn get_target(&self) -> Vec<Vec<T>>;
    fn set_input(&mut self, val: Vec<Vec<T>>);
    fn set_target(&mut self, val: Vec<Vec<T>>);
}

pub struct Data<T> {
    pub input: Vec<Vec<T>>,
    pub target: Vec<Vec<T>>,
}

impl <T: Debug + Clone> DataTrait<T> for Data<T> {
    fn new(input: Vec<Vec<T>>, target: Vec<Vec<T>>) -> Data<T> {
        Data { input, target }
    }
    fn get_input(&self) -> Vec<Vec<T>> {
        self.input.clone()
    }

    fn get_target(&self) -> Vec<Vec<T>> {
        self.target.clone()
    }

    fn set_input(&mut self, val: Vec<Vec<T>>) {
        self.input = val;
    }

    fn set_target(&mut self, val: Vec<Vec<T>>) {
        self.target = val;
    }
}