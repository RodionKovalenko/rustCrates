pub fn initialize_layer() {
    println!("crate neural networks is working");
}

trait Layer {
    type Weights = Vec<Vec<f64>>;
    type Bias = u8;
    type InaktivatedInput = Vec<Vec<f64>>;
    type AktivatedInput = Vec<Vec<f64>>;
    type Errors = Vec<Vec<f64>>;
    type Gradient = Vec<Vec<f64>>;
    type ActivationType = String;
}

pub struct InputLayer {}

pub struct HiddenLayer {}

pub struct OutputLayer {}

impl Layer for InputLayer {}

impl Layer for HiddenLayer {}

impl Layer for OutputLayer {}
