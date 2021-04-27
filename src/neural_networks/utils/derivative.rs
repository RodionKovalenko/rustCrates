use crate::network_components::layer::ActivationType;
use std::fmt::Debug;
use std::ops::{Mul, AddAssign, Sub, Add, Div};

pub fn get_derivative<T: Debug + Clone + Mul<Output=T> + From<f64> + AddAssign
+ Into<f64> + Sub<Output=T> + Add<Output=T> + Div<Output=T>>
(value: T, deriv_type: ActivationType) -> T {
    let mut derivative: T = T::from(0.0);
    if matches!(deriv_type, ActivationType::SIGMOID) {
        derivative = value.clone() * (T::from(1.0) - value.clone())
    }
    if matches!(deriv_type, ActivationType::TANH) {
        derivative = (T::from(1.0) - (value.clone() * value.clone()))
    }

    derivative
}