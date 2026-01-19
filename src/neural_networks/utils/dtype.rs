use num::Complex;

/// Scalar type used across the neural network codebase.
///
/// Default is `f32`. Enable the Cargo feature `dtype-f64` to build the whole crate in `f64`.
#[cfg(feature = "dtype-f64")]
pub type Real = f64;

#[cfg(not(feature = "dtype-f64"))]
pub type Real = f32;

pub type C = Complex<Real>;

pub type CF64 = Complex<f64>;

pub const ZERO: Real = 0.0 as Real;
pub const ONE: Real = 1.0 as Real;
pub const TWO: Real = 2.0 as Real;
pub const PI: Real = std::f64::consts::PI as Real;

#[inline]
pub fn r(x: f64) -> Real {
    x as Real
}

#[inline]
pub fn c(re: f64, im: f64) -> C {
    Complex::new(r(re), r(im))
}

#[inline]
pub fn real_to_f64(x: Real) -> f64 {
    x as f64
}

#[inline]
pub fn real_from_f64(x: f64) -> Real {
    x as Real
}

#[inline]
pub fn c_to_f64(z: C) -> CF64 {
    Complex::new(z.re as f64, z.im as f64)
}

#[inline]
pub fn c_from_f64(z: CF64) -> C {
    Complex::new(z.re as Real, z.im as Real)
}
