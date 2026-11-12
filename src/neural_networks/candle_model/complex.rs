//! Complex tensors for candle: a complex value is a pair of real tensors (real part, imaginary part) of the
//! same shape. Every operation is written with real candle ops, so candle's autograd differentiates the real
//! loss with respect to the real and imaginary parts of every parameter separately.

use candle_core::{DType, Device, Result, Tensor, Var};

/// Complex tensor `re + i im`.
#[derive(Clone, Debug)]
pub struct CTensor {
    pub re: Tensor,
    pub im: Tensor,
}

impl CTensor {
    pub fn new(re: Tensor, im: Tensor) -> Self {
        Self { re, im }
    }

    /// A real tensor as a complex one (zero imaginary part).
    pub fn from_real(re: Tensor) -> Result<Self> {
        let im = re.zeros_like()?;
        Ok(Self { re, im })
    }

    /// From interleaved (re, im) values in row-major order with the given shape.
    pub fn from_pairs(pairs: &[(f64, f64)], shape: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        let re: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let im: Vec<f64> = pairs.iter().map(|p| p.1).collect();
        // Converted on the CPU, then copied: no GPU kernel is needed to create parameters.
        let make = |v: Vec<f64>| Tensor::from_vec(v, shape, &Device::Cpu)?.to_dtype(dtype)?.to_device(device);
        Ok(Self { re: make(re)?, im: make(im)? })
    }

    pub fn dims(&self) -> &[usize] {
        self.re.dims()
    }

    pub fn add(&self, o: &CTensor) -> Result<CTensor> {
        Ok(CTensor::new(self.re.broadcast_add(&o.re)?, self.im.broadcast_add(&o.im)?))
    }

    pub fn sub(&self, o: &CTensor) -> Result<CTensor> {
        Ok(CTensor::new(self.re.broadcast_sub(&o.re)?, self.im.broadcast_sub(&o.im)?))
    }

    /// Elementwise complex product (with broadcasting).
    pub fn mul(&self, o: &CTensor) -> Result<CTensor> {
        let re = self.re.broadcast_mul(&o.re)?.broadcast_sub(&self.im.broadcast_mul(&o.im)?)?;
        let im = self.re.broadcast_mul(&o.im)?.broadcast_add(&self.im.broadcast_mul(&o.re)?)?;
        Ok(CTensor::new(re, im))
    }

    /// Elementwise complex quotient (with broadcasting).
    pub fn div(&self, o: &CTensor) -> Result<CTensor> {
        let denom = o.re.sqr()?.broadcast_add(&o.im.sqr()?)?;
        let re = self.re.broadcast_mul(&o.re)?.broadcast_add(&self.im.broadcast_mul(&o.im)?)?.broadcast_div(&denom)?;
        let im = self.im.broadcast_mul(&o.re)?.broadcast_sub(&self.re.broadcast_mul(&o.im)?)?.broadcast_div(&denom)?;
        Ok(CTensor::new(re, im))
    }

    pub fn scale(&self, s: f64) -> Result<CTensor> {
        Ok(CTensor::new((&self.re * s)?, (&self.im * s)?))
    }

    /// Multiplies by a real tensor (with broadcasting).
    pub fn mul_real(&self, r: &Tensor) -> Result<CTensor> {
        Ok(CTensor::new(self.re.broadcast_mul(r)?, self.im.broadcast_mul(r)?))
    }

    /// `self @ w` for `self` of shape (..., k) and `w` of shape (k, n): complex matrix product.
    pub fn matmul(&self, w: &CTensor) -> Result<CTensor> {
        let re = self.re.broadcast_matmul(&w.re)?.sub(&self.im.broadcast_matmul(&w.im)?)?;
        let im = self.re.broadcast_matmul(&w.im)?.add(&self.im.broadcast_matmul(&w.re)?)?;
        Ok(CTensor::new(re, im))
    }

    /// Mean over the last dimension, keeping it (size 1).
    pub fn mean_keepdim_last(&self) -> Result<CTensor> {
        let d = self.re.rank() - 1;
        Ok(CTensor::new(self.re.mean_keepdim(d)?, self.im.mean_keepdim(d)?))
    }

    /// Principal square root: sqrt(z) = sqrt((|z| + re) / 2) + i sign(im) sqrt((|z| - re) / 2), with sign(+0) = +
    /// as in num_complex. The radicands are clamped at a tiny positive value so the gradient stays finite.
    pub fn sqrt(&self) -> Result<CTensor> {
        let abs = self.re.sqr()?.add(&self.im.sqr()?)?.sqrt()?;
        let tiny = 1e-30;
        let re = (abs.add(&self.re)? * 0.5)?.maximum(tiny)?.sqrt()?;
        let im_mag = (abs.sub(&self.re)? * 0.5)?.maximum(tiny)?.sqrt()?;
        let sign = self.im.ge(0.0)?.where_cond(&im_mag.ones_like()?, &im_mag.ones_like()?.neg()?)?;
        Ok(CTensor::new(re, im_mag.mul(&sign)?))
    }

    /// Complex sigmoid 1 / (1 + exp(-z)), evaluated as exp(z) / (1 + exp(z)) where re < 0 (no overflow).
    pub fn sigmoid(&self) -> Result<CTensor> {
        // For re >= 0: e = exp(-z) = exp(-re) (cos(im) - i sin(im)); sigma = 1 / (1 + e).
        // For re <  0: e = exp(z)  = exp(re)  (cos(im) + i sin(im)); sigma = e / (1 + e).
        let positive = self.re.ge(0.0)?;
        let mag = self.re.abs()?.neg()?.exp()?; // exp(-|re|), never overflows
        let (cos, sin) = (self.im.cos()?, self.im.sin()?);
        let e_re = mag.mul(&cos)?;
        let e_im_pos = mag.mul(&sin)?.neg()?;
        let e_im_neg = mag.mul(&sin)?;
        let e_im = positive.where_cond(&e_im_pos, &e_im_neg)?;
        let one_plus = CTensor::new((&e_re + 1.0)?, e_im.clone());
        let one = CTensor::from_real(e_re.ones_like()?)?;
        let numerator = CTensor::new(positive.where_cond(&one.re, &e_re)?, positive.where_cond(&one.im, &e_im)?);
        numerator.div(&one_plus)
    }

    /// Complex leaky ReLU: z if re > 0, else slope * z (the whole complex value is scaled).
    pub fn leaky_relu(&self, slope: f64) -> Result<CTensor> {
        let positive = self.re.gt(0.0)?;
        Ok(CTensor::new(positive.where_cond(&self.re, &(&self.re * slope)?)?, positive.where_cond(&self.im, &(&self.im * slope)?)?))
    }

    /// Concatenates along the last dimension.
    pub fn cat_last(parts: &[CTensor]) -> Result<CTensor> {
        let d = parts[0].re.rank() - 1;
        let re: Vec<&Tensor> = parts.iter().map(|p| &p.re).collect();
        let im: Vec<&Tensor> = parts.iter().map(|p| &p.im).collect();
        Ok(CTensor::new(Tensor::cat(&re, d)?, Tensor::cat(&im, d)?))
    }

    /// Splits the last dimension into `[0, n)` and `[n, 2n)`.
    pub fn split_half_last(&self) -> Result<(CTensor, CTensor)> {
        let d = self.re.rank() - 1;
        let n = self.re.dim(d)? / 2;
        Ok((CTensor::new(self.re.narrow(d, 0, n)?, self.im.narrow(d, 0, n)?), CTensor::new(self.re.narrow(d, n, n)?, self.im.narrow(d, n, n)?)))
    }

    /// Interleaved (re, im) values in row-major order (for tests and saving).
    pub fn to_pairs(&self) -> Result<Vec<(f64, f64)>> {
        let re = self.re.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
        let im = self.im.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
        Ok(re.into_iter().zip(im).collect())
    }
}

/// A trainable complex parameter: two candle variables.
#[derive(Clone, Debug)]
pub struct CVar {
    pub re: Var,
    pub im: Var,
}

impl CVar {
    pub fn from_pairs(pairs: &[(f64, f64)], shape: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        let t = CTensor::from_pairs(pairs, shape, dtype, device)?;
        Ok(Self { re: Var::from_tensor(&t.re)?, im: Var::from_tensor(&t.im)? })
    }

    pub fn tensor(&self) -> CTensor {
        CTensor::new(self.re.as_tensor().clone(), self.im.as_tensor().clone())
    }

    pub fn vars(&self) -> [&Var; 2] {
        [&self.re, &self.im]
    }
}
