use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

pub trait NumTrait: Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> + Sized + Debug + Clone
{
    fn to_f64(&self) -> f64;
}

impl NumTrait for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }
}

impl NumTrait for f32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl NumTrait for i32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl NumTrait for i64 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

#[derive(Debug)]
pub enum Array {
    Array1D(Vec<f64>),
    ArrayF321D(Vec<f32>),
    ArrayI32_1D(Vec<i32>),
    ArrayI64_1D(Vec<i64>),

    ArrayI32_2D(Vec<Vec<i32>>),
    ArrayI64_2D(Vec<Vec<i64>>),
    Array2D(Vec<Vec<f64>>),
    ArrayF32_2D(Vec<Vec<f32>>),

    ArrayI32_3D(Vec<Vec<Vec<i32>>>),
    ArrayI64_3D(Vec<Vec<Vec<i64>>>),
    Array3D(Vec<Vec<Vec<f64>>>),
    ArrayF32_3D(Vec<Vec<Vec<f32>>>),

    ArrayI32_4D(Vec<Vec<Vec<Vec<i32>>>>),
    ArrayI64_4D(Vec<Vec<Vec<Vec<i64>>>>),
    Array4D(Vec<Vec<Vec<Vec<f64>>>>),
    ArrayF32_4D(Vec<Vec<Vec<Vec<f32>>>>),

    ArrayI32_5D(Vec<Vec<Vec<Vec<Vec<i32>>>>>),
    ArrayI64_5D(Vec<Vec<Vec<Vec<Vec<i64>>>>>),
    Array5D(Vec<Vec<Vec<Vec<Vec<f64>>>>>),
    ArrayF32_5D(Vec<Vec<Vec<Vec<Vec<f32>>>>>),

    ArrayI32_6D(Vec<Vec<Vec<Vec<Vec<Vec<i32>>>>>>),
    ArrayI64_6D(Vec<Vec<Vec<Vec<Vec<Vec<i64>>>>>>),
    Array6D(Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>),
    ArrayF32_6D(Vec<Vec<Vec<Vec<Vec<Vec<f32>>>>>>),

    ArrayI32_7D(Vec<Vec<Vec<Vec<Vec<Vec<Vec<i32>>>>>>>),
    ArrayI64_7D(Vec<Vec<Vec<Vec<Vec<Vec<Vec<i64>>>>>>>),
    Array7D(Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>),
    ArrayF32_7D(Vec<Vec<Vec<Vec<Vec<Vec<Vec<f32>>>>>>>),

    ArrayI32_8D(Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<i32>>>>>>>>),
    ArrayI64_8D(Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<i64>>>>>>>>),
    Array8D(Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>>),
    ArrayF32_8D(Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f32>>>>>>>>),
}

pub trait ArrayType: Debug {
    type Element;
    fn dimension(&self) -> usize;
    fn len(&self) -> usize;
    fn as_f64_array(&self) -> Array;
    fn get_element_type(&self) -> Self::Element;
}

impl<T: NumTrait, const N: usize> ArrayType for [T; N] {
    type Element = Vec<f64>;
    fn dimension(&self) -> usize {
        1
    }
    fn len(&self) -> usize {
        N
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<f64> = self.iter().map(|item| item.to_f64()).collect();
        Array::Array1D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait, const N: usize, const M: usize> ArrayType for [[T; M]; N] {
    type Element = Vec<Vec<f64>>;
    fn dimension(&self) -> usize {
        2
    }
    fn len(&self) -> usize {
        N
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<f64>> = self.iter().map(|inner_vec| inner_vec.iter().map(|item| item.to_f64()).collect()).collect();
        Array::Array2D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait, const N: usize, const M: usize, const P: usize> ArrayType for [[[T; M]; N]; P] {
    type Element = Vec<Vec<Vec<f64>>>;
    fn dimension(&self) -> usize {
        3
    }
    fn len(&self) -> usize {
        N
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<f64>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect();
        Array::Array3D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait, const N: usize, const M: usize, const P: usize, const K: usize> ArrayType for [[[[T; M]; N]; P]; K] {
    type Element = Vec<Vec<Vec<Vec<f64>>>>;
    fn dimension(&self) -> usize {
        4
    }
    fn len(&self) -> usize {
        N
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<f64>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect()).collect();
        Array::Array4D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait, const N: usize, const M: usize, const P: usize, const K: usize, const O: usize> ArrayType for [[[[[T; M]; N]; P]; K]; O] {
    type Element = Vec<Vec<Vec<Vec<Vec<f64>>>>>;
    fn dimension(&self) -> usize {
        5
    }
    fn len(&self) -> usize {
        N
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<Vec<f64>>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter()
                    .map(|inner_inner_inner_inner_vec| inner_inner_inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect()).collect()).collect();
        Array::Array5D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait, const N: usize, const M: usize, const P: usize, const K: usize, const O: usize, const H: usize> ArrayType for [[[[[[T; M]; N]; P]; K]; O]; H] {
    type Element = Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>;
    fn dimension(&self) -> usize {
        6
    }
    fn len(&self) -> usize {
        N
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter()
                    .map(|inner_inner_inner_inner_vec| inner_inner_inner_inner_vec.iter()
                        .map(|inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect()).collect()).collect()).collect();
        Array::Array6D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait, const N: usize, const M: usize, const P: usize, const K: usize, const O: usize, const H: usize, const J: usize> ArrayType for [[[[[[[T; M]; N]; P]; K]; O]; H]; J] {
    type Element = Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>;
    fn dimension(&self) -> usize {
        7
    }
    fn len(&self) -> usize {
        N
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter()
                    .map(|inner_inner_inner_inner_vec| inner_inner_inner_inner_vec.iter()
                        .map(|inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_vec.iter()
                            .map(|inner_inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_inner_vec.iter()
                                .map(|item| item.to_f64()).collect()).collect()).collect()).collect()).collect()).collect()).collect();
        Array::Array7D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait, const N: usize, const M: usize, const P: usize, const K: usize, const O: usize, const H: usize, const J: usize, const I: usize> ArrayType for [[[[[[[[T; M]; N]; P]; K]; O]; H]; J]; I] {
    type Element = Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>>;
    fn dimension(&self) -> usize {
        8
    }
    fn len(&self) -> usize {
        N
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter()
                    .map(|inner_inner_inner_inner_vec| inner_inner_inner_inner_vec.iter()
                        .map(|inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_vec.iter()
                            .map(|inner_inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_inner_vec.iter().
                                map(|inner_inner_inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect()).collect()).collect()).collect()).collect()).collect();
        Array::Array8D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait> ArrayType for Vec<T> {
    type Element = Vec<f64>;
    fn dimension(&self) -> usize {
        1
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<f64> = self.iter()
            .map(|item| item.to_f64()).collect();

        Array::Array1D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait> ArrayType for Vec<Vec<T>> {
    type Element = Vec<Vec<f64>>;
    fn dimension(&self) -> usize {
        2
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<f64>> = self.iter()
            .map(|inner_vec| inner_vec.iter().map(|item| item.to_f64()).collect())
            .collect();
        Array::Array2D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait> ArrayType for Vec<Vec<Vec<T>>> {
    type Element = Vec<Vec<Vec<f64>>>;
    fn dimension(&self) -> usize {
        3
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<f64>>> = self.iter()
            .map(|inner_vec| inner_vec.iter()
                .map(|inner_inner_vec| inner_inner_vec.iter()
                    .map(|item| item.to_f64()).collect())
                .collect()
            )
            .collect();
        Array::Array3D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait> ArrayType for Vec<Vec<Vec<Vec<T>>>> {
    type Element = Vec<Vec<Vec<Vec<f64>>>>;
    fn dimension(&self) -> usize {
        4
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<f64>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect()).collect();
        Array::Array4D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait> ArrayType for Vec<Vec<Vec<Vec<Vec<T>>>>> {
    type Element = Vec<Vec<Vec<Vec<Vec<f64>>>>>;
    fn dimension(&self) -> usize {
        5
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<Vec<f64>>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter()
                    .map(|inner_inner_inner_inner_vec| inner_inner_inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect()).collect()).collect();
        Array::Array5D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait> ArrayType for Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>> {
    type Element = Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>;
    fn dimension(&self) -> usize {
        6
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter()
                    .map(|inner_inner_inner_inner_vec| inner_inner_inner_inner_vec.iter()
                        .map(|inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect()).collect()).collect()).collect();
        Array::Array6D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait> ArrayType for Vec<Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>> {
    type Element = Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>;
    fn dimension(&self) -> usize {
        7
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter()
                    .map(|inner_inner_inner_inner_vec| inner_inner_inner_inner_vec.iter()
                        .map(|inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_vec.iter()
                            .map(|inner_inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_inner_vec.iter()
                                .map(|item| item.to_f64()).collect()).collect()).collect()).collect()).collect()).collect()).collect();
        Array::Array7D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

impl<T: NumTrait> ArrayType for Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>>> {
    type Element = Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>>;
    fn dimension(&self) -> usize {
        8
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn as_f64_array(&self) -> Array {
        let result: Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>>
            = self.iter().map(|inner_vec| inner_vec.iter()
            .map(|inner_inner_vec| inner_inner_vec.iter()
                .map(|inner_inner_inner_vec| inner_inner_inner_vec.iter()
                    .map(|inner_inner_inner_inner_vec| inner_inner_inner_inner_vec.iter()
                        .map(|inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_vec.iter()
                            .map(|inner_inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_inner_vec.iter().
                                map(|inner_inner_inner_inner_inner_inner_inner_vec| inner_inner_inner_inner_inner_inner_inner_vec.iter().map(|item| item.to_f64()).collect()).collect()).collect()).collect()).collect()).collect()).collect()).collect();
        Array::Array8D(result)
    }
    fn get_element_type(&self) -> Self::Element {
        Vec::new()
    }
}

pub fn get_num_dimensions<T: ArrayType>(list: &T) -> usize {
    list.dimension()
}