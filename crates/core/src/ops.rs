use crate::{
    errors::{NanHandling, PipeError, check_finite},
    traits::{ElementOp, False, IsTrue, TransformOp, True},
};

#[allow(unused_imports)]
use num_traits::Float as _;

/// Normalize operation (point-wise)
/// Normalizes by standard deviation and mean
#[derive(Debug, Clone)]
pub struct Normalize {
    pub mean: f32,
    pub std: f32,
    pub nan_handling: NanHandling,
}

impl Default for Normalize {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
            nan_handling: NanHandling::default(),
        }
    }
}

impl ElementOp for Normalize {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let result = (data - self.mean) / self.std;
        check_finite(result, self.nan_handling)
    }
}

/// Division operation (point-wise)
#[derive(Debug, Clone)]
pub struct Div {
    pub factor: f32,
    pub nan_handling: NanHandling,
}

impl Default for Div {
    fn default() -> Self {
        Self {
            factor: 1.0,
            nan_handling: NanHandling::default(),
        }
    }
}

impl ElementOp for Div {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let result = data / self.factor;
        check_finite(result, self.nan_handling)
    }
}

/// Multiplication operation (point-wise)
#[derive(Debug, Clone)]
pub struct Multiply {
    pub factor: f32,
    pub nan_handling: NanHandling,
}

impl Default for Multiply {
    fn default() -> Self {
        Self {
            factor: 1.0,
            nan_handling: NanHandling::default(),
        }
    }
}

impl ElementOp for Multiply {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let result = data * self.factor;
        check_finite(result, self.nan_handling)
    }
}

/// Addition operation (point-wise)
#[derive(Debug, Clone)]
pub struct Add {
    pub value: f32,
    pub nan_handling: NanHandling,
}

impl Default for Add {
    fn default() -> Self {
        Self {
            value: 0.0,
            nan_handling: NanHandling::default(),
        }
    }
}

impl ElementOp for Add {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let result = data + self.value;
        check_finite(result, self.nan_handling)
    }
}

/// Subtraction operation (point-wise)
#[derive(Debug, Clone)]
pub struct Subtract {
    pub value: f32,
    pub nan_handling: NanHandling,
}

impl Default for Subtract {
    fn default() -> Self {
        Self {
            value: 0.0,
            nan_handling: NanHandling::default(),
        }
    }
}

impl ElementOp for Subtract {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let result = data - self.value;
        check_finite(result, self.nan_handling)
    }
}

/// Clamp operation (point-wise)
/// Clamps values between min and max
#[derive(Debug, Clone)]
pub struct Clamp {
    pub min: f32,
    pub max: f32,
    pub nan_handling: NanHandling,
}

impl ElementOp for Clamp {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        // Clamp itself produces finite values if min/max are finite
        // but input could be NaN/inf
        if !data.is_finite() {
            check_finite(data, self.nan_handling)
        } else {
            Ok(data.clamp(self.min, self.max))
        }
    }
}

/// Absolute value operation (point-wise)
#[derive(Debug, Clone)]
pub struct Abs {
    pub nan_handling: NanHandling,
}

impl Default for Abs {
    fn default() -> Self {
        Self {
            nan_handling: NanHandling::default(),
        }
    }
}

impl ElementOp for Abs {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let result = data.abs();
        check_finite(result, self.nan_handling)
    }
}

/// Power operation (point-wise)
#[derive(Debug, Clone)]
pub struct Pow {
    pub exponent: f32,
    pub nan_handling: NanHandling,
}

impl Default for Pow {
    fn default() -> Self {
        Self {
            exponent: 1.0,
            nan_handling: NanHandling::default(),
        }
    }
}

impl ElementOp for Pow {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let result = data.powf(self.exponent);
        check_finite(result, self.nan_handling)
    }
}

/// Square root operation (point-wise)
#[derive(Debug, Clone)]
pub struct Sqrt {
    pub nan_handling: NanHandling,
}

impl Default for Sqrt {
    fn default() -> Self {
        Self {
            nan_handling: NanHandling::default(),
        }
    }
}

impl ElementOp for Sqrt {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let result = data.sqrt();
        check_finite(result, self.nan_handling)
    }
}

/// Truncate operation - reduces the length of the data vector
#[derive(Debug, Clone, Copy)]
pub struct Truncate<const ORIGINAL_LEN: usize, const NEW_LEN: usize>;

impl<const ORIGINAL_LEN: usize, const NEW_LEN: usize> TransformOp for Truncate<ORIGINAL_LEN, NEW_LEN> {
    type IndexRemapping = True;

    const IN_LEN: usize = ORIGINAL_LEN;
    const OUT_LEN: usize = NEW_LEN;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        out_index
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        _: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // TODO: n?
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), NEW_LEN);
        }
        Ok(out)
    }
}

/// Transpose operation for 2D matrices stored in row-major order
#[derive(Debug, Clone, Copy)]
pub struct Transpose<const ROWS: usize, const COLS: usize>;

impl<const ROWS: usize, const COLS: usize> TransformOp for Transpose<ROWS, COLS> {
    type IndexRemapping = True;

    const IN_LEN: usize = ROWS * COLS;
    const OUT_LEN: usize = ROWS * COLS;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        let out_row = out_index / ROWS;
        let out_col = out_index % ROWS;
        out_col * COLS + out_row
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> Result<f32, PipeError> {
        let in_index = self.map_index(out_index);
        Ok(unsafe { *data.get_unchecked(in_index) })
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }
}

/// Pad operation - adds padding to the data
#[derive(Debug, Clone, Copy)]
pub struct Pad<const ORIGINAL_LEN: usize, const PADDED_LEN: usize> {
    pub pad_value: f32,
}

impl<const ORIGINAL_LEN: usize, const PADDED_LEN: usize> TransformOp
    for Pad<ORIGINAL_LEN, PADDED_LEN>
{
    type IndexRemapping = False;

    const IN_LEN: usize = ORIGINAL_LEN;
    const OUT_LEN: usize = PADDED_LEN;

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> Result<f32, PipeError> {
        if out_index < ORIGINAL_LEN {
            Ok(unsafe { *data.get_unchecked(out_index) })
        } else {
            Ok(self.pad_value)
        }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        let copy_len = ORIGINAL_LEN.min(n);
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), copy_len);
        }

        for i in copy_len..n {
            unsafe {
                *out.get_unchecked_mut(i) = self.pad_value;
            }
        }
        Ok(out)
    }
}

/// Reverse operation - reverses the order of elements
#[derive(Debug, Clone, Copy)]
pub struct Reverse<const LEN: usize>;

impl<const LEN: usize> TransformOp for Reverse<LEN> {
    type IndexRemapping = True;

    const IN_LEN: usize = LEN;
    const OUT_LEN: usize = LEN;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        LEN - 1 - out_index
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> Result<f32, PipeError> {
        let in_index = self.map_index(out_index);
        Ok(unsafe { *data.get_unchecked(in_index) })
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }
}
