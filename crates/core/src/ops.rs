use crate::{
    errors::PipeError,
    traits::{ElementOp, False, IsTrue, TransformOp, True},
};

#[allow(unused_imports)]
use num_traits::Float as _;

/// Normalize operation (point-wise)
/// Normalizes by standard deviation and mean
#[derive(Debug, Clone, Default)]
pub struct Normalize {
    pub mean: f32,
    pub std: f32,
}

impl ElementOp for Normalize {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        (data - self.mean) / self.std
    }
}

/// Division operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Div {
    pub factor: f32,
}

impl ElementOp for Div {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data / self.factor
    }
}

/// Multiplication operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Multiply {
    pub factor: f32,
}

impl ElementOp for Multiply {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data * self.factor
    }
}

/// Addition operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Add {
    pub value: f32,
}

impl ElementOp for Add {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data + self.value
    }
}

/// Subtraction operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Subtract {
    pub value: f32,
}

impl ElementOp for Subtract {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data - self.value
    }
}

/// Clamp operation (point-wise)
/// Clamps values between min and max
#[derive(Debug, Clone)]
pub struct Clamp {
    pub min: f32,
    pub max: f32,
}

impl ElementOp for Clamp {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data.clamp(self.min, self.max)
    }
}

/// Absolute value operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Abs;

impl ElementOp for Abs {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data.abs()
    }
}

/// Power operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Pow {
    pub exponent: f32,
}

impl ElementOp for Pow {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data.powf(self.exponent)
    }
}

/// Square root operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Sqrt;

impl ElementOp for Sqrt {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data.sqrt()
    }
}

/// Truncate operation - reduces the length of the data vector
#[derive(Debug, Clone, Copy)]
pub struct Truncate<const NEW_LEN: usize>;

impl<const NEW_LEN: usize> TransformOp for Truncate<NEW_LEN> {
    type IndexRemapping = True;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        out_index
    }

    // TODO: check the validity of this
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        _input: &'i [f32],
        _: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() < NEW_LEN {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }
        Ok(&mut out[..NEW_LEN])
    }

    #[inline(always)]
    fn is_valid_input(&self, input_len: usize, actual_len: usize) -> bool {
        actual_len >= input_len
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        NEW_LEN
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        NEW_LEN
    }
}

/// Transpose operation for 2D matrices stored in row-major order
#[derive(Debug, Clone, Copy)]
pub struct Transpose<const ROWS: usize, const COLS: usize>;

impl<const ROWS: usize, const COLS: usize> TransformOp for Transpose<ROWS, COLS> {
    type IndexRemapping = True;

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
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let in_index = self.map_index(out_index);
        unsafe { *data.get_unchecked(in_index) }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != ROWS * COLS {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index);
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        ROWS * COLS
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        ROWS * COLS
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

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        if out_index < ORIGINAL_LEN {
            unsafe { *data.get_unchecked(out_index) }
        } else {
            self.pad_value
        }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != ORIGINAL_LEN {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

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

    #[inline(always)]
    fn is_valid_input(&self, input_len: usize, actual_len: usize) -> bool {
        actual_len >= input_len
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        ORIGINAL_LEN
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        PADDED_LEN
    }
}

/// Reverse operation - reverses the order of elements
#[derive(Debug, Clone, Copy)]
pub struct Reverse<const LEN: usize>;

impl<const LEN: usize> TransformOp for Reverse<LEN> {
    type IndexRemapping = True;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        LEN - 1 - out_index
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let in_index = self.map_index(out_index);
        unsafe { *data.get_unchecked(in_index) }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != LEN {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index);
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        LEN
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        LEN
    }
}
