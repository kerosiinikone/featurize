use crate::{
    errors::{check_finite, NanHandling, PipeError},
    traits::{ElementOp, False, Float, IsTrue, TransformOp, True},
};

#[allow(unused_imports)]
use num_traits::Float as _;

/// Normalize operation (point-wise)
/// Normalizes by standard deviation and mean
#[derive(Debug, Clone)]
pub struct Normalize<T: Float = f32> {
    pub mean: T,
    pub std: T,
    pub nan_handling: NanHandling,
}

impl<T: Float> Default for Normalize<T> {
    fn default() -> Self {
        Self {
            mean: T::zero(),
            std: T::one(),
            nan_handling: NanHandling::default(),
        }
    }
}

impl<T: Float> ElementOp<T> for Normalize<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let result = (data - self.mean) / self.std;
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> &'static str {
        "Normalize"
    }
}

/// Division operation (point-wise)
#[derive(Debug, Clone)]
pub struct Div<T: Float = f32> {
    pub factor: T,
    pub nan_handling: NanHandling,
}

impl<T: Float> Default for Div<T> {
    fn default() -> Self {
        Self {
            factor: T::one(),
            nan_handling: NanHandling::default(),
        }
    }
}

impl<T: Float> ElementOp<T> for Div<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let result = data / self.factor;
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> &'static str {
        "Div"
    }
}

/// Multiplication operation (point-wise)
#[derive(Debug, Clone)]
pub struct Multiply<T: Float = f32> {
    pub factor: T,
    pub nan_handling: NanHandling,
}

impl<T: Float> Default for Multiply<T> {
    fn default() -> Self {
        Self {
            factor: T::one(),
            nan_handling: NanHandling::default(),
        }
    }
}

impl<T: Float> ElementOp<T> for Multiply<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let result = data * self.factor;
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> &'static str {
        "Multiply"
    }
}

/// Addition operation (point-wise)
#[derive(Debug, Clone)]
pub struct Add<T: Float = f32> {
    pub value: T,
    pub nan_handling: NanHandling,
}

impl<T: Float> Default for Add<T> {
    fn default() -> Self {
        Self {
            value: T::zero(),
            nan_handling: NanHandling::default(),
        }
    }
}

impl<T: Float> ElementOp<T> for Add<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let result = data + self.value;
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> &'static str {
        "Add"
    }
}

/// Subtraction operation (point-wise)
#[derive(Debug, Clone)]
pub struct Subtract<T: Float = f32> {
    pub value: T,
    pub nan_handling: NanHandling,
}

impl<T: Float> Default for Subtract<T> {
    fn default() -> Self {
        Self {
            value: T::zero(),
            nan_handling: NanHandling::default(),
        }
    }
}

impl<T: Float> ElementOp<T> for Subtract<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let result = data - self.value;
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> &'static str {
        "Subtract"
    }
}

/// Clamp operation (point-wise)
/// Clamps values between min and max
#[derive(Debug, Clone)]
pub struct Clamp<T: Float = f32> {
    pub min: T,
    pub max: T,
    pub nan_handling: NanHandling,
}

impl<T: Float> ElementOp<T> for Clamp<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        if !data.is_finite() {
            check_finite(data, self.nan_handling)
        } else {
            Ok(data.clamp(self.min, self.max))
        }
    }

    fn op_name(&self) -> &'static str {
        "Clamp"
    }
}

/// Absolute value operation (point-wise)
#[derive(Debug, Clone)]
pub struct Abs<T: Float = f32> {
    pub nan_handling: NanHandling,
    _phantom: core::marker::PhantomData<T>,
}

impl<T: Float> Default for Abs<T> {
    fn default() -> Self {
        Self {
            nan_handling: NanHandling::default(),
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<T: Float> ElementOp<T> for Abs<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let result = data.abs();
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> &'static str {
        "Abs"
    }
}

/// Power operation (point-wise)
#[derive(Debug, Clone)]
pub struct Pow<T: Float = f32> {
    pub exponent: T,
    pub nan_handling: NanHandling,
}

impl<T: Float> Default for Pow<T> {
    fn default() -> Self {
        Self {
            exponent: T::one(),
            nan_handling: NanHandling::default(),
        }
    }
}

impl<T: Float> ElementOp<T> for Pow<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let result = data.powf(self.exponent);
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> &'static str {
        "Pow"
    }
}

/// Square root operation (point-wise)
#[derive(Debug, Clone)]
pub struct Sqrt<T: Float = f32> {
    pub nan_handling: NanHandling,
    _phantom: core::marker::PhantomData<T>,
}

impl<T: Float> Default for Sqrt<T> {
    fn default() -> Self {
        Self {
            nan_handling: NanHandling::default(),
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<T: Float> ElementOp<T> for Sqrt<T> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let result = data.sqrt();
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> &'static str {
        "Sqrt"
    }
}

/// Truncate operation - reduces the length of the data vector
#[derive(Debug, Clone, Copy)]
pub struct Truncate<const ORIGINAL_LEN: usize, const NEW_LEN: usize>;

impl<T: Float, const ORIGINAL_LEN: usize, const NEW_LEN: usize> TransformOp<T>
    for Truncate<ORIGINAL_LEN, NEW_LEN>
{
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
        out: &'o mut [T],
        input: &'i [T],
        _: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // TODO: n?
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), NEW_LEN);
        }
        Ok(out)
    }

    fn op_name(&self) -> &'static str {
        "Truncate"
    }
}

/// Transpose operation for 2D matrices stored in row-major order
#[derive(Debug, Clone, Copy)]
pub struct Transpose<const ROWS: usize, const COLS: usize>;

impl<T: Float, const ROWS: usize, const COLS: usize> TransformOp<T> for Transpose<ROWS, COLS> {
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
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let in_index = <Transpose<ROWS, COLS> as TransformOp<T>>::map_index(self, out_index);
        Ok(unsafe { *data.get_unchecked(in_index) })
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> &'static str {
        "Transpose"
    }
}

/// Pad operation - adds padding to the data
#[derive(Debug, Clone, Copy)]
pub struct Pad<T: Float = f32, const ORIGINAL_LEN: usize = 0, const PADDED_LEN: usize = 0> {
    pub pad_value: T,
}

impl<T: Float, const ORIGINAL_LEN: usize, const PADDED_LEN: usize> TransformOp<T>
    for Pad<T, ORIGINAL_LEN, PADDED_LEN>
{
    type IndexRemapping = False;

    const IN_LEN: usize = ORIGINAL_LEN;
    const OUT_LEN: usize = PADDED_LEN;

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        if out_index < ORIGINAL_LEN {
            Ok(unsafe { *data.get_unchecked(out_index) })
        } else {
            Ok(self.pad_value)
        }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
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

    fn op_name(&self) -> &'static str {
        "Pad"
    }
}

/// Reverse operation - reverses the order of elements
#[derive(Debug, Clone, Copy)]
pub struct Reverse<const LEN: usize>;

impl<T: Float, const LEN: usize> TransformOp<T> for Reverse<LEN> {
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
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let in_index = <Reverse<LEN> as TransformOp<T>>::map_index(self, out_index);
        Ok(unsafe { *data.get_unchecked(in_index) })
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> &'static str {
        "Reverse"
    }
}
