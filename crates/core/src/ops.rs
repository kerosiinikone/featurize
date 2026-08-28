use crate::{
    errors::{ErrorKind, NanHandler, PipeError},
    traits::{ElementOp, False, Float, IsTrue, TransformOp, True},
};

/// Normalize operation (point-wise)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalize<T: Float = f32> {
    mean: T,
    std: T,
}

impl<T: Float> Default for Normalize<T> {
    fn default() -> Self {
        Self {
            mean: T::zero(),
            std: T::one(),
        }
    }
}

impl<T: Float> Normalize<T> {
    pub fn new(std: T, mean: T) -> Self {
        assert!(!std.is_zero() && std.is_sign_positive() && std.is_finite());
        assert!(mean.is_finite());

        Self { mean, std }
    }
}

impl<T: Float> ElementOp<T> for Normalize<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        let result = (data - self.mean) / self.std;
        N::check_finite(result)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Normalize")
    }
}

/// Division operation (point-wise)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Div<T: Float = f32> {
    factor: T,
}

impl<T: Float> Div<T> {
    pub fn new(factor: T) -> Self {
        assert!(!factor.is_zero() && factor.is_finite());

        Self { factor }
    }
}

impl<T: Float> Default for Div<T> {
    fn default() -> Self {
        Self { factor: T::one() }
    }
}

impl<T: Float> ElementOp<T> for Div<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        let result = data / self.factor;
        N::check_finite(result)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Div")
    }
}

/// Multiplication operation (point-wise)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Multiply<T: Float = f32> {
    factor: T,
}

impl<T: Float> Multiply<T> {
    pub fn new(factor: T) -> Self {
        assert!(factor.is_finite());

        Self { factor }
    }
}

impl<T: Float> Default for Multiply<T> {
    fn default() -> Self {
        Self { factor: T::one() }
    }
}

impl<T: Float> ElementOp<T> for Multiply<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        let result = data * self.factor;
        N::check_finite(result)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Multiply")
    }
}

/// Addition operation (point-wise)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add<T: Float = f32> {
    value: T,
}

impl<T: Float> Add<T> {
    pub fn new(value: T) -> Self {
        assert!(value.is_finite());

        Self { value }
    }
}

impl<T: Float> Default for Add<T> {
    fn default() -> Self {
        Self { value: T::zero() }
    }
}

impl<T: Float> ElementOp<T> for Add<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        let result = data + self.value;
        N::check_finite(result)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Add")
    }
}

/// Subtraction operation (point-wise)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Subtract<T: Float = f32> {
    value: T,
}

impl<T: Float> Subtract<T> {
    pub fn new(value: T) -> Self {
        assert!(value.is_finite());

        Self { value }
    }
}

impl<T: Float> Default for Subtract<T> {
    fn default() -> Self {
        Self { value: T::zero() }
    }
}

impl<T: Float> ElementOp<T> for Subtract<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        let result = data - self.value;
        N::check_finite(result)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Subtract")
    }
}

/// Clamp operation (point-wise)
/// Clamps values between min and max
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clamp<T: Float = f32> {
    min: T,
    max: T,
}

impl<T: Float> Clamp<T> {
    pub fn new(min: T, max: T) -> Self {
        assert!(min.is_finite() && max.is_finite());
        assert!(min <= max);

        Self { min, max }
    }
}

impl<T: Float> ElementOp<T> for Clamp<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        // Non-finite inputs are handed to the pipeline policy verbatim
        // (clamping a NaN is meaningless); for `PropagateNan` this whole
        // branch folds away since the value is returned unchanged either way
        if !data.is_finite() {
            N::check_finite(data)
        } else {
            Ok(data.clamp(self.min, self.max))
        }
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Clamp")
    }
}

/// Absolute value operation (point-wise)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Abs<T: Float = f32> {
    marker: core::marker::PhantomData<T>,
}

impl<T: Float> Abs<T> {
    pub fn new() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<T: Float> Default for Abs<T> {
    fn default() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<T: Float> ElementOp<T> for Abs<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        let result = data.abs();
        N::check_finite(result)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Abs")
    }
}

/// Power operation (point-wise)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pow<T: Float = f32> {
    exponent: T,
}

impl<T: Float> Pow<T> {
    pub fn new(exponent: T) -> Self {
        assert!(exponent.is_finite());

        Self { exponent }
    }
}

impl<T: Float> Default for Pow<T> {
    fn default() -> Self {
        Self { exponent: T::one() }
    }
}

impl<T: Float> ElementOp<T> for Pow<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        let result = data.powf(self.exponent);
        N::check_finite(result)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Pow")
    }
}

/// Square root operation (point-wise)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sqrt<T: Float = f32> {
    marker: core::marker::PhantomData<T>,
}

impl<T: Float> Sqrt<T> {
    pub fn new() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<T: Float> Default for Sqrt<T> {
    fn default() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<T: Float> ElementOp<T> for Sqrt<T> {
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: T) -> Result<T, PipeError> {
        let result = data.sqrt();
        N::check_finite(result)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Sqrt")
    }
}

/// Truncate operation - reduces the length of the data vector
///
/// Pure index remapping operation that copies the first `NEW_LEN` elements
/// from an input of length `ORIGINAL_LEN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Truncate<const ORIGINAL_LEN: usize, const NEW_LEN: usize>;

impl<const ORIGINAL_LEN: usize, const NEW_LEN: usize> Truncate<ORIGINAL_LEN, NEW_LEN> {
    pub fn new() -> Self {
        Self
    }
}

impl<T: Float, const ORIGINAL_LEN: usize, const NEW_LEN: usize> TransformOp<T>
    for Truncate<ORIGINAL_LEN, NEW_LEN>
{
    type IndexRemapping = True;

    const IN_LEN: usize = ORIGINAL_LEN;
    const OUT_LEN: usize = NEW_LEN;
    /// A truncation must never *grow* the data: copying `NEW_LEN` elements
    /// out of an `ORIGINAL_LEN` input would otherwise read out of bounds.
    /// Asserted at every pipe-construction site.
    const INTERNAL_IS_VALID: bool = NEW_LEN <= ORIGINAL_LEN;

    #[inline(always)]
    fn map_index(&self, out_index: usize, _default_len: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        out_index
    }

    /// Pure index remapping: never produces a new value, so the NaN policy
    /// `N` is irrelevant here
    #[inline(always)]
    fn execute<'o, N: NanHandler>(
        &self,
        out: &'o mut [T],
        input: &[T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // The stage guarantees `n == out_len(..) == NEW_LEN`
        debug_assert_eq!(n, NEW_LEN);

        // Cheap once-per-call guard so the bulk copy below can be unchecked
        if input.len() < NEW_LEN || out.len() < NEW_LEN {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        // SAFETY: `INTERNAL_IS_VALID` guarantees `NEW_LEN <= ORIGINAL_LEN`
        // at pipe-construction time and both slice lengths were verified
        // above, so the copy stays within both allocations. `input` and
        // `out` come from distinct buffers, so the ranges cannot overlap
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), NEW_LEN);
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Truncate")
    }
}

/// Transpose operation for 2D matrices stored in row-major order
///
/// Pure index remapping that transposes a `ROWS × COLS` matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Transpose<const ROWS: usize, const COLS: usize>;

impl<const ROWS: usize, const COLS: usize> Transpose<ROWS, COLS> {
    pub fn new() -> Self {
        Self
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> TransformOp<T> for Transpose<ROWS, COLS> {
    type IndexRemapping = True;

    const IN_LEN: usize = ROWS * COLS;
    const OUT_LEN: usize = ROWS * COLS;

    // The logic equates to pure index remapping
    #[inline(always)]
    fn map_index(&self, out_index: usize, _default_len: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        // For `out_index < ROWS * COLS`: `out_row < COLS`, `out_col < ROWS`,
        // hence the result is `< ROWS * COLS` (the map_index contract)
        let out_row = out_index / ROWS;
        let out_col = out_index % ROWS;
        out_col * COLS + out_row
    }

    /// Pure index remapping: `N` is unused
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        // Transposing requires knowing the dimensions and therefore is not dynamic
        let in_index = <Transpose<ROWS, COLS> as TransformOp<T>>::map_index(self, out_index, 0);
        debug_assert!(in_index < data.len());
        // SAFETY: caller contract: `out_index < out_len(..) == ROWS * COLS`
        // and `data.len() == in_len(..) == ROWS * COLS`; `map_index` then
        // stays within `[0, ROWS * COLS)` (see above)
        Ok(unsafe { *data.get_unchecked(in_index) })
    }

    #[inline(always)]
    fn execute<'o, N: NanHandler>(
        &self,
        out: &'o mut [T],
        input: &[T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        if n > ROWS * COLS || input.len() < ROWS * COLS || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            // SAFETY: `out_index < n <= out.len()` (checked above); the read
            // is bounded by `map_index` (see `compute`)
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute::<N>(input, out_index)?;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Transpose")
    }
}

/// Pad operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pad<T: Float = f32, const ORIGINAL_LEN: usize = 0, const PADDED_LEN: usize = 0> {
    pad_value: T,
}

impl<T: Float, const ORIGINAL_LEN: usize, const PADDED_LEN: usize>
    Pad<T, ORIGINAL_LEN, PADDED_LEN>
{
    pub fn new(pad_value: T) -> Self {
        Self { pad_value }
    }
}

impl<T: Float, const ORIGINAL_LEN: usize, const PADDED_LEN: usize> TransformOp<T>
    for Pad<T, ORIGINAL_LEN, PADDED_LEN>
{
    type IndexRemapping = False;

    const IN_LEN: usize = ORIGINAL_LEN;
    const OUT_LEN: usize = PADDED_LEN;
    /// Padding must never shrink the data (use `Truncate` for that)
    const INTERNAL_IS_VALID: bool = PADDED_LEN >= ORIGINAL_LEN;

    // Copies / fills only: `N` is unused
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        if out_index < ORIGINAL_LEN {
            debug_assert!(out_index < data.len());
            // SAFETY: caller contract guarantees
            // `data.len() == in_len(..) == ORIGINAL_LEN`, and
            // `out_index < ORIGINAL_LEN` was just checked
            Ok(unsafe { *data.get_unchecked(out_index) })
        } else {
            Ok(self.pad_value)
        }
    }

    #[inline(always)]
    fn execute<'o, N: NanHandler>(
        &self,
        out: &'o mut [T],
        input: &[T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        let copy_len = ORIGINAL_LEN.min(n);

        // Cheap once-per-call guard so the bulk copy / fill below can be
        // unchecked
        if input.len() < copy_len || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        // SAFETY: `copy_len <= input.len()` and `copy_len <= n <= out.len()`
        // were verified above; the buffers are distinct allocations, so the
        // ranges cannot overlap
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), copy_len);
        }

        for i in copy_len..n {
            // SAFETY: `i < n <= out.len()` (checked above)
            unsafe {
                *out.get_unchecked_mut(i) = self.pad_value;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Pad")
    }
}

/// Reverse operation - reverses the order of elements
///
/// Pure index remapping that reverses the order of elements in a vector.
/// Supports both static (`LEN > 0`) and dynamic (`LEN = 0`) lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Reverse<const LEN: usize>;

impl<const LEN: usize> Reverse<LEN> {
    pub fn new() -> Self {
        Self
    }
}

impl<T: Float, const LEN: usize> TransformOp<T> for Reverse<LEN> {
    type IndexRemapping = True;

    const IN_LEN: usize = LEN;
    const OUT_LEN: usize = LEN;

    #[inline(always)]
    fn map_index(&self, out_index: usize, default_len: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        // Contract: `out_index < LEN`, otherwise this underflows
        // Has to check for dynamicity
        if LEN > 0 {
            LEN - 1 - out_index
        } else {
            default_len - 1 - out_index
        }
    }

    /// Dynamic/static reverse - length is either the given or constant length
    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        if LEN > 0 {
            LEN
        } else {
            default_len
        }
    }

    /// Dynamic/static reverse - length is either the given or constant length
    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        if LEN > 0 {
            LEN
        } else {
            default_len
        }
    }

    /// Pure index remapping: `N` is unused
    #[inline(always)]
    fn compute<N: NanHandler>(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        // Inlined call, should not cause too much overhead: `data.len()`
        let in_index = <Reverse<LEN> as TransformOp<T>>::map_index(self, out_index, data.len());
        debug_assert!(in_index < data.len());
        // SAFETY: caller contract: `out_index < out_len(..) == LEN`, so
        // `in_index = LEN - 1 - out_index < LEN == in_len(..) == data.len()`
        Ok(unsafe { *data.get_unchecked(in_index) })
    }

    #[inline(always)]
    fn execute<'o, N: NanHandler>(
        &self,
        out: &'o mut [T],
        input: &[T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        // (`map_index` would underflow for out_index >= LEN)
        // Has to go through the `in_len()` to validate static / dynamic
        let in_len = <Reverse<LEN> as TransformOp<T>>::in_len(self, input.len());
        if n > in_len || input.len() < in_len || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            // SAFETY: `out_index < n <= out.len()` (checked above); the read
            // is bounded by `map_index` (see `compute`)
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute::<N>(input, out_index)?;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Reverse")
    }
}
