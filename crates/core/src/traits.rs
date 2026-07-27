use crate::{
    _const_max_usize,
    errors::{ErrorKind, NanHandling, PipeError},
};

/// Float trait bound for pipeline operations
pub trait Float: num_traits::Float + num_traits::FloatConst + Default + Copy + 'static {}
impl Float for f32 {}
impl Float for f64 {}

/// Link<Head<RootOp>, PipeOp>
/// The pipeline wrapper encloses these 'stages'
pub trait Stage<T: Float = f32> {
    const IN_LEN: usize;
    const OUT_LEN: usize;
    const MAX_BUF_SIZE: usize;

    fn execute<'i, 'o>(
        &self,
        data: &[T],
        in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError>;

    /// Computes the maximum buffer size dynamically
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize;

    /// Computes the pipe output length dynamically OR
    /// the allocated buffer size
    fn out_len_dynamic(&self, input_len: usize) -> usize;
}

/// Stage marks (curr op)
// TODO: rename!
pub struct TMark;
pub struct EMark;

/// Generic over the root operation
pub struct Head<T, Mark, const INPUT_LEN: usize = 0> {
    pub root_op: T,
    pub marker: core::marker::PhantomData<Mark>,
}

// TODO: simplify the type signature!
/// Generic over the previous stage, current operation
pub struct Link<T, S, Mark, F>
where
    F: Float,
    S: Stage<F>,
{
    pub prev_stage: S,
    pub curr_op: T,
    pub marker: core::marker::PhantomData<Mark>,
    // !!!
    pub _float_marker: core::marker::PhantomData<F>,
}

/// Point-wise operations that work on individual elements
pub trait ElementOp<T: Float = f32> {
    fn compute(&self, data: T) -> Result<T, PipeError>;

    fn setup(&self) {}

    /// Get the NaN handling policy for this operation
    fn nan_handling(&self) -> NanHandling {
        NanHandling::Fail
    }

    #[inline(always)]
    fn fuse_element<U>(self, op: U) -> Fused<Self, U, ElementElement>
    where
        Self: Sized,
        U: ElementOp<T>,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: core::marker::PhantomData {},
        }
    }

    #[inline(always)]
    fn fuse_transform<U>(self, op: U) -> Fused<U, Self, TransformElement>
    where
        Self: Sized,
        U: TransformOp<T>,
    {
        Fused {
            prev_op: op,
            curr_op: self,
            marker: core::marker::PhantomData {},
        }
    }
}

// Associated types
pub struct True;
pub struct False;
pub trait IsTrue {}
impl IsTrue for True {}

/// Spatial transformation operations that map from output index to computed value by sampling from input
pub trait TransformOp<T: Float = f32> {
    /// Define whether an operation is a pure index remapping (can be fused)
    type IndexRemapping;

    // Leaving these at zero might result in out-of-bounds accesses
    // Creating ops with helpers (either sized or not sized, global options?)
    const IN_LEN: usize = 0;
    const OUT_LEN: usize = 0;
    /// Check the validity of passed in data compared to what is known
    const INTERNAL_IS_VALID: bool = true;

    /// Get the NaN handling policy for this operation
    fn nan_handling(&self) -> NanHandling {
        NanHandling::Fail
    }

    /// Map output index to input index (for pure index-remapping operations)
    /// Default implementation is identity mapping
    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        out_index
    }

    /// Compute output value at given output index by sampling from input data
    #[inline(always)]
    fn compute(&self, data: &[T], index: usize) -> Result<T, PipeError> {
        Ok(unsafe { *data.get_unchecked(index) })
    }

    /// Execute the transformation using chunk-based iteration (for stride operations)
    /// or index-based iteration (for non-linear operations)
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError>;

    /// Runtime setup / initialization method for operations
    fn setup(&self) {}

    /// When defining dynamic operations, overwrite this
    #[inline(always)]
    fn in_len(&self, _default_len: usize) -> usize {
        Self::IN_LEN
    }

    /// When defining dynamic operations, overwrite this
    #[inline(always)]
    fn out_len(&self, _default_len: usize) -> usize {
        Self::OUT_LEN
    }

    #[inline(always)]
    fn fuse_element<U>(self, op: U) -> Fused<Self, U, TransformElement>
    where
        Self: Sized,
        U: ElementOp<T>,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: core::marker::PhantomData {},
        }
    }
}

/// Extension trait for index-remapping TransformOps that can be fused
pub trait IndexRemappable<T: Float = f32>: TransformOp<T>
where
    Self::IndexRemapping: IsTrue,
{
    #[inline(always)]
    fn fuse_transform<U>(self, op: U) -> Fused<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp<T>,
        U::IndexRemapping: IsTrue,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: core::marker::PhantomData {},
        }
    }
}

/// Blanket implementation: any TransformOp with IndexRemapping = True is IndexRemappable
impl<T: Float, U> IndexRemappable<T> for U
where
    U: TransformOp<T>,
    U::IndexRemapping: IsTrue,
{
}

// Type aliases for convenience
pub type PipelineStatic32 = crate::pipeline::PipelineStatic<f32>;
pub type PipelineStatic64 = crate::pipeline::PipelineStatic<f64>;
pub type PipelineDynamic32 = crate::pipeline::PipelineDynamic<f32>;
pub type PipelineDynamic64 = crate::pipeline::PipelineDynamic<f64>;

/// Markers for assuring typestate
pub struct ElementElement;
pub struct TransformTransform;
pub struct TransformElement;

/// Generic over the previous operation and current
/// Allows for fusing the operations
#[derive(Debug, Clone, Copy)]
pub struct Fused<T, S, FusedState = ElementElement> {
    pub prev_op: T,
    pub curr_op: S,

    marker: core::marker::PhantomData<FusedState>,
}

impl<T: Float, U: ElementOp<T>, S: ElementOp<T>> ElementOp<T> for Fused<U, S, ElementElement> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let prev = self.prev_op.compute(data)?;
        self.curr_op.compute(prev)
    }
}

impl<T: Float, U: ElementOp<T>, S: ElementOp<T>> TransformOp<T> for Fused<U, S, ElementElement> {
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[T], index: usize) -> Result<T, PipeError> {
        let prev = self
            .prev_op
            .compute(unsafe { *data.get_unchecked(index) })?;
        self.curr_op.compute(prev)
    }

    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = TransformOp::compute(self, input, out_index)?;
            }
        }
        Ok(out)
    }
}

impl<T: Float, U: TransformOp<T>, S: ElementOp<T>> TransformOp<T>
    for Fused<U, S, TransformElement>
{
    type IndexRemapping = U::IndexRemapping;

    const IN_LEN: usize = U::IN_LEN;
    const OUT_LEN: usize = U::OUT_LEN;
    const INTERNAL_IS_VALID: bool = U::INTERNAL_IS_VALID;

    #[inline(always)]
    fn compute(&self, data: &[T], index: usize) -> Result<T, PipeError> {
        let prev = self.prev_op.compute(data, index)?;
        self.curr_op.compute(prev)
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
                *out.get_unchecked_mut(out_index) = TransformOp::compute(self, input, out_index)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        self.prev_op.in_len(default_len)
    }

    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        self.prev_op.out_len(default_len)
    }
}

impl<T: Float, U: TransformOp<T>, S: TransformOp<T>> TransformOp<T>
    for Fused<U, S, TransformTransform>
where
    S::IndexRemapping: IsTrue,
    U::IndexRemapping: IsTrue,
{
    type IndexRemapping = True;

    const OUT_LEN: usize = S::OUT_LEN;
    const IN_LEN: usize = U::IN_LEN;
    const INTERNAL_IS_VALID: bool = S::IN_LEN == U::OUT_LEN || S::IN_LEN == 0 || U::IN_LEN == 0;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        let intermediate_index = self.curr_op.map_index(out_index);
        self.prev_op.map_index(intermediate_index)
    }

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let input_index = self.map_index(out_index);
        Ok(unsafe { *data.get_unchecked(input_index) })
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

    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        self.prev_op.in_len(default_len)
    }

    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        self.curr_op.out_len(default_len)
    }
}

impl<T: Float, U: ElementOp<T>, const INPUT_LEN: usize> Stage<T> for Head<U, EMark, INPUT_LEN> {
    const IN_LEN: usize = INPUT_LEN;
    const OUT_LEN: usize = INPUT_LEN;
    const MAX_BUF_SIZE: usize = INPUT_LEN;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[T],
        _in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError> {
        self.root_op.setup();

        let exec_len = if INPUT_LEN > 0 { INPUT_LEN } else { data.len() };
        let out_len = if Self::OUT_LEN > 0 {
            Self::OUT_LEN
        } else {
            // Dynamically allocated once all stage buffer
            // sizes are known
            out_buf.len()
        };

        if exec_len != data.len() {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        if out_len < exec_len {
            return Err(PipeError::new(ErrorKind::InvalidOutputSize));
        }

        for i in 0..exec_len {
            unsafe {
                *out_buf.get_unchecked_mut(i) = self.root_op.compute(*data.get_unchecked(i))?;
            }
        }
        Ok(&out_buf[..exec_len])
    }

    #[inline(always)]
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize {
        if INPUT_LEN > 0 {
            INPUT_LEN
        } else {
            input_len
        }
    }

    #[inline(always)]
    fn out_len_dynamic(&self, input_len: usize) -> usize {
        if INPUT_LEN > 0 {
            INPUT_LEN
        } else {
            input_len
        }
    }
}

impl<T: Float, U: TransformOp<T>, const INPUT_LEN: usize> Stage<T> for Head<U, TMark, INPUT_LEN> {
    const IN_LEN: usize = U::IN_LEN;
    const OUT_LEN: usize = U::OUT_LEN;
    const MAX_BUF_SIZE: usize = _const_max_usize(INPUT_LEN, U::OUT_LEN);

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[T],
        _in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError> {
        let data_len = if Self::IN_LEN > 0 {
            Self::IN_LEN
        } else {
            data.len()
        };

        // Runtime guardrail, optimized if the last operation was dynamic
        if data.len() != data_len {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        let expected_in = self.root_op.in_len(data_len);

        // The operation in_len must never be zero as it is
        // either statically or dynamically set
        if expected_in == 0 || data_len != expected_in {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        let out_len = if Self::OUT_LEN > 0 {
            Self::OUT_LEN
        } else {
            out_buf.len()
        };
        let n = self.root_op.out_len(data_len);

        if n == 0 || out_len < n {
            return Err(PipeError::new(ErrorKind::InvalidOutputSize));
        }

        let out = &mut out_buf[0..n];
        Ok(self.root_op.execute(out, data, n)?)
    }

    #[inline(always)]
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize {
        self.root_op.out_len(input_len)
    }

    #[inline(always)]
    fn out_len_dynamic(&self, input_len: usize) -> usize {
        self.root_op.out_len(input_len)
    }
}

impl<T: Float, U: TransformOp<T>, S: Stage<T>> Stage<T> for Link<U, S, TMark, T> {
    const IN_LEN: usize = S::OUT_LEN;
    const OUT_LEN: usize = U::OUT_LEN;
    const MAX_BUF_SIZE: usize = _const_max_usize(S::MAX_BUF_SIZE, U::OUT_LEN);

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[T],
        in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError> {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf)?;
        self.curr_op.setup();

        let data_len = if Self::IN_LEN > 0 {
            Self::IN_LEN
        } else {
            prev_out.len()
        };

        // Runtime guardrail, optimized if the last operation was dynamic
        if prev_out.len() != data_len {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        let expected_in = self.curr_op.in_len(data_len);

        // The operation in_len must never be zero as it is
        // either statically or dynamically set
        if expected_in == 0 || data_len != expected_in {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        let out_len = if Self::OUT_LEN > 0 {
            Self::OUT_LEN
        } else {
            out_buf.len()
        };
        let n = self.curr_op.out_len(data_len);

        if n == 0 || out_len < n {
            return Err(PipeError::new(ErrorKind::InvalidOutputSize));
        }

        let out = &mut out_buf[0..n];
        Ok(self.curr_op.execute(out, prev_out, n)?)
    }

    #[inline(always)]
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize {
        let prev_max = self.prev_stage.max_buf_size_dynamic(input_len);
        let prev_out_len = self.prev_stage.out_len_dynamic(input_len);
        let curr_out = self.curr_op.out_len(prev_out_len);
        core::cmp::max(prev_max, curr_out)
    }

    #[inline(always)]
    fn out_len_dynamic(&self, input_len: usize) -> usize {
        let prev_out_len = self.prev_stage.out_len_dynamic(input_len);
        self.curr_op.out_len(prev_out_len)
    }
}

impl<T: Float, U: ElementOp<T>, S: Stage<T>> Stage<T> for Link<U, S, EMark, T> {
    const IN_LEN: usize = S::OUT_LEN;
    const OUT_LEN: usize = S::OUT_LEN;
    const MAX_BUF_SIZE: usize = S::MAX_BUF_SIZE;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[T],
        in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError> {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf)?;
        self.curr_op.setup();

        let out_len = if Self::OUT_LEN > 0 {
            Self::OUT_LEN
        } else {
            out_buf.len()
        };
        let n = prev_out.len();

        if out_len < n {
            return Err(PipeError::new(ErrorKind::InvalidOutputSize));
        }

        for i in 0..n {
            unsafe {
                *out_buf.get_unchecked_mut(i) = self.curr_op.compute(*prev_out.get_unchecked(i))?;
            }
        }
        Ok(&out_buf[..n])
    }

    #[inline(always)]
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize {
        self.prev_stage.max_buf_size_dynamic(input_len)
    }

    #[inline(always)]
    fn out_len_dynamic(&self, input_len: usize) -> usize {
        self.prev_stage.out_len_dynamic(input_len)
    }
}
