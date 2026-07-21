use core::marker::PhantomData;

use crate::{
    _const_max_usize,
    errors::{NanHandling, PipeError},
};

/// Link<Head<RootOp>, PipeOp>
/// The pipeline wrapper encloses these 'stages'
pub trait Stage {
    const IN_LEN: usize;
    const OUT_LEN: usize;
    const MAX_BUF_SIZE: usize;

    // Return back to pass-through upon build?
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError>;
}

/// Stage marks (curr op)
pub struct TMark;
pub struct EMark;

/// Generic over the root operation
pub struct Head<T, Mark, const INPUT_LEN: usize = 0> {
    pub root_op: T,
    pub marker: PhantomData<Mark>,
}

/// Generic over the previous stage, current operation
pub struct Link<T, S, Mark>
where
    S: Stage,
{
    pub prev_stage: S,
    pub curr_op: T,
    pub marker: PhantomData<Mark>,
}

/// Point-wise operations that work on individual elements
pub trait ElementOp {
    fn compute(&self, data: f32) -> Result<f32, PipeError>;

    fn setup(&self) {}

    /// Get the NaN handling policy for this operation
    fn nan_handling(&self) -> NanHandling {
        NanHandling::Fail
    }

    #[inline(always)]
    fn fuse_element<U>(self, op: U) -> Fused<Self, U, ElementElement>
    where
        Self: Sized,
        U: ElementOp,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    #[inline(always)]
    fn fuse_transform<U>(self, op: U) -> Fused<U, Self, TransformElement>
    where
        Self: Sized,
        U: TransformOp,
    {
        Fused {
            prev_op: op,
            curr_op: self,
            marker: PhantomData {},
        }
    }
}

// Associated types
pub struct True;
pub struct False;
pub trait IsTrue {}
impl IsTrue for True {}

/// Spatial transformation operations that map from output index to computed value by sampling from input
pub trait TransformOp {
    /// Define whether an operation is a pure index remapping (can be fused)
    type IndexRemapping;

    // Leaving these at 0 might result in out-of-bounds accesses
    // Creating ops with helpers (either sized or not sized, global options?)
    const IN_LEN: usize = 0;
    const OUT_LEN: usize = 0;
    /// Check the validity of passed in data compared to what is known
    // ONLY set for Fused<T, T> operations
    const INTERNAL_IS_VALID: bool = true;

    /// Get the NaN handling policy for this operation (default: FailFast)
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
    fn compute(&self, data: &[f32], index: usize) -> Result<f32, PipeError> {
        Ok(unsafe { *data.get_unchecked(index) })
    }

    /// Execute the transformation using chunk-based iteration (for stride operations)
    /// or index-based iteration (for non-linear operations)
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError>;

    /// Runtime setup / initialization method for operations
    fn setup(&self) {}

    #[inline(always)]
    fn fuse_element<U>(self, op: U) -> Fused<Self, U, TransformElement>
    where
        Self: Sized,
        U: ElementOp,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }
}

/// Extension trait for index-remapping TransformOps that can be fused
pub trait IndexRemappable: TransformOp
where
    Self::IndexRemapping: IsTrue,
{
    #[inline(always)]
    fn fuse_transform<U>(self, op: U) -> Fused<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
        U::IndexRemapping: IsTrue,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }
}

/// Blanket implementation: any TransformOp with IndexRemapping = True is IndexRemappable
impl<T> IndexRemappable for T
where
    T: TransformOp,
    T::IndexRemapping: IsTrue,
{
}

/// Markers for assuring typestate
pub struct ElementElement;
pub struct TransformTransform;
pub struct TransformElement;

/// Generic over the previous operation and current
/// Allows for fusing the operations
/// In the future -> 1:1 operations can be fused if they are of TransformOp type
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Fused<T, S, FusedState = ElementElement> {
    pub prev_op: T,
    pub curr_op: S,

    marker: PhantomData<FusedState>,
}

impl<T: ElementOp, S: ElementOp> ElementOp for Fused<T, S, ElementElement> {
    #[inline(always)]
    fn compute(&self, data: f32) -> Result<f32, PipeError> {
        let prev = self.prev_op.compute(data)?;
        self.curr_op.compute(prev)
    }
}

impl<T: ElementOp, S: ElementOp> TransformOp for Fused<T, S, ElementElement> {
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[f32], index: usize) -> Result<f32, PipeError> {
        let prev = self
            .prev_op
            .compute(unsafe { *data.get_unchecked(index) })?;
        self.curr_op.compute(prev)
    }

    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = TransformOp::compute(self, input, out_index)?;
            }
        }
        Ok(out)
    }
}

impl<T: TransformOp, S: ElementOp> TransformOp for Fused<T, S, TransformElement> {
    type IndexRemapping = T::IndexRemapping;

    const IN_LEN: usize = T::IN_LEN;
    const OUT_LEN: usize = T::OUT_LEN;
    const INTERNAL_IS_VALID: bool = T::INTERNAL_IS_VALID;

    #[inline(always)]
    fn compute(&self, data: &[f32], index: usize) -> Result<f32, PipeError> {
        let prev = self.prev_op.compute(data, index)?;
        self.curr_op.compute(prev)
    }

    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = TransformOp::compute(self, input, out_index)?;
            }
        }
        Ok(out)
    }
}

impl<T: TransformOp, S: TransformOp> TransformOp for Fused<T, S, TransformTransform>
where
    S::IndexRemapping: IsTrue,
    T::IndexRemapping: IsTrue,
{
    type IndexRemapping = True;

    const OUT_LEN: usize = S::OUT_LEN;
    const IN_LEN: usize = T::IN_LEN;
    const INTERNAL_IS_VALID: bool = S::IN_LEN == T::OUT_LEN || S::IN_LEN == 0 || T::IN_LEN == 0;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        let intermediate_index = self.curr_op.map_index(out_index);
        self.prev_op.map_index(intermediate_index)
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> Result<f32, PipeError> {
        let input_index = self.map_index(out_index);
        Ok(unsafe { *data.get_unchecked(input_index) })
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

impl<T: ElementOp, const INPUT_LEN: usize> Stage for Head<T, EMark, INPUT_LEN> {
    const IN_LEN: usize = INPUT_LEN;
    const OUT_LEN: usize = INPUT_LEN;
    const MAX_BUF_SIZE: usize = INPUT_LEN;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError> {
        self.root_op.setup();

        for i in 0..INPUT_LEN {
            unsafe {
                *out_buf.get_unchecked_mut(i) = self.root_op.compute(*data.get_unchecked(i))?;
            }
        }
        Ok(out_buf)
    }
}

impl<T: TransformOp, const INPUT_LEN: usize> Stage for Head<T, TMark, INPUT_LEN> {
    const IN_LEN: usize = T::IN_LEN;
    const MAX_BUF_SIZE: usize = T::OUT_LEN;
    const OUT_LEN: usize = T::OUT_LEN;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError> {
        const {
            let is_valid = T::INTERNAL_IS_VALID;
            assert!(is_valid, "Invalid input length");
        }

        self.root_op.setup();
        let n = T::OUT_LEN;

        let out = &mut out_buf[0..n];
        Ok(self.root_op.execute(out, data, n)?)
    }
}

impl<T: TransformOp, S: Stage> Stage for Link<T, S, TMark> {
    const IN_LEN: usize = S::OUT_LEN;
    const OUT_LEN: usize = T::OUT_LEN;
    const MAX_BUF_SIZE: usize = _const_max_usize(S::MAX_BUF_SIZE, T::OUT_LEN);

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError> {
        const {
            let is_valid = T::INTERNAL_IS_VALID;
            let prev_out = Self::IN_LEN;
            let curr_in = T::IN_LEN;
            assert!(
                is_valid && prev_out == curr_in || prev_out == 0 || curr_in == 0,
                "Invalid input length"
            );
        }

        let prev_out = self.prev_stage.execute(data, out_buf, in_buf)?;
        self.curr_op.setup();

        let n = T::OUT_LEN;
        let out = &mut out_buf[0..n];

        Ok(self.curr_op.execute(out, prev_out, n)?)
    }
}

impl<T: ElementOp, S: Stage> Stage for Link<T, S, EMark> {
    const IN_LEN: usize = S::OUT_LEN;
    const OUT_LEN: usize = S::OUT_LEN;
    const MAX_BUF_SIZE: usize = S::MAX_BUF_SIZE;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError> {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf)?;
        self.curr_op.setup();

        for i in 0..Self::OUT_LEN {
            unsafe {
                *out_buf.get_unchecked_mut(i) = self.curr_op.compute(*prev_out.get_unchecked(i))?;
            }
        }
        Ok(out_buf)
    }
}
