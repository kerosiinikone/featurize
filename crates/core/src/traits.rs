use core::{cmp, marker::PhantomData};

use crate::errors::{ErrorKind, PipeError};

/// Link<Head<RootOp>, PipeOp>
/// The pipeline wrapper encloses these 'stages'
pub trait Stage {
    fn execute<'i, 'o, const LEN: usize>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError>;

    // TODO: redundant?
    fn buf_size<const LEN: usize>(&self) -> usize;

    fn output_len<const LEN: usize>(&self) -> usize;
}

/// Stage marks (curr op)
pub struct TMark;
pub struct EMark;

/// Generic over the root operation
pub struct Head<T, Mark> {
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
    fn compute(&self, data: f32) -> f32;

    fn setup(&self) {}

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
    // TODO: can fail?
    #[inline(always)]
    fn compute(&self, data: &[f32], index: usize) -> f32 {
        unsafe { *data.get_unchecked(index) }
    }

    /// Execute the transformation using chunk-based iteration (for stride operations)
    /// or index-based iteration (for non-linear operations)
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError>;

    /// Check the validity of passed in data compared to what is known
    // TODO: better checks
    #[inline(always)]
    fn is_valid_input(&self, input_len: usize, output_len: usize) -> bool {
        input_len == output_len
    }

    fn input_len(&self) -> usize {
        0
    }

    fn output_len(&self) -> usize {
        0
    }

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
    fn compute(&self, data: f32) -> f32 {
        let prev = self.prev_op.compute(data);
        self.curr_op.compute(prev)
    }
}

impl<T: ElementOp, S: ElementOp> TransformOp for Fused<T, S, ElementElement> {
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[f32], index: usize) -> f32 {
        let prev = self.prev_op.compute(unsafe { *data.get_unchecked(index) });
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
                *out.get_unchecked_mut(out_index) = TransformOp::compute(self, input, out_index);
            }
        }
        Ok(out)
    }
}

impl<T: TransformOp, S: ElementOp> TransformOp for Fused<T, S, TransformElement> {
    type IndexRemapping = T::IndexRemapping;

    #[inline(always)]
    fn compute(&self, data: &[f32], index: usize) -> f32 {
        let prev = self.prev_op.compute(data, index);
        self.curr_op.compute(prev)
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        self.prev_op.input_len()
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        self.prev_op.output_len()
    }

    #[inline(always)]
    fn is_valid_input(&self, input_len: usize, output_len: usize) -> bool {
        self.prev_op.is_valid_input(input_len, output_len)
    }

    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = TransformOp::compute(self, input, out_index);
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

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        let intermediate_index = self.curr_op.map_index(out_index);
        self.prev_op.map_index(intermediate_index)
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let input_index = self.map_index(out_index);
        unsafe { *data.get_unchecked(input_index) }
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        self.prev_op.input_len()
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        self.curr_op.output_len()
    }

    // TODO: FOR NOW -> fix later
    #[inline(always)]
    fn is_valid_input(&self, curr_input_len: usize, prev_output_len: usize) -> bool {
        // let prev_curr_input = match self.curr_op.input_len() {
        //     0 => {
        //         return true;
        //     }
        //     val => val,
        // };
        // let prev_prev_output = match self.prev_op.output_len() {
        //     0 => prev_curr_input,
        //     val => val,
        // };
        // let prev_is_valid = self
        //     .prev_op
        //     .is_valid_input(prev_curr_input, prev_prev_output);
        //
        // let curr_input = match curr_input_len {
        //     0 => {
        //         return prev_is_valid;
        //     }
        //     val => val,
        // };
        // let prev_output = match prev_output_len {
        //     0 => curr_input,
        //     val => val,
        // };
        // // If, say, two Truncated are fused -> both need to be validated
        // self.curr_op.is_valid_input(curr_input, prev_output) && prev_is_valid
        true
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
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index);
            }
        }
        Ok(out)
    }
}

impl<T: ElementOp> Stage for Head<T, EMark> {
    #[inline(always)]
    fn execute<'i, 'o, const LEN: usize>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError> {
        self.root_op.setup();

        for i in 0..LEN {
            unsafe {
                *out_buf.get_unchecked_mut(i) = self.root_op.compute(*data.get_unchecked(i));
            }
        }
        Ok(out_buf)
    }

    #[inline(always)]
    fn buf_size<const LEN: usize>(&self) -> usize {
        LEN
    }

    #[inline(always)]
    fn output_len<const LEN: usize>(&self) -> usize {
        LEN
    }
}

impl<T: TransformOp> Stage for Head<T, TMark> {
    #[inline(always)]
    fn execute<'i, 'o, const LEN: usize>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError> {
        self.root_op.setup();

        let input_len = match self.root_op.input_len() {
            0 => LEN,
            val => val,
        };
        if !self.root_op.is_valid_input(input_len, LEN) {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        let n = self.root_op.output_len();
        let out = &mut out_buf[0..n];

        Ok(self.root_op.execute(out, data, n)?)
    }

    #[inline(always)]
    fn buf_size<const LEN: usize>(&self) -> usize {
        self.root_op.output_len()
    }

    #[inline(always)]
    fn output_len<const LEN: usize>(&self) -> usize {
        self.root_op.output_len()
    }
}

impl<T: TransformOp, S: Stage> Stage for Link<T, S, TMark> {
    #[inline(always)]
    fn execute<'i, 'o, const LEN: usize>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError> {
        let prev_out = self.prev_stage.execute::<LEN>(data, out_buf, in_buf)?;
        self.curr_op.setup();

        let prev_output_len = match self.prev_stage.output_len::<LEN>() {
            0 => LEN,
            val => val,
        };
        let curr_input_len = match self.curr_op.input_len() {
            0 => prev_output_len,
            val => val,
        };

        if !self.curr_op.is_valid_input(curr_input_len, prev_output_len) {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        let n = self.curr_op.output_len();
        let out = &mut out_buf[0..n];

        Ok(self.curr_op.execute(out, prev_out, n)?)
    }

    #[inline(always)]
    fn buf_size<const LEN: usize>(&self) -> usize {
        cmp::max(self.prev_stage.buf_size::<LEN>(), self.curr_op.output_len())
    }

    #[inline(always)]
    fn output_len<const LEN: usize>(&self) -> usize {
        self.curr_op.output_len()
    }
}

impl<T: ElementOp, S: Stage> Stage for Link<T, S, EMark> {
    #[inline(always)]
    fn execute<'i, 'o, const LEN: usize>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> Result<&'o [f32], PipeError> {
        let prev_out = self.prev_stage.execute::<LEN>(data, out_buf, in_buf)?;
        self.curr_op.setup();

        let n = self.buf_size::<LEN>();

        for i in 0..n {
            unsafe {
                *out_buf.get_unchecked_mut(i) = self.curr_op.compute(*prev_out.get_unchecked(i));
            }
        }
        Ok(out_buf)
    }

    #[inline(always)]
    fn buf_size<const LEN: usize>(&self) -> usize {
        self.prev_stage.buf_size::<LEN>()
    }

    #[inline(always)]
    fn output_len<const LEN: usize>(&self) -> usize {
        self.prev_stage.output_len::<LEN>()
    }
}
