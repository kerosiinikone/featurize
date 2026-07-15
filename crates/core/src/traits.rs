use core::{cmp, marker::PhantomData};

/// Link<Head<RootOp>, PipeOp>
/// The pipeline wrapper encloses these 'stages'
#[allow(dead_code)]
pub trait Stage {
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32];

    fn buf_size(&self) -> usize;

    fn output_shape(&self) -> usize;
}

/// Stage marks (curr op)
pub struct TMark;
pub struct EMark;

/// Generic over the root operation
#[allow(dead_code)]
pub struct Head<T, Mark, const LEN: usize> {
    pub root_op: T,
    pub marker: PhantomData<Mark>,
}

/// Generic over the previous stage, current operation
#[allow(dead_code)]
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

/// Spatial transformation operations that map from output index to computed value by sampling from input
pub trait TransformOp {
    /// Compute output value at given output index by sampling from input data
    fn compute(&self, _: &[f32], __: usize) -> f32 {
        unimplemented!()
    }

    /// Execute the transformation using chunk-based iteration (for stride operations)
    /// or index-based iteration (for non-linear operations)
    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32];

    fn buffer_size(&self) -> usize;
    fn output_shape(&self) -> usize;

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

    // TODO: Implementable through single operations types (1:1)
    #[inline(always)]
    #[allow(dead_code)]
    fn fuse_transform<U>(self, _: U) -> Fused<U, Self, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        unimplemented!()
    }
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
    #[inline(always)]
    fn compute(&self, data: &[f32], index: usize) -> f32 {
        let prev = self.prev_op.compute(data[index]);
        self.curr_op.compute(prev)
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        0
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        0
    }

    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32] {
        for (out_index, out_pixel) in out[0..n].iter_mut().enumerate() {
            *out_pixel = TransformOp::compute(self, input, out_index);
        }
        out
    }
}

impl<T: TransformOp, S: ElementOp> TransformOp for Fused<T, S, TransformElement> {
    #[inline(always)]
    fn compute(&self, data: &[f32], index: usize) -> f32 {
        let prev = self.prev_op.compute(data, index);
        self.curr_op.compute(prev)
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        self.prev_op.buffer_size()
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.prev_op.output_shape()
    }

    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32] {
        for (out_index, out_pixel) in out[0..n].iter_mut().enumerate() {
            *out_pixel = TransformOp::compute(self, input, out_index);
        }
        out
    }
}

impl<T: ElementOp, const LEN: usize> Stage for Head<T, EMark, LEN> {
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32] {
        self.root_op.setup();

        for (out_pixel, in_pixel) in out_buf[0..LEN].iter_mut().zip(data.iter()) {
            *out_pixel = self.root_op.compute(*in_pixel);
        }
        out_buf
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        LEN
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        LEN
    }
}

impl<T: TransformOp, const LEN: usize> Stage for Head<T, TMark, LEN> {
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32] {
        self.root_op.setup();
        let n = self.root_op.buffer_size();
        let out = &mut out_buf[0..n];

        self.root_op.execute(out, data, n)
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self.root_op.buffer_size()
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.root_op.output_shape()
    }
}

impl<T: TransformOp, S: Stage> Stage for Link<T, S, TMark> {
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32] {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf);
        self.curr_op.setup();
        let n = self.curr_op.buffer_size();
        let out = &mut out_buf[0..n];

        self.curr_op.execute(out, prev_out, n)
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        cmp::max(self.prev_stage.buf_size(), self.curr_op.buffer_size())
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.curr_op.output_shape()
    }
}

impl<T: ElementOp, S: Stage> Stage for Link<T, S, EMark> {
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32] {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf);
        self.curr_op.setup();
        let n = self.buf_size();

        for (out_pixel, in_pixel) in out_buf[0..n].iter_mut().zip(prev_out.iter()) {
            *out_pixel = self.curr_op.compute(*in_pixel);
        }
        out_buf
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self.prev_stage.buf_size()
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.prev_stage.output_shape()
    }
}
