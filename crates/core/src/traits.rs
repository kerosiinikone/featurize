use core::{cmp, marker::PhantomData};

use alloc::vec::Vec;

/// Link<Head<RootOp>, PipeOp>
/// The pipeline wrapper encloses these 'stages'
#[allow(dead_code)]
pub trait Stage {
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
        shape: &[usize],
        stride: usize,
    ) -> &'o [f32];

    fn buf_size(&self) -> usize;
}

/// Expects data from previous stages (Ref)
/// These are the top-level operations that sit inside
/// the stages
// DERIVE?
#[allow(dead_code)]
pub trait PipeOp {
    // FOR NOW
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32;
    fn buf_size(&self) -> usize;
}

/// Does not expect data from previous stages (&)
/// These are the top-level operations that sit inside
/// the stages
// DERIVE?
#[allow(dead_code)]
pub trait RootOp {
    // FOR NOW
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32;
    fn buf_size(&self) -> usize;
}

pub struct ResampleMark;
pub struct ElementMark;

/// Generic over the root operation
#[allow(dead_code)]
pub struct Head<T, Mark>
where
    T: RootOp,
{
    pub root_op: T,
    pub marker: PhantomData<Mark>,
}

/// Generic over the previous stage, current pipe operation
#[allow(dead_code)]
pub struct Link<T, S, Mark>
where
    S: Stage,
    T: PipeOp,
{
    pub prev_stage: S,
    pub curr_op: T,
    pub marker: PhantomData<Mark>,
}

/// Normalize, Grayscale
pub trait ElementOp {
    fn compute(&self, data: f32) -> f32;
    // CAN ONLY BE FUSED THIS WAY
    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U>
    where
        Self: Sized,
        U: ElementOp;
    // TODO: later -> order can be switched to produce a fusion
}

/// Scale
pub trait ResampleOp {
    fn compute(&self, data: &[f32], index: usize, in_shape: &[usize]) -> (f32, usize);
    // CAN ONLY BE FUSED THIS WAY (for now)
    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, ResampleElement>
    where
        Self: Sized,
        U: ElementOp;
    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, ResampleResample>
    where
        Self: Sized,
        U: ResampleOp;
    fn _shape(&self) -> &[usize];
}

// pub trait PixelOp {
//     fn apply(&self, data: &[f32], stride: usize) -> f32;
// }
//
//
// pub trait NeighborOp {
//     fn apply(&self, data: &[f32]) -> f32;
// }

/// Markers for assuring typestate
pub struct ElementElement;
pub struct ResampleResample;
pub struct ResampleElement;

/// Generic over the previous operation and current
/// Allows for fusing the operations
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct FusedPoint<T, S, FusedState = ElementElement> {
    pub prev_op: T,
    pub curr_op: S,

    marker: PhantomData<FusedState>,
}

// TODO: generic impl
// impl<T, S> FusedPoint<T, S> {
//     #[inline(always)]
//     fn buf_size(&self) -> usize {
//         cmp::max(self.curr_op.buf_size(), self.prev_op.buf_size())
//     }
// }

impl<T: ElementOp, S: ElementOp> FusedPoint<T, S, ElementElement> {
    #[inline(always)]
    fn _compute_el(&self, data: f32) -> f32 {
        let prev = self.prev_op.compute(data);
        self.curr_op.compute(prev)
    }
}

impl<T: ResampleOp, S: ElementOp> FusedPoint<T, S, ResampleElement> {
    #[inline(always)]
    fn _compute_res_el(&self, data: &[f32], index: usize, in_shape: &[usize]) -> (f32, usize) {
        let prev = self.prev_op.compute(data, index, in_shape);
        (self.curr_op.compute(prev.0), prev.1)
    }

    #[inline(always)]
    fn _shape_res_el(&self) -> &[usize] {
        self.prev_op._shape()
    }
}

impl<T: ElementOp, S: ElementOp> ElementOp for FusedPoint<T, S, ElementElement> {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        self._compute_el(data)
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U>
    where
        Self: Sized,
        U: ElementOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }
}

// TODO: later -> Resample + Element <-> Element + Resample (commutativity)

impl<T: ElementOp, S: ElementOp> ResampleOp for FusedPoint<T, S, ElementElement> {
    fn _shape(&self) -> &[usize] {
        todo!()
    }

    // ???
    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, ResampleResample>
    where
        Self: Sized,
        U: ResampleOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, ResampleElement>
    where
        Self: Sized,
        U: ElementOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    fn compute(&self, data: &[f32], index: usize, _in_shape: &[usize]) -> (f32, usize) {
        (self._compute_el(data[index]), index)
    }
}

impl<T: ResampleOp, S: ElementOp> ResampleOp for FusedPoint<T, S, ResampleElement> {
    fn _shape(&self) -> &[usize] {
        todo!()
    }

    // ???
    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, ResampleResample>
    where
        Self: Sized,
        U: ResampleOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, ResampleElement>
    where
        Self: Sized,
        U: ElementOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    fn compute(&self, data: &[f32], index: usize, in_shape: &[usize]) -> (f32, usize) {
        self._compute_res_el(data, index, in_shape)
    }
}

impl<T: ElementOp + PipeOp, S: ElementOp + PipeOp> PipeOp for FusedPoint<T, S, ElementElement> {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        self._compute_el(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        cmp::max(self.curr_op.buf_size(), self.prev_op.buf_size())
    }
}

impl<T: ResampleOp + PipeOp, S: ElementOp + PipeOp> PipeOp for FusedPoint<T, S, ResampleElement> {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        self._compute_res_el(data, index, in_shape).0
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        cmp::max(self.curr_op.buf_size(), self.prev_op.buf_size())
    }
}

impl<T: ElementOp + RootOp, S: ElementOp + RootOp> RootOp for FusedPoint<T, S, ElementElement> {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        self._compute_el(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        cmp::max(self.curr_op.buf_size(), self.prev_op.buf_size())
    }
}

impl<T: ResampleOp + RootOp, S: ElementOp + RootOp> RootOp for FusedPoint<T, S, ResampleElement> {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        self._compute_res_el(data, index, in_shape).0
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        cmp::max(self.curr_op.buf_size(), self.prev_op.buf_size())
    }
}

/// Normalize operation (point)
/// Norm by std and mean
#[derive(Debug, Clone, Default)]
pub struct Normalize {
    mean: f32,
    std: f32
}

/// Div operation (point)
#[derive(Debug, Clone, Default)]
pub struct Div {
    factor: f32
}

/// Scale operation (resample)
#[derive(Debug, Clone, Default)]
pub struct Scale {
    pub out_shape: Vec<usize>,
}

impl Scale {
    #[inline(always)]
    fn _compute(&self, data: &[f32], index: usize, in_shape: &[usize]) -> (f32, usize) {
        let in_dim = in_shape
            .iter()
            .copied()
            .reduce(|a, b| a * b)
            .expect("panic");
        let out_dim = self._size();
        (data[index * (out_dim / in_dim)], (out_dim / in_dim))
    }

    #[inline(always)]
    fn _size(&self) -> usize {
        self.out_shape
            .iter()
            .copied()
            .reduce(|a, b| a * b)
            .expect("panic")
    }

    // #[inline(always)]
    // pub fn offset(&self, coords: usize) -> usize {
    //     self.out_strides * coords
    // }
}

impl RootOp for Scale {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        self._compute(data, index, in_shape).0
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl PipeOp for Scale {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        self._compute(data, index, in_shape).0
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl ResampleOp for Scale {
    #[inline(always)]
    fn compute(&self, data: &[f32], index: usize, in_shape: &[usize]) -> (f32, usize) {
        self._compute(data, index, in_shape)
    }

    // MACRO MAGIC for creating the applies?
    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, ResampleElement>
    where
        Self: Sized,
        U: ElementOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    // MACRO MAGIC?
    #[inline(always)]
    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, ResampleResample>
    where
        Self: Sized,
        U: ResampleOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    #[inline(always)]
    fn _shape(&self) -> &[usize] {
        self.out_shape.as_slice()
    }
}

/// Example element-wise (point) operation
/// on the data vector
#[derive(Debug, Clone, Copy)]
pub struct _TestOp {
    pub size: usize,
}

impl _TestOp {
    #[inline(always)]
    fn _compute(&self, data: f32) -> f32 {
        data * 2.0
    }

    #[inline(always)]
    fn _size(&self) -> usize {
        self.size
    }
}

impl RootOp for _TestOp {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        self._compute(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl PipeOp for _TestOp {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        self._compute(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl ElementOp for _TestOp {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        self._compute(data)
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U>
    where
        Self: Sized,
        U: ElementOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }
}

impl<T: RootOp, State> Stage for Head<T, State> {
    // Output acts as either the temp buffer (rotated) OR the output
    // operation
    // TODO: get the right iterator form / shape with stride from the struct method -> output.iterator?
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
        shape: &[usize],
        _stride: usize,
    ) -> &'o [f32] {
        let n = self.root_op.buf_size();
        let out = &mut out_buf[0..n];

        for i in 0..n {
            out[i] = RootOp::compute(&self.root_op, i, data, shape);
        }
        out
    }

    fn buf_size(&self) -> usize {
        self.root_op.buf_size()
    }
}

impl<T: PipeOp, S: Stage, Mark> Stage for Link<T, S, Mark> {
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
        shape: &[usize],
        stride: usize,
    ) -> &'o [f32] {
        let prev_out = self
            .prev_stage
            .execute(data, out_buf, in_buf, shape, stride);

        let n = self.curr_op.buf_size();
        let out = &mut out_buf[0..n];

        for i in 0..n {
            out[i] = PipeOp::compute(&self.curr_op, i, prev_out, shape);
        }
        out
    }

    fn buf_size(&self) -> usize {
        cmp::max(self.prev_stage.buf_size(), self.curr_op.buf_size())
    }
}
