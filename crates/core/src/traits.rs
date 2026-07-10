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

pub struct ResampleMark;
pub struct ElementMark;

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

/// Normalize, Grayscale
/// Point-wise operations that work on individual elements
pub trait ElementOp {
    fn compute(&self, data: f32) -> f32;
    fn _setup(&self) {}
    fn _apply_point<U>(self, op: U) -> Fused<Self, U>
    where
        Self: Sized,
        U: ElementOp;
}

/// Scale and other spatial transformation operations
/// Maps from output index to computed value by sampling from input
pub trait TransformOp {
    /// Compute output value at given output index by sampling from input data
    fn compute(&self, data: &[f32], out_index: usize) -> f32;
    fn buf_size(&self) -> usize;
    fn output_shape(&self) -> usize;
    fn _setup(&self) {}
    fn _apply_point<U>(self, op: U) -> Fused<Self, U, TransformElement>
    where
        Self: Sized,
        U: ElementOp;
    fn _apply_resample<U>(self, op: U) -> Fused<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp;
}

/// Markers for assuring typestate
pub struct ElementElement;
pub struct TransformTransform;
pub struct TransformElement;

/// Generic over the previous operation and current
/// Allows for fusing the operations
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Fused<T, S, FusedState = ElementElement> {
    pub prev_op: T,
    pub curr_op: S,

    marker: PhantomData<FusedState>,
}

impl<T: ElementOp, S: ElementOp> Fused<T, S, ElementElement> {
    #[inline(always)]
    fn _compute_el(&self, data: f32) -> f32 {
        let prev = self.prev_op.compute(data);
        self.curr_op.compute(prev)
    }
}

impl<T: TransformOp, S: ElementOp> Fused<T, S, TransformElement> {
    #[inline(always)]
    fn _compute_res_el(&self, data: &[f32], index: usize) -> f32 {
        let prev = self.prev_op.compute(data, index);
        self.curr_op.compute(prev)
    }
}

impl<T: ElementOp, S: ElementOp> ElementOp for Fused<T, S, ElementElement> {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        self._compute_el(data)
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> Fused<Self, U>
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

// TODO: later -> Resample + Element <-> Element + Resample (commutativity)

impl<T: ElementOp, S: ElementOp> TransformOp for Fused<T, S, ElementElement> {
    #[inline(always)]
    fn _apply_resample<U>(self, op: U) -> Fused<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> Fused<Self, U, TransformElement>
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
    fn compute(&self, data: &[f32], index: usize) -> f32 {
        self._compute_el(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        0
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        0
    }
}

impl<T: TransformOp, S: ElementOp> TransformOp for Fused<T, S, TransformElement> {
    #[inline(always)]
    fn _apply_resample<U>(self, op: U) -> Fused<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> Fused<Self, U, TransformElement>
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
    fn compute(&self, data: &[f32], index: usize) -> f32 {
        self._compute_res_el(data, index)
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self.prev_op.buf_size()
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.prev_op.output_shape()
    }
}

/// Normalize operation (point)
/// Norm by std and mean
#[derive(Debug, Clone, Default)]
pub struct Normalize {
    pub mean: f32,
    pub std: f32,
}

impl Normalize {
    #[inline(always)]
    fn _compute(&self, data: f32) -> f32 {
        (data - self.mean) / self.std
    }
}

impl ElementOp for Normalize {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        self._compute(data)
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> Fused<Self, U>
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

/// Div operation (point)
#[derive(Debug, Clone, Default)]
pub struct Div {
    pub factor: f32,
}

impl Div {
    #[inline(always)]
    fn _compute(&self, data: f32) -> f32 {
        data / self.factor
    }
}

impl ElementOp for Div {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        self._compute(data)
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> Fused<Self, U>
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

/// Grayscale operation (channel reduction)
/// Converts multi-channel data to single channel using luminance weights for RGB
/// or averaging for other channel counts
#[derive(Debug, Clone, Default)]
pub struct Grayscale<const IN_W: usize, const IN_H: usize, const IN_C: usize> {
    pub invert: bool,
}

impl<const IN_W: usize, const IN_H: usize, const IN_C: usize> TransformOp
    for Grayscale<IN_W, IN_H, IN_C>
{
    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let base_idx = out_index * IN_C;

        if base_idx >= data.len() {
            return 0.0;
        }
        let maybe_invert = |val: f32| -> f32 { if self.invert { 255.0 - val } else { val } };

        if IN_C == 3 && base_idx + 2 < data.len() {
            let r = maybe_invert(data[base_idx]);
            let g = maybe_invert(data[base_idx + 1]);
            let b = maybe_invert(data[base_idx + 2]);
            0.299 * r + 0.587 * g + 0.114 * b
        } else if IN_C == 4 && base_idx + 2 < data.len() {
            let r = maybe_invert(data[base_idx]);
            let g = maybe_invert(data[base_idx + 1]);
            let b = maybe_invert(data[base_idx + 2]);
            0.299 * r + 0.587 * g + 0.114 * b
        } else if IN_C == 1 {
            maybe_invert(data[base_idx])
        } else {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..IN_C {
                if base_idx + i < data.len() {
                    sum += maybe_invert(data[base_idx + i]);
                    count += 1;
                }
            }
            if count > 0 { sum / count as f32 } else { 0.0 }
        }
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.buf_size()
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        IN_W * IN_H * 1
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> Fused<Self, U, TransformElement>
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
    fn _apply_resample<U>(self, op: U) -> Fused<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Scale<
    const IN_W: usize,
    const IN_H: usize,
    const IN_C: usize,
    const OUT_W: usize,
    const OUT_H: usize,
    const OUT_C: usize,
>;

impl<
    const IN_W: usize,
    const IN_H: usize,
    const IN_C: usize,
    const OUT_W: usize,
    const OUT_H: usize,
    const OUT_C: usize,
> TransformOp for Scale<IN_W, IN_H, IN_C, OUT_W, OUT_H, OUT_C>
{
    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let in_size: usize = IN_W * IN_H * IN_C;
        let scale_x: f32 = IN_W as f32 / OUT_W as f32;
        let scale_y: f32 = IN_H as f32 / OUT_H as f32;

        let out_c = if OUT_C > 1 { out_index % OUT_C } else { 0 };
        let pixel_index = if OUT_C > 1 {
            out_index / OUT_C
        } else {
            out_index
        };

        let out_x = pixel_index % OUT_W;
        let out_y = pixel_index / OUT_W;

        if out_y >= OUT_H || out_x >= OUT_W {
            return 0.0;
        }

        let in_y = ((out_y as f32 * scale_y) as usize).min(IN_H - 1);
        let in_x = ((out_x as f32 * scale_x) as usize).min(IN_W - 1);

        let in_idx = (in_y * IN_W + in_x) * IN_C + out_c;

        if in_idx < in_size { data[in_idx] } else { 0.0 }
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.buf_size()
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        OUT_W * OUT_H * OUT_C
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> Fused<Self, U, TransformElement>
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
    fn _apply_resample<U>(self, op: U) -> Fused<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }
}

impl<T: ElementOp, const LEN: usize> Stage for Head<T, ElementMark, LEN> {
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32] {
        self.root_op._setup();

        let out = &mut out_buf[0..LEN];
        for i in 0..LEN {
            out[i] = self.root_op.compute(data[i]);
        }
        out
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

impl<T: TransformOp, const LEN: usize> Stage for Head<T, ResampleMark, LEN> {
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        _in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32] {
        self.root_op._setup();

        let n = self.root_op.buf_size();
        let out = &mut out_buf[0..n];

        for i in 0..n {
            out[i] = self.root_op.compute(data, i);
        }
        out
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self.root_op.buf_size()
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.root_op.output_shape()
    }
}

impl<T: TransformOp, S: Stage> Stage for Link<T, S, ResampleMark> {
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32] {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf);

        self.curr_op._setup();

        let n = self.curr_op.buf_size();
        let out = &mut out_buf[0..n];

        for i in 0..n {
            out[i] = self.curr_op.compute(prev_out, i);
        }
        out
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        cmp::max(self.prev_stage.buf_size(), self.curr_op.buf_size())
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.curr_op.output_shape()
    }
}

impl<T: ElementOp, S: Stage> Stage for Link<T, S, ElementMark> {
    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[f32],
        in_buf: &'i mut [f32],
        out_buf: &'o mut [f32],
    ) -> &'o [f32] {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf);

        self.curr_op._setup();

        let n = self.buf_size();
        let out = &mut out_buf[0..n];

        for i in 0..n {
            out[i] = self.curr_op.compute(prev_out[i]);
        }
        out
    }

    fn buf_size(&self) -> usize {
        self.prev_stage.buf_size()
    }

    fn output_shape(&self) -> usize {
        self.prev_stage.output_shape()
    }
}
