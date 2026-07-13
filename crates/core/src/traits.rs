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
    fn fuse_after_transform<U>(self, op: U) -> Fused<U, Self, TransformElement>
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
    fn compute(&self, data: &[f32], out_index: usize) -> f32;

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

/// Grayscale operation (channel reduction)
/// Converts multi-channel data to single channel using luminance weights for RGB
/// or averaging for other channel counts
#[derive(Debug, Clone, Default)]
pub struct Grayscale<const IN_W: usize, const IN_H: usize, const IN_C: usize> {
    pub invert: bool,
}

impl<const IN_W: usize, const IN_H: usize, const IN_C: usize> Grayscale<IN_W, IN_H, IN_C> {
    #[inline(always)]
    fn apply_inversion(&self, val: f32) -> f32 {
        if self.invert { 255.0 - val } else { val }
    }

    #[inline(always)]
    fn compute_luminance(&self, channels: &[f32]) -> f32 {
        match IN_C {
            3 | 4 if channels.len() >= 3 => {
                let r = self.apply_inversion(channels[0]);
                let g = self.apply_inversion(channels[1]);
                let b = self.apply_inversion(channels[2]);
                0.299 * r + 0.587 * g + 0.114 * b
            }
            1 => self.apply_inversion(channels[0]),
            _ => {
                let sum: f32 = channels.iter().map(|&v| self.apply_inversion(v)).sum();
                sum / channels.len() as f32
            }
        }
    }
}

impl<const IN_W: usize, const IN_H: usize, const IN_C: usize> TransformOp
    for Grayscale<IN_W, IN_H, IN_C>
{
    #[inline(always)]
    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32] {
        for (out_pixel, in_chunk) in out[0..n].iter_mut().zip(input.chunks_exact(IN_C)) {
            *out_pixel = self.compute_luminance(in_chunk);
        }
        out
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let base_idx = out_index * IN_C;

        if base_idx >= data.len() {
            return 0.0;
        }

        let end_idx = (base_idx + IN_C).min(data.len());
        self.compute_luminance(&data[base_idx..end_idx])
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.buffer_size()
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        IN_W * IN_H
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
    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32] {
        for (out_index, out_pixel) in out[0..n].iter_mut().enumerate() {
            *out_pixel = self.compute(input, out_index);
        }
        out
    }

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
        self.buffer_size()
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        OUT_W * OUT_H * OUT_C
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

impl<T: TransformOp, const LEN: usize> Stage for Head<T, ResampleMark, LEN> {
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

impl<T: TransformOp, S: Stage> Stage for Link<T, S, ResampleMark> {
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

impl<T: ElementOp, S: Stage> Stage for Link<T, S, ElementMark> {
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
