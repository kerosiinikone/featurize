use core::{cmp, marker::PhantomData};

use alloc::vec::{Vec};
use alloc::vec;

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
    
    /// Get the output shape after this stage's transformation
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize>;
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

/// Scale and other spatial transformation operations
/// Maps from output index to computed value by sampling from input
pub trait TransformOp {
    /// Compute output value at given output index by sampling from input data
    fn compute(&self, data: &[f32], out_index: usize, in_shape: &[usize], stride: usize) -> f32;
    
    /// Get the output shape given input shape
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize>;
    
    // CAN ONLY BE FUSED THIS WAY (for now)
    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, TransformElement>
    where
        Self: Sized,
        U: ElementOp;
    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp;
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
pub struct TransformTransform;
pub struct TransformElement;

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

impl<T: TransformOp, S: ElementOp> FusedPoint<T, S, TransformElement> {
    #[inline(always)]
    fn _compute_res_el(&self, data: &[f32], index: usize, in_shape: &[usize], stride: usize) -> f32 {
        let prev = self.prev_op.compute(data, index, in_shape, stride);
        self.curr_op.compute(prev)
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

impl<T: ElementOp, S: ElementOp> TransformOp for FusedPoint<T, S, ElementElement> {
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize> {
        in_shape.to_vec()
    }

    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, TransformElement>
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

    fn compute(&self, data: &[f32], index: usize, _in_shape: &[usize], _stride: usize) -> f32 {
        self._compute_el(data[index])
    }
}

impl<T: TransformOp, S: ElementOp> TransformOp for FusedPoint<T, S, TransformElement> {
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize> {
        self.prev_op.output_shape(in_shape)
    }

    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }

    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, TransformElement>
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

    fn compute(&self, data: &[f32], index: usize, in_shape: &[usize], stride: usize) -> f32 {
        self._compute_res_el(data, index, in_shape, stride)
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

impl<T: TransformOp + PipeOp, S: ElementOp + PipeOp> PipeOp for FusedPoint<T, S, TransformElement> {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        let stride = if in_shape.len() >= 3 { in_shape[2] } else { 1 };
        self._compute_res_el(data, index, in_shape, stride)
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

impl<T: TransformOp + RootOp, S: ElementOp + RootOp> RootOp for FusedPoint<T, S, TransformElement> {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        let stride = if in_shape.len() >= 3 { in_shape[2] } else { 1 };
        self._compute_res_el(data, index, in_shape, stride)
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
    pub mean: f32,
    pub std: f32,
    pub size: usize,
}

impl Normalize {
    #[inline(always)]
    fn _compute(&self, data: f32) -> f32 {
        (data - self.mean) / self.std
    }

    #[inline(always)]
    fn _size(&self) -> usize {
        self.size
    }
}

impl ElementOp for Normalize {
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

impl RootOp for Normalize {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        self._compute(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl PipeOp for Normalize {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        self._compute(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

/// Div operation (point)
#[derive(Debug, Clone, Default)]
pub struct Div {
    pub factor: f32,
    pub size: usize,
}

impl Div {
    #[inline(always)]
    fn _compute(&self, data: f32) -> f32 {
        data / self.factor
    }

    #[inline(always)]
    fn _size(&self) -> usize {
        self.size
    }
}

impl ElementOp for Div {
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

impl RootOp for Div {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        self._compute(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl PipeOp for Div {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        self._compute(data[index])
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

/// Grayscale operation (channel reduction)
/// Converts multi-channel data to single channel using luminance weights for RGB
/// or averaging for other channel counts
#[derive(Debug, Clone, Default)]
pub struct Grayscale {
    pub in_channels: usize,
    pub out_size: usize,
    pub invert: bool,
}

impl Grayscale {
    #[inline(always)]
    fn _compute(&self, data: &[f32], out_index: usize, stride: usize) -> f32 {
        let base_idx = out_index * stride;
        
        if base_idx >= data.len() {
            return 0.0;
        }
        
        let maybe_invert = |val: f32| -> f32 {
            if self.invert {
                255.0 - val
            } else {
                val
            }
        };
        
        // Standard luminance weights for RGB: 0.299*R + 0.587*G + 0.114*B
        if stride == 3 && base_idx + 2 < data.len() {
            let r = maybe_invert(data[base_idx]);
            let g = maybe_invert(data[base_idx + 1]);
            let b = maybe_invert(data[base_idx + 2]);
            0.299 * r + 0.587 * g + 0.114 * b
        } else if stride == 4 && base_idx + 2 < data.len() {
            // RGBA: ignore alpha channel
            let r = maybe_invert(data[base_idx]);
            let g = maybe_invert(data[base_idx + 1]);
            let b = maybe_invert(data[base_idx + 2]);
            0.299 * r + 0.587 * g + 0.114 * b
        } else if stride == 1 {
            // Already grayscale
            maybe_invert(data[base_idx])
        } else {
            // Generic: average all channels
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..stride {
                if base_idx + i < data.len() {
                    sum += maybe_invert(data[base_idx + i]);
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f32
            } else {
                0.0
            }
        }
    }

    #[inline(always)]
    fn _size(&self) -> usize {
        self.out_size
    }
}

impl TransformOp for Grayscale {
    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize, _in_shape: &[usize], stride: usize) -> f32 {
        self._compute(data, out_index, stride)
    }

    #[inline(always)]
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize> {
        // Reduce channels: [H, W, C] -> [H, W, 1]
        // Always output 3D shape for consistency
        if in_shape.len() >= 2 {
            vec![in_shape[0], in_shape[1], 1]
        } else {
            // Fallback for 1D input
            in_shape.to_vec()
        }
    }

    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, TransformElement>
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

    #[inline(always)]
    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
    }
}

impl RootOp for Grayscale {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        // For RootOp, in_shape is the original input shape
        let stride = if in_shape.len() >= 3 { 
            in_shape[2] 
        } else { 
            self.in_channels 
        };
        self._compute(data, index, stride)
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl PipeOp for Grayscale {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], _in_shape: &[usize]) -> f32 {
        // For PipeOp, data comes from previous stage output
        // We use in_channels which should match the previous stage's output channels
        self._compute(data, index, self.in_channels)
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

/// Scale operation (resample with nearest-neighbor interpolation)
/// Handles multi-dimensional data with proper coordinate mapping
#[derive(Debug, Clone, Default)]
pub struct Scale {
    pub out_shape: Vec<usize>, // [out_height, out_width, channels]
    pub in_channels: Option<usize>, // Input channels (if known), otherwise extracted from shape
}

impl Scale {
    #[inline(always)]
    fn _compute(&self, data: &[f32], out_index: usize, in_shape: &[usize], stride: usize) -> f32 {
        // Handle 1D or malformed shapes
        if in_shape.len() < 2 || self.out_shape.len() < 2 {
            let in_size = in_shape.iter().product::<usize>();
            let out_size = self.out_shape.iter().product::<usize>();
            if out_size == 0 {
                return 0.0;
            }
            let scale = in_size as f32 / out_size as f32;
            let in_idx = ((out_index as f32 * scale) as usize).min(in_size.saturating_sub(1));
            return if in_idx < data.len() { data[in_idx] } else { 0.0 };
        }

        let in_height = in_shape[0];
        let in_width = in_shape[1];
        let out_height = self.out_shape[0];
        let out_width = self.out_shape[1];
        
        // For grayscale (stride=1), treat as single channel
        let channels = if stride == 1 { 1 } else if self.out_shape.len() >= 3 {
            self.out_shape[2]
        } else {
            stride
        };

        // Convert flat output index to 2D coordinates
        // For grayscale: out_index directly maps to pixel position
        let out_c = if channels > 1 { out_index % channels } else { 0 };
        let pixel_index = if channels > 1 { out_index / channels } else { out_index };
        let out_x = pixel_index % out_width;
        let out_y = pixel_index / out_width;

        if out_y >= out_height || out_x >= out_width {
            return 0.0;
        }

        // Map output coordinates to input coordinates (nearest neighbor)
        let scale_y = in_height as f32 / out_height as f32;
        let scale_x = in_width as f32 / out_width as f32;
        
        let in_y = ((out_y as f32 * scale_y) as usize).min(in_height - 1);
        let in_x = ((out_x as f32 * scale_x) as usize).min(in_width - 1);

        // Calculate input index
        let in_idx = if stride == 1 {
            // Grayscale: simple 2D indexing
            in_y * in_width + in_x
        } else {
            // Multi-channel: include channel offset
            (in_y * in_width + in_x) * stride + out_c
        };

        if in_idx < data.len() {
            data[in_idx]
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn _size(&self) -> usize {
        self.out_shape
            .iter()
            .copied()
            .reduce(|a, b| a * b)
            .unwrap_or(0)
    }
}

impl RootOp for Scale {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        let stride = if in_shape.len() >= 3 { in_shape[2] } else { 1 };
        self._compute(data, index, in_shape, stride)
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl PipeOp for Scale {
    #[inline(always)]
    fn compute(&self, index: usize, data: &[f32], in_shape: &[usize]) -> f32 {
        // Use stored in_channels if available, otherwise extract from shape
        let stride = if let Some(channels) = self.in_channels {
            channels
        } else if in_shape.len() >= 3 {
            in_shape[2]
        } else {
            1
        };
        self._compute(data, index, in_shape, stride)
    }

    #[inline(always)]
    fn buf_size(&self) -> usize {
        self._size()
    }
}

impl TransformOp for Scale {
    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize, in_shape: &[usize], stride: usize) -> f32 {
        self._compute(data, out_index, in_shape, stride)
    }

    #[inline(always)]
    fn output_shape(&self, _in_shape: &[usize]) -> Vec<usize> {
        self.out_shape.clone()
    }

    // MACRO MAGIC for creating the applies?
    #[inline(always)]
    fn _apply_point<U>(self, op: U) -> FusedPoint<Self, U, TransformElement>
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
    fn _apply_resample<U>(self, op: U) -> FusedPoint<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp,
    {
        FusedPoint {
            prev_op: self,
            curr_op: op,
            marker: PhantomData {},
        }
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

impl<T: RootOp + ElementOp> Stage for Head<T, ElementMark> {
    #[inline(always)]
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
    
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize> {
        // ElementOp doesn't change shape
        in_shape.to_vec()
    }
}

impl<T: RootOp + TransformOp> Stage for Head<T, ResampleMark> {
    #[inline(always)]
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
    
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize> {
        // TransformOp changes shape - delegate to the operation
        self.root_op.output_shape(in_shape)
    }
}

impl<T: PipeOp + TransformOp, S: Stage> Stage for Link<T, S, ResampleMark> {
    #[inline(always)]
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

        let prev_shape = self.prev_stage.output_shape(shape);

        let n = self.curr_op.buf_size();
        let out = &mut out_buf[0..n];

        for i in 0..n {
            out[i] = PipeOp::compute(&self.curr_op, i, prev_out, &prev_shape);
        }
        out
    }

    fn buf_size(&self) -> usize {
        cmp::max(self.prev_stage.buf_size(), self.curr_op.buf_size())
    }
    
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize> {
        let prev_shape = self.prev_stage.output_shape(in_shape);
        self.curr_op.output_shape(&prev_shape)
    }
}

impl<T: PipeOp + ElementOp, S: Stage> Stage for Link<T, S, ElementMark> {
    // inline
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

        let prev_shape = self.prev_stage.output_shape(shape);

        let n = self.curr_op.buf_size();
        let out = &mut out_buf[0..n];

        for i in 0..n {
            out[i] = PipeOp::compute(&self.curr_op, i, prev_out, &prev_shape);
        }
        out
    }

    fn buf_size(&self) -> usize {
        cmp::max(self.prev_stage.buf_size(), self.curr_op.buf_size())
    }
    
    fn output_shape(&self, in_shape: &[usize]) -> Vec<usize> {
        self.prev_stage.output_shape(in_shape)
    }
}
