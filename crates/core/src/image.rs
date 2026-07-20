use crate::{
    errors::PipeError,
    traits::{False, TransformOp},
};

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
    type IndexRemapping = False;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != IN_W * IN_H * IN_C {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

        for i in 0..n {
            let base_idx = i * IN_C;
            unsafe {
                let in_chunk = core::slice::from_raw_parts(input.as_ptr().add(base_idx), IN_C);
                *out.get_unchecked_mut(i) = self.compute_luminance(in_chunk);
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let base_idx = out_index * IN_C;
        unsafe {
            let in_chunk = core::slice::from_raw_parts(data.as_ptr().add(base_idx), IN_C);
            self.compute_luminance(in_chunk)
        }
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        IN_W * IN_H * IN_C
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        IN_W * IN_H
    }
}

/// 2D Scale operation for images
/// Scales an image from input dimensions to output dimensions using nearest neighbor sampling
#[derive(Debug, Clone, Copy)]
pub struct Scale2D<
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
> TransformOp for Scale2D<IN_W, IN_H, IN_C, OUT_W, OUT_H, OUT_C>
{
    type IndexRemapping = False;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != IN_W * IN_H * IN_C {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index);
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let out_c = if OUT_C > 1 { out_index % OUT_C } else { 0 };
        let pixel_index = if OUT_C > 1 {
            out_index / OUT_C
        } else {
            out_index
        };

        let out_x = pixel_index % OUT_W;
        let out_y = pixel_index / OUT_W;

        let in_x = (out_x * IN_W) / OUT_W;
        let in_y = (out_y * IN_H) / OUT_H;
        let in_idx = (in_y * IN_W + in_x) * IN_C + out_c;

        unsafe { *data.get_unchecked(in_idx) }
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        IN_W * IN_H * IN_C
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        OUT_W * OUT_H * OUT_C
    }
}

/// Crop operation for images
/// Extracts a rectangular region from an image
#[derive(Debug, Clone, Copy)]
pub struct Crop<
    const IN_W: usize,
    const IN_H: usize,
    const IN_C: usize,
    const OUT_W: usize,
    const OUT_H: usize,
> {
    pub offset_x: usize,
    pub offset_y: usize,
}

impl<
    const IN_W: usize,
    const IN_H: usize,
    const IN_C: usize,
    const OUT_W: usize,
    const OUT_H: usize,
> TransformOp for Crop<IN_W, IN_H, IN_C, OUT_W, OUT_H>
{
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let out_c = out_index % IN_C;
        let pixel_index = out_index / IN_C;
        let out_x = pixel_index % OUT_W;
        let out_y = pixel_index / OUT_W;

        let in_x = out_x + self.offset_x;
        let in_y = out_y + self.offset_y;

        let in_idx = (in_y * IN_W + in_x) * IN_C + out_c;

        unsafe { *data.get_unchecked(in_idx) }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != IN_W * IN_H * IN_C {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index);
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        IN_W * IN_H * IN_C
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        OUT_W * OUT_H * IN_C
    }
}

/// Rotate90 operation - rotates image 90 degrees clockwise
#[derive(Debug, Clone, Copy)]
pub struct Rotate90<const W: usize, const H: usize, const C: usize>;

impl<const W: usize, const H: usize, const C: usize> TransformOp for Rotate90<W, H, C> {
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let out_c = out_index % C;
        let pixel_index = out_index / C;
        let out_x = pixel_index % H;
        let out_y = pixel_index / H;

        let in_x = H - 1 - out_y;
        let in_y = out_x;

        let in_idx = (in_y * W + in_x) * C + out_c;

        unsafe { *data.get_unchecked(in_idx) }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != W * H * C {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index);
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        W * H * C
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        W * H * C
    }
}

/// Flip horizontal operation
#[derive(Debug, Clone, Copy)]
pub struct FlipHorizontal<const W: usize, const H: usize, const C: usize>;

impl<const W: usize, const H: usize, const C: usize> TransformOp for FlipHorizontal<W, H, C> {
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let out_c = out_index % C;
        let pixel_index = out_index / C;
        let out_x = pixel_index % W;
        let out_y = pixel_index / W;

        let in_x = W - 1 - out_x;
        let in_y = out_y;

        let in_idx = (in_y * W + in_x) * C + out_c;

        unsafe { *data.get_unchecked(in_idx) }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != W * H * C {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index);
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        W * H * C
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        W * H * C
    }
}

/// Flip vertical operation
#[derive(Debug, Clone, Copy)]
pub struct FlipVertical<const W: usize, const H: usize, const C: usize>;

impl<const W: usize, const H: usize, const C: usize> TransformOp for FlipVertical<W, H, C> {
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let out_c = out_index % C;
        let pixel_index = out_index / C;
        let out_x = pixel_index % W;
        let out_y = pixel_index / W;

        let in_x = out_x;
        let in_y = H - 1 - out_y;

        let in_idx = (in_y * W + in_x) * C + out_c;

        unsafe { *data.get_unchecked(in_idx) }
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        // if input.len() != W * H * C {
        //     return Err(PipeError::new(ErrorKind::InvalidInputSize));
        // }

        for out_index in 0..n {
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index);
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn input_len(&self) -> usize {
        W * H * C
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        W * H * C
    }
}
