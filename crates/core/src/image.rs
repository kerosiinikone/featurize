use crate::{
    errors::{check_finite, ErrorKind, NanHandling, PipeError},
    traits::{False, Float, IsTrue, TransformOp, True},
};

/// Channel layout of an image buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ChannelLayout {
    /// Interleaved
    #[default]
    Hwc,
    /// Planar
    Chw,
}

/// Grayscale operation (channel reduction)
/// Converts multi-channel data to single channel using luminance weights for RGB
/// or averaging for other channel counts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grayscale<const IN_W: usize, const IN_H: usize, const IN_C: usize, T: Float> {
    invert: bool,
    nan_handling: NanHandling,
    marker: core::marker::PhantomData<T>,
}

impl<const IN_W: usize, const IN_H: usize, const IN_C: usize, T: Float>
    Grayscale<IN_W, IN_H, IN_C, T>
{
    pub fn new() -> Self {
        Self {
            invert: false,
            marker: core::marker::PhantomData,
            nan_handling: NanHandling::default(),
        }
    }

    pub fn with_inversion(mut self) -> Self {
        self.invert = true;
        self
    }

    pub fn set_nan_handling(mut self, nan_handling: NanHandling) -> Self {
        self.nan_handling = nan_handling;
        self
    }
}

impl<const IN_W: usize, const IN_H: usize, const IN_C: usize, T: Float> Default
    for Grayscale<IN_W, IN_H, IN_C, T>
{
    fn default() -> Self {
        Self {
            invert: false,
            marker: core::marker::PhantomData,
            nan_handling: NanHandling::default(),
        }
    }
}

impl<const IN_W: usize, const IN_H: usize, const IN_C: usize, T: Float>
    Grayscale<IN_W, IN_H, IN_C, T>
{
    #[inline(always)]
    fn apply_inversion(&self, val: T) -> T {
        if self.invert {
            // This should never fail due to `f32` -> `f64` being a safe
            // operation. T is always either `f32` or `f64`.
            num_traits::cast::<f32, T>(255.0).unwrap() - val
        } else {
            val
        }
    }

    #[inline(always)]
    fn compute_luminance(&self, channels: &[T]) -> T {
        const {
            assert!(IN_C != 0, "No channels");
        }
        match IN_C {
            3 | 4 if channels.len() >= 3 => {
                let r = self.apply_inversion(channels[0]);
                let g = self.apply_inversion(channels[1]);
                let b = self.apply_inversion(channels[2]);
                // This should never fail due to `f32` -> `f64` being a safe
                // operation. T is always either `f32` or `f64`.
                num_traits::cast::<f32, T>(0.299).unwrap() * r
                    + num_traits::cast::<f32, T>(0.587).unwrap() * g
                    + num_traits::cast::<f32, T>(0.114).unwrap() * b
            }
            1 => self.apply_inversion(channels[0]),
            _ => {
                let sum: T = channels.iter().map(|&v| self.apply_inversion(v)).sum();
                sum / num_traits::cast::<usize, T>(channels.len()).unwrap_or(T::zero())
            }
        }
    }
}

impl<const IN_W: usize, const IN_H: usize, const IN_C: usize, T: Float> TransformOp<T>
    for Grayscale<IN_W, IN_H, IN_C, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = IN_W * IN_H * IN_C;
    const OUT_LEN: usize = IN_W * IN_H;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the per-pixel loop below can be
        // unchecked
        if input.len() < n * IN_C || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for i in 0..n {
            let base_idx = i * IN_C;
            // SAFETY: `base_idx + IN_C <= n * IN_C <= input.len()` and
            // `i < n <= out.len()` (both checked above)
            unsafe {
                let in_chunk = core::slice::from_raw_parts(input.as_ptr().add(base_idx), IN_C);
                let luminance = self.compute_luminance(in_chunk);
                *out.get_unchecked_mut(i) = check_finite(luminance, self.nan_handling)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let base_idx = out_index * IN_C;
        debug_assert!(base_idx + IN_C <= data.len());
        // SAFETY: caller contract: `out_index < out_len(..) == IN_W * IN_H`,
        // so `base_idx + IN_C <= IN_W * IN_H * IN_C == in_len(..) ==
        // data.len()`
        let luminance = unsafe {
            let in_chunk = core::slice::from_raw_parts(data.as_ptr().add(base_idx), IN_C);
            self.compute_luminance(in_chunk)
        };
        check_finite(luminance, self.nan_handling)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Grayscale")
    }
}

/// Layout conversion: HWC (interleaved) -> CHW (planar)
/// Pure index remapping, fuses with adjacent transforms.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HwcToChw<const W: usize, const H: usize, const C: usize, T: Float> {
    marker: core::marker::PhantomData<T>,
}

impl<const W: usize, const H: usize, const C: usize, T: Float> HwcToChw<W, H, C, T> {
    pub fn new() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<const W: usize, const H: usize, const C: usize, T: Float> TransformOp<T>
    for HwcToChw<W, H, C, T>
{
    type IndexRemapping = True;

    const IN_LEN: usize = W * H * C;
    const OUT_LEN: usize = W * H * C;

    #[inline(always)]
    fn map_index(&self, out_index: usize, _default_len: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        let c = out_index / (W * H);
        let pixel_index = out_index % (W * H);
        pixel_index * C + c
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        debug_assert!(out.len() >= n);
        for out_index in 0..n {
            // SAFETY: `out_index < n == out.len()` (guaranteed by the stage);
            // the input access is bounded by the `map_index` contract (see
            // `compute`)
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let in_index = <HwcToChw<W, H, C, T> as TransformOp<T>>::map_index(self, out_index, 0);
        debug_assert!(in_index < data.len());
        // SAFETY: caller contract: `out_index < out_len(..) == W * H * C`,
        // so `c < C` and `pixel_index < W * H`, hence
        // `in_index < W * H * C == in_len(..) == data.len()`
        Ok(unsafe { *data.get_unchecked(in_index) })
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("HwcToChw")
    }
}

/// Layout conversion: CHW (planar) -> HWC (interleaved)
/// Pure index remapping, fuses with adjacent transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ChwToHwc<const W: usize, const H: usize, const C: usize, T: Float> {
    marker: core::marker::PhantomData<T>,
}

impl<const W: usize, const H: usize, const C: usize, T: Float> ChwToHwc<W, H, C, T> {
    pub fn new() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<const W: usize, const H: usize, const C: usize, T: Float> TransformOp<T>
    for ChwToHwc<W, H, C, T>
{
    type IndexRemapping = True;

    const IN_LEN: usize = W * H * C;
    const OUT_LEN: usize = W * H * C;

    #[inline(always)]
    fn map_index(&self, out_index: usize, _default_len: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        // Output (HWC): out_index = (y * W + x) * C + c
        let c = out_index % C;
        let pixel_index = out_index / C;
        // Input (CHW): c * (W * H) + y * W + x
        c * (W * H) + pixel_index
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        debug_assert!(out.len() >= n);
        for out_index in 0..n {
            // SAFETY: `out_index < n == out.len()` (guaranteed by the stage);
            // the input access is bounded by the `map_index` contract (see
            // `compute`)
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let in_index = <ChwToHwc<W, H, C, T> as TransformOp<T>>::map_index(self, out_index, 0);
        debug_assert!(in_index < data.len());
        // SAFETY: caller contract: `out_index < out_len(..) == W * H * C`,
        // so `c < C` and `pixel_index < W * H`, hence
        // `in_index < W * H * C == in_len(..) == data.len()`
        Ok(unsafe { *data.get_unchecked(in_index) })
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("ChwToHwc")
    }
}

/// Per-channel normalization: `(x - mean[c]) / std[c]`
/// ImageNet-style preprocessing; works on HWC or CHW buffers (see `layout`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizePerChannel<const W: usize, const H: usize, const C: usize, T: Float> {
    mean: [T; C],
    std: [T; C],
    layout: ChannelLayout,
    nan_handling: NanHandling,
}

impl<const W: usize, const H: usize, const C: usize, T: Float> NormalizePerChannel<W, H, C, T> {
    pub fn new(mean: [T; C], std: [T; C]) -> Self {
        Self {
            mean,
            std,
            layout: ChannelLayout::default(),
            nan_handling: NanHandling::default(),
        }
    }

    // Takes ownership in the `TransformOp<_>` impl, hence not a mutable reference.
    pub fn with_layout(mut self, layout: ChannelLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn set_nan_handling(mut self, nan_handling: NanHandling) -> Self {
        self.nan_handling = nan_handling;
        self
    }

    #[inline(always)]
    fn channel_of(&self, index: usize) -> usize {
        match self.layout {
            ChannelLayout::Hwc => index % C,
            ChannelLayout::Chw => index / (W * H),
        }
    }
}

impl<const W: usize, const H: usize, const C: usize, T: Float> TransformOp<T>
    for NormalizePerChannel<W, H, C, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = W * H * C;
    const OUT_LEN: usize = W * H * C;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        if n > W * H * C || input.len() < n || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for i in 0..n {
            let c = self.channel_of(i);
            debug_assert!(c < C);
            // SAFETY: `i < n <= input.len()` and `i < n <= out.len()`
            // (checked above); `c < C` for both layouts since
            // `i < n <= W * H * C`
            unsafe {
                let result = (*input.get_unchecked(i) - *self.mean.get_unchecked(c))
                    / *self.std.get_unchecked(c);
                *out.get_unchecked_mut(i) = check_finite(result, self.nan_handling)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let c = self.channel_of(out_index);
        debug_assert!(out_index < data.len());
        debug_assert!(c < C);
        // SAFETY: caller contract: `out_index < out_len(..) == W * H * C ==
        // in_len(..) == data.len()`; `c < C` for both layouts
        let result = unsafe {
            (*data.get_unchecked(out_index) - *self.mean.get_unchecked(c))
                / *self.std.get_unchecked(c)
        };
        check_finite(result, self.nan_handling)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("NormalizePerChannel")
    }
}

/// 2D Scale operation for images
/// Scales an image from input dimensions to output dimensions using
/// **nearest neighbor** sampling. For bilinear interpolation (the usual ML
/// preprocessing default) use [`Scale2DBilinear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Scale2D<
    const IN_W: usize,
    const IN_H: usize,
    const IN_C: usize,
    const OUT_W: usize,
    const OUT_H: usize,
    const OUT_C: usize,
    T: Float,
> {
    marker: core::marker::PhantomData<T>,
}

impl<
        const IN_W: usize,
        const IN_H: usize,
        const IN_C: usize,
        const OUT_W: usize,
        const OUT_H: usize,
        const OUT_C: usize,
        T: Float,
    > Scale2D<IN_W, IN_H, IN_C, OUT_W, OUT_H, OUT_C, T>
{
    pub fn new() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<
        const IN_W: usize,
        const IN_H: usize,
        const IN_C: usize,
        const OUT_W: usize,
        const OUT_H: usize,
        const OUT_C: usize,
        T: Float,
    > TransformOp<T> for Scale2D<IN_W, IN_H, IN_C, OUT_W, OUT_H, OUT_C, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = IN_W * IN_H * IN_C;
    const OUT_LEN: usize = OUT_W * OUT_H * OUT_C;
    /// The output channel index is used to sample the *input* channels, so
    /// `OUT_C > IN_C` would read out of bounds. Asserted at every
    /// pipe-construction site.
    const INTERNAL_IS_VALID: bool = OUT_C <= IN_C;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        if n > OUT_W * OUT_H * OUT_C || input.len() < IN_W * IN_H * IN_C || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            // SAFETY: `out_index < n <= out.len()` (checked above); the read
            // is bounded by the index math in `compute`
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        debug_assert!(out_index < OUT_W * OUT_H * OUT_C);

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

        debug_assert!(in_idx < data.len());
        // SAFETY: caller contract: `out_index < OUT_LEN`, so `out_x < OUT_W`
        // and `out_y < OUT_H`, hence `in_x < IN_W` and `in_y < IN_H`;
        // `out_c < OUT_C <= IN_C` (INTERNAL_IS_VALID, asserted at
        // construction), therefore
        // `in_idx < IN_W * IN_H * IN_C == in_len(..) == data.len()`
        Ok(unsafe { *data.get_unchecked(in_idx) })
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Scale2D")
    }
}

/// 2D Scale operation for images using **bilinear** interpolation
/// (half-pixel-centers convention, matching common ML frameworks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Scale2DBilinear<
    const IN_W: usize,
    const IN_H: usize,
    const IN_C: usize,
    const OUT_W: usize,
    const OUT_H: usize,
    const OUT_C: usize,
    T: Float,
> {
    marker: core::marker::PhantomData<T>,
}

impl<
        const IN_W: usize,
        const IN_H: usize,
        const IN_C: usize,
        const OUT_W: usize,
        const OUT_H: usize,
        const OUT_C: usize,
        T: Float,
    > Scale2DBilinear<IN_W, IN_H, IN_C, OUT_W, OUT_H, OUT_C, T>
{
    pub fn new() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<
        const IN_W: usize,
        const IN_H: usize,
        const IN_C: usize,
        const OUT_W: usize,
        const OUT_H: usize,
        const OUT_C: usize,
        T: Float,
    > TransformOp<T> for Scale2DBilinear<IN_W, IN_H, IN_C, OUT_W, OUT_H, OUT_C, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = IN_W * IN_H * IN_C;
    const OUT_LEN: usize = OUT_W * OUT_H * OUT_C;
    /// See `Scale2D`; additionally all dimensions must be non-zero for the
    /// sampling math below.
    const INTERNAL_IS_VALID: bool = OUT_C <= IN_C && IN_W > 0 && IN_H > 0 && OUT_W > 0 && OUT_H > 0;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        if n > OUT_W * OUT_H * OUT_C || input.len() < IN_W * IN_H * IN_C || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            // SAFETY: `out_index < n <= out.len()` (checked above); the reads
            // are bounded by the index math in `compute`
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    // TODO: see the casts!
    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        debug_assert!(out_index < OUT_W * OUT_H * OUT_C);

        let out_c = if OUT_C > 1 { out_index % OUT_C } else { 0 };
        let pixel_index = if OUT_C > 1 {
            out_index / OUT_C
        } else {
            out_index
        };

        let out_x = pixel_index % OUT_W;
        let out_y = pixel_index / OUT_W;

        let scale_x = num_traits::cast::<usize, T>(IN_W).unwrap()
            / num_traits::cast::<usize, T>(OUT_W).unwrap();
        let scale_y = num_traits::cast::<usize, T>(IN_H).unwrap()
            / num_traits::cast::<usize, T>(OUT_H).unwrap();

        let half = num_traits::cast::<f32, T>(0.5).unwrap();
        // Half-pixel centers; clamped to 0 so truncation below equals floor
        let src_x =
            ((num_traits::cast::<usize, T>(out_x).unwrap() + half) * scale_x - half).max(T::zero());
        let src_y =
            ((num_traits::cast::<usize, T>(out_y).unwrap() + half) * scale_y - half).max(T::zero());

        let x0 = (num_traits::cast::<T, usize>(src_x).unwrap()).min(IN_W - 1);
        let y0 = (num_traits::cast::<T, usize>(src_y).unwrap()).min(IN_H - 1);
        let x1 = (x0 + 1).min(IN_W - 1);
        let y1 = (y0 + 1).min(IN_H - 1);

        let fx = src_x - num_traits::cast::<usize, T>(x0).unwrap();
        let fy = src_y - num_traits::cast::<usize, T>(y0).unwrap();

        let idx = |x: usize, y: usize| (y * IN_W + x) * IN_C + out_c;

        debug_assert!(idx(x1, y1) < data.len());
        // SAFETY: `x0 <= x1 <= IN_W - 1` and `y0 <= y1 <= IN_H - 1` (clamped
        // above); `out_c < OUT_C <= IN_C` (INTERNAL_IS_VALID, asserted at
        // construction), therefore each sampled index is
        // `< IN_W * IN_H * IN_C == in_len(..) == data.len()`
        let (v00, v10, v01, v11) = unsafe {
            (
                *data.get_unchecked(idx(x0, y0)),
                *data.get_unchecked(idx(x1, y0)),
                *data.get_unchecked(idx(x0, y1)),
                *data.get_unchecked(idx(x1, y1)),
            )
        };

        let top = v00 + (v10 - v00) * fx;
        let bottom = v01 + (v11 - v01) * fx;
        Ok(top + (bottom - top) * fy)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Scale2DBilinear")
    }
}

/// Crop operation for images
/// Extracts a rectangular region from an image
///
/// Is only static for now
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Crop<
    const IN_W: usize,
    const IN_H: usize,
    const IN_C: usize,
    const OUT_W: usize,
    const OUT_H: usize,
    T: Float,
> {
    offset_x: usize,
    offset_y: usize,
    marker: core::marker::PhantomData<T>,
}

impl<
        const IN_W: usize,
        const IN_H: usize,
        const IN_C: usize,
        const OUT_W: usize,
        const OUT_H: usize,
        T: Float,
    > Crop<IN_W, IN_H, IN_C, OUT_W, OUT_H, T>
{
    pub fn new(offset_x: usize, offset_y: usize) -> Self {
        assert!(offset_x + OUT_W <= IN_W);
        assert!(offset_y + OUT_H <= IN_H);

        Self {
            offset_x,
            offset_y,
            marker: core::marker::PhantomData,
        }
    }

    /// Center crop: offsets computed so the crop window is centered in the
    /// input image.
    pub fn centered() -> Self {
        Self::new((IN_W - OUT_W) / 2, (IN_H - OUT_H) / 2)
    }
}

impl<
        const IN_W: usize,
        const IN_H: usize,
        const IN_C: usize,
        const OUT_W: usize,
        const OUT_H: usize,
        T: Float,
    > TransformOp<T> for Crop<IN_W, IN_H, IN_C, OUT_W, OUT_H, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = IN_W * IN_H * IN_C;
    const OUT_LEN: usize = OUT_W * OUT_H * IN_C;
    /// The crop window must fit into the input even before applying the
    /// runtime offsets (those are validated in `execute` / `compute`)
    const INTERNAL_IS_VALID: bool = OUT_W <= IN_W && OUT_H <= IN_H;

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let out_c = out_index % IN_C;
        let pixel_index = out_index / IN_C;
        let out_x = pixel_index % OUT_W;
        let out_y = pixel_index / OUT_W;

        let in_x = out_x + self.offset_x;
        let in_y = out_y + self.offset_y;

        let in_idx = (in_y * IN_W + in_x) * IN_C + out_c;

        // Safe with bound checks
        data.get(in_idx)
            .copied()
            .ok_or_else(|| PipeError::new(ErrorKind::InvalidInputSize))
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Validate the *runtime* crop window once, so the loop below can run
        // without per-element bound checks
        if self.offset_x + OUT_W > IN_W
            || self.offset_y + OUT_H > IN_H
            || n > OUT_W * OUT_H * IN_C
            || input.len() < IN_W * IN_H * IN_C
            || out.len() < n
        {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            let out_c = out_index % IN_C;
            let pixel_index = out_index / IN_C;
            let out_x = pixel_index % OUT_W;
            let out_y = pixel_index / OUT_W;

            let in_x = out_x + self.offset_x;
            let in_y = out_y + self.offset_y;

            let in_idx = (in_y * IN_W + in_x) * IN_C + out_c;

            // SAFETY: the window validation above guarantees
            // `in_x < offset_x + OUT_W <= IN_W` and
            // `in_y < offset_y + OUT_H <= IN_H`, hence
            // `in_idx < IN_W * IN_H * IN_C <= input.len()`;
            // `out_index < n <= out.len()` was also checked above
            unsafe {
                *out.get_unchecked_mut(out_index) = *input.get_unchecked(in_idx);
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Crop")
    }
}

/// Letterbox: aspect-preserving scale (nearest neighbor) into a centered
/// window of the output, padding the remainder with `pad_value`
/// (YOLO-style pad-to-aspect). Scaled size and offsets are computed once in
/// the constructor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox<
    const IN_W: usize,
    const IN_H: usize,
    const C: usize,
    const OUT_W: usize,
    const OUT_H: usize,
    T: Float,
> {
    pad_value: T,
    scaled_w: usize,
    scaled_h: usize,
    offset_x: usize,
    offset_y: usize,
}

impl<
        const IN_W: usize,
        const IN_H: usize,
        const C: usize,
        const OUT_W: usize,
        const OUT_H: usize,
        T: Float,
    > Letterbox<IN_W, IN_H, C, OUT_W, OUT_H, T>
{
    pub fn new(pad_value: T) -> Self {
        const {
            assert!(
                IN_W > 0 && IN_H > 0 && OUT_W > 0 && OUT_H > 0,
                "Zero dimension"
            );
        }

        // Type hacking, no need to cast to T as only known floats and integers
        // are in question.
        let scale = core::cmp::min(OUT_W / IN_W, OUT_H / IN_H) as f64;
        let half = 0.5f64;
        // Truncation of `x + 0.5` == round-half-up for non-negative values
        let scaled_w = ((IN_W as f64 * scale + half) as usize).clamp(1, OUT_W);
        let scaled_h = ((IN_H as f64 * scale + half) as usize).clamp(1, OUT_H);

        Self {
            pad_value,
            scaled_w,
            scaled_h,
            offset_x: (OUT_W - scaled_w) / 2,
            offset_y: (OUT_H - scaled_h) / 2,
        }
    }

    pub fn scaled_size(&self) -> (usize, usize) {
        (self.scaled_w, self.scaled_h)
    }

    pub fn offsets(&self) -> (usize, usize) {
        (self.offset_x, self.offset_y)
    }
}

impl<
        const IN_W: usize,
        const IN_H: usize,
        const C: usize,
        const OUT_W: usize,
        const OUT_H: usize,
        T: Float,
    > TransformOp<T> for Letterbox<IN_W, IN_H, C, OUT_W, OUT_H, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = IN_W * IN_H * C;
    const OUT_LEN: usize = OUT_W * OUT_H * C;

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        debug_assert!(out_index < OUT_W * OUT_H * C);

        let out_c = out_index % C;
        let pixel_index = out_index / C;
        let out_x = pixel_index % OUT_W;
        let out_y = pixel_index / OUT_W;

        if out_x < self.offset_x
            || out_x >= self.offset_x + self.scaled_w
            || out_y < self.offset_y
            || out_y >= self.offset_y + self.scaled_h
        {
            return Ok(self.pad_value);
        }

        let scaled_x = out_x - self.offset_x;
        let scaled_y = out_y - self.offset_y;

        let in_x = (scaled_x * IN_W) / self.scaled_w;
        let in_y = (scaled_y * IN_H) / self.scaled_h;
        let in_idx = (in_y * IN_W + in_x) * C + out_c;

        debug_assert!(in_idx < data.len());
        // SAFETY: `scaled_x < scaled_w` and `scaled_y < scaled_h` (checked
        // above), so `in_x < IN_W` and `in_y < IN_H`; `out_c < C`, therefore
        // `in_idx < IN_W * IN_H * C == in_len(..) == data.len()`
        Ok(unsafe { *data.get_unchecked(in_idx) })
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        if n > OUT_W * OUT_H * C || input.len() < IN_W * IN_H * C || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            // SAFETY: `out_index < n <= out.len()` (checked above); the read
            // is bounded by the index math in `compute`
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Letterbox")
    }
}

/// Rotate90 operation - rotates image 90 degrees clockwise
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Rotate90<const W: usize, const H: usize, const C: usize, T: Float> {
    marker: core::marker::PhantomData<T>,
}

impl<const W: usize, const H: usize, const C: usize, T: Float> Rotate90<W, H, C, T> {
    pub fn new() -> Self {
        Self {
            marker: core::marker::PhantomData,
        }
    }
}

impl<const W: usize, const H: usize, const C: usize, T: Float> TransformOp<T>
    for Rotate90<W, H, C, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = W * H * C;
    const OUT_LEN: usize = W * H * C;

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let out_c = out_index % C;
        let pixel_index = out_index / C;
        // The rotated image is `H` pixels wide and `W` pixels tall
        let out_x = pixel_index % H;
        let out_y = pixel_index / H;

        // Clockwise rotation: output(x, y) <- input(y, H - 1 - x)
        let in_x = out_y;
        let in_y = H - 1 - out_x;

        let in_idx = (in_y * W + in_x) * C + out_c;

        debug_assert!(in_idx < data.len());
        // SAFETY: caller contract: `out_index < out_len(..) == W * H * C`,
        // so `out_x < H` and `out_y < W`, hence `in_x < W` and `in_y < H`,
        // therefore `in_idx < W * H * C == in_len(..) == data.len()`
        Ok(unsafe { *data.get_unchecked(in_idx) })
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        if n > W * H * C || input.len() < W * H * C || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            // SAFETY: `out_index < n <= out.len()` (checked above); the read
            // is bounded by the index math in `compute`
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("Rotate90")
    }
}

/// Flip horizontal operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FlipHorizontal<const W: usize, const H: usize, const C: usize, T: Float> {
    marker: core::marker::PhantomData<T>,
}

impl<const W: usize, const H: usize, const C: usize, T: Float> TransformOp<T>
    for FlipHorizontal<W, H, C, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = W * H * C;
    const OUT_LEN: usize = W * H * C;

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let out_c = out_index % C;
        let pixel_index = out_index / C;
        let out_x = pixel_index % W;
        let out_y = pixel_index / W;

        let in_x = W - 1 - out_x;
        let in_y = out_y;

        let in_idx = (in_y * W + in_x) * C + out_c;

        debug_assert!(in_idx < data.len());
        // SAFETY: caller contract: `out_index < out_len(..) == W * H * C`,
        // so `out_x < W` (hence `in_x < W`) and `out_y < H`, therefore
        // `in_idx < W * H * C == in_len(..) == data.len()`
        Ok(unsafe { *data.get_unchecked(in_idx) })
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        if n > W * H * C || input.len() < W * H * C || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            // SAFETY: `out_index < n <= out.len()` (checked above); the read
            // is bounded by the index math in `compute`
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("FlipHorizontal")
    }
}

/// Flip vertical operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FlipVertical<const W: usize, const H: usize, const C: usize, T: Float> {
    marker: core::marker::PhantomData<T>,
}

impl<const W: usize, const H: usize, const C: usize, T: Float> TransformOp<T>
    for FlipVertical<W, H, C, T>
{
    type IndexRemapping = False;

    const IN_LEN: usize = W * H * C;
    const OUT_LEN: usize = W * H * C;

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let out_c = out_index % C;
        let pixel_index = out_index / C;
        let out_x = pixel_index % W;
        let out_y = pixel_index / W;

        let in_x = out_x;
        let in_y = H - 1 - out_y;

        let in_idx = (in_y * W + in_x) * C + out_c;

        debug_assert!(in_idx < data.len());
        // SAFETY: caller contract: `out_index < out_len(..) == W * H * C`,
        // so `out_x < W` and `out_y < H` (hence `in_y < H`), therefore
        // `in_idx < W * H * C == in_len(..) == data.len()`
        Ok(unsafe { *data.get_unchecked(in_idx) })
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // Cheap once-per-call guard so the loop below can be unchecked
        if n > W * H * C || input.len() < W * H * C || out.len() < n {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        for out_index in 0..n {
            // SAFETY: `out_index < n <= out.len()` (checked above); the read
            // is bounded by the index math in `compute`
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from("FlipVertical")
    }
}
