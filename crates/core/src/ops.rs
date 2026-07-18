use crate::traits::{ElementOp, False, TransformOp, True, IsTrue};

#[allow(unused_imports)]
use num_traits::Float as _;

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

/// Multiplication operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Multiply {
    pub factor: f32,
}

impl ElementOp for Multiply {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data * self.factor
    }
}

/// Addition operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Add {
    pub value: f32,
}

impl ElementOp for Add {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data + self.value
    }
}

/// Subtraction operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Subtract {
    pub value: f32,
}

impl ElementOp for Subtract {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data - self.value
    }
}

/// Clamp operation (point-wise)
/// Clamps values between min and max
#[derive(Debug, Clone)]
pub struct Clamp {
    pub min: f32,
    pub max: f32,
}

impl ElementOp for Clamp {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data.clamp(self.min, self.max)
    }
}

/// Absolute value operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Abs;

impl ElementOp for Abs {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data.abs()
    }
}

/// Power operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Pow {
    pub exponent: f32,
}

impl ElementOp for Pow {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data.powf(self.exponent)
    }
}

/// Square root operation (point-wise)
#[derive(Debug, Clone, Default)]
pub struct Sqrt;

impl ElementOp for Sqrt {
    #[inline(always)]
    fn compute(&self, data: f32) -> f32 {
        data.sqrt()
    }
}

/// Truncate operation - reduces the length of the data vector
#[derive(Debug, Clone, Copy)]
pub struct Truncate<const NEW_LEN: usize>;

impl<const NEW_LEN: usize> TransformOp for Truncate<NEW_LEN> {
    type IndexRemapping = True;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        // Truncate is identity mapping for valid indices
        out_index
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        data[out_index]
    }

    #[inline(always)]
    fn execute<'i, 'o>(&self, out: &'o mut [f32], _: &'i [f32], __: usize) -> &'o mut [f32] {
        &mut out[..NEW_LEN]
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        NEW_LEN
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        NEW_LEN
    }
}

/// Transpose operation for 2D matrices stored in row-major order
#[derive(Debug, Clone, Copy)]
pub struct Transpose<const ROWS: usize, const COLS: usize>;

impl<const ROWS: usize, const COLS: usize> TransformOp for Transpose<ROWS, COLS> {
    type IndexRemapping = True;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        // Transpose: output[i,j] = input[j,i]
        let out_row = out_index / ROWS;
        let out_col = out_index % ROWS;
        out_col * COLS + out_row
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let in_index = self.map_index(out_index);

        if in_index < COLS * ROWS && in_index < data.len() {
            data[in_index]
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32] {
        for (out_index, out_pixel) in out[0..n].iter_mut().enumerate() {
            *out_pixel = self.compute(input, out_index);
        }
        out
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        ROWS * COLS
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        ROWS * COLS
    }
}

/// Reshape operation - changes the logical shape without moving data
#[derive(Debug, Clone, Copy)]
pub struct Reshape<const NEW_SHAPE: usize>;

impl<const NEW_SHAPE: usize> TransformOp for Reshape<NEW_SHAPE> {
    type IndexRemapping = True;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        // Reshape is identity mapping
        out_index
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        if out_index < data.len() {
            data[out_index]
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32] {
        let copy_len = n.min(input.len());
        out[..copy_len].copy_from_slice(&input[..copy_len]);
        out
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        NEW_SHAPE
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        NEW_SHAPE
    }
}

/// Pad operation - adds padding to the data
#[derive(Debug, Clone, Copy)]
pub struct Pad<const ORIGINAL_LEN: usize, const PADDED_LEN: usize> {
    pub pad_value: f32,
}

impl<const ORIGINAL_LEN: usize, const PADDED_LEN: usize> TransformOp
    for Pad<ORIGINAL_LEN, PADDED_LEN>
{
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        if out_index < ORIGINAL_LEN && out_index < data.len() {
            data[out_index]
        } else {
            self.pad_value
        }
    }

    #[inline(always)]
    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32] {
        let copy_len = ORIGINAL_LEN.min(input.len()).min(n);
        out[..copy_len].copy_from_slice(&input[..copy_len]);

        for i in copy_len..n {
            out[i] = self.pad_value;
        }
        out
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        PADDED_LEN
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        PADDED_LEN
    }
}

/// Reverse operation - reverses the order of elements
#[derive(Debug, Clone, Copy)]
pub struct Reverse<const LEN: usize>;

impl<const LEN: usize> TransformOp for Reverse<LEN> {
    type IndexRemapping = True;

    #[inline(always)]
    fn map_index(&self, out_index: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        // Reverse: output[i] = input[LEN - 1 - i]
        LEN - 1 - out_index
    }

    #[inline(always)]
    fn compute(&self, data: &[f32], out_index: usize) -> f32 {
        let in_index = self.map_index(out_index);
        if in_index < data.len() {
            data[in_index]
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], n: usize) -> &'o mut [f32] {
        for (out_index, out_pixel) in out[0..n].iter_mut().enumerate() {
            *out_pixel = self.compute(input, out_index);
        }
        out
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        LEN
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        LEN
    }
}
