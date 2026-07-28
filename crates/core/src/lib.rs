#![no_std]

extern crate alloc;

pub mod errors;
pub mod image;
pub mod ops;
pub mod pipeline;
pub mod prelude;
pub mod traits;
pub mod tensors;

pub const DYNAMIC_SIZE: usize = 0;

pub(crate) const fn _const_max_usize(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

#[cfg(test)]
struct Identity;

#[cfg(test)]
impl<T: crate::traits::Float> crate::traits::TransformOp<T> for Identity {
    const OUT_LEN: usize = 0;
    const INTERNAL_IS_VALID: bool = true;
    const IN_LEN: usize = 0;

    type IndexRemapping = crate::traits::False;

    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        default_len
    }

    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        default_len
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], errors::PipeError> {
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), n);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::{errors::ErrorKind, ops::*, pipeline::Pipeline, Identity};

    #[test]
    fn test_static_single_element_op() {
        let mut out_buf = alloc::vec![0f32; 10];
        let in_buf = alloc::vec![2.0f32; 10];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 10>(Multiply {
                factor: 3.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 10);
        assert_eq!(out_buf[0], 6.0);
        assert_eq!(out_buf[9], 6.0);
    }

    #[test]
    fn test_static_single_transform_op() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut pipe = Pipeline::new().apply_transform(Truncate::<10, 5>).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[4], 5.0);
    }

    #[test]
    fn test_static_element_chain() {
        let mut out_buf = alloc::vec![0f32; 4];
        let in_buf = alloc::vec![10.0, 20.0, 30.0, 40.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 4>(Div {
                factor: 2.0,
                ..Default::default()
            })
            .apply_point(Add {
                value: 5.0,
                ..Default::default()
            })
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf[0], 20.0);
        assert_eq!(out_buf[1], 30.0);
    }

    #[test]
    fn test_dynamic_single_element_op() {
        let mut out_buf = alloc::vec![0f32; 100];
        let in_buf = alloc::vec![5.0f32; 100];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(100);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 100);
        assert_eq!(out_buf[0], 10.0);
    }

    #[test]
    fn test_dynamic_transform_op_static() {
        let mut out_buf = alloc::vec![0f32; 1024];
        let in_buf = alloc::vec![1f32; 1024];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 1024>(Div {
                factor: 1f32,
                ..Default::default()
            })
            .apply_transform(Identity)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(out_buf[0], in_buf[0]);
        assert_eq!(n, out_buf.len());
    }

    #[test]
    fn test_dynamic_transform_op_dynamic() {
        let mut out_buf = alloc::vec![0f32; 1024];
        let in_buf = alloc::vec![1f32; 1024];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Identity)
            .apply_point(Div {
                factor: 1f32,
                ..Default::default()
            })
            .build_dynamic(4096);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(out_buf[0], in_buf[0]);
        assert_eq!(n, out_buf.len());
    }

    #[test]
    fn test_dynamic_varying_input_sizes() {
        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(1000);

        let small_in = alloc::vec![1.0f32; 10];
        let mut small_out = alloc::vec![0f32; 10];
        let n = pipe.execute(&small_in, &mut small_out).unwrap();
        assert_eq!(n, 10);
        assert_eq!(small_out[0], 2.0);

        let large_in = alloc::vec![3.0f32; 500];
        let mut large_out = alloc::vec![0f32; 500];
        let n = pipe.execute(&large_in, &mut large_out).unwrap();
        assert_eq!(n, 500);
        assert_eq!(large_out[0], 6.0);
    }

    #[test]
    fn test_transform_chain_dimension_change() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Truncate::<10, 8>)
            .apply_transform(Truncate::<8, 5>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[4], 5.0);
    }

    #[test]
    fn test_transform_element_transform_chain() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Truncate::<10, 8>)
            .apply_point(Div {
                factor: 2.0,
                ..Default::default()
            })
            .apply_transform(Truncate::<8, 5>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[4], 5.0);
    }

    #[test]
    fn test_fusable_transform_chain() {
        let mut out_buf = alloc::vec![0f32; 8];
        let in_buf: alloc::vec::Vec<f32> = (0..10).map(|i| i as f32).collect();

        let mut pipe = Pipeline::new()
            .apply_transform(Reverse::<10>)
            .apply_transform_fusable(Truncate::<10, 8>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 8);
        assert_eq!(out_buf[0], 9.0);
        assert_eq!(out_buf[7], 2.0);
    }

    #[test]
    fn test_element_element_fusion() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 5>(Div {
                factor: 10.0,
                ..Default::default()
            })
            .apply_point(Add {
                value: 1.0,
                ..Default::default()
            })
            .apply_point(Multiply {
                factor: 10.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 20.0);
    }

    #[test]
    fn test_transform_element_fusion() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Truncate::<10, 5>)
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[4], 10.0);
    }

    #[test]
    fn test_long_fusion_chain() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![100.0, 200.0, 300.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 3>(Div {
                factor: 100.0,
                ..Default::default()
            })
            .apply_point(Add {
                value: 1.0,
                ..Default::default()
            })
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .apply_point(Subtract {
                value: 1.0,
                ..Default::default()
            })
            .apply_point(Pow {
                exponent: 2.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 9.0);
    }

    #[test]
    fn test_single_element_input() {
        let mut out_buf = alloc::vec![0f32; 1];
        let in_buf = alloc::vec![42.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 1>(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 1);
        assert_eq!(out_buf[0], 84.0);
    }

    #[test]
    fn test_large_buffer() {
        const SIZE: usize = 10000;
        let mut out_buf = alloc::vec![0f32; SIZE];
        let in_buf = alloc::vec![1.0f32; SIZE];

        let mut pipe = Pipeline::new()
            .apply_point::<_, SIZE>(Add {
                value: 1.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, SIZE);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[SIZE - 1], 2.0);
    }

    #[test]
    fn test_pad_operation() {
        let mut out_buf = alloc::vec![0f32; 10];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Pad::<f32, 5, 10> { pad_value: 0.0 })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 10);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[4], 5.0);
        assert_eq!(out_buf[5], 0.0);
        assert_eq!(out_buf[9], 0.0);
    }

    #[test]
    fn test_reverse_operation() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new().apply_transform(Reverse::<5>).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 5.0);
        assert_eq!(out_buf[4], 1.0);
    }

    #[test]
    fn test_transpose_square_matrix() {
        let mut out_buf = alloc::vec![0f32; 9];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,];

        let mut pipe = Pipeline::new().apply_transform(Transpose::<3, 3>).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 9);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[1], 4.0);
        assert_eq!(out_buf[2], 7.0);
        assert_eq!(out_buf[3], 2.0);
    }

    #[test]
    fn test_transpose_rectangular_matrix() {
        let mut out_buf = alloc::vec![0f32; 6];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0,];

        let mut pipe = Pipeline::new().apply_transform(Transpose::<2, 3>).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[1], 4.0);
        assert_eq!(out_buf[2], 2.0);
        assert_eq!(out_buf[3], 5.0);
    }

    #[test]
    fn test_output_buffer_too_small() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0f32; 10];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 10>(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build();

        let result = pipe.execute(&in_buf, &mut out_buf);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    #[test]
    fn test_nan_handling_fail() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![1.0, f32::NAN, 3.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 3>(Multiply {
                factor: 2.0,
                nan_handling: crate::errors::NanHandling::Fail,
            })
            .build();

        let result = pipe.execute(&in_buf, &mut out_buf);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::NaN));
        }
    }

    #[test]
    fn test_nan_handling_zero() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![1.0, f32::NAN, 3.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 3>(Multiply {
                factor: 2.0,
                nan_handling: crate::errors::NanHandling::Zero,
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[1], 0.0);
        assert_eq!(out_buf[2], 6.0);
    }

    #[test]
    fn test_infinity_handling() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![1.0, f32::INFINITY, 3.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 3>(Multiply {
                factor: 2.0,
                nan_handling: crate::errors::NanHandling::Zero,
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[1], 0.0);
        assert_eq!(out_buf[2], 6.0);
    }

    #[test]
    fn test_image_normalization_pipeline() {
        const SIZE: usize = 8 * 8 * 3;
        let mut out_buf = alloc::vec![0f32; SIZE];
        let mut in_buf = alloc::vec![0f32; SIZE];

        for i in 0..SIZE {
            in_buf[i] = (i % 256) as f32;
        }

        let mut pipe = Pipeline::new()
            .apply_point::<_, SIZE>(Div {
                factor: 255.0,
                ..Default::default()
            })
            .apply_point(Subtract {
                value: 0.5,
                ..Default::default()
            })
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, SIZE);
        assert!((out_buf[0] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_audio_preprocessing_pipeline() {
        const ORIGINAL: usize = 16000;
        const PADDED: usize = 16384;
        const TRUNCATED: usize = 8000;

        let mut out_buf = alloc::vec![0f32; TRUNCATED];
        let in_buf = alloc::vec![0.5f32; ORIGINAL];

        let mut pipe = Pipeline::new()
            .apply_transform(Pad::<f32, ORIGINAL, PADDED> { pad_value: 0.0 })
            .apply_point(Normalize {
                mean: 0.0,
                std: 0.5,
                ..Default::default()
            })
            .apply_transform(Truncate::<PADDED, TRUNCATED>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, TRUNCATED);
        assert_eq!(out_buf[0], 1.0);
    }

    #[test]
    fn test_batch_normalization_scenario() {
        const BATCH_SIZE: usize = 32;
        const FEATURES: usize = 128;
        const SIZE: usize = BATCH_SIZE * FEATURES;

        let mut out_buf = alloc::vec![0f32; SIZE];
        let in_buf = alloc::vec![10.0f32; SIZE];

        let mut pipe = Pipeline::new()
            .apply_point::<_, SIZE>(Normalize {
                mean: 5.0,
                std: 2.0,
                ..Default::default()
            })
            .apply_point(Clamp {
                min: -3.0,
                max: 3.0,
                nan_handling: Default::default(),
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, SIZE);
        assert_eq!(out_buf[0], 2.5);
    }

    #[test]
    fn test_feature_extraction_pipeline() {
        const INPUT_SIZE: usize = 1000;
        const REDUCED_SIZE: usize = 100;

        let mut out_buf = alloc::vec![0f32; REDUCED_SIZE];
        let in_buf: alloc::vec::Vec<f32> = (0..INPUT_SIZE).map(|i| i as f32).collect();

        let mut pipe = Pipeline::new()
            .apply_point::<_, INPUT_SIZE>(Normalize {
                mean: 500.0,
                std: 100.0,
                ..Default::default()
            })
            .apply_transform(Truncate::<INPUT_SIZE, REDUCED_SIZE>)
            .apply_point(Abs::default())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, REDUCED_SIZE);
        assert_eq!(out_buf[0], 5.0);
    }

    #[test]
    fn test_alternating_transform_element_chain() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Truncate::<10, 8>)
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .apply_transform(Truncate::<8, 5>)
            .apply_point(Add {
                value: 1.0,
                ..Default::default()
            })
            .apply_transform(Truncate::<5, 3>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 3.0);
    }

    #[test]
    fn test_reverse_then_operations() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Reverse::<5>)
            .apply_point(Multiply {
                factor: 10.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 50.0);
        assert_eq!(out_buf[4], 10.0);
    }

    #[test]
    fn test_pad_then_truncate() {
        let mut out_buf = alloc::vec![0f32; 8];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Pad::<f32, 5, 10> { pad_value: 0.0 })
            .apply_transform(Truncate::<10, 8>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 8);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[4], 5.0);
        assert_eq!(out_buf[5], 0.0);
        assert_eq!(out_buf[7], 0.0);
    }

    #[test]
    fn test_division_by_very_small_number() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![1.0, 2.0, 3.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 3>(Div {
                factor: 1e-10,
                nan_handling: crate::errors::NanHandling::Zero,
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert!(out_buf[0].is_finite() || out_buf[0] == 0.0);
    }

    #[test]
    fn test_sqrt_of_negative() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![-1.0, 4.0, -9.0];

        let mut pipe = Pipeline::new().apply_point::<_, 3>(Sqrt::default()).build();

        let n = pipe.execute(&in_buf, &mut out_buf);
        assert!(n.is_err())
    }

    #[test]
    fn test_clamp_with_inverted_bounds() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 5>(Clamp {
                min: 3.0,
                max: 3.0,
                nan_handling: Default::default(),
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        for i in 0..5 {
            assert_eq!(out_buf[i], 3.0);
        }
    }

    #[test]
    fn test_power_with_zero_exponent() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_point::<_, 5>(Pow {
                exponent: 0.0,
                ..Default::default()
            })
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        for i in 0..5 {
            assert_eq!(out_buf[i], 1.0);
        }
    }

    #[test]
    fn test_dynamic_input_adapts_to_size() {
        let mut out_buf = alloc::vec![0f32; 100];
        let in_buf = alloc::vec![1.0f32; 50];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(100);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 50);
        assert_eq!(out_buf[0], 2.0);
    }

    #[test]
    fn test_dynamic_output_buffer_too_small() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0f32; 100];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(100);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    #[test]
    fn test_dynamic_transform_dimension_mismatch() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0f32; 8];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Truncate::<10, 5>)
            .build_dynamic(10);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }

    #[test]
    fn test_dynamic_chained_transform_correct_sizes() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![1.0f32; 10];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Truncate::<10, 8>)
            .apply_transform(Truncate::<8, 5>)
            .apply_transform(Truncate::<5, 3>)
            .build_dynamic(10);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn test_dynamic_pad_then_process_large() {
        const ORIGINAL: usize = 100;
        const PADDED: usize = 1000;

        let mut out_buf = alloc::vec![0f32; PADDED];
        let in_buf = alloc::vec![1.0f32; ORIGINAL];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Pad::<f32, ORIGINAL, PADDED> { pad_value: 0.0 })
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(ORIGINAL);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, PADDED);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[ORIGINAL], 0.0);
    }

    #[test]
    fn test_dynamic_empty_input() {
        let mut out_buf = alloc::vec![0f32; 10];
        let in_buf = alloc::vec![];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(10);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_dynamic_zero_output_buffer() {
        let mut out_buf = alloc::vec![];
        let in_buf = alloc::vec![1.0f32; 10];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(10);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    #[test]
    fn test_dynamic_transform_with_intermediate_buffer() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0f32; 10];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Pad::<_, 10, 20> { pad_value: 0.0 })
            .apply_transform(Truncate::<20, 5>)
            .build_dynamic(10);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn test_dynamic_multiple_dimension_changes() {
        let mut out_buf = alloc::vec![0f32; 2];
        let in_buf = alloc::vec![1.0f32; 5];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Pad::<_, 5, 10> { pad_value: 0.0 })
            .apply_transform(Truncate::<10, 7>)
            .apply_transform(Truncate::<7, 4>)
            .apply_transform(Truncate::<4, 2>)
            .build_dynamic(5);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[1], 1.0);
    }

    #[test]
    fn test_dynamic_reverse_with_wrong_size() {
        let mut out_buf = alloc::vec![0f32; 10];
        let in_buf = alloc::vec![1.0f32; 8];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Reverse::<10>)
            .build_dynamic(10);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }

    #[test]
    fn test_dynamic_transpose_with_wrong_dimensions() {
        let mut out_buf = alloc::vec![0f32; 6];
        let in_buf = alloc::vec![1.0f32; 8];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Transpose::<2, 3>)
            .build_dynamic(6);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }

    #[test]
    fn test_dynamic_nan_propagation_through_chain() {
        let mut out_buf = alloc::vec![0f32; 5];
        let in_buf = alloc::vec![1.0, 2.0, f32::NAN, 4.0, 5.0];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                nan_handling: crate::errors::NanHandling::Fail,
            })
            .apply_point(Add {
                value: 1.0,
                nan_handling: crate::errors::NanHandling::Fail,
            })
            .build_dynamic(5);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::NaN));
        }
    }

    #[test]
    fn test_dynamic_infinity_in_division() {
        let mut out_buf = alloc::vec![0f32; 3];
        let in_buf = alloc::vec![1.0, 0.0, 3.0];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Div {
                factor: 0.0,
                nan_handling: crate::errors::NanHandling::Fail,
            })
            .build_dynamic(3);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::NaN));
        }
    }

    #[test]
    fn test_dynamic_buffer_reuse_correctness() {
        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(10);

        let in_buf1 = alloc::vec![1.0f32; 10];
        let mut out_buf1 = alloc::vec![0f32; 10];
        let n1 = pipe.execute(&in_buf1, &mut out_buf1).unwrap();
        assert_eq!(n1, 10);
        assert_eq!(out_buf1[0], 2.0);

        let in_buf2 = alloc::vec![5.0f32; 10];
        let mut out_buf2 = alloc::vec![0f32; 10];
        let n2 = pipe.execute(&in_buf2, &mut out_buf2).unwrap();
        assert_eq!(n2, 10);
        assert_eq!(out_buf2[0], 10.0);
    }

    #[test]
    fn test_dynamic_very_large_pad() {
        const SMALL: usize = 10;
        const HUGE: usize = 10000;

        let mut out_buf = alloc::vec![0f32; HUGE];
        let in_buf = alloc::vec![1.0f32; SMALL];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_transform(Pad::<_, SMALL, HUGE> { pad_value: -1.0 })
            .build_dynamic(SMALL);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, HUGE);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[SMALL], -1.0);
        assert_eq!(out_buf[HUGE - 1], -1.0);
    }

    #[test]
    fn test_dynamic_exceeds_max_expected_input() {
        let mut out_buf = alloc::vec![0f32; 200];
        let in_buf = alloc::vec![1.0f32; 200];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply {
                factor: 2.0,
                ..Default::default()
            })
            .build_dynamic(100);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }
}
