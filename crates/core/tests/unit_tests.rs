#![allow(clippy::needless_range_loop)]

use featurize_core::errors::NanHandler;
use featurize_core::prelude::*;

#[allow(clippy::empty_line_after_doc_comments)]
/// Unit tests for the core library functionality
///
/// Contains tests for basic lib operations, image operations
/// and other simple pipeline properties

#[cfg(test)]
struct Identity;

#[cfg(test)]
impl<T: Float> TransformOp<T> for Identity {
    const OUT_LEN: usize = 0;
    const INTERNAL_IS_VALID: bool = true;
    const IN_LEN: usize = 0;

    type IndexRemapping = featurize_core::traits::False;

    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        default_len
    }

    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        default_len
    }

    /// Pure copy: the pipeline NaN policy `N` is irrelevant here
    #[inline(always)]
    fn execute<'o, N: NanHandler>(
        &self,
        out: &'o mut [T],
        input: &[T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        debug_assert!(input.len() >= n && out.len() >= n);
        // SAFETY: the stage guarantees `input.len() == in_len(..) == n` and
        // `out.len() == n` before dispatching here; the buffers are distinct
        // allocations, so the ranges cannot overlap
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), n);
        }
        Ok(out)
    }
}

#[cfg(test)]
struct Doubler;

#[cfg(test)]
impl<T: Float> TransformOp<T> for Doubler {
    const IN_LEN: usize = 0;
    const OUT_LEN: usize = 0;
    const INTERNAL_IS_VALID: bool = true;

    type IndexRemapping = featurize_core::traits::False;

    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        default_len * 2
    }

    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        default_len
    }

    #[inline(always)]
    fn execute<'o, N: NanHandler>(
        &self,
        out: &'o mut [T],
        input: &[T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        // `n` is always the size of the output buffer
        debug_assert!(input.len() * 2 >= n && out.len() >= n);
        // Duplicate each element
        for i in 0..n {
            out[i] = input[i / 2];
        }
        Ok(out)
    }
}

#[cfg(test)]
struct ErrorOp;

#[cfg(test)]
impl<T: Float> TransformOp<T> for ErrorOp {
    const OUT_LEN: usize = 0;
    const INTERNAL_IS_VALID: bool = true;
    const IN_LEN: usize = 0;

    type IndexRemapping = featurize_core::traits::False;

    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        default_len
    }

    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        default_len
    }

    #[inline(always)]
    fn compute<N: NanHandler>(&self, _data: &[T], _index: usize) -> Result<T, PipeError> {
        Err(PipeError::new(ErrorKind::InvalidInputSize))
    }

    #[inline(always)]
    fn execute<'o, N: NanHandler>(
        &self,
        _out: &'o mut [T],
        _input: &[T],
        _n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        Err(PipeError::new(ErrorKind::InvalidInputSize))
    }
}

#[cfg(test)]
mod unit_tests {
    use featurize_core::errors::{PropagateNan, ZeroOnNan};
    use featurize_core::prelude::*;

    // NOTE: empty static inputs get a compile error and are
    // therefore not testable via unit tests
    //
    // #[test]
    // fn test_static_empty_input() {}

    #[test]
    fn test_static_input_size_mismatch() {
        let mut out_buf = vec![0f32; 1];
        let in_buf = vec![];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 1>(Multiply::new(3.0))
            .build();

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }

    #[test]
    fn test_static_single_element_op() {
        let mut out_buf = vec![0f32; 10];
        let in_buf = vec![2.0f32; 10];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 10>(Multiply::new(3.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 10);
        assert_eq!(out_buf[0], 6.0);
        assert_eq!(out_buf[9], 6.0);
    }

    #[test]
    fn test_static_single_transform_op() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut pipe = Pipeline::new().apply_transform(Truncate::<10, 5>).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[4], 5.0);
    }

    #[test]
    fn test_static_pipeline_reuse() {
        let mut out_buf = vec![0f32; 10];
        let in_buf = vec![2.0f32; 10];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 10>(Multiply::new(3.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 10);
        assert_eq!(out_buf[0], 6.0);
        assert_eq!(out_buf[9], 6.0);

        let mut out_buf2 = vec![0f32; 10];
        let in_buf2 = vec![3.0f32; 10];

        let n = pipe.execute(&in_buf2, &mut out_buf2).unwrap();

        assert_eq!(n, 10);
        assert_eq!(out_buf2[0], 9.0);
        assert_eq!(out_buf2[9], 9.0);
    }

    #[test]
    fn test_static_element_chain() {
        let mut out_buf = vec![0f32; 4];
        let in_buf = vec![10.0, 20.0, 30.0, 40.0];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 4>(Div::new(2.0))
            .apply_element(Add::new(5.0))
            .apply_element(Multiply::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf[0], 20.0);
        assert_eq!(out_buf[1], 30.0);
    }

    #[test]
    fn test_dynamic_single_element_op() {
        let mut out_buf = vec![0f32; 100];
        let in_buf = vec![5.0f32; 100];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(100);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 100);
        assert_eq!(out_buf[0], 10.0);
    }

    #[test]
    fn test_dynamic_transform_op_static() {
        let mut out_buf = vec![0f32; 1024];
        let in_buf = vec![1f32; 1024];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 1024>(Div::new(1f32))
            .apply_transform(crate::Identity)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(out_buf[0], in_buf[0]);
        assert_eq!(n, out_buf.len());
    }

    #[test]
    fn test_dynamic_transform_op_dynamic() {
        let mut out_buf = vec![0f32; 1024];
        let in_buf = vec![1f32; 1024];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(crate::Identity)
            .apply_element(Div::new(1f32))
            .build_dynamic(4096);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(out_buf[0], in_buf[0]);
        assert_eq!(n, out_buf.len());
    }

    #[test]
    fn test_dynamic_varying_input_sizes() {
        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(1000);

        let small_in = vec![1.0f32; 10];
        let mut small_out = vec![0f32; 10];
        let n = pipe.execute(&small_in, &mut small_out).unwrap();
        assert_eq!(n, 10);
        assert_eq!(small_out[0], 2.0);

        let large_in = vec![3.0f32; 500];
        let mut large_out = vec![0f32; 500];
        let n = pipe.execute(&large_in, &mut large_out).unwrap();
        assert_eq!(n, 500);
        assert_eq!(large_out[0], 6.0);
    }

    #[test]
    fn test_transform_chain_dimension_change() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

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
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Truncate::<10, 8>)
            .apply_element(Div::new(2.0))
            .apply_transform(Truncate::<8, 5>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[4], 5.0);
    }

    #[test]
    fn test_fusable_transform_chain() {
        let mut out_buf = vec![0f32; 8];
        let in_buf: Vec<f32> = (0..10).map(|i| i as f32).collect();

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
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 5>(Div::new(10.0))
            .apply_element(Add::new(1.0))
            .apply_element(Multiply::new(10.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 20.0);
    }

    #[test]
    fn test_transform_element_fusion() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Truncate::<10, 5>)
            .apply_element(Multiply::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[4], 10.0);
    }

    #[test]
    fn test_long_fusion_chain() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![100.0, 200.0, 300.0];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 3>(Div::new(100.0))
            .apply_element(Add::new(1.0))
            .apply_element(Multiply::new(2.0))
            .apply_element(Subtract::new(1.0))
            .apply_element(Pow::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 9.0);
    }

    #[test]
    fn test_large_buffer() {
        const SIZE: usize = 10000;
        let mut out_buf = vec![0f32; SIZE];
        let in_buf = vec![1.0f32; SIZE];

        let mut pipe = Pipeline::new()
            .apply_element::<_, SIZE>(Add::new(1.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, SIZE);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[SIZE - 1], 2.0);
    }

    #[test]
    fn test_output_buffer_too_small() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0f32; 10];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 10>(Multiply::new(2.0))
            .build();

        let result = pipe.execute(&in_buf, &mut out_buf);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    // Ops

    #[test]
    fn test_pad_operation() {
        let mut out_buf = vec![0f32; 10];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_transform(Pad::<f32, 5, 10>::new(0.0))
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
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new().apply_transform(Reverse::<5>).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 5.0);
        assert_eq!(out_buf[4], 1.0);
    }

    #[test]
    fn test_transpose_square_matrix() {
        let mut out_buf = vec![0f32; 9];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

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
        let mut out_buf = vec![0f32; 6];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut pipe = Pipeline::new().apply_transform(Transpose::<2, 3>).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[1], 4.0);
        assert_eq!(out_buf[2], 2.0);
        assert_eq!(out_buf[3], 5.0);
    }

    // Err

    #[test]
    fn test_nan_handling_fail() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![1.0, f32::NAN, 3.0];

        // `Pipeline::new()` defaults to the fail-fast policy
        let mut pipe = Pipeline::new()
            .apply_element::<_, 3>(Multiply::new(2.0))
            .build();

        assert_eq!(pipe.nan_handling(), NanHandling::Fail);

        let result = pipe.execute(&in_buf, &mut out_buf);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::NaN));
        }
    }

    #[test]
    fn test_nan_handling_zero() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![1.0, f32::NAN, 3.0];

        // Pipeline-wide, compile-time policy
        let mut pipe = Pipeline::new_with::<f32, ZeroOnNan>()
            .apply_element::<_, 3>(Multiply::new(2.0))
            .build();

        assert_eq!(pipe.nan_handling(), NanHandling::Zero);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[1], 0.0);
        assert_eq!(out_buf[2], 6.0);
    }

    #[test]
    fn test_nan_handling_propagate() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![1.0, f32::NAN, 3.0];

        let mut pipe = Pipeline::new_with::<f32, PropagateNan>()
            .apply_element::<_, 3>(Multiply::new(2.0))
            .build();

        assert_eq!(pipe.nan_handling(), NanHandling::Propagate);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 2.0);
        assert!(out_buf[1].is_nan());
        assert_eq!(out_buf[2], 6.0);
    }

    #[test]
    fn test_nan_handling_selected_via_builder() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![1.0, f32::NAN, 3.0];

        // Equivalent to `Pipeline::new_with::<f32, ZeroOnNan>()`
        let mut pipe = Pipeline::new::<f32>()
            .nan_handling::<ZeroOnNan>()
            .apply_element::<_, 3>(Multiply::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[1], 0.0);
    }

    #[test]
    fn test_infinity_handling() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![1.0, f32::INFINITY, 3.0];

        let mut pipe = Pipeline::new_with::<f32, ZeroOnNan>()
            .apply_element::<_, 3>(Multiply::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[1], 0.0);
        assert_eq!(out_buf[2], 6.0);
    }

    // Edge cases

    #[test]
    fn test_division_by_very_small_number() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![1.0, 2.0, 3.0];

        let mut pipe = Pipeline::new_with::<f32, ZeroOnNan>()
            .apply_element::<_, 3>(Div::new(1e-10))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert!(out_buf[0].is_finite() || out_buf[0] == 0.0);
    }

    #[test]
    fn test_sqrt_of_negative() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![-1.0, 4.0, -9.0];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 3>(Sqrt::default())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf);
        assert!(n.is_err())
    }

    #[test]
    fn test_clamp_with_inverted_bounds() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 5>(Clamp::new(3.0, 3.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        for i in 0..5 {
            assert_eq!(out_buf[i], 3.0);
        }
    }

    #[test]
    fn test_power_with_zero_exponent() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new().apply_element::<_, 5>(Pow::new(0.0)).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        for i in 0..5 {
            assert_eq!(out_buf[i], 1.0);
        }
    }

    #[test]
    fn test_dynamic_input_adapts_to_size() {
        let mut out_buf = vec![0f32; 100];
        let in_buf = vec![1.0f32; 50];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(100);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 50);
        assert_eq!(out_buf[0], 2.0);
    }

    #[test]
    fn test_dynamic_output_buffer_too_small() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0f32; 100];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(100);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    #[test]
    fn test_dynamic_transform_dimension_mismatch() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0f32; 8];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Truncate::<10, 5>)
            .build_dynamic(10);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }

    #[test]
    fn test_dynamic_empty_input() {
        let mut out_buf = vec![0f32; 10];
        let in_buf = vec![];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(10);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_dynamic_zero_output_buffer() {
        let mut out_buf = vec![];
        let in_buf = vec![1.0f32; 10];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(10);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    #[test]
    fn test_dynamic_reverse_with_wrong_size() {
        let mut out_buf = vec![0f32; 10];
        let in_buf = vec![1.0f32; 8];

        let mut pipe = Pipeline::with_dynamic()
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
        let mut out_buf = vec![0f32; 6];
        let in_buf = vec![1.0f32; 8];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Transpose::<2, 3>)
            .build_dynamic(6);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }

    // NOTE: division by zero is a compile error
    //
    // #[test]
    // fn test_dynamic_infinity_in_division() {}

    #[test]
    fn test_dynamic_very_large_pad() {
        const SMALL: usize = 10;
        const HUGE: usize = 10000;

        let mut out_buf = vec![0f32; HUGE];
        let in_buf = vec![1.0f32; SMALL];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Pad::<_, SMALL, HUGE>::new(-1.0))
            .build_dynamic(SMALL);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, HUGE);
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[SMALL], -1.0);
        assert_eq!(out_buf[HUGE - 1], -1.0);
    }

    #[test]
    fn test_dynamic_exceeds_max_expected_input() {
        let mut out_buf = vec![0f32; 200];
        let in_buf = vec![1.0f32; 200];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(100);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    // NOTE: offsets are checked on compile time
    //
    // #[test]
    // fn test_crop_invalid_offsets_error() {
    //     let mut out_buf = vec![0f32; 4];
    //     let in_buf = vec![1.0f32; 16];
    //
    //     let mut pipe = Pipeline::new()
    //         .apply_transform(Crop::<4, 4, 1, 2, 2, _>::new(3, 3))
    //         .build();
    //
    //     let result = pipe.execute(&in_buf, &mut out_buf);
    //     assert!(result.is_err());
    //     if let Err(e) = result {
    //         assert!(matches!(
    //             e.kind(),
    //             ErrorKind::InvalidInputSize
    //         ));
    //     }
    // }

    // Image operations

    #[test]
    fn test_rotate90_rectangular() {
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_buf = vec![0f32; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(Rotate90::<3, 2, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
    }

    #[test]
    fn test_rotate90_twice_rectangular() {
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_buf = vec![0f32; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(Rotate90::<3, 2, 1, _>::new())
            .apply_transform_fusable(Rotate90::<2, 3, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_horizontal_rectangular() {
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_buf = vec![0f32; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(FlipHorizontal::<3, 2, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_flip_horizontal_twice_rectangular() {
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_buf = vec![0f32; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(FlipHorizontal::<3, 2, 1, _>::new())
            .apply_transform_fusable(FlipHorizontal::<3, 2, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_flip_vertical_rectangular() {
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_buf = vec![0f32; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(FlipVertical::<3, 2, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_flip_vertical_twice_rectangular() {
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_buf = vec![0f32; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(FlipVertical::<3, 2, 1, _>::new())
            .apply_transform_fusable(FlipVertical::<3, 2, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_hwc_to_chw() {
        // 2x2 image, 3 channels (HWC interleaved):
        // pixel(0,0)=[1,2,3] pixel(1,0)=[4,5,6]
        // pixel(0,1)=[7,8,9] pixel(1,1)=[10,11,12]
        let in_buf = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut out_buf = vec![0f32; 12];

        let mut pipe = Pipeline::new()
            .apply_transform(HwcToChw::<2, 2, 3, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 12);
        assert_eq!(
            out_buf,
            vec![1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]
        );
    }

    #[test]
    fn test_chw_to_hwc_roundtrip() {
        let in_buf: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut out_buf = vec![0f32; 24];

        // CHW -> HWC must be the identity
        let mut pipe = Pipeline::new()
            .apply_transform(HwcToChw::<4, 2, 3, _>::new())
            .apply_transform_fusable(ChwToHwc::<4, 2, 3, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 24);
        assert_eq!(out_buf, in_buf);
    }

    #[test]
    fn test_normalize_per_channel_hwc() {
        // 1x2 image, 3 channels (HWC)
        let in_buf = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut out_buf = vec![0f32; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(NormalizePerChannel::<2, 1, 3, _>::new(
                [10.0, 20.0, 30.0],
                [2.0, 5.0, 10.0],
            ))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![0.0, 0.0, 0.0, 15.0, 6.0, 3.0]);
    }

    #[test]
    fn test_normalize_per_channel_chw() {
        // 1x2 image, 3 channels (CHW planar): [c0 c0 | c1 c1 | c2 c2]
        let in_buf = vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0];
        let mut out_buf = vec![0f32; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(
                NormalizePerChannel::<2, 1, 3, _>::new([10.0, 20.0, 30.0], [2.0, 5.0, 10.0])
                    .with_layout(ChannelLayout::Chw),
            )
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![0.0, 15.0, 0.0, 6.0, 0.0, 3.0]);
    }

    #[test]
    fn test_scale2d_nearest_identity() {
        // Same input/output size must be the identity
        let in_buf = vec![1.0, 2.0, 3.0, 4.0];
        let mut out_buf = vec![0f32; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2D::<2, 2, 1, 2, 2, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf, in_buf);
    }

    #[test]
    fn test_scale2d_nearest_downscale() {
        // 4x4 single-channel image with values 0..16 -> 2x2
        // Nearest neighbor samples (0,0), (2,0), (0,2), (2,2)
        let in_buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out_buf = vec![0f32; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2D::<4, 4, 1, 2, 2, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf, vec![0.0, 2.0, 8.0, 10.0]);
    }

    #[test]
    fn test_scale2d_nearest_upscale() {
        // 2x1 -> 4x1: nearest neighbor duplicates samples (no interpolation),
        // in contrast to `Scale2DBilinear`
        let in_buf = vec![0.0, 10.0];
        let mut out_buf = vec![0f32; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2D::<2, 1, 1, 4, 1, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf, vec![0.0, 0.0, 10.0, 10.0]);
    }

    #[test]
    fn test_scale2d_nearest_multichannel() {
        // 2x1 image, 3 channels (HWC): pixel0 = [1,2,3], pixel1 = [4,5,6]
        // Upscaled to 4x1: each source pixel is duplicated, channels intact
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_buf = vec![0f32; 12];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2D::<2, 1, 3, 4, 1, 3, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 12);
        assert_eq!(
            out_buf,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_scale2d_nearest_channel_reduction() {
        // 2x1 image, 3 channels -> 2x1 image, 1 channel: keeps channel 0
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_buf = vec![0f32; 2];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2D::<2, 1, 3, 2, 1, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 2);
        assert_eq!(out_buf, vec![1.0, 4.0]);
    }

    #[test]
    fn test_scale2d_nearest_in_chain() {
        // Element ops before and after the (non-fusable) nearest-neighbor scale
        let in_buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out_buf = vec![0f32; 4];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 16>(Add::new(1.0))
            .apply_transform(Scale2D::<4, 4, 1, 2, 2, 1, _>::new())
            .apply_element(Multiply::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        // (0, 2, 8, 10) + 1 = (1, 3, 9, 11), then * 2
        assert_eq!(out_buf, vec![2.0, 6.0, 18.0, 22.0]);
    }

    #[test]
    fn test_scale2d_nearest_output_buffer_too_small() {
        let in_buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out_buf = vec![0f32; 3];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2D::<4, 4, 1, 2, 2, 1, _>::new())
            .build();

        let result = pipe.execute(&in_buf, &mut out_buf);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    #[test]
    fn test_scale2d_nearest_dynamic() {
        let in_buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out_buf = vec![0f32; 4];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Scale2D::<4, 4, 1, 2, 2, 1, _>::new())
            .build_dynamic(16);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf, vec![0.0, 2.0, 8.0, 10.0]);
    }

    #[test]
    fn test_scale2d_bilinear_identity() {
        // Same input/output size must be the identity
        let in_buf = vec![1.0, 2.0, 3.0, 4.0];
        let mut out_buf = vec![0f32; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2DBilinear::<2, 2, 1, 2, 2, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf, in_buf);
    }

    #[test]
    fn test_scale2d_bilinear_upscale() {
        // 2x1 -> 4x1: interpolated values must lie between neighbors and be
        // monotonic
        let in_buf = vec![0.0, 10.0];
        let mut out_buf = vec![0f32; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2DBilinear::<2, 1, 1, 4, 1, 1, _>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf[0], 0.0);
        assert_eq!(out_buf[3], 10.0);
        assert!(out_buf[1] > 0.0 && out_buf[1] < 10.0);
        assert!(out_buf[2] > out_buf[1] && out_buf[2] < 10.0);
    }

    #[test]
    fn test_center_crop() {
        // 4x4 single-channel image with values 0..16
        let in_buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out_buf = vec![0f32; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Crop::<4, 4, 1, 2, 2, _>::centered())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        // Center 2x2 of the 4x4 grid: rows 1-2, cols 1-2
        assert_eq!(out_buf, vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_letterbox_wide_input() {
        // 4x2 input into 4x4 output: scale = 1, pad 1 row on top and bottom
        let in_buf = vec![1.0; 8];
        let mut out_buf = vec![0f32; 16];

        let letterbox = Letterbox::<4, 2, 1, 4, 4, _>::new(0.5);
        assert_eq!(letterbox.scaled_size(), (4, 2));
        assert_eq!(letterbox.offsets(), (0, 1));

        let mut pipe = Pipeline::new().apply_transform(letterbox).build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 16);
        // Top
        assert_eq!(&out_buf[0..4], &[0.5, 0.5, 0.5, 0.5]);
        assert_eq!(&out_buf[4..8], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&out_buf[8..12], &[1.0, 1.0, 1.0, 1.0]);
        // Bottom
        assert_eq!(&out_buf[12..16], &[0.5, 0.5, 0.5, 0.5]);
    }

    // f64

    #[test]
    fn test_f64_grayscale() {
        let in_buf = vec![
            100.0f64, 150.0, 200.0, 50.0, 100.0, 150.0, 200.0, 100.0, 50.0, 0.0, 128.0, 255.0,
        ];
        let mut out_buf = vec![0f64; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Grayscale::<2, 2, 3, f64>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        // L = 0.299*R + 0.587*G + 0.114*B
        let expected_0 = 0.299 * 100.0 + 0.587 * 150.0 + 0.114 * 200.0;
        assert!((out_buf[0] - expected_0).abs() < 0.01);
    }

    #[test]
    fn test_f64_grayscale_with_inversion() {
        // Test that inversion (255 - v) works correctly with f64
        let in_buf = vec![100.0f64, 100.0, 100.0, 200.0, 200.0, 200.0];
        let mut out_buf = vec![0f64; 2];

        let mut pipe = Pipeline::new()
            .apply_transform(Grayscale::<2, 1, 3, f64>::new().with_inversion())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 2);
        // With inversion, each channel is (255 - value) before luminance
        // For [100, 100, 100], inverted = [155, 155, 155]
        let expected_0 = 0.299 * 155.0 + 0.587 * 155.0 + 0.114 * 155.0;
        assert!((out_buf[0] - expected_0).abs() < 0.01);
    }

    #[test]
    fn test_f64_normalize_per_channel() {
        let in_buf = vec![10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut out_buf = vec![0f64; 6];

        let mut pipe = Pipeline::new()
            .apply_transform(NormalizePerChannel::<2, 1, 3, f64>::new(
                [10.0, 20.0, 30.0],
                [2.0, 5.0, 10.0],
            ))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 6);
        assert_eq!(out_buf, vec![0.0, 0.0, 0.0, 15.0, 6.0, 3.0]);
    }

    #[test]
    fn test_f64_scale2d_nearest() {
        let in_buf = vec![0.0f64, 10.0];
        let mut out_buf = vec![0f64; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2D::<2, 1, 1, 4, 1, 1, f64>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        // Nearest neighbor: no interpolated (intermediate) values
        assert_eq!(out_buf, vec![0.0, 0.0, 10.0, 10.0]);
    }

    #[test]
    fn test_f64_scale2d_bilinear() {
        let in_buf = vec![0.0f64, 10.0];
        let mut out_buf = vec![0f64; 4];

        let mut pipe = Pipeline::new()
            .apply_transform(Scale2DBilinear::<2, 1, 1, 4, 1, 1, f64>::new())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf[0], 0.0);
        assert_eq!(out_buf[3], 10.0);
        // Intermediate values should be interpolated
        assert!(out_buf[1] > 0.0 && out_buf[1] < 10.0);
        assert!(out_buf[2] > out_buf[1] && out_buf[2] < 10.0);
    }

    #[test]
    fn test_f64_complex_pipeline() {
        const SIZE: usize = 12;
        let mut out_buf = vec![0f64; SIZE];
        let in_buf: Vec<f64> = (0..SIZE).map(|i| i as f64 * 10.0).collect();

        let mut pipe = Pipeline::new()
            .apply_element::<_, SIZE>(Div::new(10.0f64))
            .apply_element(Add::new(1.0f64))
            .apply_transform(Reverse::<SIZE>)
            .apply_element(Multiply::new(2.0f64))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, SIZE);
        // First element after reverse should be last input element
        // (11 * 10) / 10 + 1 = 12, then * 2 = 24
        assert_eq!(out_buf[0], 24.0);
        // Last element after reverse should be first input element
        // (0 * 10) / 10 + 1 = 1, then * 2 = 2
        assert_eq!(out_buf[SIZE - 1], 2.0);
    }

    // Dynamic pipeline coverage

    #[test]
    fn test_dynamic_build_with_zero_size() {
        // build_dynamic(0) should handle empty pipelines
        let mut out_buf = vec![0f32; 0];
        let in_buf = vec![];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(0);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_dynamic_nan_policy_selection() {
        let mut out_buf = vec![0f32; 4];
        let in_buf = vec![1.0, f32::NAN, 3.0, f32::NEG_INFINITY];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(8);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 4);
        assert_eq!(out_buf, vec![2.0, 0.0, 6.0, 0.0]);
    }

    // Buffer safety / aliasing tests

    #[test]
    fn test_output_buffer_exact_size() {
        // out_buf.len() == n exactly (not larger) succeeds for static pipeline
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 5>(Multiply::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(out_buf.len(), 5);
        assert_eq!(out_buf, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_output_buffer_exact_size_dynamic() {
        // out_buf.len() == n exactly (not larger) succeeds for dynamic pipeline
        let mut out_buf = vec![0f32; 7];
        let in_buf = vec![1.0f32; 7];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(3.0))
            .build_dynamic(10);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 7);
        assert_eq!(out_buf.len(), 7);
        assert_eq!(out_buf[0], 3.0);
    }

    #[test]
    fn test_oversized_output_untouched_tail() {
        let mut out_buf = vec![99.0f32; 10];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 5>(Multiply::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 5);
        // First 5 elements should be modified
        assert_eq!(&out_buf[0..5], &[2.0, 4.0, 6.0, 8.0, 10.0]);
        // Remaining elements should be untouched
        assert_eq!(&out_buf[5..10], &[99.0, 99.0, 99.0, 99.0, 99.0]);
    }

    // Custom TransformOp contract

    // NOTE: compile time check
    //
    // #[test]
    // fn test_custom_op_identity_static() {
    //     // Custom Identity op in static pipeline
    //     let mut out_buf = vec![0f32; 10];
    //     let in_buf: Vec<f32> = (0..10).map(|i| i as f32).collect();
    //
    //     let mut pipe = Pipeline::new()
    //         .apply_transform::<crate::Identity>(crate::Identity)
    //         .build();
    //
    //     let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
    //     assert_eq!(n, 10);
    //     assert_eq!(out_buf, in_buf);
    // }

    #[test]
    fn test_custom_op_identity_dynamic() {
        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(crate::Identity)
            .build_dynamic(100);

        let in_buf1 = vec![1.0f32; 20];
        let mut out_buf1 = vec![0f32; 20];
        let n = pipe.execute(&in_buf1, &mut out_buf1).unwrap();
        assert_eq!(n, 20);
        assert_eq!(out_buf1, in_buf1);

        let in_buf2 = vec![2.0f32; 50];
        let mut out_buf2 = vec![0f32; 50];
        let n = pipe.execute(&in_buf2, &mut out_buf2).unwrap();
        assert_eq!(n, 50);
        assert_eq!(out_buf2, in_buf2);
    }

    #[test]
    fn test_custom_op_in_pipeline_chain() {
        let mut out_buf = vec![0f32; 10];
        let in_buf: Vec<f32> = (0..10).map(|i| i as f32).collect();

        let mut pipe = Pipeline::new()
            .apply_element::<_, 10>(Multiply::new(2.0))
            .apply_transform(crate::Identity)
            .apply_element(Add::new(1.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 10);
        // (i * 2) + 1
        assert_eq!(out_buf[0], 1.0);
        assert_eq!(out_buf[5], 11.0);
        assert_eq!(out_buf[9], 19.0);
    }

    // NOTE: compile time check
    //
    // #[test]
    // fn test_custom_op_in_len_out_len_contract_static() {
    //     // Custom op whose out_len differs from in_len (doubling) in static pipeline
    //     let mut out_buf = vec![0f32; 10];
    //     let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    //
    //     let mut pipe = Pipeline::new()
    //         .apply_transform::<crate::Doubler>(crate::Doubler)
    //         .build();
    //
    //     let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
    //     assert_eq!(n, 10);
    //     assert_eq!(out_buf, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]);
    // }

    #[test]
    fn test_custom_op_in_len_out_len_contract_dynamic() {
        // Custom op whose out_len differs from in_len (doubling) in dynamic pipeline
        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(crate::Doubler)
            .build_dynamic(10);

        let in_buf1 = vec![1.0f32; 5];
        let mut out_buf1 = vec![0f32; 10];
        let n = pipe.execute(&in_buf1, &mut out_buf1).unwrap();
        assert_eq!(n, 10);
        for i in 0..10 {
            assert_eq!(out_buf1[i], 1.0);
        }

        let in_buf2 = vec![2.0f32; 3];
        let mut out_buf2 = vec![0f32; 6];
        let n = pipe.execute(&in_buf2, &mut out_buf2).unwrap();
        assert_eq!(n, 6);
        for i in 0..6 {
            assert_eq!(out_buf2[i], 2.0);
        }
    }

    #[test]
    fn test_custom_op_reports_error() {
        let mut out_buf = vec![0f32; 10];
        let in_buf = vec![1.0f32; 10];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform::<crate::ErrorOp>(crate::ErrorOp)
            .build_dynamic(10);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }

    #[test]
    fn test_custom_op_error_in_chain() {
        let mut out_buf = vec![0f32; 10];
        let in_buf = vec![1.0f32; 10];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 10>(Multiply::new(2.0))
            .apply_transform(crate::ErrorOp)
            .apply_element(Add::new(1.0))
            .build();

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }
}
