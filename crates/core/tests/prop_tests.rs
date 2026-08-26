use featurize_core::{
    errors::{ErrorKind, PropagateNan, ZeroOnNan},
    image::*,
    prelude::*,
};
use proptest::prelude::*;

/// Pipeline prop tests for input variety checking
///
/// NOTE: the NaN / infinity policy is a *pipeline-wide, compile-time*
/// choice now, so it is selected at construction:
///
/// * `Pipeline::new()` / `Pipeline::with_dynamic()` -> fail fast,
/// * `Pipeline::new_with::<F, ZeroOnNan>()` /
///   `Pipeline::with_dynamic_and::<F, ZeroOnNan>()` -> replace with zero,
/// * `..::<F, PropagateNan>()` -> IEEE 754 passthrough.

fn valid_f32() -> impl Strategy<Value = f32> {
    -1e6f32..1e6f32
}

fn safe_f32() -> impl Strategy<Value = f32> {
    -1e3f32..1e3f32
}

#[allow(dead_code)]
fn small_size() -> impl Strategy<Value = usize> {
    1usize..=100
}

proptest! {
    #[test]
    fn prop_element_op_preserves_length(
        input in prop::collection::vec(safe_f32(), 1..100),
        factor in 0.01f32..100.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        // Default policy is fail-fast
        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(factor))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), len);
    }

    #[test]
    fn prop_chained_element_ops_preserve_length(
        input in prop::collection::vec(safe_f32(), 1..100),
        factor1 in 0.1f32..10.0f32,
        factor2 in 0.1f32..10.0f32,
        add_val in -100.0f32..100.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor1))
            .apply_element(Div::new(factor2))
            .apply_element(Add::new(add_val))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), len);
    }

    #[test]
    fn prop_nan_handling_zero_replaces_nan(
        mut input in prop::collection::vec(valid_f32(), 1..100),
        nan_index in 0usize..100
    ) {
        if input.is_empty() { return Ok(()); }
        let len = input.len();
        let idx = nan_index % len;
        input[idx] = f32::NAN;

        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(out_buf[idx], 0.0);

        for i in 0..len {
            if i != idx {
                prop_assert_eq!(out_buf[i], input[i]);
            }
        }
    }

    #[test]
    fn prop_nan_handling_fail_detects_nan(
        mut input in prop::collection::vec(valid_f32(), 1..100),
        nan_index in 0usize..100
    ) {
        if input.is_empty() { return Ok(()); }
        let len = input.len();
        let idx = nan_index % len;
        input[idx] = f32::NAN;

        let mut out_buf = vec![0f32; len];

        // Default policy is fail-fast
        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_err());
        if let Err(e) = result {
            prop_assert!(matches!(e.kind(), ErrorKind::NaN));
        }
    }

    #[test]
    fn prop_nan_handling_propagate_passes_nan_through(
        mut input in prop::collection::vec(safe_f32(), 1..100),
        nan_index in 0usize..100
    ) {
        if input.is_empty() { return Ok(()); }
        let len = input.len();
        let idx = nan_index % len;
        input[idx] = f32::NAN;

        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, PropagateNan>()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert!(out_buf[idx].is_nan());

        for i in 0..len {
            if i != idx {
                prop_assert_eq!(out_buf[i], input[i] * 2.0);
            }
        }
    }

    #[test]
    fn prop_reverse_preserves_values(
        input in prop::collection::vec(safe_f32(), 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Reverse::<0>::new())
            .apply_transform(Reverse::<0>::new())
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());

        for i in 0..len {
            prop_assert_eq!(out_buf[i], input[i]);
        }
    }

    #[test]
    fn prop_multiply_associative(
        input in prop::collection::vec(safe_f32(), 1..100),
        a in 0.1f32..10.0f32,
        b in 0.1f32..10.0f32
    ) {
        let len = input.len();

        let mut out1 = vec![0f32; len];
        let mut pipe1 = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(a))
            .apply_element(Multiply::new(b))
            .build_dynamic(len);
        pipe1.execute(&input, &mut out1)?;

        let mut out2 = vec![0f32; len];
        let mut pipe2 = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(a*b))
            .build_dynamic(len);
        pipe2.execute(&input, &mut out2)?;

        for i in 0..len {
            prop_assert!(out1[i].is_finite());
            prop_assert!(out2[i].is_finite());

            let diff = (out1[i] - out2[i]).abs();
            // NOTE: The diff gets quite large
            prop_assert!(diff < 1e-4);
        }
    }

    #[test]
    fn prop_add_subtract_inverse(
        input in prop::collection::vec(safe_f32(), 1..100),
        value in -1000.0f32..1000.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Add::new(value))
            .apply_element(Subtract::new(value))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            // NOTE: The difference is quite large, 1e-3 - 1e-4
            println!("got: {}, wanted: {}, plain: {}", out_buf[i], input[i], (input[i] + value) - value * (1.0 + f32::MIN_POSITIVE));
            let diff = (out_buf[i] - input[i]).abs();
            prop_assert!(diff < 1e-5);
        }
    }

    #[test]
    fn prop_multiply_divide_inverse(
        input in prop::collection::vec(safe_f32(), 1..100),
        factor in 0.1f32..100.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor))
            .apply_element(Div::new(factor))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let diff = (out_buf[i] - input[i]).abs();
            let rel_diff = if input[i].abs() > 1e-6 {
                diff / input[i].abs()
            } else {
                diff
            };
            prop_assert!(rel_diff < 1e-3);
        }
    }

    #[test]
    fn prop_abs_non_negative(
        input in prop::collection::vec(safe_f32(), 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Abs::default())
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            prop_assert!(out_buf[i] >= 0.0);
        }
    }

    #[test]
    fn prop_clamp_respects_bounds(
        input in prop::collection::vec(safe_f32(), 1..100),
        min in -100.0f32..100.0f32,
        max in -100.0f32..100.0f32
    ) {
        let (min, max) = if min <= max { (min, max) } else { (max, min) };
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Clamp::new(min, max))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            prop_assert!(out_buf[i] >= min);
            prop_assert!(out_buf[i] <= max);
        }
    }

    #[test]
    fn prop_sqrt_square_inverse(
        input in prop::collection::vec(0.0f32..1000.0f32, 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Pow::new(2.0))
            .apply_element(Sqrt::default())
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let diff = (out_buf[i] - input[i]).abs();
            let rel_diff = if input[i] > 1e-6 {
                diff / input[i]
            } else {
                diff
            };
            prop_assert!(rel_diff < 1e-3);
        }
    }

    #[test]
    fn prop_output_buffer_too_small_fails(
        input in prop::collection::vec(safe_f32(), 10..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len / 2];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_err());
        if let Err(e) = result {
            prop_assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    #[test]
    fn prop_normalize_range(
        input in prop::collection::vec(safe_f32(), 1..100),
        mean in -100.0f32..100.0f32,
        std in 0.1f32..100.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Normalize::new(std, mean))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let expected = (input[i] - mean) / std;
            let diff = (out_buf[i] - expected).abs();
            prop_assert!(diff < 1e-5);
        }
    }

    #[test]
    fn prop_dynamic_varying_sizes(
        size1 in 1usize..50,
        size2 in 1usize..50,
        factor in 0.1f32..10.0f32
    ) {
        let max_size = size1.max(size2);

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor))
            .build_dynamic(max_size);

        let input1 = vec![1.0f32; size1];
        let mut out1 = vec![0f32; size1];
        let result1 = pipe.execute(&input1, &mut out1);
        prop_assert!(result1.is_ok());
        prop_assert_eq!(result1.unwrap(), size1);

        let input2 = vec![2.0f32; size2];
        let mut out2 = vec![0f32; size2];
        let result2 = pipe.execute(&input2, &mut out2);
        prop_assert!(result2.is_ok());
        prop_assert_eq!(result2.unwrap(), size2);
    }

    #[test]
    fn prop_infinity_handling_zero(
        mut input in prop::collection::vec(safe_f32(), 1..100),
        inf_index in 0usize..100
    ) {
        if input.is_empty() { return Ok(()); }
        let len = input.len();
        let idx = inf_index % len;
        input[idx] = f32::INFINITY;

        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(out_buf[idx], 0.0);
    }

    #[test]
    fn prop_very_small_values(
        input in prop::collection::vec(-1e-10f32..1e-10f32, 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());

        for val in &out_buf[..len] {
            prop_assert!(val.is_finite());
        }
    }

    #[test]
    fn prop_moderate_values_stay_finite(
        input in prop::collection::vec(-1e6f32..1e6f32, 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());

        for val in &out_buf[..len] {
            prop_assert!(val.is_finite());
        }
    }

    #[test]
    fn prop_fusion_equivalence(
        input in prop::collection::vec(safe_f32(), 1..100),
        factor1 in 0.1f32..10.0f32,
        factor2 in 0.1f32..10.0f32
    ) {
        let len = input.len();

        let mut out_fused = vec![0f32; len];
        let mut pipe_fused = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor1))
            .apply_element(Multiply::new(factor2))
            .build_dynamic(len);
        pipe_fused.execute(&input, &mut out_fused)?;

        let mut out_manual = vec![0f32; len];
        for i in 0..len {
            out_manual[i] = input[i] * factor1 * factor2;
        }

        for i in 0..len {
            let diff = (out_fused[i] - out_manual[i]).abs();
            let rel_diff = if out_manual[i].abs() > 1e-6 {
                diff / out_manual[i].abs()
            } else {
                diff
            };
            prop_assert!(rel_diff < 1e-5);
        }
    }

    #[test]
    fn prop_multiply_overflow_to_zero(
        input in prop::collection::vec(1e30f32..1e38f32, 1..100),
        factor in 1e5f32..1e10f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let expected = input[i] * factor;
            if expected.is_infinite() {
                prop_assert_eq!(out_buf[i], 0.0, "Expected zero for overflow at index {}", i);
            } else {
                prop_assert!(out_buf[i].is_finite());
                let diff = (out_buf[i] - expected).abs();
                let rel_diff = if expected.abs() > 1e-6 {
                    diff / expected.abs()
                } else {
                    diff
                };
                prop_assert!(rel_diff < 1e-4);
            }
        }
    }

    #[test]
    fn prop_add_overflow_to_zero(
        input in prop::collection::vec(1e37f32..3e38f32, 1..100),
        value in 1e37f32..3e38f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Add::new(value))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let expected = input[i] + value;
            if expected.is_infinite() {
                prop_assert_eq!(out_buf[i], 0.0, "Expected zero for overflow at index {}", i);
            } else {
                prop_assert!(out_buf[i].is_finite());
            }
        }
    }

    #[test]
    fn prop_pow_overflow_to_zero(
        input in prop::collection::vec(10.0f32..100.0f32, 1..100),
        exponent in 10.0f32..50.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Pow::new(exponent))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let expected = input[i].powf(exponent);
            if expected.is_infinite() {
                prop_assert_eq!(out_buf[i], 0.0, "Expected zero for overflow at index {}", i);
            } else {
                prop_assert!(out_buf[i].is_finite());
            }
        }
    }

    #[test]
    fn prop_chained_multiply_overflow_to_zero(
        input in prop::collection::vec(1e15f32..1e20f32, 1..100),
        factor1 in 1e10f32..1e15f32,
        factor2 in 1e10f32..1e15f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor1))
            .apply_element(Multiply::new(factor2))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let step1 = input[i] * factor1;
            let expected = if step1.is_infinite() {
                0.0
            } else {
                let step2 = step1 * factor2;
                if step2.is_infinite() { 0.0 } else { step2 }
            };

            if expected == 0.0 {
                prop_assert_eq!(out_buf[i], 0.0, "Expected zero at index {}", i);
            } else {
                let diff = (out_buf[i] - expected).abs();
                let rel_diff = diff / expected.abs();
                prop_assert!(rel_diff < 1e-4, "Mismatch at index {}: got {}, expected {}", i, out_buf[i], expected);
            }
        }
    }

    #[test]
    fn prop_normalize_overflow_to_zero(
        input in prop::collection::vec(1e30f32..1e37f32, 1..100),
        mean in -1e10f32..1e10f32,
        std in 1e-35f32..1e-25f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Normalize::new(std, mean))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let expected = (input[i] - mean) / std;
            if expected.is_infinite() {
                prop_assert_eq!(out_buf[i], 0.0, "Expected zero for overflow at index {}", i);
            } else {
                prop_assert!(out_buf[i].is_finite());
            }
        }
    }

    #[test]
    fn prop_div_by_small_overflow_to_zero(
        input in prop::collection::vec(1e30f32..1e38f32, 1..100),
        factor in 1e-38f32..1e-30f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Div::new(factor))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let expected = input[i] / factor;
            if expected.is_infinite() {
                prop_assert_eq!(out_buf[i], 0.0, "Expected zero for overflow at index {}", i);
            } else {
                prop_assert!(out_buf[i].is_finite());
            }
        }
    }

    #[test]
    fn prop_subtract_underflow_to_zero(
        input in prop::collection::vec(-1e37f32..-1e30f32, 1..100),
        value in 1e37f32..3e38f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Subtract::new(value))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let expected = input[i] - value;
            if expected.is_infinite() {
                prop_assert_eq!(out_buf[i], 0.0, "Expected zero for underflow at index {}", i);
            } else {
                prop_assert!(out_buf[i].is_finite());
            }
        }
    }

    #[test]
    fn prop_reverse_preserves_multiset(
        input in prop::collection::vec(safe_f32(), 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Reverse::<0>::new())
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        let mut input_sorted = input.clone();
        let mut output_sorted = out_buf[..len].to_vec();
        input_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        output_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for i in 0..len {
            prop_assert_eq!(input_sorted[i], output_sorted[i]);
        }
    }

    #[test]
    fn prop_transpose_preserves_multiset(
        input in prop::collection::vec(safe_f32(), 12..=12)
    ) {
        const ROWS: usize = 3;
        const COLS: usize = 4;
        let mut out_buf = vec![0f32; ROWS * COLS];

        let mut pipe = Pipeline::new()
            .apply_transform(Transpose::<ROWS, COLS>::new())
            .build();

        pipe.execute(&input, &mut out_buf)?;

        let mut input_sorted = input.clone();
        let mut output_sorted = out_buf.to_vec();
        input_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        output_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for i in 0..input_sorted.len() {
            prop_assert_eq!(input_sorted[i], output_sorted[i]);
        }
    }

    #[test]
    fn prop_index_remapping_is_bijective(
        input in prop::collection::vec(safe_f32(), 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Reverse::<0>::new())
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let expected = input[len - 1 - i];
            prop_assert_eq!(out_buf[i], expected);
        }
    }

    #[test]
    fn prop_clamp_is_idempotent(
        input in prop::collection::vec(safe_f32(), 1..100),
        min in -100.0f32..100.0f32,
        max in -100.0f32..100.0f32
    ) {
        let (min, max) = if min <= max { (min, max) } else { (max, min) };
        let len = input.len();
        let mut out_buf1 = vec![0f32; len];
        let mut out_buf2 = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Clamp::new(min, max))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf1)?;
        pipe.execute(&out_buf1, &mut out_buf2)?;

        for i in 0..len {
            prop_assert_eq!(out_buf1[i], out_buf2[i]);
        }
    }

    #[test]
    fn prop_abs_is_idempotent(
        input in prop::collection::vec(safe_f32(), 1..100)
    ) {
        let len = input.len();
        let mut out_buf1 = vec![0f32; len];
        let mut out_buf2 = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Abs::default())
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf1)?;
        pipe.execute(&out_buf1, &mut out_buf2)?;

        for i in 0..len {
            prop_assert_eq!(out_buf1[i], out_buf2[i]);
        }
    }

    #[test]
    fn prop_normalize_denormalize_inverse(
        input in prop::collection::vec(safe_f32(), 1..100),
        mean in -100.0f32..100.0f32,
        std in 0.1f32..100.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Normalize::new(std, mean))
            .apply_element(Multiply::new(std))
            .apply_element(Add::new(mean))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let diff = (out_buf[i] - input[i]).abs();
            let rel_diff = if input[i].abs() > 1e-6 {
                diff / input[i].abs()
            } else {
                diff
            };
            prop_assert!(rel_diff < 1e-3);
        }
    }

    #[test]
    fn prop_div_multiply_inverse(
        input in prop::collection::vec(safe_f32(), 1..100),
        factor in 0.1f32..100.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Div::new(factor))
            .apply_element(Multiply::new(factor))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let diff = (out_buf[i] - input[i]).abs();
            let rel_diff = if input[i].abs() > 1e-6 {
                diff / input[i].abs()
            } else {
                diff
            };
            prop_assert!(rel_diff < 1e-3);
        }
    }

    #[test]
    fn prop_pow_one_is_identity(
        input in prop::collection::vec(safe_f32(), 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Pow::new(1.0))
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            let diff = (out_buf[i] - input[i]).abs();
            let rel_diff = if input[i].abs() > 1e-6 {
                diff / input[i].abs()
            } else {
                diff
            };
            prop_assert!(rel_diff < 1e-5);
        }
    }

    #[test]
    fn prop_transform_fusion_equivalence(
        input in prop::collection::vec(safe_f32(), 12..=12)
    ) {
        const ROWS: usize = 3;
        const COLS: usize = 4;
        let mut out_fused = vec![0f32; ROWS * COLS];
        let mut out_separate = vec![0f32; ROWS * COLS];

        // Fused transform chain
        let mut pipe_fused = Pipeline::new()
            .apply_transform(Transpose::<ROWS, COLS>::new())
            .apply_transform_fusable(Reverse::<{ROWS * COLS}>::new())
            .build();

        pipe_fused.execute(&input, &mut out_fused)?;

        // Separate stages
        let mut temp = vec![0f32; ROWS * COLS];
        let mut pipe1 = Pipeline::new()
            .apply_transform(Transpose::<ROWS, COLS>::new())
            .build();
        pipe1.execute(&input, &mut temp)?;

        let mut pipe2 = Pipeline::new()
            .apply_transform(Reverse::<{ROWS * COLS}>::new())
            .build();
        pipe2.execute(&temp, &mut out_separate)?;

        for i in 0..out_fused.len() {
            prop_assert_eq!(out_fused[i], out_separate[i]);
        }
    }

    #[test]
    fn prop_static_dynamic_equivalence(
        input in prop::collection::vec(safe_f32(), 10..=10),
        factor in 0.1f32..10.0f32
    ) {
        const LEN: usize = 10;
        let mut out_static = vec![0f32; LEN];
        let mut out_dynamic = vec![0f32; LEN];

        let mut pipe_static = Pipeline::new_with::<f32, ZeroOnNan>()
            .apply_element::<_, LEN>(Multiply::new(factor))
            .build();

        let mut pipe_dynamic = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor))
            .build_dynamic(LEN);

        pipe_static.execute(&input, &mut out_static)?;
        pipe_dynamic.execute(&input, &mut out_dynamic)?;

        for i in 0..LEN {
            prop_assert_eq!(out_static[i], out_dynamic[i]);
        }
    }

    #[test]
    fn prop_element_fusion_order(
        input in prop::collection::vec(safe_f32(), 1..100),
        add_val in -100.0f32..100.0f32,
        mul_val in 0.1f32..10.0f32
    ) {
        let len = input.len();
        let mut out_add_mul = vec![0f32; len];
        let mut out_mul_add = vec![0f32; len];

        // Add then Multiply
        let mut pipe1 = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Add::new(add_val))
            .apply_element(Multiply::new(mul_val))
            .build_dynamic(len);
        pipe1.execute(&input, &mut out_add_mul)?;

        // Multiply then Add
        let mut pipe2 = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(mul_val))
            .apply_element(Add::new(add_val))
            .build_dynamic(len);
        pipe2.execute(&input, &mut out_mul_add)?;

        // They should differ (unless add_val is 0 or mul_val is 1)
        if add_val.abs() > 1e-6 && (mul_val - 1.0).abs() > 1e-6 {
            let mut found_difference = false;
            for i in 0..len {
                if (out_add_mul[i] - out_mul_add[i]).abs() > 1e-5 {
                    found_difference = true;
                    break;
                }
            }
            prop_assert!(found_difference, "Expected different results for non-commutative operations");
        }
    }

    #[test]
    fn prop_fusion_across_transform_boundary(
        input in prop::collection::vec(safe_f32(), 1..100),
        factor in 0.1f32..10.0f32
    ) {
        let len = input.len();
        let mut out_fused = vec![0f32; len];
        let mut out_separate = vec![0f32; len];

        // Transform + Element in one pipeline
        let mut pipe_fused = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_transform(Reverse::<0>::new())
            .apply_element(Multiply::new(factor))
            .build_dynamic(len);
        pipe_fused.execute(&input, &mut out_fused)?;

        // Separate pipelines
        let mut temp = vec![0f32; len];
        let mut pipe1 = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_transform(Reverse::<0>::new())
            .build_dynamic(len);
        pipe1.execute(&input, &mut temp)?;

        let mut pipe2 = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor))
            .build_dynamic(len);
        pipe2.execute(&temp, &mut out_separate)?;

        for i in 0..len {
            prop_assert_eq!(out_fused[i], out_separate[i]);
        }
    }

    #[test]
    fn prop_multiple_nans_all_replaced(
        mut input in prop::collection::vec(valid_f32(), 10..100),
        nan_indices in prop::collection::vec(0usize..100, 2..5)
    ) {
        let len = input.len();
        let mut actual_indices = vec![];
        for &idx in &nan_indices {
            let actual_idx = idx % len;
            input[actual_idx] = f32::NAN;
            actual_indices.push(actual_idx);
        }

        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());

        for &idx in &actual_indices {
            prop_assert_eq!(out_buf[idx], 0.0, "NaN at index {} not replaced", idx);
        }

        for i in 0..len {
            if !actual_indices.contains(&i) {
                prop_assert_eq!(out_buf[i], input[i], "Non-NaN value at index {} changed", i);
            }
        }
    }

    #[test]
    fn prop_neg_infinity_handling_zero(
        mut input in prop::collection::vec(safe_f32(), 1..100),
        inf_index in 0usize..100
    ) {
        if input.is_empty() { return Ok(()); }
        let len = input.len();
        let idx = inf_index % len;
        input[idx] = f32::NEG_INFINITY;

        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(out_buf[idx], 0.0);
    }

    #[test]
    fn prop_nan_in_transform_stage(
        mut input in prop::collection::vec(valid_f32(), 10..100),
        nan_index in 0usize..100
    ) {
        let len = input.len();
        let idx = nan_index % len;
        input[idx] = f32::NAN;

        let mut out_buf = vec![0f32; len];

        // NaN passes through Reverse, then gets caught by element op
        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_transform(Reverse::<0>::new())
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());

        // The NaN should be at the reversed position and zeroed
        let reversed_idx = len - 1 - idx;
        prop_assert_eq!(out_buf[reversed_idx], 0.0);
    }

    #[test]
    fn prop_nan_produced_mid_chain(
        mut input in prop::collection::vec(-10.0f32..10.0f32, 1..100),
        neg_index in 0usize..100
    ) {
        let len = input.len();
        let idx = neg_index % len;
        input[idx] = -1.0;

        let mut out_buf = vec![0f32; len];

        // Sqrt of negative produces NaN, which should be caught
        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Sqrt::new())
            .apply_element(Multiply::new(2.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(out_buf[idx], 0.0);
    }

    #[test]
    fn prop_subnormal_values(
        input in prop::collection::vec(
            prop_oneof![
                Just(f32::MIN_POSITIVE),
                Just(f32::MIN_POSITIVE / 2.0),
                Just(f32::MIN_POSITIVE / 10.0),
                -f32::MIN_POSITIVE..f32::MIN_POSITIVE
            ],
            1..100
        )
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        // Fail-fast policy: subnormals must not be mistaken for non-finite
        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok(), "Subnormal values should not trigger NaN handling");
    }

    #[test]
    fn prop_input_size_mismatch_fails(
        input in prop::collection::vec(safe_f32(), 1..50)
    ) {
        const EXPECTED_LEN: usize = 100;
        let len = input.len();

        if len <= EXPECTED_LEN {
            return Ok(());
        }

        let mut out_buf = vec![0f32; EXPECTED_LEN];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(EXPECTED_LEN);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_err());
        if let Err(e) = result {
            prop_assert!(matches!(e.kind(), ErrorKind::InvalidInputSize));
        }
    }

    #[test]
    fn prop_f64_pipeline_matches_f32(
        input_f32 in prop::collection::vec(safe_f32(), 1..100),
        factor in 0.1f32..10.0f32
    ) {
        let len = input_f32.len();
        let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

        let mut out_f32 = vec![0f32; len];
        let mut pipe_f32 = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor))
            .build_dynamic(len);
        pipe_f32.execute(&input_f32, &mut out_f32)?;

        let mut out_f64 = vec![0f64; len];
        let mut pipe_f64 = Pipeline::with_dynamic_and::<f64, ZeroOnNan>()
            .apply_element(Multiply::new(factor as f64))
            .build_dynamic(len);
        pipe_f64.execute(&input_f64, &mut out_f64)?;

        for i in 0..len {
            let diff = ((out_f64[i] as f32) - out_f32[i]).abs();
            let rel_diff = if out_f32[i].abs() > 1e-6 {
                diff / out_f32[i].abs()
            } else {
                diff
            };
            prop_assert!(rel_diff < 1e-5);
        }
    }

    #[test]
    fn prop_oversized_output_buffer(
        input in prop::collection::vec(safe_f32(), 10..100),
        extra_size in 1usize..50
    ) {
        let len = input.len();
        let buffer_size = len + extra_size;
        let mut out_buf = vec![99.0f32; buffer_size];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), len);

        for i in len..buffer_size {
            prop_assert_eq!(out_buf[i], 99.0, "Buffer written past index {}", i);
        }
    }

    #[test]
    fn prop_repeated_execute_deterministic(
        input in prop::collection::vec(safe_f32(), 1..100),
        factor in 0.1f32..10.0f32,
        iterations in 2usize..10
    ) {
        let len = input.len();
        let mut outputs = vec![];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(factor))
            .build_dynamic(len);

        for _ in 0..iterations {
            let mut out_buf = vec![0f32; len];
            pipe.execute(&input, &mut out_buf)?;
            outputs.push(out_buf);
        }

        for i in 1..iterations {
            for j in 0..len {
                prop_assert_eq!(outputs[0][j], outputs[i][j],
                    "Output differs between iterations at index {}", j);
            }
        }
    }

    #[test]
    fn prop_dynamic_input_larger_than_built_size_fails(
        small_size in 10usize..50,
        large_size in 51usize..100
    ) {
        let input = vec![1.0f32; large_size];
        let mut out_buf = vec![0f32; large_size];

        let mut pipe = Pipeline::with_dynamic_and::<f32, ZeroOnNan>()
            .apply_element(Multiply::new(1.0))
            .build_dynamic(small_size);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_err());
        if let Err(e) = result {
            prop_assert!(matches!(e.kind(), ErrorKind::InvalidOutputSize));
        }
    }

    #[test]
    fn prop_normalize_per_channel_hwc_chw_agree(
        input in prop::collection::vec(0.0f32..255.0f32, 12..=12)
    ) {
        const W: usize = 2;
        const H: usize = 2;
        const C: usize = 3;
        let mean = [100.0, 110.0, 120.0];
        let std = [50.0, 55.0, 60.0];

        // HWC layout
        let mut out_hwc = vec![0f32; W * H * C];
        let mut pipe_hwc = Pipeline::new_with::<f32, ZeroOnNan>()
            .apply_transform(NormalizePerChannel::<W, H, C, f32>::new(mean, std)
                .with_layout(ChannelLayout::Hwc))
            .build();
        pipe_hwc.execute(&input, &mut out_hwc)?;

        // CHW layout: convert input, normalize, convert back
        let mut temp_chw = vec![0f32; W * H * C];
        let mut pipe_to_chw = Pipeline::new()
            .apply_transform(HwcToChw::<W, H, C, f32>::new())
            .build();
        pipe_to_chw.execute(&input, &mut temp_chw)?;

        let mut normalized_chw = vec![0f32; W * H * C];
        let mut pipe_norm_chw = Pipeline::new_with::<f32, ZeroOnNan>()
                .apply_transform(NormalizePerChannel::<W, H, C, f32>::new(mean, std)
                .with_layout(ChannelLayout::Chw))
            .build();
        pipe_norm_chw.execute(&temp_chw, &mut normalized_chw)?;

        let mut out_chw_back = vec![0f32; W * H * C];
        let mut pipe_to_hwc = Pipeline::new()
            .apply_transform(ChwToHwc::<W, H, C, f32>::new())
            .build();
        pipe_to_hwc.execute(&normalized_chw, &mut out_chw_back)?;

        for i in 0..out_hwc.len() {
            let diff = (out_hwc[i] - out_chw_back[i]).abs();
            prop_assert!(diff < 1e-4, "Mismatch at index {}: hwc={}, chw={}", i, out_hwc[i], out_chw_back[i]);
        }
    }

    #[test]
    fn prop_grayscale_luminance_bounds(
        input in prop::collection::vec(0.0f32..255.0f32, 12..=12),
        invert in prop::bool::ANY
    ) {
        const W: usize = 2;
        const H: usize = 2;
        const C: usize = 3;
        let mut out_buf = vec![0f32; W * H];

        let grayscale = if invert {
            Grayscale::<W, H, C, f32>::new().with_inversion()
        } else {
            Grayscale::<W, H, C, f32>::new()
        };

        let mut pipe = Pipeline::new()
            .apply_transform(grayscale)
            .build();

        pipe.execute(&input, &mut out_buf)?;

        for &val in &out_buf {
            prop_assert!(val >= 0.0 && val <= 255.0,
                "Grayscale output {} out of bounds [0, 255]", val);
        }
    }
}
