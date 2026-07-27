use featurize_core::prelude::*;
use proptest::prelude::*;

fn valid_f32() -> impl Strategy<Value = f32> {
    -1e6f32..1e6f32
}

fn safe_f32() -> impl Strategy<Value = f32> {
    -1e3f32..1e3f32
}

fn any_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        prop::num::f32::NORMAL,
        Just(f32::NAN),
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
        Just(0.0),
        Just(-0.0),
    ]
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: factor1, nan_handling: NanHandling::Zero })
            .apply_point(Div { factor: factor2, nan_handling: NanHandling::Zero })
            .apply_point(Add { value: add_val, nan_handling: NanHandling::Zero })
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), len);
    }

    #[test]
    fn prop_nan_handling_zero_replaces_nan(
        mut input in prop::collection::vec(any_f32(), 1..100),
        nan_index in 0usize..100
    ) {
        if input.is_empty() { return Ok(()); }
        let len = input.len();
        let idx = nan_index % len;
        input[idx] = f32::NAN;

        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: 1.0, nan_handling: NanHandling::Zero })
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_ok());
        prop_assert_eq!(out_buf[idx], 0.0);
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: 1.0, nan_handling: NanHandling::Fail })
            .build_dynamic(len);

        let result = pipe.execute(&input, &mut out_buf);
        prop_assert!(result.is_err());
        if let Err(e) = result {
            prop_assert!(matches!(e.kind(), ErrorKind::NaN));
        }
    }

    #[test]
    fn prop_reverse_preserves_values(
        input in prop::collection::vec(safe_f32(), 1..100)
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: 1.0, nan_handling: NanHandling::Zero })
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
        let mut pipe1 = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: a, nan_handling: NanHandling::Zero })
            .apply_point(Multiply { factor: b, nan_handling: NanHandling::Zero })
            .build_dynamic(len);
        pipe1.execute(&input, &mut out1)?;

        let mut out2 = vec![0f32; len];
        let mut pipe2 = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: a * b, nan_handling: NanHandling::Zero })
            .build_dynamic(len);
        pipe2.execute(&input, &mut out2)?;

        for i in 0..len {
            prop_assert!(out1[i].is_finite());
            prop_assert!(out2[i].is_finite());

            let diff = (out1[i] - out2[i]).abs();
            let rel_diff = if out1[i].abs() > 1e-6 {
                diff / out1[i].abs()
            } else {
                diff
            };
            prop_assert!(rel_diff < 1e-4);
        }
    }

    #[test]
    fn prop_add_subtract_inverse(
        input in prop::collection::vec(safe_f32(), 1..100),
        value in -1000.0f32..1000.0f32
    ) {
        let len = input.len();
        let mut out_buf = vec![0f32; len];

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Add { value, nan_handling: NanHandling::Zero })
            .apply_point(Subtract { value, nan_handling: NanHandling::Zero })
            .build_dynamic(len);

        pipe.execute(&input, &mut out_buf)?;

        for i in 0..len {
            // The difference is quite large, 1e-3 - 1e-4
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor, nan_handling: NanHandling::Zero })
            .apply_point(Div { factor, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Abs::default())
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Clamp { min, max, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Pow { exponent: 2.0, nan_handling: NanHandling::Zero })
            .apply_point(Sqrt::default())
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: 1.0, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Normalize { mean, std, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: 1.0, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: 1.0, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: 2.0, nan_handling: NanHandling::Zero })
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
        let mut pipe_fused = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: factor1, nan_handling: NanHandling::Zero })
            .apply_point(Multiply { factor: factor2, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Add { value, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Pow { exponent, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Multiply { factor: factor1, nan_handling: NanHandling::Zero })
            .apply_point(Multiply { factor: factor2, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Normalize { mean, std, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Div { factor, nan_handling: NanHandling::Zero })
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

        let mut pipe = Pipeline::new_with_dynamic()
            .apply_point(Subtract { value, nan_handling: NanHandling::Zero })
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
}
