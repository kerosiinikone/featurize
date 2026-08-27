#![allow(clippy::needless_range_loop)]

#[allow(clippy::empty_line_after_doc_comments)]
/// Pipeline integration tests for core lib functionality
///
/// Contains multi-property pipeline tests for basic lib operations,
/// image operations and more complex pipeline scenarios

#[cfg(test)]
mod unit_tests {
    use featurize_core::prelude::*;

    #[test]
    fn test_image_normalization_pipeline() {
        const SIZE: usize = 8 * 8 * 3;
        let mut out_buf = vec![0f32; SIZE];
        let mut in_buf = vec![0f32; SIZE];

        for i in 0..SIZE {
            in_buf[i] = (i % 256) as f32;
        }

        // RGBA values; normalize, sample point operations
        let mut pipe = Pipeline::new()
            .apply_element::<_, SIZE>(Div::new(255.0))
            .apply_element(Subtract::new(0.5))
            .apply_element(Multiply::new(2.0))
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

        let mut out_buf = vec![0f32; TRUNCATED];
        let in_buf = vec![0.5f32; ORIGINAL];

        // Audio; pad zeros to default length, normalize and truncate
        let mut pipe = Pipeline::new()
            .apply_transform(Pad::<f32, ORIGINAL, PADDED>::new(0.0))
            .apply_element(Normalize::new(0.5, 0.0))
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

        let mut out_buf = vec![0f32; SIZE];
        let in_buf = vec![10.0f32; SIZE];

        // Normalize and clamp point values
        let mut pipe = Pipeline::new()
            .apply_element::<_, SIZE>(Normalize::new(2.0, 5.0))
            .apply_element(Clamp::new(-3.0, 3.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, SIZE);
        assert_eq!(out_buf[0], 2.5);
    }

    #[test]
    fn test_feature_extraction_pipeline() {
        const INPUT_SIZE: usize = 1000;
        const REDUCED_SIZE: usize = 100;

        let mut out_buf = vec![0f32; REDUCED_SIZE];
        let in_buf: Vec<f32> = (0..INPUT_SIZE).map(|i| i as f32).collect();

        // Extract a portion of the input dataset
        let mut pipe = Pipeline::new()
            .apply_element::<_, INPUT_SIZE>(Normalize::new(100.0, 500.0))
            .apply_transform(Truncate::<INPUT_SIZE, REDUCED_SIZE>)
            .apply_element(Abs::default())
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, REDUCED_SIZE);
        assert_eq!(out_buf[0], 5.0);
    }

    #[test]
    fn test_alternating_transform_element_chain() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Multistage; truncate several times, apply points in between
        let mut pipe = Pipeline::new()
            .apply_transform(Truncate::<10, 8>)
            .apply_element(Multiply::new(2.0))
            .apply_transform(Truncate::<8, 5>)
            .apply_element(Add::new(1.0))
            .apply_transform(Truncate::<5, 3>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 3);
        assert_eq!(out_buf[0], 3.0);
    }

    #[test]
    fn test_reverse_then_operations() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Single stage; transform + point
        let mut pipe = Pipeline::new()
            .apply_transform(Reverse::<5>)
            .apply_element(Multiply::new(10.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 50.0);
        assert_eq!(out_buf[4], 10.0);
    }

    #[test]
    fn test_dynamic_chained_transform_correct_sizes() {
        let mut out_buf = vec![0f32; 3];
        let in_buf = vec![1.0f32; 10];

        // Multistage; 3 x transform
        let mut pipe = Pipeline::with_dynamic()
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

        let mut out_buf = vec![0f32; PADDED];
        let in_buf = vec![1.0f32; ORIGINAL];

        // Dynamic single stage; pad, apply point
        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Pad::<_, ORIGINAL, PADDED>::new(0.0))
            .apply_element(Multiply::new(2.0))
            .build_dynamic(ORIGINAL);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, PADDED);
        assert_eq!(out_buf[0], 2.0);
        assert_eq!(out_buf[ORIGINAL], 0.0);
    }

    #[test]
    fn test_dynamic_transform_with_intermediate_buffer() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0f32; 10];

        // Multistage; transfrom then transform
        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Pad::<_, 10, 20>::new(0.0))
            .apply_transform(Truncate::<20, 5>)
            .build_dynamic(10);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn test_dynamic_multiple_dimension_changes() {
        let mut out_buf = vec![0f32; 2];
        let in_buf = vec![1.0f32; 5];

        // Multistage; 4 x transform (dim change)
        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Pad::<_, 5, 10>::new(0.0))
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
    fn test_dynamic_nan_propagation_through_chain() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0, 2.0, f32::NAN, 4.0, 5.0];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .apply_element(Add::new(1.0))
            .build_dynamic(5);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind(), ErrorKind::NaN));
        }
    }

    #[test]
    fn test_dynamic_buffer_reuse_correctness() {
        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(10);

        let in_buf1 = vec![1.0f32; 10];
        let mut out_buf1 = vec![0f32; 10];
        let n1 = pipe.execute(&in_buf1, &mut out_buf1).unwrap();
        assert_eq!(n1, 10);
        assert_eq!(out_buf1[0], 2.0);

        let in_buf2 = vec![5.0f32; 10];
        let mut out_buf2 = vec![0f32; 10];
        let n2 = pipe.execute(&in_buf2, &mut out_buf2).unwrap();
        assert_eq!(n2, 10);
        assert_eq!(out_buf2[0], 10.0);
    }

    // f64

    #[test]
    fn test_f64_element_pipeline() {
        let mut out_buf = vec![0f64; 5];
        let in_buf = vec![10.0f64, 20.0, 30.0, 40.0, 50.0];

        let mut pipe = Pipeline::new()
            .apply_element::<_, 5>(Div::new(10.0f64))
            .apply_element(Add::new(1.0f64))
            .apply_element(Multiply::new(2.0f64))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(out_buf[0], 4.0);
        assert_eq!(out_buf[1], 6.0);
        assert_eq!(out_buf[2], 8.0);
        assert_eq!(out_buf[3], 10.0);
        assert_eq!(out_buf[4], 12.0);
    }

    // Fusion coverage

    #[test]
    fn test_fused_transform_transform_transform() {
        let mut out_buf = vec![0f32; 5];
        let in_buf: Vec<f32> = (0..10).map(|i| i as f32).collect();

        let mut pipe = Pipeline::new()
            .apply_transform(Reverse::<10>)
            .apply_transform_fusable(Truncate::<10, 8>)
            .apply_transform_fusable(Truncate::<8, 5>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        // Reverse gives [9,8,7,6,5,4,3,2,1,0]
        // Truncate to 8 gives [9,8,7,6,5,4,3,2]
        // Truncate to 5 gives [9,8,7,6,5]
        assert_eq!(out_buf, vec![9.0, 8.0, 7.0, 6.0, 5.0]);
    }

    #[test]
    fn test_fusable_transform_then_element() {
        // Element op fused after a fused transform pair
        let mut out_buf = vec![0f32; 5];
        let in_buf: Vec<f32> = (0..10).map(|i| i as f32).collect();

        let mut pipe = Pipeline::new()
            .apply_transform(Reverse::<10>)
            .apply_transform_fusable(Truncate::<10, 5>)
            .apply_element(Multiply::new(2.0))
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        // Reverse + Truncate gives [9,8,7,6,5], then multiply by 2
        assert_eq!(out_buf, vec![18.0, 16.0, 14.0, 12.0, 10.0]);
    }

    #[test]
    fn test_element_before_fusable_transform() {
        // Element op precedes apply_transform_fusable
        let mut out_buf = vec![0f32; 5];
        let in_buf: Vec<f32> = (0..10).map(|i| i as f32).collect();

        let mut pipe = Pipeline::new()
            .apply_element::<_, 10>(Add::new(1.0))
            .apply_transform(Reverse::<10>)
            .apply_transform_fusable(Truncate::<10, 5>)
            .build();

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();

        assert_eq!(n, 5);
        // Add 1 gives [1,2,3,4,5,6,7,8,9,10]
        // Reverse gives [10,9,8,7,6,5,4,3,2,1]
        // Truncate to 5 gives [10,9,8,7,6]
        assert_eq!(out_buf, vec![10.0, 9.0, 8.0, 7.0, 6.0]);
    }

    // Dynamic pipeline coverage

    #[test]
    fn test_dynamic_static_transform_after_dynamic_head() {
        let mut out_buf = vec![0f32; 5];
        let in_buf = vec![1.0f32; 8];

        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .apply_transform(Truncate::<10, 5>)
            .build_dynamic(10);

        let result = pipe.execute(&in_buf, &mut out_buf);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(
                e.kind(),
                ErrorKind::InvalidInputSize
            ));
        }
    }

    #[test]
    fn test_dynamic_intermediate_buffer_sizing() {
        const ORIGINAL: usize = 5;
        const PADDED: usize = 20;
        const FINAL: usize = 3;

        let mut out_buf = vec![0f32; FINAL];
        let in_buf = vec![1.0f32; ORIGINAL];

        let mut pipe = Pipeline::with_dynamic()
            .apply_transform(Pad::<_, ORIGINAL, PADDED>::new(0.0))
            .apply_transform(Truncate::<PADDED, FINAL>)
            .build_dynamic(ORIGINAL);

        let n = pipe.execute(&in_buf, &mut out_buf).unwrap();
        assert_eq!(n, FINAL);
        assert_eq!(out_buf, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_dynamic_grow_then_shrink_repeated() {
        let mut pipe = Pipeline::with_dynamic()
            .apply_element(Multiply::new(2.0))
            .build_dynamic(100);

        // Small input
        let small_in = vec![1.0f32; 10];
        let mut small_out = vec![0f32; 10];
        let n = pipe.execute(&small_in, &mut small_out).unwrap();
        assert_eq!(n, 10);
        assert_eq!(small_out[0], 2.0);

        // Large input
        let large_in = vec![3.0f32; 80];
        let mut large_out = vec![0f32; 80];
        let n = pipe.execute(&large_in, &mut large_out).unwrap();
        assert_eq!(n, 80);
        assert_eq!(large_out[0], 6.0);

        // Small again
        let small_in2 = vec![5.0f32; 15];
        let mut small_out2 = vec![0f32; 15];
        let n = pipe.execute(&small_in2, &mut small_out2).unwrap();
        assert_eq!(n, 15);
        assert_eq!(small_out2[0], 10.0);

        // Large again
        let large_in2 = vec![7.0f32; 90];
        let mut large_out2 = vec![0f32; 90];
        let n = pipe.execute(&large_in2, &mut large_out2).unwrap();
        assert_eq!(n, 90);
        assert_eq!(large_out2[0], 14.0);
    }
}
