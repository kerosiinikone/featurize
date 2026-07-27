use featurize_core::prelude::*;

const VECTOR_LEN: usize = 64;
const TRUNCATED_LEN: usize = 32;

fn main() {
    let mut input_data = vec![0.0f32; VECTOR_LEN];

    for i in 0..VECTOR_LEN {
        input_data[i] = (i as f32).sin() * 100.0 + 50.0;
    }

    let mut pipe = Pipeline::new()
        .apply_transform(Reverse::<{ VECTOR_LEN }>)
        .apply_transform_fusable(Truncate::<VECTOR_LEN, TRUNCATED_LEN>)
        .apply_point(Normalize {
            mean: 50.0,
            std: 25.0,
            ..Default::default()
        })
        .apply_point(Add {
            value: 1.0,
            ..Default::default()
        })
        .apply_point(Abs::default())
        .apply_point(Pow {
            exponent: 0.5,
            ..Default::default()
        })
        .apply_point(Multiply {
            factor: 2.0,
            ..Default::default()
        })
        .build();

    let mut output = vec![0.0f32; VECTOR_LEN];
    pipe.execute(&input_data, &mut output).unwrap();
}
