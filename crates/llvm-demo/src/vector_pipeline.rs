use featurize_core::prelude::*;

const VECTOR_LEN: usize = 64;
const TRUNCATED_LEN: usize = 32;

fn main() {
    let mut input_data = vec![0.0f32; VECTOR_LEN];

    for i in 0..VECTOR_LEN {
        input_data[i] = (i as f32).sin() * 100.0 + 50.0;
    }

    let mut pipe = Pipeline::new()
        .apply_point(Normalize {
            mean: 50.0,
            std: 25.0,
        })
        .apply_point(Add { value: 1.0 })
        .apply_point(Abs)
        .apply_point(Pow { exponent: 0.5 })
        .apply_point(Multiply { factor: 2.0 })
        // TODO: this is not currently being checked
        .apply_transform(Truncate::<{ VECTOR_LEN + 1 }>)
        .apply_transform_fusable(Reverse::<{ VECTOR_LEN + 2 }>)
        .build::<VECTOR_LEN>();

    let output_size = pipe.output_len();
    let mut output = vec![0.0f32; output_size];

    pipe.execute(&input_data, &mut output);

    println!("Vector Pipeline Demo");
    println!("  output = {:?}", output);
    println!("  input = {:?}", input_data);
}
