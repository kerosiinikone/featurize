use featurize_core::prelude::*;

const VECTOR_LEN: usize = 1024;
const TRUNCATED_LEN: usize = 512;

fn main() {
    let mut input_data = vec![0.0f32; VECTOR_LEN];
    
    for i in 0..VECTOR_LEN {
        input_data[i] = (i as f32).sin() * 100.0 + 50.0;
    }

    // 1. Normalize (subtract mean, divide by std)
    // 2. Add offset
    // 3. Take absolute value
    // 4. Apply power transformation
    // 5. Multiply by factor
    // 6. Truncate to half length
    // 7. Reverse the order
    let mut pipe = Pipeline::new()
        .apply_point::<Normalize, VECTOR_LEN>(Normalize { mean: 50.0, std: 25.0 })
        .apply_point(Add { value: 1.0 })
        .apply_point(Abs)
        .apply_point(Pow { exponent: 0.5 })
        .apply_point(Multiply { factor: 2.0 })
        .apply_transform(Truncate::<TRUNCATED_LEN>)
        .apply_transform(Reverse::<TRUNCATED_LEN>)
        .build();

    let output_size = pipe.output_shape();
    let mut output = vec![0.0f32; output_size];

    pipe.execute(&input_data, &mut output);

    println!("Vector Pipeline Demo");
    println!("Input length: {}", VECTOR_LEN);
    println!("Output length: {}", output_size);
    println!("First 10 output values:");
    for i in 0..10.min(output.len()) {
        println!("  output[{}] = {:.4}", i, output[i]);
    }
    println!("Last 10 output values:");
    for i in (output.len().saturating_sub(10))..output.len() {
        println!("  output[{}] = {:.4}", i, output[i]);
    }
    println!("Sum of all outputs: {:.4}", output.iter().sum::<f32>());
    println!("Mean of outputs: {:.4}", output.iter().sum::<f32>() / output.len() as f32);
}
