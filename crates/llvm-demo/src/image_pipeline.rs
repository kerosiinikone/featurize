use featurize_core::prelude::*;

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const CHANNELS: usize = 4;
const SCALED_WIDTH: usize = 32;
const SCALED_HEIGHT: usize = 32;

fn main() {
    let mut input_data = vec![0.0f32; WIDTH * HEIGHT * CHANNELS];

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y * WIDTH + x) * CHANNELS;
            input_data[idx] = (x as f32 / WIDTH as f32) * 255.0; // R
            input_data[idx + 1] = (y as f32 / HEIGHT as f32) * 255.0; // G
            input_data[idx + 2] = 128.0; // B
            input_data[idx + 3] = 255.0; // A
        }
    }

    // 1. Scale down from 64x64 to 32x32
    // 2. Convert to grayscale (4 channels -> 1 channel)
    // 3. Normalize (divide by 255)
    // 4. Apply contrast adjustment (multiply by 1.2)
    // 5. Clamp values to [0, 1]
    let mut pipe = Pipeline::new()
        .apply_transform::<Scale2D<WIDTH, HEIGHT, CHANNELS, SCALED_WIDTH, SCALED_HEIGHT, CHANNELS>, { WIDTH * HEIGHT * CHANNELS }>(Scale2D)
        .apply_transform(Grayscale::<SCALED_WIDTH, SCALED_HEIGHT, CHANNELS> { invert: false })
        .apply_point(Div { factor: 255.0 })
        .apply_point(Multiply { factor: 1.2 })
        .apply_point(Clamp { min: 0.0, max: 1.0 })
        .build();

    let output_size = pipe.output_shape();
    let mut output = vec![0.0f32; output_size];

    pipe.execute(&input_data, &mut output);

    println!("Image Pipeline Demo");
    println!("Input size: {}x{}x{}", WIDTH, HEIGHT, CHANNELS);
    println!(
        "Output size: {}x{} (grayscale)",
        SCALED_WIDTH, SCALED_HEIGHT
    );
    println!("First 10 output values:");
    for i in 0..10.min(output.len()) {
        println!("  output[{}] = {:.4}", i, output[i]);
    }
    println!("Sum of all outputs: {:.4}", output.iter().sum::<f32>());
}
