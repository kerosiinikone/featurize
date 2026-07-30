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
            input_data[idx] = (x as f32 / WIDTH as f32) * 255.0;
            input_data[idx + 1] = (y as f32 / HEIGHT as f32) * 255.0;
            input_data[idx + 2] = 128.0;
            input_data[idx + 3] = 255.0;
        }
    }

    let mut pipe = Pipeline::new()
        .apply_transform(
            Scale2D::<WIDTH, HEIGHT, CHANNELS, SCALED_WIDTH, SCALED_HEIGHT, CHANNELS> {},
        )
        .apply_transform(Grayscale::<SCALED_WIDTH, SCALED_HEIGHT, CHANNELS> {
            invert: false,
            ..Default::default()
        })
        .apply_point(Div {
            factor: 255.0,
            ..Default::default()
        })
        .apply_point(Multiply {
            factor: 1.2,
            ..Default::default()
        })
        .apply_transform(
            Truncate::<{ SCALED_WIDTH * SCALED_HEIGHT }, { SCALED_WIDTH * SCALED_HEIGHT }>,
        )
        .apply_transform_fusable(Truncate::<{ SCALED_WIDTH * SCALED_HEIGHT }, SCALED_WIDTH>)
        .apply_point(Clamp {
            min: 0.0,
            max: 1.0,
            nan_handling: featurize_core::errors::NanHandling::Fail,
        })
        .build();

    let mut output = vec![0.0f32; WIDTH * HEIGHT * 10];
    pipe.execute(&input_data, &mut output).unwrap();
}
