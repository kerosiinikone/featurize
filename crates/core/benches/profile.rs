use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use featurize_core::prelude::*;

const SRC_W: usize = 112;
const SRC_H: usize = 112;
const SRC_C: usize = 4;
const SRC_LEN: usize = SRC_W * SRC_H * SRC_C;

const WIDTH: usize = 28;
const HEIGHT: usize = 28;
const CHANNELS: usize = 1;
const IMAGE_LEN: usize = WIDTH * HEIGHT * CHANNELS;

const SCALED_LEN: usize = WIDTH * HEIGHT * SRC_C;

const BATCH: usize = 256;
const CANVAS_BATCH: usize = 64;

const MNIST_MEAN: f32 = 0.1307;
const MNIST_STD: f32 = 0.3081;

const CLAMP_MIN: f32 = -3.0;
const CLAMP_MAX: f32 = 3.0;

const LUMA_R: f32 = 0.299;
const LUMA_G: f32 = 0.587;
const LUMA_B: f32 = 0.114;

const EPSILON: f32 = 1e-5;

fn make_batch(len: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    let mut data = vec![0.0f32; len];

    for pixel in data.iter_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *pixel = ((state >> 16) & 0xFF) as f32;
    }

    data
}

fn set_opaque_alpha(data: &mut [f32]) {
    for pixel in data.chunks_exact_mut(SRC_C) {
        pixel[SRC_C - 1] = 255.0;
    }
}

#[inline(always)]
fn src_coord(out: usize, out_dim: usize, in_dim: usize) -> usize {
    out * in_dim / out_dim
}

fn naive_scale2d(input: &[f32], output: &mut [f32]) {
    for out_y in 0..HEIGHT {
        let in_y = src_coord(out_y, HEIGHT, SRC_H);

        for out_x in 0..WIDTH {
            let in_x = src_coord(out_x, WIDTH, SRC_W);

            let in_base = (in_y * SRC_W + in_x) * SRC_C;
            let out_base = (out_y * WIDTH + out_x) * SRC_C;

            output[out_base..(SRC_C + out_base)]
                .copy_from_slice(&input[in_base..(SRC_C + in_base)]);
        }
    }
}

fn naive_grayscale(input: &[f32], output: &mut [f32]) {
    for (pixel, dst) in output.iter_mut().enumerate() {
        let base = pixel * SRC_C;
        *dst = LUMA_R * input[base] + LUMA_G * input[base + 1] + LUMA_B * input[base + 2];
    }
}

fn naive_mnist(input: &[f32], output: &mut [f32]) {
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        let scaled = src / 255.0;
        let normalized = (scaled - MNIST_MEAN) / MNIST_STD;
        *dst = normalized.clamp(CLAMP_MIN, CLAMP_MAX);
    }
}

fn naive_canvas_mnist(input: &[f32], scratch: &mut [f32], output: &mut [f32]) {
    naive_scale2d(input, scratch);
    naive_grayscale(scratch, output);
    for value in output.iter_mut() {
        let scaled = *value / 255.0;
        let normalized = (scaled - MNIST_MEAN) / MNIST_STD;
        *value = normalized.clamp(CLAMP_MIN, CLAMP_MAX);
    }
}

fn verify_equivalence(label: &str, pipeline_out: &[f32], naive_out: &[f32]) {
    assert_eq!(
        pipeline_out.len(),
        naive_out.len(),
        "[{label}] pipeline and naive baseline produced different output lengths"
    );

    for (i, (a, b)) in pipeline_out.iter().zip(naive_out.iter()).enumerate() {
        assert!(
            (a - b).abs() <= EPSILON,
            "[{label}] pipeline diverged from the naive baseline at index {i}: {a} vs {b}"
        );
    }
}

fn bench_element_path(c: &mut Criterion) {
    let input = make_batch(BATCH * IMAGE_LEN);
    let mut pipeline_out = vec![0.0f32; BATCH * IMAGE_LEN];
    let mut naive_out = vec![0.0f32; BATCH * IMAGE_LEN];

    let mut pipe = Pipeline::new()
        .apply_element::<Div, IMAGE_LEN>(Div::new(255.0))
        .apply_element(Normalize::new(MNIST_STD, MNIST_MEAN))
        .apply_element(Clamp::new(CLAMP_MIN, CLAMP_MAX))
        .build();

    for (src, dst) in input
        .chunks_exact(IMAGE_LEN)
        .zip(pipeline_out.chunks_exact_mut(IMAGE_LEN))
    {
        pipe.execute(src, dst).unwrap();
    }
    naive_mnist(&input, &mut naive_out);
    verify_equivalence("mnist_preprocess", &pipeline_out, &naive_out);

    let mut group = c.benchmark_group("mnist_preprocess");
    group.throughput(Throughput::Elements((BATCH * IMAGE_LEN) as u64));

    group.bench_function("featurize_pipeline", |b| {
        b.iter(|| {
            for (src, dst) in std::hint::black_box(&input)
                .chunks_exact(IMAGE_LEN)
                .zip(pipeline_out.chunks_exact_mut(IMAGE_LEN))
            {
                pipe.execute(src, dst).unwrap();
            }
            std::hint::black_box(&pipeline_out);
        })
    });

    group.bench_function("naive_sequential_loop", |b| {
        b.iter(|| {
            for (src, dst) in std::hint::black_box(&input)
                .chunks_exact(IMAGE_LEN)
                .zip(naive_out.chunks_exact_mut(IMAGE_LEN))
            {
                naive_mnist(src, dst);
            }
            std::hint::black_box(&naive_out);
        })
    });

    group.finish();
}

/// Direct comparison of the monomorphized NaN policies on an otherwise
/// identical element pipeline.
///
/// `FailOnNan` keeps an early exit in the loop body (blocking
/// vectorization), `ZeroOnNan` lowers to a branchless select, and
/// `PropagateNan` emits no check at all -- it should track the naive
/// baseline in `mnist_preprocess`.
fn bench_nan_policy(c: &mut Criterion) {
    let input = make_batch(BATCH * IMAGE_LEN);

    let mut out_fail = vec![0.0f32; BATCH * IMAGE_LEN];
    let mut out_zero = vec![0.0f32; BATCH * IMAGE_LEN];
    let mut out_propagate = vec![0.0f32; BATCH * IMAGE_LEN];

    let mut pipe_fail = Pipeline::new_with::<f32, FailOnNan>()
        .apply_element::<Div, IMAGE_LEN>(Div::new(255.0))
        .apply_element(Normalize::new(MNIST_STD, MNIST_MEAN))
        .apply_element(Clamp::new(CLAMP_MIN, CLAMP_MAX))
        .build();

    let mut pipe_zero = Pipeline::new_with::<f32, ZeroOnNan>()
        .apply_element::<Div, IMAGE_LEN>(Div::new(255.0))
        .apply_element(Normalize::new(MNIST_STD, MNIST_MEAN))
        .apply_element(Clamp::new(CLAMP_MIN, CLAMP_MAX))
        .build();

    let mut pipe_propagate = Pipeline::new_with::<f32, PropagateNan>()
        .apply_element::<Div, IMAGE_LEN>(Div::new(255.0))
        .apply_element(Normalize::new(MNIST_STD, MNIST_MEAN))
        .apply_element(Clamp::new(CLAMP_MIN, CLAMP_MAX))
        .build();

    // The input is entirely finite, so all three policies must agree
    for (src, dst) in input
        .chunks_exact(IMAGE_LEN)
        .zip(out_fail.chunks_exact_mut(IMAGE_LEN))
    {
        pipe_fail.execute(src, dst).unwrap();
    }
    for (src, dst) in input
        .chunks_exact(IMAGE_LEN)
        .zip(out_zero.chunks_exact_mut(IMAGE_LEN))
    {
        pipe_zero.execute(src, dst).unwrap();
    }
    for (src, dst) in input
        .chunks_exact(IMAGE_LEN)
        .zip(out_propagate.chunks_exact_mut(IMAGE_LEN))
    {
        pipe_propagate.execute(src, dst).unwrap();
    }
    verify_equivalence("nan_policy(zero vs fail)", &out_zero, &out_fail);
    verify_equivalence("nan_policy(propagate vs fail)", &out_propagate, &out_fail);

    let mut group = c.benchmark_group("nan_policy");
    group.throughput(Throughput::Elements((BATCH * IMAGE_LEN) as u64));

    group.bench_function("fail_on_nan", |b| {
        b.iter(|| {
            for (src, dst) in std::hint::black_box(&input)
                .chunks_exact(IMAGE_LEN)
                .zip(out_fail.chunks_exact_mut(IMAGE_LEN))
            {
                pipe_fail.execute(src, dst).unwrap();
            }
            std::hint::black_box(&out_fail);
        })
    });

    group.bench_function("zero_on_nan", |b| {
        b.iter(|| {
            for (src, dst) in std::hint::black_box(&input)
                .chunks_exact(IMAGE_LEN)
                .zip(out_zero.chunks_exact_mut(IMAGE_LEN))
            {
                pipe_zero.execute(src, dst).unwrap();
            }
            std::hint::black_box(&out_zero);
        })
    });

    group.bench_function("propagate_nan", |b| {
        b.iter(|| {
            for (src, dst) in std::hint::black_box(&input)
                .chunks_exact(IMAGE_LEN)
                .zip(out_propagate.chunks_exact_mut(IMAGE_LEN))
            {
                pipe_propagate.execute(src, dst).unwrap();
            }
            std::hint::black_box(&out_propagate);
        })
    });

    group.finish();
}

fn bench_canvas_path(c: &mut Criterion) {
    let mut input = make_batch(CANVAS_BATCH * SRC_LEN);
    set_opaque_alpha(&mut input);

    let mut pipeline_out = vec![0.0f32; CANVAS_BATCH * IMAGE_LEN];
    let mut naive_out = vec![0.0f32; CANVAS_BATCH * IMAGE_LEN];
    let mut naive_scratch = vec![0.0f32; SCALED_LEN];

    let mut pipe = Pipeline::new_with::<f32, PropagateNan>()
        .apply_transform(Scale2D::<SRC_W, SRC_H, SRC_C, WIDTH, HEIGHT, SRC_C, f32>::new())
        .apply_transform(Grayscale::<WIDTH, HEIGHT, SRC_C, f32>::new())
        .apply_element(Div::new(255.0))
        .apply_element(Normalize::new(MNIST_STD, MNIST_MEAN))
        .apply_element(Clamp::new(CLAMP_MIN, CLAMP_MAX))
        .build();

    for (src, dst) in input
        .chunks_exact(SRC_LEN)
        .zip(pipeline_out.chunks_exact_mut(IMAGE_LEN))
    {
        pipe.execute(src, dst).unwrap();
    }
    for (src, dst) in input
        .chunks_exact(SRC_LEN)
        .zip(naive_out.chunks_exact_mut(IMAGE_LEN))
    {
        naive_canvas_mnist(src, &mut naive_scratch, dst);
    }
    verify_equivalence("mnist_canvas_rgba", &pipeline_out, &naive_out);

    let mut group = c.benchmark_group("mnist_canvas_rgba");
    group.throughput(Throughput::Elements((CANVAS_BATCH * SRC_LEN) as u64));

    group.bench_function("featurize_pipeline", |b| {
        b.iter(|| {
            for (src, dst) in std::hint::black_box(&input)
                .chunks_exact(SRC_LEN)
                .zip(pipeline_out.chunks_exact_mut(IMAGE_LEN))
            {
                pipe.execute(src, dst).unwrap();
            }
            std::hint::black_box(&pipeline_out);
        })
    });

    group.bench_function("naive_sequential_loop", |b| {
        b.iter(|| {
            for (src, dst) in std::hint::black_box(&input)
                .chunks_exact(SRC_LEN)
                .zip(naive_out.chunks_exact_mut(IMAGE_LEN))
            {
                naive_canvas_mnist(src, &mut naive_scratch, dst);
            }
            std::hint::black_box(&naive_out);
        })
    });

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_element_path(c);
    bench_nan_policy(c);
    bench_canvas_path(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
